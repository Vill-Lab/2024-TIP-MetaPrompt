import os.path as osp
import random

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from copy import deepcopy



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class VisionEncoder(nn.Module):
    def __init__(self, cfg, clip_model): #, image_weight
        super().__init__()
        visual = clip_model.visual  # CLIP's visual encoder
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.layers = len(self.transformer)
        self.n_pro = cfg.TRAINER.META.N_PRO
        self.layer_p = cfg.TRAINER.META.LAYERS

    def forward(self, x, ctx_v):
        x = torch.cat([x, ctx_v[:, 0, :, :]], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        for i in range(self.layers):
            if 1 <= i < self.layer_p:
                ctx = ctx_v[:, i].permute(1, 0, 2)
                prefix = x[:-self.n_pro, :, :]
                x = torch.cat([prefix, ctx], dim=0)
            x = self.transformer[i](x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_pro = cfg.TRAINER.META.N_PRO
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.visual = clip_model.visual
        self.conv1 = self.visual.conv1
        self.class_embedding = self.visual.class_embedding
        self.positional_embedding = self.visual.positional_embedding
        self.layers = len(self.visual.transformer.resblocks)
        self.layer_p = cfg.TRAINER.META.LAYERS

        ctx_vectors = torch.empty(self.layer_p, n_pro, ctx_dim, dtype=self.dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        
    def forward(self, x):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(len(x), -1, -1, -1)

        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.type(self.dtype)
        
        return x, ctx


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.layers = len(clip_model.transformer.resblocks)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.layers = len(self.transformer)
        self.n_ctx = cfg.TRAINER.META.N_CTX
        self.layer_p = cfg.TRAINER.META.LAYERS

    def forward(self, prompts, tokenized_prompts, ctx_t):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        for i in range(self.layers):
            if 1 <= i < self.layer_p:
                ctx = ctx_t[:, i].permute(1, 0, 2)
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx:, :, :]
                x = torch.cat([prefix, ctx, suffix], dim=0) 
            x = self.transformer[i](x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.META.N_CTX
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = len(clip_model.transformer.resblocks)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.layer_p = cfg.TRAINER.META.LAYERS

        ctx_vectors = torch.empty(self.layer_p, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + n_ctx :, :]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx[:, 0], suffix], dim=1)
        return prompts, ctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        for p in clip_model.parameters():
            p.requires_grad = False

        t = {
            'StanfordCars':'a photo of a {}',
            'Caltech101':'a photo of a {}',
            'OxfordFlowers':'a photo of a {}, a type of flower',
            'EuroSAT':'a centered satellite photo of {}',
            'DescribableTextures':'{} texture',
            'Food101':'a photo of a {}, a type of food',
            'FGVCAircraft':'a photo of a {}, a type of aircraft',
            'OxfordPets':'a photo of a {}, a type of pet',
            'UCF101':'a photo of a person doing {}',
            'SUN397':'a photo of a {}',
            'PACS':'a photo of a {}',
            'OfficeHomeFS':'a photo of a {}',
            'PACSFS':'a photo of a {}',
            'VLCSFS':'a photo of a {}',
            'DomainNetFS':'a photo of a {}',
            'TERRAFS':'a photo of a {}'
        }

        if 'ImageNet' not in cfg.DATASET.NAME:
            templates = [t[cfg.DATASET.NAME]]
        else:
            templates = [
                'itap of a {}',
                'a bad photo of the {}',
                'a origami {}',
                'a photo of the large {}',
                'a {} in a video game',
                'art of the {}',
                'a photo of the small {}',
                'a photo of a {}'
            ]

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.vision_prompt_learner = VisionPromptLearner(cfg, clip_model)
        self.image_encoder = VisionEncoder(cfg, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.model = clip_model
        self.classname = classnames

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts) 
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self.text_features_zs = torch.stack(zeroshot_weights, dim=1).cuda()
        
        

    def forward(self, image, label_idx = None):
        if label_idx != None:
            text_features_zs = self.text_features_zs[:, label_idx]
        else:
            text_features_zs = self.text_features_zs

        image_features_zs = self.model.encode_image(image.type(self.dtype))
        image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)

        x, ctx_v = self.vision_prompt_learner(image)
        image_features = self.image_encoder(x, ctx_v)
        prompts, ctx_t = self.prompt_learner()
        
        tokenized_prompts = self.tokenized_prompts
        if label_idx != None:
            prompts = prompts[label_idx]
            ctx_t = ctx_t[label_idx]
            tokenized_prompts = tokenized_prompts[label_idx]

        text_features = self.text_encoder(prompts, tokenized_prompts, ctx_t)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        logits_i = logit_scale * (image_features @ text_features_zs)
        logits_t = logit_scale * (image_features_zs @ text_features.t())
        logits = 0.5*(logits_i + logits_t)

        if self.training:
            return logits , logits_i, logits_t
        else:
            return logits



@TRAINER_REGISTRY.register()
class Meta_B2N(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.META.PREC in ["fp16", "fp32", "amp"]


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).cuda()
        
        if cfg.TRAINER.META.PREC == "fp32" or cfg.TRAINER.META.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        self.meta_step = cfg.TRAINER.META.META_STEP

        self.scaler = GradScaler() if cfg.TRAINER.META.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1: 
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits_mix, logits_a, logits_b = self.model(image)
        loss = F.cross_entropy(logits_mix, label)
        loss_a = F.cross_entropy(logits_a, label)
        loss_b = F.cross_entropy(logits_b, label)
        loss = loss + loss_a + loss_b

        self.model_backward_and_update(loss)
        label_mix = label

        label_lst = label.tolist()
        label_set = list(range(len(self.dm.dataset.classnames)))
        random.shuffle(label_set)

        N = 2
        l = len(label_set)
        m = (l-1) // N + 1
        cls_li = [label_set[i*m:(i+1)*m] for i in range(N)]
        relabeler = [{y: y_new for y_new, y in enumerate(cls_li[i])} for i in range(N)]
        index = [[j for j, l in enumerate(label_lst) if l in cls_li[i]] for i in range(N)]
        label_li = [torch.tensor([relabeler[i][j] for j in label[index[i]].tolist()]).cuda() for i in range(N)]
        image_li = [image[index[i]].cuda() for i in range(N)]

        self.model.vision_prompt_learner.ctx.requires_grad_(False)
        textual_prompt = deepcopy(self.model.prompt_learner.ctx)

        grad_l = []
        for i in range(N):

            image = torch.cat([image_li[j] for j in range(N) if j != i])
            label = torch.cat([label_li[j] for j in range(N) if j != i])
            cls = [item for j in range(N) if j != i for item in cls_li[j]]
            
            if len(cls_li[i]) < 2 or len(image_li[i]) == 0 or len(image) == 0:
                continue

            _, _, logits_b = self.model(image, cls)
            loss = F.cross_entropy(logits_b, label)
            self.model_backward_and_update(loss)

            _, _, logits_b = self.model(image_li[i], cls_li[i])
            loss = F.cross_entropy(logits_b, label_li[i])

            grads = autograd.grad(loss, self.model.prompt_learner.ctx, create_graph=False)[0]
            grad_l.append(grads)
            self.model.prompt_learner.ctx.data = textual_prompt

        for grad in grad_l:
            self.model.prompt_learner.ctx.data -= self.get_current_lr()*grad*self.meta_step
        
        self.model.vision_prompt_learner.ctx.requires_grad_(True)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits_mix, label_mix)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            self._models[name].load_state_dict(state_dict, strict=False)



@TRAINER_REGISTRY.register()
class Meta_DG(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.META.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).cuda()
        
        if cfg.TRAINER.META.PREC == "fp32" or cfg.TRAINER.META.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        self.meta_step = cfg.TRAINER.META.META_STEP

        self.scaler = GradScaler() if cfg.TRAINER.META.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1: 
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)

        logits_mix, logits_a, logits_b = self.model(image)
        loss = F.cross_entropy(logits_mix, label)
        loss_a = F.cross_entropy(logits_a, label)
        loss_b = F.cross_entropy(logits_b, label)
        loss = loss + loss_a + loss_b
        self.model_backward_and_update(loss)

        dd = domain.tolist()
        N = len(set(dd))
        
        train_dm = [list(range(N)) for _ in range(N)]
        test_dm = [[train_dm[i].pop(i)] for i in range(N)]

        index_train = [[i for i, l in enumerate(dd) if l in train_dm[j]] for j in range(N)]
        index_test = [[i for i, l in enumerate(dd) if l in test_dm[j]] for j in range(N)]

        label_train = [label[index_train[i]] for i in range(N)]
        label_test = [label[index_test[i]] for i in range(N)]

        image_train = [image[index_train[i]] for i in range(N)]
        image_test = [image[index_test[i]] for i in range(N)]

        self.model.prompt_learner.ctx.requires_grad_(False)
        vision_prompt = deepcopy(self.model.vision_prompt_learner.ctx)

        grad_l = []
        for i in range(N): 
            
            if image_train[i].shape[0] == 0 or image_test[i].shape[0] == 0:
                continue
            
            _, logits_a, _ = self.model(image_train[i])
            loss = F.cross_entropy(logits_a, label_train[i])
            self.model_backward_and_update(loss)

            _, logits_a, _ = self.model(image_test[i])
            loss = F.cross_entropy(logits_a, label_test[i])

            grads = autograd.grad(loss, self.model.vision_prompt_learner.ctx, create_graph=False)[0]
            grad_l.append(grads)
            self.model.vision_prompt_learner.ctx.data = vision_prompt
            
        for grad in grad_l:
            self.model.vision_prompt_learner.ctx.data -= self.meta_step*self.get_current_lr()*grad
        
        self.model.prompt_learner.ctx.requires_grad_(True)
        
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits_mix, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        domain = batch['domain'].to(self.device)
        return input, label, domain

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
