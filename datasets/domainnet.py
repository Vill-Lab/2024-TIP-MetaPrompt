import os.path as osp
import os
import pickle
import random
from collections import defaultdict

from dassl.data.datasets import Datum, DatasetBase, DATASET_REGISTRY
from dassl.utils import mkdir_if_missing, listdir_nohidden


@DATASET_REGISTRY.register()
class DomainNetFS(DatasetBase):
    """DomainNet.
    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.
    Special note: the t-shirt class (327) is missing in painting_train.txt.
    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    dataset_dir = "domainnet"
    domains = [
        "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits")

        SOURCE_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        TARGET_DOMAINS = [SOURCE_DOMAINS.pop(cfg.DATASET.ID)]

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(SOURCE_DOMAINS, split="train")
        val = self._read_data(SOURCE_DOMAINS, split="test")
        test = self._read_data(TARGET_DOMAINS, split="test")
        
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"{cfg.DATASET.ID}_shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self._generate_fewshot_dataset(train, num_shots=num_shots)
                val = self._generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items
    
    
    def _generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).
        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.
        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset(data_source)
            dataset = []
            for domain, dict in tracker.items():
                for label, items in dict.items():
                    if len(items) >= num_shots:
                        sampled_items = random.sample(items, num_shots)
                    else:
                        if repeat:
                            sampled_items = random.choices(items, k=num_shots)
                        else:
                            sampled_items = items
                    dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.
        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(dict)

        for item in data_source:
            if item.label not in output[item.domain]:
                output[item.domain][item.label] = [item]
            else:
                output[item.domain][item.label].append(item)

        return output