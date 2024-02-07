import os
import pickle
import os.path as osp
from collections import defaultdict
import random

from dassl.data.datasets import Datum, DatasetBase, DATASET_REGISTRY
from dassl.utils import mkdir_if_missing



@DATASET_REGISTRY.register()
class PACSFS(DatasetBase):
    """PACS.
    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.
    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """

    dataset_dir = "pacs"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        SOURCE_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
        TARGET_DOMAINS = [SOURCE_DOMAINS.pop(cfg.DATASET.ID)]

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            SOURCE_DOMAINS, TARGET_DOMAINS
        )

        train = self._read_data(SOURCE_DOMAINS, "train")
        val = self._read_data(SOURCE_DOMAINS, "crossval")
        test = self._read_data(TARGET_DOMAINS, "all")

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

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(
                    self.split_dir, dname + "_train_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + "_crossval_kfold.txt"
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + "_" + split + "_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

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