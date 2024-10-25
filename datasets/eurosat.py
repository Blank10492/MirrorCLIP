import json
import re
import ssl
from pathlib import Path

import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.eurosat_config import templates
from datasets.utils.make_dataset_train import make_image_text


def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSAT(Dataset.EuroSAT):
    def __init__(
        self,
        root: str,
        transform = None,
        target_transform = None,
        download: bool = False,
        split: str = 'train'
    ) -> None:
        ssl._create_default_https_context = ssl._create_unverified_context

        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        
        self._base_folder = Path(root) / "eurosat"
        self._data_folder = Path(self._base_folder) / "2750"
        self._typographic_data_folder = Path(self._base_folder) / "2750_typographic_images"
        self._split = Path(self.root) / "configs" / "split_zhou_EuroSAT.json"

        self.classes = [pretify_classname(c) for c in self.classes]
        self.templates = templates

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        dict((v, k)
                            for k, v in self.class_to_idx.items())

        with open(self._split) as f:
            split_dict = json.load(f)
        self._all_image_files = [v[0] for v in split_dict["train"]] + [v[0] for v in split_dict["test"]]
        self._all_labels = [v[1] for v in split_dict["train"]] + [v[1] for v in split_dict["test"]]
        
        if split == 'train':
            self._split_image_files = [v[0] for v in split_dict["train"]]
            self._split_labels = [v[1] for v in split_dict["train"]]
        else:
            self._split_image_files = [v[0] for v in split_dict["test"]]
            self._split_labels = [v[1] for v in split_dict["test"]]

        self._make_typographic_attack_dataset()

    def __len__(self) -> int:
        return len(self._split_image_files)

    def __getitem__(self, idx):
        image_file, typographic_image_file, label = self._data_folder / self._split_image_files[idx], self._typographic_data_folder / self._split_image_files[idx], self._split_labels[idx]
        image, typographic_image = Image.open(image_file).convert("RGB"), Image.open(typographic_image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)
            typographic_image = self.transform(typographic_image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, typographic_image ,label
    
    def _check_exists_synthesized_dataset(self) -> bool:
        return self._typographic_data_folder.is_dir()

    def _make_typographic_attack_dataset(self) -> None:
        if self._check_exists_synthesized_dataset():
            return
        for i, file in enumerate(self._split_image_files):
            print(file)
            
            make_image_text(file, self.classes, self._data_folder, self._typographic_data_folder, self._split_labels[i])
            