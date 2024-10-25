import os
from pathlib import Path

import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.fgvcaircraft_config import templates
from datasets.utils.make_dataset_train import make_image_text


class FGVCAircraft(Dataset.FGVCAircraft):
    def __init__(self, root, split='test', transform=None, download=False):
        super().__init__(root, split, transform=transform, download=download)

        self._image_data_folder = os.path.join(self._data_path, "data", "images")
        self._typographic_image_data_folder = os.path.join(self._data_path, "data", "typographic_images")
  
        labels_file = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self._split}.txt")
        self._typographic_image_files = []
        self._base_image_files = []

        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._typographic_image_files.append(os.path.join(self._typographic_image_data_folder, f"{image_name}.jpg"))
                self._base_image_files.append(f"{image_name}.jpg")

        self._make_typographic_attack_dataset()

        self.templates = templates

    def __getitem__(self, idx):
        image_file, typographic_image_file, label = self._image_files[idx], self._typographic_image_files[idx], self._labels[idx]
        image, typographic_image = Image.open(image_file).convert("RGB"), Image.open(typographic_image_file).convert("RGB")

        if self.transform:
            image, typographic_image = self.transform(image), self.transform(typographic_image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, typographic_image, label

    def _check_exists_synthesized_dataset(self):
        return os.path.exists(self._typographic_image_data_folder)

    def _make_typographic_attack_dataset(self):
        if self._check_exists_synthesized_dataset():
            return
        for i, file in enumerate(self._base_image_files):
            make_image_text(file, self.classes, Path(self._image_data_folder), Path(self._typographic_image_data_folder), self._labels[i])
            
            self._typographic_image_files.append(os.path.join(self._typographic_image_data_folder, os.path.basename(file)))