from pathlib import Path

import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.dtd_config import templates
from datasets.utils.make_dataset_train import make_image_text


class DTD(Dataset.DTD):
    def __init__(self, root, split='test', transform=None, download=False):
        super().__init__(root, split, transform=transform, download=download)
     
        self._typographic_images_folder = self._data_folder / "typographic_images"
  
        self._typographic_image_files = []
        self._base_image_files = []

        with open(self._meta_folder / f"{self._split}{self._partition}.txt") as file:
            for line in file:
                cls, name = line.strip().split("/")
                self._typographic_image_files.append(self._typographic_images_folder.joinpath(cls, name))
                self._base_image_files.append(Path(cls+'/'+name))

        self.classes = [' '.join(class_i.split('_')) for class_i in self.classes]
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

    def _check_exists_synthesized_dataset(self) -> bool:
        return self._typographic_images_folder.is_dir()

    def _make_typographic_attack_dataset(self):
        if self._check_exists_synthesized_dataset():
            return
        for i, file in enumerate(self._base_image_files):
            print(file)
            print(self._typographic_images_folder)
            
            make_image_text(str(file), self.classes, self._images_folder, self._typographic_images_folder, self._labels[i])
 