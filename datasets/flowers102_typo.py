import torchvision.datasets as Dataset
from PIL import Image
import os
import json

from datasets.configs.flowers102_config import classes, templates
from datasets.utils.make_dataset_train import make_image_text


class Flowers102Typo(Dataset.Flowers102):
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False, make_typographic_dataset=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        self._typographic_images_folder = self._base_folder / 'typographic_images_typo'
        self._typographic_image_files = [self._typographic_images_folder / image.name for image in self._image_files]

        self.classes = classes 
        # self.templates = templates
        self.templates = ['text of {}.']

        if make_typographic_dataset:
            self.make_typographic_dataset()
        
        with open (os.path.join(self._base_folder, 'typo_dict.json'),'r') as f:
            self.typo_dict = json.load(f)
    
    def __getitem__(self, idx):
        image_file, typographic_image_file ,label = self._image_files[idx], self._typographic_image_files[idx] ,self._labels[idx]
        image = Image.open(image_file).convert("RGB")
        typographic_image_file = Image.open(typographic_image_file).convert("RGB")
        
        label = self.classes.index(self.typo_dict[str(image_file.name)])
        if self.transform:
            image = self.transform(image)
            typographic_image_file = self.transform(typographic_image_file)

        if self.target_transform:
            label = self.target_transform(label)

        return image, typographic_image_file, label
    
    def make_typographic_dataset(self):
        if self._check_typographic_exists():
            return 

        typo_dict = {}
        self._typographic_images_folder.mkdir()
        for i, file in enumerate(self._image_files):
            text = make_image_text(file.name, self.classes, self._images_folder, self._typographic_images_folder, self._labels[i])
            typo_dict[str(file.name)] = text
        
        with open (os.path.join(self._base_folder, 'typo_dict.json'),'w') as f:
            json.dump(typo_dict,f)
            
    def _check_typographic_exists(self):
        return self._typographic_images_folder.exists()
