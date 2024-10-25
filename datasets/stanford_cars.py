import os

import scipy.io as sio
import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.stanfordcars_config import classes, templates
from datasets.utils.make_dataset_train import make_image_text


class StanfordCars(Dataset.StanfordCars):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, make_typographic_dataset=False):
        super().__init__(root, split, transform, target_transform, download)
        
        self.original_classes = self.classes
        self._typographic_images_base_path = self._base_folder / 'typographic_images'
        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                str(self._typographic_images_base_path / annotation["fname"]),
                classes.index(self.original_classes[annotation["class"] - 1]),  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]
        self.classes = classes 
        self.templates = templates

        if make_typographic_dataset:
            self.make_typographic_dataset()
    
    def __getitem__(self, idx):
        image_path, typographic_image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert('RGB')
        typographic_pil_image = Image.open(typographic_image_path).convert('RGB')
        if self.transform is not None:
            pil_image, typographic_pil_image = self.transform(pil_image), self.transform(typographic_pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return pil_image, typographic_pil_image, target

    def make_typographic_dataset(self):
        if self._check_typographic_exists():
            return 

        self._typographic_images_base_path.mkdir()
        for i, (file, _, label_idx) in enumerate(self._samples):
            make_image_text(os.path.basename(file), self.classes, self._images_base_path, self._typographic_images_base_path, label_idx)
            
    def _check_typographic_exists(self):
        return self._typographic_images_base_path.exists()
