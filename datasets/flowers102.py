import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.flowers102_config import classes, templates
from datasets.utils.make_dataset_train import make_image_text


class Flowers102(Dataset.Flowers102):
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False, make_typographic_dataset=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        self._typographic_images_folder = self._base_folder / 'typographic_images'
        self._typographic_image_files = [self._typographic_images_folder / image.name for image in self._image_files]

        self.classes = classes 
        self.templates = templates

        if make_typographic_dataset:
            self.make_typographic_dataset()
    
    def __getitem__(self, idx):
        image_file, typographic_image_file ,label = self._image_files[idx], self._typographic_image_files[idx] ,self._labels[idx]
        image = Image.open(image_file).convert("RGB")
        typographic_image_file = Image.open(typographic_image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)
            typographic_image_file = self.transform(typographic_image_file)

        if self.target_transform:
            label = self.target_transform(label)

        return image, typographic_image_file, label
    
    def make_typographic_dataset(self):
        if self._check_typographic_exists():
            return 

        self._typographic_images_folder.mkdir()
        for i, file in enumerate(self._image_files):
            make_image_text(file.name, self.classes, self._images_folder, self._typographic_images_folder, self._labels[i])
            
    def _check_typographic_exists(self):
        return self._typographic_images_folder.exists()
