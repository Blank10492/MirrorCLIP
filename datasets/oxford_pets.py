import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.oxfordpets_config import CNAME, classes, templates
from datasets.utils.make_dataset_train import make_image_text


class OxfordIIITPet(Dataset.OxfordIIITPet):
    def __init__(self, root, split='trainval', transform=None, target_transform=None, download=False, make_typographic_dataset=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        
        self.original_classes = self.classes

        self._typographic_images_folder = self._base_folder / 'typographic_images'
        self._typographic_images = [self._typographic_images_folder / image.name for image in self._images]
        
        self._labels = [
            classes.index(CNAME[self.original_classes[label]]) 
            if self.original_classes[label] in CNAME.keys() 
            else classes.index(self.original_classes[label])
                for label in self._labels
        ]

        self.classes = classes 
        self.templates = templates

        if make_typographic_dataset:
            self.make_typographic_dataset()
    
    def __getitem__(self, idx):
        image = Image.open(self._images[idx]).convert("RGB")
        typographic_image = Image.open(self._typographic_images[idx]).convert("RGB")
        target = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transform:
            image, typographic_image = self.transform(image), self.transform(typographic_image)

        return image, typographic_image, target

    def make_typographic_dataset(self):
        if self._check_typographic_exists():
            return 

        self._typographic_images_folder.mkdir()
        for i, file in enumerate(self._images):
            make_image_text(file.name, self.classes, self._images_folder, self._typographic_images_folder, self._labels[i])
            
    def _check_typographic_exists(self):
        return self._typographic_images_folder.exists()