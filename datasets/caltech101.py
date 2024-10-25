import os
from pathlib import Path

import torchvision.datasets as Dataset
from PIL import Image

from datasets.configs.caltech101_config import CNAME, classes, templates
from datasets.utils.make_dataset_train import make_image_text


class Caltech101(Dataset.Caltech101):
    def __init__(self, root, transform=None, target_transform=None, download=False, make_typographic_dataset=False):
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        
        self.original_categories = self.categories

        self.original_y = self.y
        self.y = [
            classes.index(CNAME[self.original_categories[y_i]])
            if self.original_categories[y_i] in CNAME.keys()
            else classes.index(self.original_categories[y_i])
                for y_i in self.original_y
        ]

        self.classes = classes 
        self.templates = templates

        if make_typographic_dataset:
            self.make_typographic_dataset()
    
    def __getitem__(self, index):
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.original_y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )
        typographic_img = Image.open(
            os.path.join(
                self.root,
                "typographic_images",
                self.categories[self.original_y[index]],
                f"image_{self.index[index]:04d}.jpg"
            )
        )

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)
            typographic_img = self.transform(typographic_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, typographic_img, target

    def make_typographic_dataset(self):
        typographic_dir = os.path.join(self.root, "typographic_images")
        if self._check_typographic_exists(typographic_dir):
            return 

        os.mkdir(typographic_dir)
        for index in range(len(self.original_y)):
            images_folder = os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.original_y[index]]
            )
            target_dir = os.path.join(typographic_dir, self.categories[self.original_y[index]])
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            make_image_text(f"image_{self.index[index]:04d}.jpg", self.classes, Path(images_folder), Path(target_dir), self.original_y[index])


    
    def _check_typographic_exists(self, typographic_dir):
        return os.path.exists(typographic_dir)