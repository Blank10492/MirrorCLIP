import os
import pathlib
import shutil
import tarfile

import requests
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from datasets.configs.imagenet_config import classes, template

from .utils.make_dataset_train import make_image_text

# URLS = {"matched-frequency" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
#         "threshold-0.7" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold0.7.tar.gz",
#         "top-images": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-top-images.tar.gz",
#         "val": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenet_validation.tar.gz"}
URLS = {"matched-frequency" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main/imagenetv2-top-images.tar.gz",
        "val": "https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}


V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000

class ImageNetValDataset(Dataset):
    def __init__(self, transform=None, location="."):
        os.makedirs(f"{location}/imagenetv2", exist_ok=True)

        self.dataset_root = pathlib.Path(f"{location}/imagenetv2/imagenet_validation/")
        self.typographic_root = pathlib.Path(f"{location}/imagenetv2/typographic_images/")
        self.tar_root = pathlib.Path(f"{location}/imagenetv2/imagenet_validation.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.JPEG"))
        self.transform = transform
        if not self.dataset_root.exists() or len(self.fnames) != VAL_DATASET_SIZE:
            print(len(self.fnames))
            if not self.tar_root.exists():
                print("Dataset imagenet-val not found on disk, downloading....")
                # response = requests.get(URLS["val"], stream=True)
                response = requests.get(URLS["top-images"], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, "Downloading failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES['val']}", self.dataset_root)

        self.typographic_fnames = [self.typographic_root / file.relative_to(self.dataset_root) for file in self.fnames]
        self.classes = classes
        self.templates = template

        self.dataset = ImageFolder(str(self.dataset_root))
        self._labels = []
        for image, label in self.dataset:
            self._labels.append(label)


        self._make_typographic_attack_dataset()

        self.typographic_dataset = ImageFolder(str(self.typographic_root))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        (img, label), (typographic_img, label2) = self.dataset[i], self.typographic_dataset[i]
        if label != label2:
            print("error")
        if self.transform is not None:
            img = self.transform(img)
            typographic_img = self.transform(typographic_img)
        return img, typographic_img, label

    def _check_exists_synthesized_dataset(self) -> bool:
        return self.typographic_root.is_dir()

    def _make_typographic_attack_dataset(self) -> None:
        if self._check_exists_synthesized_dataset():
            return
        for i, file in enumerate(self.fnames):
            print(file)
            make_image_text(file.relative_to(self.dataset_root), self.classes, self.dataset_root, self.typographic_root, self._labels[i])