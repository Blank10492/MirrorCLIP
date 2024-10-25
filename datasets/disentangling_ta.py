import json
import os

from PIL import Image
from torch.utils.data import Dataset


class DisentanglingDataset(Dataset):
    def __init__(self, root, transform=None):
        root_dir = os.path.join(root, 'typographic_attack_dataset')
        annotations_file = os.path.join(root_dir, 'annotations.json')
        classes_file =  os.path.join(root_dir, 'classes.json')
        img_dir = os.path.join(root_dir, 'images')

        self.img_dir = img_dir 
        self.transform = transform 
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        with open(classes_file) as f:
            self.classes_dict = json.load(f)
        self.classes = list(self.classes_dict.keys())
        self.templates = ["a photo of {}."]
        self.img_files = []
        for key in self.annotations:
            self.img_files.append(key)
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
            
        true_label = self.annotations[img_file]['true object']
        self.annotations[img_file]['typographic attack label']

        return (image, image, self.classes.index(true_label))