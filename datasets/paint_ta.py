import os

from PIL import Image
from torch.utils.data import Dataset


class PAINTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'typographic_real_data_from_PAINT/typographic_attacks')

        self.transform = transform 

        self.templates = ["a photo of a {}."]
        self.img_files = []
        
        self.true_labels = []
        class_set = set()
        for img in os.listdir(self.img_dir):
            self.img_files.append(img)
            label = img.split('_')[0].split('=')[1]
            text = img.split('_')[1].split('=')[1][:-4]
            self.true_labels.append(label)
            class_set.add(label)
            class_set.add(text)
        self.classes = list(class_set)
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        true_label = self.true_labels[idx]
        return (image, image, self.classes.index(true_label))