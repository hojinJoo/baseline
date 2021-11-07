import csv
import json
import random
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class KoreanFoodDataset(Dataset):

    __version__ = 'v1.20210914'

    def __init__(self, json_path, root_dir, transform=None):
        """
        Args:
            json_path (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_path,'r') as json_file:
            self.metadata = json.load(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata['annotations'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = self.metadata["annotations"][idx]["file_name"]
        category_idx = self.metadata["annotations"][idx]["category_idx"]

        img_name = os.path.join(self.root_dir, file_name)
        try:
            image = Image.open(img_name)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = category_idx
            label = np.array([label])
            label = label.astype('int')
        except Exception as E:
            print(f'[WARN] Wrong image on {file_name}, dummy image will be provided, related error is, "{E}"')
            image = torch.randn(3, 224, 224)
            label = np.random.randn(1).astype('int')
        return image, label


class KoreanFoodInferenceDataset(Dataset):

    __version__ = 'v1.20210914'

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_path_list = list(sorted(os.listdir(root_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = self.image_path_list[idx]

        img_name = os.path.join(self.root_dir, file_name)
        image_id = img_name.split('/')[-1]
        try:
            image = Image.open(img_name)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as E:
            print(f'[WARN] Wrong image on {file_name}, dummy image will be provided, related error is, "{E}"')
            image = torch.randn(3, 224, 224)
        return image, image_id
    

if __name__ == '__main__':
    kfd = KoreanFoodDataset("data/train/train/train.json", "data/train/train")
    print(len(kfd))
    print(kfd[0])
    kfd_inference = KoreanFoodInferenceDataset("data/test/test")
    print(len(kfd_inference))
    print(kfd_inference[0])