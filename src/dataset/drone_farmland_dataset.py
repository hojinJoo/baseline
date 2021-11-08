import json
import tifffile
import os
import numpy as np
import torch
import pprint
from torch.utils.data import Dataset


class DroneFarmlandDataset(Dataset):

    __version__ = 'v1.20211108'

    def __init__(self, meta_json_path, root_dir, transform=None):
        """
        Args:
            meta_json_path (string): Path to the meta json file with paths
            root_dir (string): Directory with images, labels, json files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(meta_json_path, 'r') as json_file:
            self.metadata = json.load(json_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        meta_path = self.metadata[idx]
        image_path = os.path.join(self.root_dir, meta_path['image_path'])
        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()
            image = image.astype(np.float64)
            image = image - image.min()
            image = image / image.max()

        label_path = os.path.join(self.root_dir, meta_path['label_path'])
        with tifffile.TiffFile(label_path) as tif:
            label = tif.asarray()
            label = label.astype(np.float64)
            label = label - label.min()
            label = label / label.max()
        
        json_path = os.path.join(self.root_dir, meta_path['json_path'])
        with open(json_path, 'r') as json_file:
            meta_info = json.load(json_file)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        else:
            image = torch.Tensor(image)
            label = torch.Tensor(label)
        
        return image, label, meta_info


if __name__ == '__main__':
    dfd = DroneFarmlandDataset(
        "data/drone_farmland_semantic_segmentation/meta_valid_v1.json",
        "data/drone_farmland_semantic_segmentation",
    )
    print(len(dfd))
    print(pprint.pformat(dfd[0]))
