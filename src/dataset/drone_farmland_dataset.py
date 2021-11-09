import json
import tifffile
import os
import numpy as np
import torch
import logging
import pprint
import cv2
from pathlib import Path
from torch.utils.data import Dataset


class DroneFarmlandDataset(Dataset):

    __version__ = 'v1.20211108'

    def __init__(self, cfg, meta_json_path, root_dir, transform=None, inference=False):
        """
        Args:
            meta_json_path (string): Path to the meta json file with paths
            root_dir (string): Directory with images, labels, json files
            transform (callable, optional): Optional transform to be applied
                on a sample.
            inference (bool): If True, provide input only
        """
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.inference = inference
        if not self.inference:
            with open(meta_json_path, 'r') as json_file:
                self.metadata = json.load(json_file)
        else:
            self.metadata = []
            for image_path in list(sorted(list(Path(self.root_dir).iterdir()))):
                self.metadata.append(
                    dict(
                        image_path=image_path,
                    )
                )
        self.length = len(self.metadata)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        if self.inference:

            meta_path = self.metadata[idx]
            image_path = meta_path['image_path']
            with tifffile.TiffFile(image_path) as tif:
                image = tif.asarray()
                image = image.astype(np.float64)
                image = image - image.min()
                max_value = image.max()
                if max_value > 0: image = image / image.max()
            
            # FIXME: Not utilzing full band
            C, H, W = image.shape
            if C > self.cfg.DATA.INPUT_BAND:
                image = image[:self.cfg.DATA.INPUT_BAND,:,:]
                logging.debug(f"Truncated({C}-->{self.cfg.DATA.INPUT_BAND}) image at {image_path}")
        
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.Tensor(image)
            
            image_id = Path(image_path).stem
            return image, str(image_id)
        
        meta_path = self.metadata[idx]
        image_path = os.path.join(self.root_dir, meta_path['image_path'])
        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()
            image = image.astype(np.float64)
            image = image - image.min()
            max_value = image.max()
            if max_value > 0: image = image / image.max()

        label_path = os.path.join(self.root_dir, meta_path['label_path'])
        with tifffile.TiffFile(label_path) as tif:
            label = tif.asarray()
            label = label.astype(np.float64)
            label = label - label.min()
            max_value = label.max()
            if max_value > 0: label = label / label.max()
        
        json_path = os.path.join(self.root_dir, meta_path['json_path'])
        with open(json_path, 'r') as json_file:
            meta_info = json.load(json_file)
        
        # FIXME: Not utilzing full band
        C, H, W = image.shape
        if C > self.cfg.DATA.INPUT_BAND:
            image = image[:self.cfg.DATA.INPUT_BAND,:,:]
            logging.debug(f"Truncated({C}-->{self.cfg.DATA.INPUT_BAND}) image at {image_path}")
        
        # Generate semantic segmentation label with json information
        C = self.cfg.DATA.SEMANTIC_CLASS
        H, W = self.cfg.DATA.RESOLUTION
        new_label = np.zeros((C, H, W))
        num_semantic_class = len(meta_info['annotations'])
        if num_semantic_class == 1:
            semantic_class = int(meta_info['annotations'][0]['properties']['cropsid'][-3:])
            if np.isnan(label.max()):
                logging.debug(f"nan label at {label_path}")
            else:
                new_label[semantic_class,:,:] = label
        elif num_semantic_class > 1:
            for meta_each in meta_info['annotations']:
                layer_polyline = np.zeros((H, W, 3))
                polylines = meta_each['points']
                for polyline in polylines:
                    polyline = np.array(polyline)
                    layer_polyline = cv2.fillPoly(layer_polyline, [polyline], (255, 255, 255))
                semantic_class = int(meta_each['properties']['cropsid'][-3:])
                layer_polyline = np.mean(layer_polyline, axis=-1) / 255.0
                if np.isnan(layer_polyline.max()):
                    logging.debug(f"nan label at {label_path}")
                else:
                    new_label[semantic_class,:,:] += layer_polyline
                    new_label = np.clip(new_label, 0.0, 1.0)
        else:
            # FIXME: Dummy background map will be passed
            pass
        new_label[0,:,:] = 1 - np.sum(new_label[1:,:,:], axis=0)
        
        if self.transform:
            image = self.transform(image)
            new_label = self.transform(new_label)
        else:
            image = torch.Tensor(image)
            new_label = torch.Tensor(new_label)
        
        return image, new_label


if __name__ == '__main__':
    dfd = DroneFarmlandDataset(
        "data/drone_farmland_semantic_segmentation/meta_valid_v1.json",
        "data/drone_farmland_semantic_segmentation",
    )
    print(len(dfd))
    print(pprint.pformat(dfd[0]))
