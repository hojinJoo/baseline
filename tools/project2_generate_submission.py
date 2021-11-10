import tifffile
import numpy as np
import json
import pprint
import random
import csv
import cv2
import uuid
import shutil
import torch
from pathlib import Path
from tqdm import tqdm

from src.utils.visualize import Visualizer

root_dir = "data/drone_farmland_semantic_segmentation"
test_dir = f"{root_dir}/test"
test_vis_dir = f"{root_dir}/test_vis"
meta_json_path = f"{root_dir}/meta_all_v1.json"
meta_json_train_path = f"{root_dir}/meta_train_v1.json"
meta_json_valid_path = f"{root_dir}/meta_valid_v1.json"
test_ratio = 0.01
test_csv_file = f"{root_dir}/test.csv"

Path(test_dir).mkdir(parents=True, exist_ok=True)
Path(test_vis_dir).mkdir(parents=True, exist_ok=True)

print("Meta All Json File")
with open(meta_json_path, 'r') as json_file:
    data = json.load(json_file)
    print(f"data: {pprint.pformat(data[:5])}")
    print(".\n.\n.\n")
    print(f"data: {pprint.pformat(data[-5:])}")
print(f"length of all data: {len(data)}")

C, H, W = 25, 128, 128
csvfile = open(test_csv_file, 'w', newline='\n')
writer = csv.writer(csvfile)
writer.writerow(["Id", "Expected"])
for row in tqdm(data):
    if random.random() < test_ratio:
        image_id = uuid.uuid4()
        new_image_path = str(Path(test_dir) / f"{image_id}.tif")
        pre_image_path = f"{root_dir}/{row['image_path']}"
        shutil.copy(pre_image_path, new_image_path)

        json_path = f"{root_dir}/{row['json_path']}"
        with open(json_path, 'r') as json_file:
            meta_info = json.load(json_file)
            label = np.zeros((C, H, W))
            for meta_each in meta_info['annotations']:
                layer_polyline = np.zeros((H, W, 3))
                polylines = meta_each['points']
                for polyline in polylines:
                    polyline = np.array(polyline)
                    layer_polyline = cv2.fillPoly(layer_polyline, [polyline], (255, 255, 255))
                semantic_class = int(meta_each['properties']['cropsid'][-3:])
                layer_polyline = np.mean(layer_polyline, axis=-1) / 255.0
                label[semantic_class,:,:] += layer_polyline
                label = np.clip(label, 0.0, 1.0)
            label[0,:,:] = 1 - np.sum(label[1:,:,:], axis=0)
            label_vis = label.copy()
            label = np.argmax(label, axis=0)
        H, W = label.shape
        for i in range(H):
            for j in range(W):
                key = f"{image_id}_{i:03d}_{j:03d}"
                value = label[i,j].item()
                writer.writerow([key, value])

        with tifffile.TiffFile(str(pre_image_path)) as tif:
            image = tif.asarray()
            image = image.astype(np.float64)
            image = image - image.min()
            max_value = image.max()
            if max_value > 0: image = image / image.max()
        test_input_vis_path = f"{test_vis_dir}/{image_id}.png"
        Visualizer.save_multi_channel_as_png(torch.Tensor(image)[None,:,:,:], test_input_vis_path)
        test_label_vis_path = f"{test_vis_dir}/{image_id}_gt.png"
        Visualizer.save_multi_channel_as_png(torch.Tensor(label_vis)[None,:,:,:], test_label_vis_path)

csvfile.close()