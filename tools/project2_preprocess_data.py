import tifffile
import numpy as np
import json
import cv2
import pprint
import random
from pathlib import Path
from tqdm import tqdm

root_dir = "data/drone_farmland_semantic_segmentation"
meta_json_path = f"{root_dir}/meta_all_v2.json"
meta_json_train_path = f"{root_dir}/meta_train_v2.json"
meta_json_valid_path = f"{root_dir}/meta_valid_v2.json"
valid_ratio = 0.1
image_dir = f"{root_dir}/train/images"
json_dir = f"{root_dir}/train/jsons"
label_dir = f"{root_dir}/train/labels"

json_data = {}

image_file_list = list(sorted(list(Path(image_dir).rglob("*"))))
print(f"image_file_list: {len(image_file_list)}")
for file_path in tqdm(image_file_list):
    data_id = file_path.stem
    if file_path.suffix == '.tif':
        # with tifffile.TiffFile(str(file_path)) as tif:
        #     data = tif.asarray()
        #     if np.isnan(data.max()):
        #         print(f"Nan image at {file_path}")
        #         continue
        file_path = '/'.join(str(file_path).split('/')[2:])
        if data_id in json_data:
            json_data[data_id]['image_path']=file_path
        else:
            json_data[data_id] = dict(image_path=file_path)

label_file_list = list(sorted(list(Path(label_dir).rglob("*"))))
print(f"label_file_list: {len(label_file_list)}")
for file_path in tqdm(label_file_list):
    data_id = file_path.stem[:-3]
    if file_path.suffix == '.tif':
        # with tifffile.TiffFile(str(file_path)) as tif:
        #     data = tif.asarray()
        #     if np.isnan(data.max()):
        #         print(f"Nan label at {file_path}")
        #         continue
        file_path = '/'.join(str(file_path).split('/')[2:])
        if data_id in json_data:
            json_data[data_id]['label_path']=file_path
        else:
            json_data[data_id] = dict(label_path=file_path)

json_file_list = list(sorted(list(Path(json_dir).rglob("*"))))
print(f"json_file_list: {len(json_file_list)}")
for file_path in tqdm(json_file_list):
    data_id = file_path.stem
    if file_path.suffix == '.json':
        file_path = '/'.join(str(file_path).split('/')[2:])
        if data_id in json_data:
            json_data[data_id]['json_path']=file_path
        else:
            json_data[data_id] = dict(json_path=file_path)

json_data_all = []
for _, v in json_data.items():
    if len(v.keys()) == 3:
        json_data_all.append(v)
print(f"length of all: {len(json_data_all)}")

with open(meta_json_path, 'w') as json_file:
    json.dump(json_data_all, json_file, indent=4)

print("Meta All Json File")
with open(meta_json_path, 'r') as json_file:
    data = json.load(json_file)
    print(f"data: {pprint.pformat(data[:5])}")
    print(".\n.\n.\n")
    print(f"data: {pprint.pformat(data[-5:])}")

json_train_data = []
json_valid_data = []
for json_line in json_data_all:
    if random.random() > valid_ratio:
        json_train_data.append(json_line)
    else:
        json_valid_data.append(json_line)
with open(meta_json_train_path, 'w') as json_file:
    json.dump(json_train_data, json_file, indent=4)
with open(meta_json_valid_path, 'w') as json_file:
    json.dump(json_valid_data, json_file, indent=4)
print("Meta Train Json File")
with open(meta_json_train_path, 'r') as json_file:
    data = json.load(json_file)
    print(f"data: {pprint.pformat(data[:5])}")
    print(".\n.\n.\n")
    print(f"data: {pprint.pformat(data[-5:])}")
print("Meta Valid Json File")
with open(meta_json_valid_path, 'r') as json_file:
    data = json.load(json_file)
    print(f"data: {pprint.pformat(data[:5])}")
    print(".\n.\n.\n")
    print(f"data: {pprint.pformat(data[-5:])}")

