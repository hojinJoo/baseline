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

count_image_tif = 0
image_file_list = list(sorted(list(Path(image_dir).rglob("*"))))
print(f"image_file_list: {len(image_file_list)}")
for file_path in tqdm(image_file_list):
    if file_path.suffix == '.tif':
        count_image_tif += 1

        if 'Zone-A-001_000_011' in str(file_path):
            with tifffile.TiffFile(str(file_path)) as tif:
                data = tif.asarray()
                print(f"data shape of Zone-A-001_000_011: {data.shape}")

        # with tifffile.TiffFile(str(file_path)) as tif:
        #     data = tif.asarray()
        
    else:
        pass
        # print(f"Not tif image founded, {file_path}", flush=True)

count_label_tif = 0
label_file_list = list(sorted(list(Path(label_dir).rglob("*"))))
print(f"label_file_list: {len(label_file_list)}")
for file_path in tqdm(label_file_list):
    if file_path.suffix == '.tif':
        count_label_tif += 1

        if 'Zone-A-001_000_011' in str(file_path):
            with tifffile.TiffFile(str(file_path)) as tif:
                data = tif.asarray()
                print(f"label shape of Zone-A-001_000_011: {data.shape}, {data.min()}, {data.max()}")
                data = data.astype(np.float64)
                data = data - data.min()
                data = data / data.max() * 255.0
                cv2.imwrite("debug_vis.png", data)

        # with tifffile.TiffFile(str(file_path)) as tif:
        #     data = tif.asarray()
        #     # data = data[::50,:,:]
        #     # data = data[::-1,:,:]
        #     data = data.astype(np.float64)
        #     data = data - data.min()
        #     data = data / data.max() * 255.0
        #     # data = data.transpose(1, 2, 0)
        
    else:
        pass
        # print(f"Not tif image founded, {file_path}")
    

count_json = 0
json_file_list = list(sorted(list(Path(json_dir).rglob("*"))))
for file_path in tqdm(json_file_list):
    if file_path.suffix == '.json':
        count_json += 1

        with open(str(file_path), 'r') as json_file:
            data = json.load(json_file)
            if len(data['annotations']) > 1:
                print(f"label: {pprint.pformat(data)}")
                break
    else:
        pass
        # print(f"Not json label found, {file_path}")
    
print(f"count image tif: {count_image_tif}, label tif: {count_label_tif}, json: {count_json}")

json_data = []
for image_path, label_path, json_path in zip(image_file_list, label_file_list, json_file_list):
    image_path = '/'.join(str(image_path).split('/')[2:])
    label_path = '/'.join(str(label_path).split('/')[2:])
    json_path = '/'.join(str(json_path).split('/')[2:])
    json_data.append(dict(
        image_path=image_path,
        label_path=label_path,
        json_path=json_path,
    ))

with open(meta_json_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
print("Meta All Json File")
with open(meta_json_path, 'r') as json_file:
    data = json.load(json_file)
    print(f"data: {pprint.pformat(data[:5])}")
    print(".\n.\n.\n")
    print(f"data: {pprint.pformat(data[-5:])}")

json_train_data = []
json_valid_data = []
for json_line in json_data:
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

