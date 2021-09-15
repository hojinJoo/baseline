import os
import json
import random
import string
from collections import defaultdict

from detectron2.utils.file_io import PathManager

CHAR_SET = string.ascii_uppercase + string.digits
RATIO = (0.7, 0.15, 0.15) # ratio of train/val/test respectively.
TEMO_DIR = 'temp'
assert sum(RATIO) == 1, "ratio of train/val/test must be 1"


# Initialize necessary info.
curpath = os.path.curdir
category_idx = 0
category_dict = {}
collected_data = []
rand_names = set()

# Make directory for aggregating collected data.
data_save_path = os.path.join(curpath, TEMO_DIR)
if not PathManager.isdir(data_save_path):
    PathManager.mkdirs(data_save_path)

# Iterate over main categories.
main_categories = PathManager.ls(curpath)
for main_cat_name in main_categories:
    print("Handling: {}".format(main_cat_name))
    main_cat_path = os.path.join(curpath, main_cat_name)

    # Skip if encountered .zip file
    if PathManager.isfile(main_cat_path) or main_cat_name == TEMO_DIR:
        continue

    # Iterate over sub categories.
    sub_categories = PathManager.ls(main_cat_path)
    for sub_cat_name in sub_categories:
        sub_cat_path = os.path.join(main_cat_path, sub_cat_name)

        # Collect image files. 
        for f in PathManager.ls(sub_cat_path):
            if f.endswith('.jpg'): # NOTE: not using the crop information.
                # Generate random file name ensuring that there is no duplicates.
                while True:
                    new_filename = ''.join(random.sample(CHAR_SET*8, 8))
                    if new_filename not in rand_names:
                        rand_names.add(new_filename)
                        break

                new_file_path = new_filename + ".jpg"
                collected_data.append(
                    {
                        "file_name": new_file_path,
                        "category_idx": category_idx,
                        "main_category": main_cat_name,
                        "sub_category": sub_cat_name
                    }
                )

                # Copy image files to 'data_save_path'.
                PathManager.copy(
                    os.path.join(sub_cat_path, f), os.path.join(data_save_path, new_file_path)
                )

        # Save category index info for labeling.
        category_dict[category_idx] = {'sub': sub_cat_name, 'main': main_cat_name}
        category_idx += 1

total_num = len(collected_data)
train_num = round(total_num * RATIO[0])
val_num = round(total_num * RATIO[1])
test_num = total_num - (train_num + val_num)

train_save_path = os.path.join(curpath, 'train')
val_save_path = os.path.join(curpath, 'val')
test_save_path = os.path.join(curpath, 'test')

for p in [train_save_path, val_save_path, test_save_path]:
    if not PathManager.isdir(p):
        PathManager.mkdirs(p)

# Shuffle collected data.
random.shuffle(collected_data)

train_data = collected_data[:train_num]
val_data = collected_data[train_num:train_num+val_num]
test_data = collected_data[train_num+val_num:]

print("Moving files...")
for idx, data in enumerate(train_data):
    assert PathManager.isfile(
        os.path.join(data_save_path, data["file_name"])
    ), "{} not collected".format(data["file_name"])
    PathManager.mv(
        os.path.join(data_save_path, data["file_name"]), os.path.join(train_save_path, data["file_name"])
    )
for idx, data in enumerate(val_data):
    assert PathManager.isfile(
        os.path.join(data_save_path, data["file_name"])
    ), "{} not collected".format(data["file_name"])
    PathManager.mv(
        os.path.join(data_save_path, data["file_name"]), os.path.join(val_save_path, data["file_name"])
    )
for idx, data in enumerate(test_data):
    assert PathManager.isfile(
        os.path.join(data_save_path, data["file_name"])
    ), "{} not collected".format(data["file_name"])
    PathManager.mv(
        os.path.join(data_save_path, data["file_name"]), os.path.join(test_save_path, data["file_name"])
    )

print("Writing json files...")

train_json_file = open("train.json", "w")
val_json_file = open("val.json", "w")
test_json_file = open("test.json", "w")

train_data = {
    "annotations": train_data,
    "catgories": category_dict,
}
val_data = {
    "annotations": val_data,
    "catgories": category_dict,
}
test_data = {
    "annotations": test_data,
    "catgories": category_dict,
}
json.dump(train_data, train_json_file)
json.dump(val_data, val_json_file)
json.dump(test_data, test_json_file)

print("Data processing done.")