from yacs.config import CfgNode as CN

_C = CN()

_C.CHECKPOINT_PATH = 'checkpoints/baseline.pth'
_C.TRAIN_JSON_PATH = 'data/korean_food_classification_data/train/train/train.json'
_C.TRAIN_DATA_PATH = 'data/korean_food_classification_data/train/train'
_C.VALID_JSON_PATH = 'data/korean_food_classification_data/val/val/val.json'
_C.VALID_DATA_PATH = 'data/korean_food_classification_data/val/val'

_C.TEST_DATA_PATH = 'data/korean_food_classification_data/test/test'

def get_cfg_project1_default():
    cfg = _C.clone()
    return cfg
