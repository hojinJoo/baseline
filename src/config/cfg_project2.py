from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.NAME = 'DroneFarmlandDataset'
_C.DATA.ROOT_DIR = "data/drone_farmland_semantic_segmentation"
_C.DATA.TRAIN_JSON_PATH = "data/drone_farmland_semantic_segmentation/meta_train_v1.json"
_C.DATA.VALID_JSON_PATH = "data/drone_farmland_semantic_segmentation/meta_valid_v1.json"
_C.DATA.TEST_DIR_PATH = "data/drone_farmland_semantic_segmentation/test"
_C.DATA.INPUT_BAND = 150
_C.DATA.SEMANTIC_CLASS = 25
_C.DATA.RESOLUTION = [128, 128]
_C.DATA.BATCH_SIZE = 64
_C.DATA.NUM_WORKERS = 8
_C.DATA.SHUFFLE = True
_C.DATA.PIN_MEMORY = True

_C.MODEL = CN()
_C.MODEL.ARCH = 'UNet'
_C.MODEL.WEIGHTS = ''
_C.MODEL.IN_CHANNEL = 150
_C.MODEL.OUT_CHANNEL = 25

_C.TRAINER = CN()
_C.TRAINER.RESUME = False
_C.TRAINER.OPTIMIZER = 'Adam'
_C.TRAINER.LOSS = 'mse_loss'
_C.TRAINER.LEARNING_RATE = 1e-3
_C.TRAINER.MAX_ITER = 10000
_C.TRAINER.PRINT_ITER = 500
_C.TRAINER.LR_STEP = [6000,8000]
_C.TRAINER.LR_GAMMA = 0.1

_C.OUTPUT_DIR = 'outputs'
_C.CHECKPOINT_PATH = f'model_final.pth'
_C.SUMMARY_DIR = f'summary'
_C.VIS_DIR = f'vis'


def get_cfg_project2_default():
    cfg = _C.clone()
    return cfg

