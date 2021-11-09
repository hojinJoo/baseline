from torch.utils.data import DataLoader

from .drone_farmland_dataset import DroneFarmlandDataset

DATASET_DICT = dict(
    DroneFarmlandDataset=DroneFarmlandDataset,
)

def get_dataset(cfg):
    
    dataset = None
    for k, v in DATASET_DICT.items():
        if cfg.DATA.NAME == k:
            dataset = v
    
    if dataset is not None:
        return dataset
    else:
        raise KeyError(f"cfg.DATA.NAME({cfg.DATA.NAME}) is not in the DATASET_DICT({DATASET_DICT.keys()})")


def get_dataloader(cfg, phase='train', transform=None):
    """
    NOTE: This function is highly entangled with DroneFamlandDataset class
    """

    dataset_cls = get_dataset(cfg)

    if phase == 'train':
        dataset = dataset_cls(
            cfg,
            cfg.DATA.TRAIN_JSON_PATH,
            cfg.DATA.ROOT_DIR,
            transform,
        )
    elif phase == 'valid':
        dataset = dataset_cls(
            cfg,
            cfg.DATA.VALID_JSON_PATH,
            cfg.DATA.ROOT_DIR,
            transform,
        )
    elif phase == 'test':
        dataset = dataset_cls(
            cfg,
            None,
            cfg.DATA.TEST_DIR_PATH,
            inference=True,
        )
    else:
        raise ValueError(f"The value of phase({phase}) should be one of ['train', 'valid']")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=cfg.DATA.SHUFFLE,
        pin_memory=cfg.DATA.PIN_MEMORY,
    )
    return dataloader