from torch.optim import (
    Adam,
)
from torch.optim.lr_scheduler import MultiStepLR


OPTIM_DICT = dict(
    Adam=Adam,
)


def get_optim_and_scheduler(cfg, parameters):
    """
    NOTE: Currently only supporting MultiStepLR
    """

    optim = None
    for k, v in OPTIM_DICT.items():
        if cfg.TRAINER.OPTIMIZER == k:
            optim = v(
                parameters,
                lr=cfg.TRAINER.LEARNING_RATE,
            )
    
    if optim is not None:
        scheduler = MultiStepLR(
            optim,
            milestones=cfg.TRAINER.LR_STEP,
            gamma=cfg.TRAINER.LR_GAMMA
        )
        return optim, scheduler
    else:
        raise KeyError(f"cfg.TRAINER.OPTIMIZER({cfg.TRAINER.OPTIMIER}) is not in the OPTIM_DICT({OPTIM_DICT.keys()})")

