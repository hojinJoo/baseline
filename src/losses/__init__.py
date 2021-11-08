import torch.nn.functional as F


def mse_loss(pred, target):

    loss = F.mse_loss(pred, target)
    return loss


def dice_loss(pred, target, smooth = 1.):

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

LOSS_DICT = dict(
    mse_loss=mse_loss,
    dice_loss=dice_loss,
)

def get_loss(cfg):

    loss = None
    for k, v in LOSS_DICT.items():
        if cfg.TRAINER.LOSS == k:
            loss = v
    
    if loss is not None:
        return loss
    else:
        raise KeyError(f"cfg.TRAINER.LOSS({cfg.TRAINER.LOSS}) is not in the LOSS_DICT({LOSS_DICT.keys()})")
    
