from .unet import UNet


MODEL_DICT = dict(
    UNet=UNet,
)

def get_model(cfg):
    model = None
    for k, v in MODEL_DICT.items():
        if cfg.MODEL.ARCH == k:
            model = v(
                cfg.MODEL.IN_CHANNEL,
                cfg.MODEL.OUT_CHANNEL,
            )
    
    if model is not None:
        return model
    else:
        raise KeyError(f"cfg.MODEL.ARCH({cfg.MODEL.ARCH}) is not in the MODEL_DICT({MODEL_DICT.keys()})")

