import argparse
import pprint

from src.config.cfg_project2 import get_cfg_project2_default
from src.predictor import DefaultPredictor


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    cfg = get_cfg_project2_default()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(cfg):

    print(pprint.pformat(cfg))

    predictor = DefaultPredictor(cfg)
    predictor.test()


if __name__ == '__main__':
    args = get_argument_parser().parse_args()
    cfg = setup_cfg(args)
    main(cfg)
