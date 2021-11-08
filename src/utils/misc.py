import logging
import sys
from pathlib import Path


def setup_logger(output_dir_path, filestem='log_in_english_is'):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    output_dir_p = Path(output_dir_path)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    filename =  output_dir_p / f"{filestem}.log"
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)

    return logger