import sys
import json
import logging




def get_logger(path):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(path)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger