import sys
import torch.distributed
import logging
import os

HINTED = set()


def hint_once(content, uid, rank=None):
    if (rank is None) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == rank:
        if uid not in HINTED:
            logging.info(content)
            HINTED.add(uid)


def get_logger(fpath=None, log_level=logging.INFO, local_rank=0, world_size=1):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter(
        f"[{os.uname()[1].split('.')[0]}]({local_rank}/{world_size})"
        f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.basicConfig(
        level=log_level,
        format=f"[{os.uname()[1].split('.')[0]}]({local_rank}/{world_size})"
               f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("Pyobj, f")
    if fpath is not None:
        # Dump log to file
        fh = logging.FileHandler(fpath)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
