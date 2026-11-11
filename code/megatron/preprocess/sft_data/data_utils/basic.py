#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def setup_logging(rich_logging: bool = False):
    kwargs = {}

    if rich_logging:
        try:
            from rich.logging import RichHandler
        except ImportError:
            pass
        else:
            kwargs['format'] = '%(message)s'
            kwargs['handlers'] = [RichHandler()]

    logging.basicConfig(
        datefmt='%m/%d %H:%M:%S',
        level=logging.INFO,
        **kwargs,
    )


def set_seed(seed):
    random.seed(seed)


def shuffle(*data_list, seed):
    rng = random.Random(seed)
    for data in data_list:
        rng.shuffle(data)


def flatten_lists(lists):
    result = []
    for lst in lists:
        result.extend(lst)
    return result


def get_langs(name: str) -> List[str]:
    if name.startswith('en:'):
        return ['en']
    elif name.startswith('cn:'):
        return ['cn']
    elif name.startswith('other'):
        return ['other']
    elif name.startswith('hotfix:'):
        return ['cn']
    elif name.startswith('all:'):
        return ['en', 'cn']
    elif name.startswith('multi:'):
        return ['en', 'cn']
    else:
        return ['en', 'cn']


def get_main_lang(name: str) -> str:
    langs = get_langs(name)
    if len(langs) > 1:
        return 'multi-lang'
    else:
        return langs[0]


################
# Discard Data #
################

FILTERED_DATA = []


def discard_init():
    FILTERED_DATA.clear()


def discard_data(item: dict, source: str, reason: str):
    """Record and discard data item."""
    FILTERED_DATA.append({'source': source, 'reason': reason, 'item': deepcopy(item)})


def save_filtered_data(file_name: Path):
    file_name.parent.mkdir(exist_ok=True, parents=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        for item in FILTERED_DATA:
            print(json.dumps(item, ensure_ascii=False), file=f)
    logger.info(f'{len(FILTERED_DATA)} filtered data saved to {file_name}.')
    discard_init()
