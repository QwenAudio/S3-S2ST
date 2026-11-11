# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import os
import site

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def find_nccl_library() -> str:
    """Return NCCL/RCCL shared library name to load.

    Uses `VLLM_NCCL_SO_PATH` if set; otherwise chooses by torch backend.
    """
    so_file = envs.VLLM_NCCL_SO_PATH
    local_so_file = os.path.join(
        torch.__path__[0], '..', 'nvdia', 'nccl', 'lib', 'libnccl.so.2'
    )
    site_so_file = os.path.join(
        site.getsitepackages()[0], 'nvidia', 'nccl', 'lib', 'libnccl.so.2'
    )
    if so_file:
        logger.info_once(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s", so_file
        )
    elif os.path.isfile(local_so_file):
        so_file = local_so_file
        logger.info_once("Found nccl from library %s", so_file)
    elif os.path.isfile(site_so_file):
        so_file = site_so_file
        logger.info_once("Found nccl from library %s", so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info_once("Found nccl from library %s", so_file)
    return so_file


def find_nccl_include_paths() -> list[str] | None:
    """Return possible include paths containing `nccl.h`.

    Considers `VLLM_NCCL_INCLUDE_PATH` and the `nvidia-nccl-cuXX` package.
    """
    paths: list[str] = []
    inc = envs.VLLM_NCCL_INCLUDE_PATH
    if inc and os.path.isdir(inc):
        paths.append(inc)

    try:
        spec = importlib.util.find_spec("nvidia.nccl")
        if spec and (locs := getattr(spec, "submodule_search_locations", None)):
            for loc in locs:
                inc_dir = os.path.join(loc, "include")
                if os.path.exists(os.path.join(inc_dir, "nccl.h")):
                    paths.append(inc_dir)
    except Exception as e:
        logger.debug("Failed to find nccl include path from nvidia.nccl package: %s", e)

    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out or None
