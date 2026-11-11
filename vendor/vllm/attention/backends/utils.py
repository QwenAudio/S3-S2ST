# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend utils"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

PAD_SLOT_ID = -1


@dataclass
class MLADims:
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


def get_mla_dims(model_config: ModelConfig) -> MLADims:
    hf_text_config = model_config.hf_text_config

    return MLADims(
        q_lora_rank=getattr(hf_text_config, "q_lora_rank", None),
        kv_lora_rank=hf_text_config.kv_lora_rank,
        qk_nope_head_dim=hf_text_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_text_config.qk_rope_head_dim,
        v_head_dim=hf_text_config.v_head_dim,
    )


def remap_dual_chunk_attention_config(
    pattern: dict[int, Any],
    num_heads: int,
    num_kv_heads: int,
) -> dict[int, Any]:
    if not pattern or len(pattern) == num_heads:
        return pattern

    orig_heads_per_kv = np.split(np.arange(len(pattern)), num_kv_heads)
    heads_per_kv = np.split(np.arange(num_heads), num_kv_heads)

    mapping = {}
    for orig_heads, heads in zip(orig_heads_per_kv, heads_per_kv):
        for orig_head, head in zip(orig_heads, heads):
            mapping[head] = orig_head
        for head in heads[len(orig_heads) :]:
            mapping[head] = orig_heads[-1]

    return {ii: pattern[mapping[ii]] for ii in range(num_heads)}
