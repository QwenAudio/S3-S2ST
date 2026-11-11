# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from vllm.triton_utils import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor,
                          chunk_size: int) -> torch.LongTensor:
    return torch.cat([
        cu_seqlens.new_tensor([0]),
        triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    ]).cumsum(-1)


@tensor_cache
def prepare_mlist(seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    nums = triton.cdiv(seqlens, chunk_size)
    nums_list = nums.tolist()

    mlist_list = []
    for idx, num in enumerate(nums_list):
        mlist_list.extend([idx] * num)

    return torch.tensor(mlist_list, dtype=torch.long, device=seqlens.device)


@tensor_cache
def prepare_offsetlist(seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    nums = triton.cdiv(seqlens, chunk_size)
    nums_list = nums.tolist()

    offsetlist_list = []
    for num in nums_list:
        offsetlist_list.extend(range(num))
    
    return torch.tensor(offsetlist_list, dtype=torch.long, device=seqlens.device)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])

@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1

@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    # if you have 3 seq: 2, 3, 4，
    # generate token_offset in seq: [0, 1]、[0, 1, 2] 和 [0, 1, 2, 3] -> tensor([0, 1, 0, 1, 2, 0, 1, 2, 3])
    position_ids = prepare_position_ids(cu_seqlens)
    # prepare_sequence_ids:
    sequence_ids = position_ids.eq(0).cumsum(0) - 1
    # and generate token's seq_id:  [0, 0, 1, 1, 1, 2, 2, 2, 2]
    return torch.stack([sequence_ids, position_ids], 1).to(cu_seqlens)

def prepare_token_indices_fast(cu_seqlens: torch.Tensor,total_len) -> torch.Tensor:
    # 确保 cu_seqlens 是长整型，以便进行 arange 操作
    # total_len = N, num_seqs = B
    total_len = cu_seqlens[-1].item()
    num_seqs = cu_seqlens.numel() - 1
    
    # 1. 获取每个序列的长度 [B]
    lens = cu_seqlens[1:] - cu_seqlens[:-1]

    # 2. 生成 position_ids [N]
    # 原理：[0, 1, 2, 3, 4, 5...] - [0, 0, 2, 2, 2, 5, 5, 5...] (序列起始偏移)
    # 这是生成 [0, 1, 0, 1, 2, 0, 1, 2, 3] 最快的方法
    full_arange = torch.arange(total_len, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    # 重复每个序列的起始位置
    start_offsets = torch.repeat_interleave(cu_seqlens[:-1], lens)
    position_ids = full_arange - start_offsets

    # 3. 生成 sequence_ids [N]
    # 直接重复 [0, 1, 2, ..., B-1]
    sequence_ids = torch.repeat_interleave(
        torch.arange(num_seqs, device=cu_seqlens.device, dtype=cu_seqlens.dtype),
        lens
    )

    # 4. 合并结果 [N, 2]
    # 使用 stack 或 cat 都可以，stack 在这里性能很好
    return torch.stack([sequence_ids, position_ids], dim=1)
