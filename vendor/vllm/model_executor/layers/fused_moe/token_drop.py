from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth,
                                signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip

    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr,
                   n_dims: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    if order == 2:
        shape: core.constexpr = [
            n_outer * 2**(n_dims - 1 - stage), 2, 2**stage
        ]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape),
            x.shape)
    else:
        flip = order
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x,
            ids,
            dim: core.constexpr = None,
            descending: core.constexpr = core.CONSTEXPR_0):
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1,
                       "only minor dimension is currently supported")
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending,
                                n_dims)
    return x, ids


@triton.jit
def _token_drop_policy(
    packed_recv_count_ptr,
    drop_allocations_ptr,
    num_avg_token_per_expert,
    num_avg_token_per_rank,
    num_min_token_per_expert,
    NUM_EXPERT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_EXPERT
    tl.store(drop_allocations_ptr + offs, 0, mask=mask)
    packed_recv_count = tl.load(packed_recv_count_ptr + offs,
                                mask=mask,
                                other=0)

    total_tokens_sum = tl.sum(packed_recv_count, axis=0)
    max_drop_total = tl.maximum((total_tokens_sum - num_avg_token_per_rank), 0)

    keep_tokens = tl.minimum(packed_recv_count, num_avg_token_per_expert)
    keep_tokens = tl.maximum(keep_tokens, num_min_token_per_expert)
    drop_counts = tl.maximum((packed_recv_count - keep_tokens), 0)

    sorted_idx = offs
    sorted_drop_counts, sorted_idx = argsort(drop_counts,
                                             sorted_idx,
                                             None,
                                             descending=True)
    cum_sorted = tl.cumsum(sorted_drop_counts, axis=0)
    t1 = cum_sorted - max_drop_total
    allocation = tl.where(t1 > 0, tl.maximum(sorted_drop_counts - t1, 0),
                          sorted_drop_counts)

    tl.store(drop_allocations_ptr + sorted_idx,
             allocation.to(tl.int32),
             mask=(mask & (allocation > 0)))


def token_drop_policy(
    packed_recv_count: torch.Tensor,
    num_avg_token_per_expert: int,
    num_avg_token_per_rank: int,
    num_min_token_per_expert: int = 4,
):
    num_expert = packed_recv_count.shape[0]
    drop_allocations = torch.empty_like(packed_recv_count, dtype=torch.int32)
    BLOCK_SIZE = triton.next_power_of_2(num_expert)
    grid = (1, )

    _token_drop_policy[grid](
        packed_recv_count_ptr=packed_recv_count,
        drop_allocations_ptr=drop_allocations,
        num_avg_token_per_expert=num_avg_token_per_expert,
        num_avg_token_per_rank=num_avg_token_per_rank,
        num_min_token_per_expert=num_min_token_per_expert,
        NUM_EXPERT=num_expert,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )

    return drop_allocations


@triton.jit
def _group_masked_topk_index_kernel(
        recv_weight_ptr, recv_count_ptr, drop_count_ptr, keep_count_ptr,
        drop_num_ptr, out_index_ptr, drop_map_ptr,
        stride_recv_weight: tl.constexpr, num_tokens: tl.constexpr,
        num_groups: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    expert_id = tl.program_id(0)
    token_id = tl.arange(0, BLOCK_SIZE)
    recv_cnt = tl.load(recv_count_ptr + expert_id)
    drop_cnt = tl.load(drop_count_ptr + expert_id)
    keep_len = recv_cnt - drop_cnt
    tl.device_assert(keep_len >= 0, "keep_len must be non-negative")

    is_valid = token_id < recv_cnt
    recv_weight_ptr_s = recv_weight_ptr + expert_id * stride_recv_weight
    x_float = tl.load(recv_weight_ptr_s + token_id, mask=is_valid, other=-1)
    scale = num_groups
    x = (x_float * scale.to(tl.float32)).to(tl.int32)
    z = tl.histogram(x, num_groups)

    cumsum = tl.cumsum(z, axis=0, reverse=True)
    k_mask = cumsum >= keep_len
    num_valid = tl.sum(k_mask.to(tl.int32), axis=0)
    idx_f = (num_valid - 1).to(tl.float32)

    keep_mask = ((x >= idx_f) & is_valid).to(tl.int32)
    keep_num = tl.sum(keep_mask, axis=0)
    tl.store(keep_count_ptr + expert_id, keep_num)
    tl.store(drop_num_ptr + expert_id, recv_cnt - keep_num)
    out_index_offsets = tl.cumsum(keep_mask) - keep_mask
    out_mask = is_valid & (keep_mask == 1)
    tl.store(out_index_ptr + expert_id * stride_recv_weight + token_id, -1)
    tl.store(out_index_ptr + expert_id * stride_recv_weight +
             out_index_offsets,
             token_id,
             mask=out_mask)
    tl.store(drop_map_ptr + expert_id * stride_recv_weight + token_id,
             out_index_offsets,
             mask=out_mask)
    tl.store(drop_map_ptr + expert_id * stride_recv_weight + token_id,
             -1,
             mask=~out_mask)


def group_masked_topk_index(
    recv_weight: torch.Tensor,
    recv_count: torch.Tensor,
    drop_count: torch.Tensor,
    num_groups: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert recv_weight.is_cuda and recv_count.is_cuda and drop_count.is_cuda
    assert recv_count.shape == drop_count.shape, (
        "recv_count and drop_count must have the same shape")
    assert recv_weight.shape[0] == recv_count.shape[
        0], "recv_weight and recv_count must have the same number of experts"

    num_expert, num_max_token = recv_weight.shape
    device = recv_weight.device
    keep_count = torch.empty_like(recv_count, dtype=torch.int32, device=device)
    drop_num = torch.empty_like(drop_count, dtype=torch.int32, device=device)
    out_index = torch.empty_like(recv_weight, dtype=torch.int32)
    drop_map = torch.empty_like(recv_weight, dtype=torch.int32)

    grid = (num_expert, )
    BLOCK_SIZE = num_max_token
    _group_masked_topk_index_kernel[grid](
        recv_weight_ptr=recv_weight,
        recv_count_ptr=recv_count,
        drop_count_ptr=drop_count,
        keep_count_ptr=keep_count,
        drop_num_ptr=drop_num,
        out_index_ptr=out_index,
        drop_map_ptr=drop_map,
        stride_recv_weight=recv_weight.stride(0),
        num_tokens=num_max_token,
        num_groups=num_groups,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return keep_count, out_index, drop_map, drop_num


@triton.jit
def _fused_fp8_gather(
        x_fp8_ptr, scale_ptr, reorder_indices_ptr, reorder_indices_stride_0,
        reorder_indices_stride_1, keep_count_ptr, out_fp8_ptr, out_scale_ptr,
        x_stride_0, x_stride_1, x_stride_2, scale_stride_0, scale_stride_1,
        scale_stride_2, num_experts, total_tokens, hidden, new_fp8_x_stride_0,
        new_fp8_x_stride_1, new_fp8_x_stride_2, new_scale_stride_0,
        new_scale_stride_1, new_scale_stride_2, HIDDEN_BLOCK: tl.constexpr):
    expert_id = tl.program_id(0)
    token_id = tl.program_id(1)
    hidden_id = tl.arange(0, HIDDEN_BLOCK)

    if expert_id >= num_experts or token_id >= total_tokens:
        return

    idx_offset = expert_id * reorder_indices_stride_0 + token_id
    src_token_idx = tl.load(reorder_indices_ptr + idx_offset)
    keep_count = tl.load(keep_count_ptr + expert_id)
    is_valid = token_id < keep_count
    tl.device_assert(src_token_idx < total_tokens,
                     "src_token_idx must be less than total_tokens")

    for h_start in range(0, hidden, HIDDEN_BLOCK):
        h_offs = h_start + hidden_id
        h_mask = h_offs < hidden

        if is_valid:
            src_offset = (expert_id * x_stride_0 +
                          src_token_idx * x_stride_1 +
                          h_offs * x_stride_2)
            dst_offset = (expert_id * new_fp8_x_stride_0 +
                          token_id * new_fp8_x_stride_1 +
                          h_offs * new_fp8_x_stride_2)

            scale_idx = h_offs // 128
            scale_offset = (expert_id * scale_stride_0 +
                            src_token_idx * scale_stride_1 +
                            scale_idx * scale_stride_2)
            scale_mask = (scale_idx < hidden // 128) & h_mask & (h_offs % 128
                                                                 == 0)
            out_scale_offset = (expert_id * new_scale_stride_0 +
                                token_id * new_scale_stride_1 +
                                scale_idx * new_scale_stride_2)

            x_fp8 = tl.load(x_fp8_ptr + src_offset, mask=h_mask)
            scale = tl.load(scale_ptr + scale_offset,
                            mask=scale_mask,
                            other=1.0).to(tl.float32)
            tl.store(out_fp8_ptr + dst_offset, x_fp8, mask=h_mask)
            tl.store(out_scale_ptr + out_scale_offset, scale, mask=scale_mask)


def moe_token_drop(
    packed_recv_topk_weights: torch.Tensor,
    packed_recv_count: torch.Tensor,
    packed_recv_x: Tuple[torch.Tensor, torch.Tensor],
    num_avg_token_per_expert: int,
    num_avg_token_per_rank: int,
):
    """
    Args:
        packed_recv_topk_weights: [num_local_experts, total_tokens]
        packed_recv_count: [num_local_experts]
        packed_recv_x[0]: [num_local_experts, total_tokens, hidden]
        packed_recv_x[1]: [num_local_experts, total_tokens, hidden // 128]
        num_avg_token_per_expert: int
        num_avg_token_per_rank: int

    Returns:
        drop_num: [num_local_experts]
        (new_fp8_x, new_scale): (
            new_fp8_x: [num_local_experts, total_tokens, hidden],
            new_scale: [num_local_experts, total_tokens, hidden // 128])
        keep_count: [num_local_experts]
        drop_map: [num_local_experts, total_tokens]
    """
    assert len(packed_recv_x) == 2, "packed_recv_x must have 2 elements"
    assert packed_recv_x[
        0].dtype == torch.float8_e4m3fn, "packed_recv_x[0] must be fp8"

    num_local_experts, total_tokens = packed_recv_topk_weights.shape
    hidden = packed_recv_x[0].shape[2]

    drop_allocations = token_drop_policy(packed_recv_count,
                                         num_avg_token_per_expert,
                                         num_avg_token_per_rank)
    keep_count, reorder_indices, drop_map, drop_num = group_masked_topk_index(
        packed_recv_topk_weights,
        packed_recv_count,
        drop_allocations,
        num_groups=64)
    grid = (num_local_experts, total_tokens)
    recv_x_0 = packed_recv_x[0].view(torch.int8)
    recv_x_1 = packed_recv_x[1]
    new_fp8_x = torch.empty_like(recv_x_0)
    new_scale = torch.empty_like(recv_x_1)
    BLOCK_SIZE = triton.next_power_of_2(hidden)
    _fused_fp8_gather[grid](
        x_fp8_ptr=recv_x_0,
        scale_ptr=recv_x_1,
        reorder_indices_ptr=reorder_indices,
        reorder_indices_stride_0=reorder_indices.stride(0),
        reorder_indices_stride_1=reorder_indices.stride(1),
        keep_count_ptr=keep_count,
        out_fp8_ptr=new_fp8_x,
        out_scale_ptr=new_scale,
        x_stride_0=recv_x_0.stride(0),
        x_stride_1=recv_x_0.stride(1),
        x_stride_2=recv_x_0.stride(2),
        scale_stride_0=recv_x_1.stride(0),
        scale_stride_1=recv_x_1.stride(1),
        scale_stride_2=recv_x_1.stride(2),
        new_fp8_x_stride_0=new_fp8_x.stride(0),
        new_fp8_x_stride_1=new_fp8_x.stride(1),
        new_fp8_x_stride_2=new_fp8_x.stride(2),
        new_scale_stride_0=new_scale.stride(0),
        new_scale_stride_1=new_scale.stride(1),
        new_scale_stride_2=new_scale.stride(2),
        num_experts=num_local_experts,
        total_tokens=total_tokens,
        hidden=hidden,
        HIDDEN_BLOCK=BLOCK_SIZE,
        num_warps=1,
    )

    return drop_num, (new_fp8_x.view(torch.float8_e4m3fn),
                      new_scale), keep_count, drop_map
