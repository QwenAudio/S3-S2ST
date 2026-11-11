# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import P2POp, batch_isend_irecv, get_global_rank

from vllm.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_varlen_func,
        is_vllm_fa,
    )

from vllm.logger import init_logger

logger = init_logger(__name__)


@torch.inference_mode()
def fa_block_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    block_seq_len: int,
    causal: bool,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = True,
    attn_func: Optional[Callable] = None,
):
    """
    q: [S,H,Hd]
    k: [S,H,Hd_k]
    v: [S,H,Hd_v]
    return: o:[S,H,Hd_v], lse:[S,H] (fp32)
    """
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]
    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2

    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
    if attn_func is None:
        attn_func = flash_attn_varlen_func
    output = attn_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
        dropout_p=0.0,
        causal=causal,
        softmax_scale=softmax_scale,
        **{
            "return_softmax_lse": True,
        } if is_vllm_fa else {
            "return_attn_probs": True,
        },
        fa_version=3,
    )
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.inference_mode()
def online_merge(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def get_half_index(cu_seqlens, *, front: bool):
    if len(cu_seqlens) == 2:
        if front:
            return slice(None, cu_seqlens[-1] // 2)
        else:
            return slice(cu_seqlens[-1] // 2, None)

    index = torch.zeros((cu_seqlens[-1].item(),), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        assert slice_ is not None and out is not None and lse is not None
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


# zigzag ring attention, pass kv in varlen format
def zigzag_ring_prefill_kv(
    q, k, v, ranks, rank, world_size, group, softmax_scale=None, attn_func=None
):
    out = None
    lse = None
    seqlen_q = q.shape[0]
    seqlens = [0, seqlen_q]
    cu_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=q.device)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    half_index0 = get_half_index(cu_seqlens, front=True)
    half_index1 = get_half_index(cu_seqlens, front=False)
    block_seq_len = seqlen_q // 2
    next_k, next_v = None, None
    q1 = q[half_index1].contiguous()

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size
    global_send_rank = ranks[send_rank]
    global_recv_rank = ranks[recv_rank]

    reqs = None
    for step in range(world_size):
        if step + 1 != world_size:
            next_k = torch.empty_like(k, device=k.device)
            next_v = torch.empty_like(v, device=v.device)
            reqs = dist.batch_isend_irecv(
                [
                    P2POp(dist.isend, k, global_send_rank, group=group),
                    P2POp(dist.irecv, next_k, global_recv_rank, group=group),
                    P2POp(dist.isend, v, global_send_rank, group=group),
                    P2POp(dist.irecv, next_v, global_recv_rank, group=group),
                ]
            )

        if step == 0:
            block_out, block_lse = fa_block_with_lse(
                q,
                k,
                v,
                cu_seqlens,
                max_seqlen,
                block_seq_len,
                causal=True,
                softmax_scale=softmax_scale,
                attn_func=attn_func,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= rank:
            k0 = k[half_index0].contiguous()
            v0 = v[half_index0].contiguous()
            block_out, block_lse = fa_block_with_lse(
                q,
                k0,
                v0,
                cu_seqlens,
                max_seqlen,
                block_seq_len,
                causal=False,
                softmax_scale=softmax_scale,
                attn_func=attn_func,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = fa_block_with_lse(
                q1,
                k,
                v,
                cu_seqlens,
                max_seqlen,
                block_seq_len,
                causal=False,
                softmax_scale=softmax_scale,
                attn_func=attn_func,
            )
            assert out is not None and lse is not None
            out[half_index1], lse[half_index1] = update_out_and_lse(
                out[half_index1].contiguous(),
                lse[half_index1].contiguous(),
                block_out,
                block_lse,
            )

        if step + 1 != world_size:
            assert reqs is not None
            for req in reqs:
                req.wait()
            k, v = next_k, next_v
            reqs = None
    assert reqs is None
    next_k = None
    next_v = None
    assert out is not None and lse is not None
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out, lse


def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.contiguous()


def inverse_extract_local(gather_list, dim=0):
    world_size = len(gather_list)
    per_rank_chunks = []
    for t in gather_list:
        front, back = t.chunk(2, dim=dim)
        per_rank_chunks.append((front, back))
    chunks = [None] * (2 * world_size)
    for r, (front, back) in enumerate(per_rank_chunks):
        chunks[r] = front
        chunks[2 * world_size - r - 1] = back
    return torch.cat(chunks, dim=dim)


# =========================
def cp_merge_with_lse(
    ctx_local: torch.Tensor,  # [B, H, 1, D]
    lse_local: torch.Tensor,  # [B, H, 1]
    world: int,
    group: torch.distributed.group,
) -> torch.Tensor:
    assert group is not None
    lse_list = [torch.empty_like(lse_local) for _ in range(world)]
    ctx_list = [torch.empty_like(ctx_local) for _ in range(world)]
    dist.all_gather(lse_list, lse_local, group=group)
    dist.all_gather(ctx_list, ctx_local, group=group)

    # (order = rank order; zigzag already folded locally)
    lse_all = torch.stack(lse_list, dim=0)  # [W,B,H,1]
    ctx_all = torch.stack(ctx_list, dim=0)  # [W,B,H,1,D]

    mL = torch.amax(lse_all, dim=0)  # [B,H,1]
    w = torch.exp(lse_all - mL)  # [W,B,H,1]
    num = torch.sum(w.unsqueeze(-1) * ctx_all, dim=0)  # [B,H,1,D]
    den = torch.sum(w, dim=0)  # [B,H,1]
    return (num / den.unsqueeze(-1)).to(ctx_local.dtype)


def extract_kv_varlen(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
):
    """
    extract k and v from paged KV-cache + block_table to varlen format, just for debug
    k_flat: [total_tokens, H, D]
    v_flat: [total_tokens, H, D]
    cu_seqlens: [B+1]
    """
    B, max_num_blocks = block_table.shape
    num_blocks, bs, nheads_headdim_k = k_cache.shape
    nheads_headdim_v = v_cache.shape[-1]

    assert bs == block_size

    device = k_cache.device
    dtype = k_cache.dtype

    cu_seqlens = torch.zeros(B + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens.to(torch.int32), dim=0)
    total_tokens = int(cu_seqlens[-1].item())

    k_flat = torch.empty(total_tokens, nheads_headdim_k, device=device, dtype=dtype)
    v_flat = torch.empty(total_tokens, nheads_headdim_v, device=device, dtype=dtype)

    for b in range(B):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue
        start = int(cu_seqlens[b].item())
        end = start + seq_len

        num_blocks_b = (seq_len + block_size - 1) // block_size
        block_ids = block_table[b, :num_blocks_b]
        block_ids = block_ids[block_ids >= 0]
        num_blocks_b = block_ids.numel()
        if num_blocks_b == 0:
            continue

        k_blocks = k_cache[block_ids]  # [num_blocks_b, block_size, H, D]
        v_blocks = v_cache[block_ids]
        k_seq = k_blocks.reshape(num_blocks_b * block_size, nheads_headdim_k)
        v_seq = v_blocks.reshape(num_blocks_b * block_size, nheads_headdim_v)

        k_seq = k_seq[:seq_len]
        v_seq = v_seq[:seq_len]

        k_flat[start:end] = k_seq
        v_flat[start:end] = v_seq

    return k_flat, v_flat, cu_seqlens
