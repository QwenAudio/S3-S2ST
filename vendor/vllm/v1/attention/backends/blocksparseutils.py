# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional

import torch

import vllm.envs as envs

try:
    from sparse_attn import sparse_attn_func
except (ImportError, ModuleNotFoundError):
    try:
        from vllm.vllm_flash_attn import sparse_attn_func
    except (ImportError, ModuleNotFoundError):
        from vllm_flash_attn import sparse_attn_func

def convert_mask_to_compact_form_simple(mask_tensor):
    batchsize, numheads, qblocknum, kblocknum = mask_tensor.shape
    
    # compute block num
    block_counts = mask_tensor.sum(dim=-1)
    maxnumblockperrow = block_counts.max().item()
    
    # create col index tensor
    col_indices = torch.arange(kblocknum, device=mask_tensor.device)
    col_indices = col_indices.view(1, 1, 1, kblocknum).expand_as(mask_tensor)
    
    # apply mask
    masked_indices = torch.where(mask_tensor.bool(), 
                                 col_indices, 
                                 torch.tensor(65536, device=mask_tensor.device))
    
    # sort each line, move -1 to end pos
    sorted_indices, _ = torch.sort(masked_indices, dim=-1, descending=False)
    
    # slice max seeqlens
    compact_indices = sorted_indices[:, :, :, :maxnumblockperrow]
    
    return compact_indices.to(torch.int32), block_counts.to(torch.int32)


def find_blocks_chunked(
    input_tensor, 
    current_index, 
    threshold, 
    num_to_choose, 
    decoding: bool, 
    mode: str = "both", causal=True
):
    """
        Finds and selects relevant blocks of attention for transformer-based 
        models based on a threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor of shape 
            (batch_size, head_num, chunk_num, block_num).
        - current_index (int): The current index in the sequence processing.
        - threshold (float or None): A threshold value used to determine the 
            minimum attention weight sum.
        - num_to_choose (int or None): The number of blocks to be selected, 
            ensuring sufficient information retrieval.
        - decoding (bool): If True, operates in decoding mode; otherwise, 
            it's in encoding mode.
        - mode (str): Defines the processing mode, either 
            'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent 
            future information leakage.

        Returns:
        - torch.Tensor: A boolean mask of shape 
            (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    # 0 -- -- -- -- current_index
    # 0 -- -- -- -- -- current_index+1
    # 0 -- -- -- -- -- ----------- current_index + chunk_num - 1
    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, 
                           device=input_tensor.device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(
                        input_tensor, 
                        dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(
                        input_tensor, 
                        dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask
    input_tensor = input_tensor.to(float)
    
    if threshold is not None:
        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(float)
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            ).expand((batch_size, head_num, chunk_num, 1)).to(input_tensor.device)
        else:
            required_sum = total_sum * threshold
        if causal:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, 0] = 1
            mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            other_values = input_tensor.masked_fill(
                mask, 0
            )
            sorted_values, _ = torch.sort(
                other_values, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)

            sorted_values = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask,index,0)
            mask = mask.view(batch_size,head_num*chunk_num,block_num)
            index = index.view(batch_size,head_num*chunk_num,block_num)
            mask[ :, torch.arange(
                    mask.shape[1], 
                    device=mask.device
                ).unsqueeze(dim=-1),index] = True
            mask = mask.view(batch_size,head_num,chunk_num,block_num)
        else:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(
                input_tensor, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        # raise NotImplementedError("block num chunk prefill not impleted")
    
        topk_values, topk_indices = torch.topk(input_tensor, num_to_choose, dim = -1)
        mask = torch.zeros_like(input_tensor, dtype=torch.bool)

        mask.scatter_(-1,  topk_indices, True)

        causalmask = ((current_index + torch.arange(chunk_num)[:, None]) 
                      >= torch.arange(block_num)[None, :]).to(mask.device)
        mask = torch.where(causalmask == 0, False, mask)
        mask[:,:,:, 0] = True

    return mask


def topp_sample(query, key, toppk, sm_scale, blocksize = 128, fp8_attention=False):
    topp, topk = toppk
    
    if fp8_attention:
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
    
    query_reshape = query.view(query.shape[0], query.shape[1], 
                               -1, blocksize, query.shape[-1])
    key_reshape = key.view(key.shape[0], key.shape[1], -1, blocksize, key.shape[-1])
    query_mean = query_reshape.mean(-2)
    key_mean = key_reshape.mean(-2)

    input_tensor = torch.matmul(query_mean, key_mean.transpose(-1,-2))* sm_scale
    qkblockdiff = key_mean.shape[-2] - query_mean.shape[-2]

    causalmask = (
        qkblockdiff
        + torch.arange(input_tensor.shape[-2], device=input_tensor.device)[:, None]
        >= torch.arange(input_tensor.shape[-1], device=input_tensor.device)[None, :]
    )
    input_tensor = torch.where(causalmask == 0, float('-inf'), input_tensor)
    input_tensor = torch.nn.functional.softmax(input_tensor, dim = -1)

    if (abs(topk - 0.0) <= 1e-2):
        retmask = find_blocks_chunked(
            input_tensor, 
            qkblockdiff, 
            topp, 
            None, 
            decoding=False,
            mode="prefill",
            causal=True,
        )
    else:
        retmask = find_blocks_chunked(
            input_tensor, 
            qkblockdiff, 
            None, 
            int(input_tensor.shape[-1] * topk), 
            decoding=False,
            mode="prefill",
            causal=True,
        )

    return retmask


def block_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_pk,
    block_size_M: int = 128,
    block_size_N: int = 128,
    sm_scale: float = 1.0,
    fp8_attention: Optional[bool] = False,
    model_dtype: torch.dtype = torch.bfloat16
):
    batch_size, num_heads, context_size, head_dim = query.shape

    if (query.shape[2] % block_size_M != 0):
        pad = block_size_M - (query.shape[2] % block_size_M)
    else:
        pad = 0
    
    if (key.shape[2] % block_size_M != 0):
        padk = block_size_M - (key.shape[2] % block_size_M)
    else:
        padk = 0

    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, padk, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, padk, 0, 0, 0, 0])

    ret_mask = topp_sample(query, key, top_pk, sm_scale, block_size_M, fp8_attention)

    block_offset, block_count = convert_mask_to_compact_form_simple(ret_mask)
    block_offset *= block_size_M
    block_offset.sort(dim = -1)
    column_count = torch.zeros_like(block_count)
    column_index = torch.zeros_like(block_offset) 
    
    q = query.transpose(1,2).contiguous()
    k = key.transpose(1,2).contiguous()
    v = value.transpose(1,2).contiguous()
    
    if envs.VLLM_FP8_SPARSE_ATTENTION and not fp8_attention:
        q = torch.clamp(q, max=448.0, min=-448.0).to(torch.float8_e4m3fn)
        k = torch.clamp(k, max=448.0, min=-448.0).to(torch.float8_e4m3fn)
        v = torch.clamp(v, max=448.0, min=-448.0).to(torch.float8_e4m3fn)
    
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=True,
        softmax_scale=sm_scale,
        return_softmax_lse=True,
    )
    
    if envs.VLLM_FP8_SPARSE_ATTENTION or fp8_attention:
        out = out.to(model_dtype)

    out = out.transpose(1,2).contiguous()
    sparsetime = None

    return out[..., :context_size, :], lse, sparsetime
