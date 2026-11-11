from typing import Optional

import torch
import triton
import triton.language as tl
from einops import repeat

from vllm.model_executor.layers.fla.ops.index import (
    prepare_token_indices_fast,
)
from vllm.platforms import current_platform


def sparse_gqa_pytorch(
    query: torch.Tensor,  # [sq, num_heads, head_size]
    key: torch.Tensor,  # [sk, num_kv_heads, head_size]
    value: torch.Tensor,  # [sk, num_kv_heads, head_size]
    query_len: int,
    kv_len: int,
    indices: torch.Tensor,  # shape [sq, topk]
    sm_scale: Optional[float] = None,
    is_causal: bool = True,
):
    input_dtype = query.dtype
    sq, hq, dim = query.shape
    sk, hkv, _ = key.shape

    mask = query.new_zeros(hkv, sq, sk + 1, dtype=torch.bool, device=indices.device)
    # (sq,topk) -> (hkv,sq,topk)
    indices = repeat(indices, "m t -> h m t", h=hkv)
    indices_long = indices.to(torch.int64)
    # invalid become sk，will drop later: (m,hkv,topk)
    indices_safe = torch.where(indices_long >= 0, indices_long, sk)
    # (hkv, sq, sk+1)
    mask.scatter_(2, indices_safe, 1)
    mask = mask[..., :-1]

    assert sk - sq >= 0
    if is_causal:
        empty_mask = torch.ones(query_len, kv_len, device=mask.device)
        causal_mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        mask = mask & (~causal_mask)

    # TODO: may be more effevtive
    # if is_causal:
    #     idx_q = torch.arange(sq, device=q.device)[:, None]
    #     idx_k = torch.arange(sk, device=q.device)[None, :]
    #     offset = max(sk - sq, 0)
    #     causal_mask = idx_k <= (idx_q + offset)
    #     mask = mask & causal_mask

    # Expand mask for grouped heads: [hkv, sq, sk] -> [hkv, 1, sq, sk]
    mask = mask.unsqueeze(1)
    # Reshape query for GQA: [sq, hq, dim] -> [sq, hkv, g, dim]
    g = hq // hkv  # heads per kv group
    query = query.view(sq, hkv, g, dim)

    # Compute attention scores in float32 for numerical stability
    # [sq, hkv, g, dim], [sk, hkv, dim] -> [hkv, g, sq, sk]
    score = torch.einsum("mghd,ngd->ghmn", query, key)

    # Apply scale and mask
    score.mul_(sm_scale)
    score.masked_fill_(~mask, float("-inf"))

    # Softmax over key positions (in float32)
    p = score.softmax(dim=-1)

    # Handle NaN (can happen if all positions are masked)
    p.nan_to_num_(nan=0.0)

    # Reshape probabilities
    p = p.view(hkv, g, sq, sk)

    # Compute output in float32, then cast back
    o = torch.einsum("ghmn,ngd->mghd", p, value)

    # Reshape back: [sq, hkv, g, dim] -> [sq, hq, dim]
    o = o.reshape(sq, hq, dim)

    return o.to(input_dtype)


def sparse_gqa_pytorch_ref(
    query: torch.Tensor,  # [sq, num_heads, head_size]
    key: torch.Tensor,  # [sk, num_kv_heads, head_size]
    value: torch.Tensor,  # [sk, num_kv_heads, head_size]
    indices: torch.Tensor,  # shape [sq, topk]
    sm_scale: Optional[float] = None,
    is_causal: bool = True,
):
    input_dtype = query.dtype  # Save input dtype for later conversion
    b, sq, hq, dim = query.shape
    _, sk, hkv, _ = key.shape

    # Expand indices for kv heads: [b, sq, topk] -> [b, hkv, sq, topk]
    indices_expanded = indices.unsqueeze(1).expand(-1, hkv, -1, -1)

    # Build sparse mask from indices
    # Create a mask of shape [b, hkv, sq, sk+1] and scatter 1s at indices
    mask = query.new_zeros(b, hkv, sq, sk + 1, dtype=torch.bool).scatter(
        3, indices_expanded.long(), 1
    )
    mask = mask[..., :-1]  # Remove the extra padding position

    # Apply causal mask if needed
    if is_causal:
        q_idx = torch.arange(sq, device=query.device)
        k_idx = torch.arange(sk, device=query.device)
        causal_mask = k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)
        mask = mask & causal_mask

    # Expand mask for grouped heads: [b, hkv, sq, sk] -> [b, hkv, 1, sq, sk]
    mask = mask.unsqueeze(2)
    # Reshape query for GQA: [b, sq, hq, dim] -> [b, sq, hkv, g, dim]
    g = hq // hkv  # heads per kv group
    query = query.view(b, sq, hkv, g, dim)

    # Compute attention scores in float32 for numerical stability
    score = torch.einsum("bmghd,bngd->bghmn", query.float(), key.float())

    # Apply scale and mask
    score = score * sm_scale
    score = score.masked_fill(~mask, float("-inf"))

    # Softmax over key positions (in float32)
    p = score.softmax(dim=-1)

    # Handle NaN (can happen if all positions are masked)
    p = torch.nan_to_num(p, nan=0.0)

    # Reshape probabilities
    p = p.view(b, hkv, g, sq, sk)

    # Compute output in float32, then cast back
    o = torch.einsum("bghmn,bngd->bmghd", p, value.float())

    # Reshape back: [b, sq, hkv, g, dim] -> [b, sq, hq, dim]
    o = o.reshape(b, sq, hq, dim)

    return o.to(input_dtype)


try:
    # Import flashinfer before tilelang to prevent flashinfer's CudaRTLibrary
    # from picking up the invalid libcudart_stub.so introduced by tilelang.
    import flashinfer.comm
    import tilelang
    from tilelang import language as T

    can_use_tl = True
except ImportError:
    can_use_tl = False

if can_use_tl:

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def sparse_gqa_prefill_nopage(
        heads,
        dim,
        topk,
        heads_kv,
        sm_scale=None,
        is_causal=True,
        block_I=32,
        num_stages=2,
        threads=128,
    ):
        assert dim == tilelang.math.next_power_of_2(dim), (
            f"haven't check padding correctness yet, dim={dim}"
        )
        assert is_causal, "non-casual is not supported"
        assert topk % block_I == 0, (
            "otherwise will load some index=0 thus causing wrong kv to be loaded"
        )
        if sm_scale is None:
            sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
        else:
            sm_scale = sm_scale * 1.44269504

        batch_plus_one = T.dynamic("batch_plus_one")
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
        heads_per_group = heads // heads_kv  # 16:2 = 8
        padded_H = max(16, ((heads_per_group + 15) // 16) * 16)  # 16
        real_H = heads_per_group  # 每个kheads,对应的qheqads数目 （在一个cudalock里算）
        indices_dtype = "int32"
        dtype = "bfloat16"
        accum_dtype = "float"

        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim

        q_shape = [seq_len, heads, dim]  # (m,qheads,h)
        k_shape = [seq_len_kv, heads_kv, dim]  # (n, kvheads,h)
        v_shape = [seq_len_kv, heads_kv, dim]
        o_shape = [seq_len, heads, dim]  # (m,qheads,h)
        indices_shape = [seq_len, topk]  # (m, topk)
        lse_shape = [seq_len, heads]
        cu_seqlens_q_shape = [batch_plus_one]  # (m+1)
        cu_seqlens_k_shape = [batch_plus_one]
        token_indices_q_shape = [seq_len, 2]  # (m,2)

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(k_shape, dtype),  # type: ignore
            V: T.Tensor(v_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            CuSeqlensQ: T.Tensor(cu_seqlens_q_shape, indices_dtype),  # type: ignore
            CuSeqlensK: T.Tensor(cu_seqlens_k_shape, indices_dtype),  # type: ignore
            TokenIndicesQ: T.Tensor(token_indices_q_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        ):
            with T.Kernel(seq_len, heads_kv, threads=threads) as (
                bx,
                by,
            ):
                #  每个cudablock,处理一个seq中的一个qtoken(1/m)。
                #  其中的一个headkv (一共2个headkv)
                # (nqheads, h)
                Q_shared = T.alloc_shared([padded_H, D], dtype)
                # (blockI, h)
                K_shared = T.alloc_shared([BI, D], dtype)
                #  (blockI, h)
                V_shared = T.alloc_shared([BI, D], dtype)
                #  (blockI)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([padded_H, D], accum_dtype)
                acc_s = T.alloc_fragment([padded_H, BI], accum_dtype)
                S_shared = T.alloc_shared([padded_H, BI], dtype)
                sumexp = T.alloc_fragment([padded_H], accum_dtype)
                sumexp_i = T.alloc_fragment([padded_H], accum_dtype)
                alpha = T.alloc_fragment([padded_H], accum_dtype)
                m_i = T.alloc_fragment([padded_H], accum_dtype)
                m_i_prev = T.alloc_fragment(
                    [padded_H], accum_dtype
                )  # 累积每行qk的最大值。一个qtoken在一个cudablock里算，kv不分块

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

                # 该seq的bi, si
                # token_q, 所属的seq_id/batch_id; token_q, 在本seq中的id
                # (bx是该token在所有seq所有token中的id)
                b_i, s_i = TokenIndicesQ[bx, 0], TokenIndicesQ[bx, 1]
                # 该seq,qk本身的起止 （在传入的q,k中。传入的qk含所有seq）
                bos_q = CuSeqlensQ[b_i]
                bos_k = CuSeqlensK[b_i]
                eos_q = CuSeqlensQ[b_i + 1]
                eos_k = CuSeqlensK[b_i + 1]
                # 本cudablock的 headkv的id: g_i
                g_i = by
                # 该qtoken,在本seq中的qoffset
                q_i = s_i

                # 本seq的qlen,klen
                q_len = eos_q - bos_q
                kv_len = eos_k - bos_k
                # kv_len - q_len： 本seq, 首个q之前的kvtoken数目
                kv_q_offset = T.if_then_else(
                    (kv_len - q_len) > 0,
                    kv_len - q_len,
                    0,
                )
                # 该qtoken, 在kv方向的offset, 超过的看不到
                max_kv_i = kv_q_offset + q_i

                H0 = g_i * real_H

                # Q(m,qheads,h) -> (qheads,h),  所有16个head先都copy进来
                #   "num_attention_heads": 16,"num_key_value_heads": 2,
                for h_i, d_i in T.Parallel(padded_H, D):
                    Q_shared[h_i, d_i] = T.if_then_else(
                        h_i < real_H,
                        # Q(m,qheads,h) -> (qheads,h),
                        # 处理的m是bos_q + s_i对应的q_token
                        Q[bos_q + s_i, H0 + h_i, d_i],
                        0,
                    )

                # load K:
                # topk个token, 划分成NI个block_I大小的块。一次load。
                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    # load第i_i个topk block_I

                    for bi_i in T.Parallel(BI):
                        #  其中每个kvtoken, 对应的mask. 这些topk,有超过本qtoken的，
                        # mask置为1; buffer里就是-1的，也不算，置为1
                        # Indices(m, topk)
                        # q_token:bos_q + s_i
                        # kvtoken: i_i*block_I + bi  (该kvtoken，在本seq中的offset)
                        #  blockI（block_I）： 本迭代最多要算的topk中的block_I个token, 
                        #   对应的mask (不算的置1)
                        mask[bi_i] = (
                            Indices[bos_q + s_i, i_i * BI + bi_i] <= max_kv_i
                        ) & (Indices[bos_q + s_i, i_i * BI + bi_i] != -1)

                    # load第i_i个topk block_I
                    # K(klen, numkvhead,h)
                    for bi_i, d_i in T.Parallel(BI, D):
                        K_shared[bi_i, d_i] = K[
                            #  kv方向，qtoken[bos_q + s_i]对应的所有topk 
                            #   token在本seq中的offset:Indices[q_token]
                            # 其中本次迭代，加载topk中的block_I个kvtoken, 
                            # token bi_i 在本seq中的offset是
                            #   Indices[q_token，bi_i + i_i*block_I]
                            #  根据Indices，得到该token在本seq中的
                            #   offset: Indices[qtoken,kvtoken],
                            #   加上其他seq在kv方向的offset,
                            #  得到本kvtoken在所有seq中的offset
                            #  K(klen, numkvhead,h) 
                            #   -> kshares(ktoken_offset_in_allseqs, h) 
                            #   -> (blockI,h)
                            bos_k + Indices[bos_q + s_i, i_i * BI + bi_i], g_i, d_i
                        ]

                    for bi_i, d_i in T.Parallel(BI, D):  # v相同
                        V_shared[bi_i, d_i] = V[
                            bos_k + Indices[bos_q + s_i, i_i * BI + bi_i], g_i, d_i
                        ]

                    # acc_s（nqheads, blockI）. 
                    # blockI中，mask掉不算的KVlen位置，置为-inf
                    for h_i, bi_i in T.Parallel(padded_H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            (h_i < real_H) & mask[bi_i],
                            0,
                            -T.infinity(acc_s.dtype),
                        )

                    #  qk -> acc_s（nqheads, blockI）.  topk token提前mask掉的位置，
                    #   值还是-inf
                    # acc_s = acc_s + AB
                    T.gemm(
                        Q_shared,  # (qheads,h),  所有16个head先都copy进来
                        K_shared,  # (blockI, h)
                        acc_s,  # (qheads,blockI)
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(
                        m_i, m_i_prev
                    )  # -inf。 截止上次kvblock迭代的最大值。复制到m_i_prev中

                    # 截止当前每行的最大值。q方向。
                    # acc_s（nqheads, blockI） -> (nqheads)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)

                    # alpha: e(m-i1 - m)  (nqheads(m),)
                    for h_i in T.Parallel(padded_H):
                        alpha[h_i] = T.if_then_else(  # （m,）
                            h_i < real_H,
                            # e(m-i1 - m)
                            T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale),
                            0,
                        )

                    # e (acc_s（nqheads, blockI） - max) ->  (nqheads(m),blockI）)
                    for h_i, bi_i in T.Parallel(padded_H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            h_i < real_H,
                            T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale),
                            0,
                        )

                    # 每行的sumexp_i
                    #  (nqheads(m),)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)

                    # 修正sumexp_i：  (nqheads(m),)
                    for h_i in T.Parallel(padded_H):
                        if h_i < real_H:
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    # 修正acc_o （nqheads, blockI）
                    for h_i, d_i in T.Parallel(padded_H, D):
                        if h_i < real_H:
                            acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)

                    # pv
                    # p:（nqheads, blockI）  v(blockI, h) -> acc_o(nqheads,h)
                    # 其中V_shared也只加载了本次迭代blockI的所有topk token
                    #  S_shared里，不要的都已经是0了（e-inf）.
                    #   所以acc_o里，不要的kv没有加进最后结果中来。
                    T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # kvload迭代结束，最后rescale一次
                # Rescale ，acc_o和sumexp
                for h_i, d_i in T.Parallel(padded_H, D):
                    if h_i < real_H:
                        acc_o[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(padded_H):
                    if h_i < real_H:
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

                # acc_o(nqheads,h)， 写出到该qtoken对应位置
                for h_i, d_i in T.Parallel(real_H, D):
                    Output[bos_q + s_i, H0 + h_i, d_i] = acc_o[h_i, d_i]
                for h_i in T.Parallel(real_H):
                    Lse[bos_q + s_i, H0 + h_i] = sumexp[h_i]

        return main

    def sparse_gqa_tl_nopaged_interface(
        q, k, v, indices, cu_seqlens_q, cu_seqlens_k, sm_scale=None, is_causal=True
    ):

        assert (
            q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and indices.is_contiguous()
        )
        seq_len, heads, dim = q.shape
        _, heads_kv, _ = k.shape
        topk = indices.shape[1]
        assert indices.shape == (seq_len, topk)

        assert cu_seqlens_q.shape == cu_seqlens_k.shape, (
            "cu_seqlens_q and cu_seqlens_k must have identical shape"
        )

        # cu_seqlens_q： （bs+1）， each seq的start,end
        # token_indices: (bs,2), each token's offset_in_seq, seq_id
        # token_indices = prepare_token_indices(cu_seqlens_q)
        token_indices = prepare_token_indices_fast(cu_seqlens_q, seq_len)

        kernel = sparse_gqa_prefill_nopage(
            heads, dim, topk, heads_kv, sm_scale, is_causal
        )
        out, lse = kernel(q, k, v, indices, cu_seqlens_q, cu_seqlens_k, token_indices)
        return out, lse

    @tilelang.jit
    def sparse_kv_extraction(
        topk,
        n_kv_heads,
        head_dim,
        block_size=544,
        threads=512,
    ):
        num_blocks = T.dynamic("num_blocks")
        batch_size = T.dynamic("batch_size")
        max_blocks_per_seq = T.dynamic("max_blocks")
        dtype = "bfloat16"
        indices_dtype = "int32"

        @T.prim_func
        def kernel(
            KV_CACHE: T.Tensor((num_blocks, 2,block_size, n_kv_heads, head_dim), dtype),  # type: ignore
            BlockTables: T.Tensor((batch_size, max_blocks_per_seq), indices_dtype),  # type: ignore
            TopkIndices: T.Tensor((batch_size, topk), indices_dtype),  # type: ignore
            SeqlensK: T.Tensor((batch_size,), indices_dtype),  # type: ignore
            NewK: T.Tensor((batch_size, topk, n_kv_heads, head_dim), dtype),  # type: ignore
            NewV: T.Tensor((batch_size, topk, n_kv_heads, head_dim), dtype),  # type: ignore
        ):
            # Grid: 每个 CUDA Block 处理一个 sequence (batch) 中的一个 KV Head
            # bx: batch_idx, by: head_idx
            with T.Kernel(batch_size, n_kv_heads, threads=threads) as (bx, kvhead_i):
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                actual_k_len = SeqlensK[bx]

                warp_id = tx // 32  # 0, 1, 2, 3
                lane_id = tx % 32  # 0 ~ 31
                num_warps = threads // 32

                for j_base in T.serial(T.ceildiv(topk, num_warps)):
                    curr_j = j_base * num_warps + warp_id
                    token_pos = TopkIndices[bx, curr_j]
                    # token_pos = TopkIndices[bx, j]
                    # 合法性检查
                    if 0 <= token_pos < actual_k_len:
                        # 计算逻辑与物理映射
                        logical_blk_idx = token_pos // block_size
                        blk_offset = token_pos % block_size
                        p_blk_id = BlockTables[bx, logical_blk_idx]

                        for d_base in T.serial(
                            T.ceildiv(head_dim, 256)
                        ):  # 处理 head_dim=256 或 128
                            d_idx = d_base * 256 + lane_id * 8
                            if d_idx < head_dim:
                                # 编译器通常会将这两行编译为成对的 LDG.E.128 指令
                                k_reg = KV_CACHE[
                                    p_blk_id, 0, blk_offset, kvhead_i, d_idx : d_idx + 8
                                ]
                                v_reg = KV_CACHE[
                                    p_blk_id, 1, blk_offset, kvhead_i, d_idx : d_idx + 8
                                ]
                                NewK[bx, curr_j, kvhead_i, d_idx : d_idx + 8] = k_reg
                                NewV[bx, curr_j, kvhead_i, d_idx : d_idx + 8] = v_reg

        return kernel

    def sparse_kv_extraction_interface(kv_cache, block_tables, topk_indices, seqused_k):
        batch_size = topk_indices.shape[0]
        topk = topk_indices.shape[1]
        _, _, block_size, n_kv_heads, head_dim = kv_cache.shape

        kv_cache_physical = kv_cache.permute(1, 0, 2, 3, 4)
        assert kv_cache_physical.is_contiguous()

        # 输出 buffer：每个 q_token 抽取出的 topk 个 KV
        # 逻辑上对应新的 block_table 每一行就是一个连续的 topk 块
        decode_key_cache = torch.zeros(
            (batch_size, topk, n_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )
        decode_val_cache = torch.zeros(
            (batch_size, topk, n_kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )

        sparse_kv_extraction_kernel = sparse_kv_extraction(
            topk=topk, n_kv_heads=n_kv_heads, head_dim=head_dim, block_size=block_size
        )

        sparse_kv_extraction_kernel(
            kv_cache_physical,
            block_tables,
            topk_indices,
            seqused_k,
            decode_key_cache,
            decode_val_cache,
        )

        return decode_key_cache, decode_val_cache


NVIDIA_H20_TRITON_QSA_CONFIGS = [
    # (BLOCK_SIZE_N, num_warps, num_stages=2)
    (32, (32, 8, 2)),  # total_q <= 32
    (64, (64, 8, 2)),  # total_q <= 64
    (1024, (32, 4, 2)),  # total_q <= 1024  [128-1k]
    (float("inf"), (16, 1, 2)),  # total_q > 1024 (2k, 4k, 8k, 16k)
]

NVIDIA_L20C_TRITON_QSA_CONFIGS = [
    # (BLOCK_SIZE_N, num_warps, num_stages=2)
    (32, (32, 8, 2)),  # total_q <= 32
    (64, (64, 8, 2)),  # 64
    (128, (64, 4, 2)),  # 128
    (512, (32, 4, 2)),  # total_q <= 1024  256 512
    (float("inf"), (16, 1, 2)),  # total_q > 512 (1k 2k, 4k, 8k, 16k)
]


def get_best_config(total_q):
    device_name = current_platform.get_device_name().replace(" ", "_")
    if device_name == "NVIDIA_H20" or device_name == "NVIDIA_H20-3e":
        for limit, config in NVIDIA_H20_TRITON_QSA_CONFIGS:
            if total_q <= limit:
                return config
        return NVIDIA_H20_TRITON_QSA_CONFIGS[-1][1]
    elif (
        device_name == "NVIDIA_L20C"
        or device_name == "NVIDIA_L20D"
        or device_name == "NVIDIA_GB200"
    ):  # B200/GB200/B300  TODO:opt GB200/B300
        for limit, config in NVIDIA_L20C_TRITON_QSA_CONFIGS:
            if total_q <= limit:
                return config
        return NVIDIA_L20C_TRITON_QSA_CONFIGS[-1][1]
    else:
        raise ValueError(
            f"{device_name} not support yet, need to get best_config first,"
            f" for perf in vllm may not stable"
        )


@triton.jit
def sparse_gqa_fwd_kernel_triton(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    idx_ptr,
    cu_seqlens_ptr,
    scale,
    topk_val,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_im,
    stride_in,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_bh = tl.program_id(1)
    off_g = off_bh % NUM_KV_HEADS
    off_b = off_bh // NUM_KV_HEADS

    seq_start = tl.load(cu_seqlens_ptr + off_b).to(tl.int64)
    seq_end = tl.load(cu_seqlens_ptr + off_b + 1).to(tl.int64)
    off_m_rel = tl.program_id(0).to(tl.int64)
    off_m_abs = seq_start + off_m_rel

    if off_m_abs >= seq_end:
        return

    # --- 2. 主要优化：动态计算该行的有效 TopK 边界 ---
    #  假设 Prefill 是 Casual 的，Q 能看到的有效 KV 数量不会超过它在序列中的位置，
    # 可以利用这一信息直接缩减 for 循环范围，不需要访问 indices 内存，非常快
    q_idx_in_seq = off_m_rel
    #  如果 topk_val 是 2048，但当前 Token 是序列第 100 个，
    # 循环范围直接缩减到 112 (对齐后)
    row_topk = tl.minimum(topk_val, q_idx_in_seq + 1)
    # 向上对齐到 BLOCK_SIZE_N
    row_topk_aligned = ((row_topk + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * BLOCK_SIZE_N
    row_topk_limit = tl.minimum(topk_val, row_topk_aligned)

    # --- 3. 预加载 Q ---
    offs_h = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_head_start = off_g * GROUP_SIZE
    q_ptrs = (
        q_ptr
        + off_m_abs * stride_qm
        + (q_head_start + offs_h[:, None]) * stride_qh
        + offs_d[None, :] * stride_qd
    )
    mask_m = offs_h < GROUP_SIZE
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale * 1.4426950408).to(q.dtype)

    k_base = k_ptr + seq_start * stride_kn + off_g * stride_kh
    v_base = v_ptr + seq_start * stride_vn + off_g * stride_vh
    idx_row_ptr = idx_ptr + off_m_abs * stride_im

    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # --- 5. kv迭代  ---
    # range 现在是动态的 (row_topk_limit)，跳过后面的 -1
    for start_n in range(0, row_topk_limit, BLOCK_SIZE_N):
        curr_n = start_n + offs_n

        # 加载索引
        rel_idx = tl.load(
            idx_row_ptr + curr_n * stride_in, mask=curr_n < topk_val, other=-1
        )
        is_valid = rel_idx >= 0

        # 稀疏加载 K, V
        k = tl.load(
            k_base + rel_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=is_valid[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        v = tl.load(
            v_base + rel_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=is_valid[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )

        # QK 乘法
        s = tl.dot(q, k)
        s = tl.where(is_valid[None, :], s, -float("inf"))

        # Online Softmax
        m_next = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_next)
        p = tl.math.exp2(s - m_next[:, None])

        # 累加
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    # --- 6. 写回 ---
    acc = acc / l_i[:, None]
    o_ptrs = (
        o_ptr
        + off_m_abs * stride_om
        + (q_head_start + offs_h[:, None]) * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(o_ptr.type.element_ty), mask=mask_m[:, None])


def sparse_gqa_fwd_interface_triton(
    q, k, v, max_seqlen_k_prefill, indices, cu_seqlens, scale
):
    total_q = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    max_seqlen = max_seqlen_k_prefill

    # 一个 Block 处理一个 KV 组
    block_m = triton.next_power_of_2(group_size)
    if block_m < 16:
        block_m = 16

    o = torch.empty_like(q)
    grid = (max_seqlen, batch_size * num_kv_heads)

    #  assert num_q_heads==16 and num_kv_heads==2 and 
    #   indices.shape[1]==2048 and q.shape[-1]==256
    bn, warps, stages = get_best_config(total_q)
    fixed_config = {"BLOCK_SIZE_N": bn, "num_stages": stages, "num_warps": warps}

    sparse_gqa_fwd_kernel_triton[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=o,
        idx_ptr=indices,
        cu_seqlens_ptr=cu_seqlens,
        scale=scale,
        topk_val=indices.shape[1],
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_om=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        stride_im=indices.stride(0),
        stride_in=indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=block_m,
        HEAD_DIM=q.shape[2],
        **fixed_config,
    )
    return o


@triton.jit
def sparse_gqa_fwd_kernel_triton_ck(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    idx_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_aligned_ptr,
    actual_kv_lens_ptr,
    scale,
    topk_val,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_im,
    stride_in,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # --- 1. 定位 ---
    off_m_rel = tl.program_id(0).to(tl.int64)
    off_bh = tl.program_id(1)
    off_g = off_bh % NUM_KV_HEADS
    off_b = off_bh // NUM_KV_HEADS

    # 尽可能合并读取
    s_q = tl.load(cu_seqlens_q_ptr + off_b)
    e_q = tl.load(cu_seqlens_q_ptr + off_b + 1)

    off_m_abs = (s_q + off_m_rel).to(tl.int64)
    if off_m_abs >= e_q:
        return

    # 只加载必要的标量
    k_start = tl.load(cu_seqlens_k_aligned_ptr + off_b).to(tl.int64)  # 改int64
    kv_len = tl.load(actual_kv_lens_ptr + off_b).to(tl.int64)

    # --- 2. 预加载 Q ---
    offs_h = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # 这里的 mask_m 仅用于加载 Q，不进入循环
    mask_m = offs_h < GROUP_SIZE
    q_ptrs = (
        q_ptr
        + off_m_abs * stride_qm
        + (off_g * GROUP_SIZE + offs_h[:, None]) * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * (scale * 1.4426950408)).to(q.dtype)

    # --- 3. 初始化 ---
    # 保证 k_base 指向的是本 Batch 的开头
    k_base = k_ptr + off_g * stride_kh + k_start * stride_kn
    v_base = v_ptr + off_g * stride_vh + k_start * stride_vn
    idx_row_ptr = idx_ptr + off_m_abs * stride_im

    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)

    # 简化 row_topk 逻辑：如果是 Prefill (q=kv)，history_len 就是 0
    # 我们直接利用传入的 kv_len 和 e_q/s_q 计算
    row_topk = tl.minimum(topk_val, off_m_rel + (kv_len - (e_q - s_q)) + 1)
    row_topk_limit = tl.minimum(
        topk_val, ((row_topk + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * BLOCK_SIZE_N
    )

    # --- 4. For 循环 (追求极致精简) ---
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, row_topk_limit, BLOCK_SIZE_N):
        curr_n = start_n + offs_n

        # 1. 加载索引 (保持 int32)
        rel_idx = tl.load(
            idx_row_ptr + curr_n * stride_in, mask=curr_n < topk_val, other=-1
        )

        # 2. 只有最简单的有效性掩码
        is_valid = rel_idx >= 0

        # 3. 稀疏加载 K, V (直接使用 rel_idx，Triton 内部会自动处理 int32 乘法)
        k = tl.load(
            k_base + rel_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=is_valid[None, :],
            other=0.0,
        )
        v = tl.load(
            v_base + rel_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=is_valid[:, None],
            other=0.0,
        )

        # 4. QK 乘法
        s = tl.dot(q, k)
        s = tl.where(
            is_valid[None, :], s, -float("inf")
        )  # 移除 head_mask，只在这里判断 is_valid

        # 5. Softmax 迭代
        m_next = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_next)
        p = tl.math.exp2(s - m_next[:, None])

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    # --- 5. 写回 ---
    acc = acc / l_i[:, None]
    o_ptrs = (
        o_ptr
        + off_m_abs * stride_om
        + (off_g * GROUP_SIZE + offs_h[:, None]) * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(o_ptr.type.element_ty), mask=mask_m[:, None])


def sparse_gqa_fwd_interface_triton_ck(
    q,  # (m, qheads,h)
    k,
    v,  # (allneedtoken,kvheads,h)
    indices,
    prefill_cu_seqlens_q,
    cu_seqlens_k_aligned,
    prefill_seqused_k,
    scale,
):
    v = v.contiguous()
    k = k.contiguous()
    o = torch.empty_like(q)
    total_q = q.shape[0]
    num_kv_heads = k.shape[1]
    batch_size = prefill_cu_seqlens_q.shape[0] - 1
    max_q_len = (prefill_cu_seqlens_q[1:] - prefill_cu_seqlens_q[:-1]).max().item()
    # 一个 Block 处理一个 KV 组
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads

    # 一个 Block 处理一个 KV 组
    block_m = triton.next_power_of_2(group_size)
    if block_m < 16:
        block_m = 16
    grid = (max_q_len, batch_size * num_kv_heads)

    bn, warps, stages = get_best_config(total_q)
    fixed_config = {"BLOCK_SIZE_N": bn, "num_stages": stages, "num_warps": warps}

    sparse_gqa_fwd_kernel_triton_ck[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=o,
        idx_ptr=indices,
        cu_seqlens_q_ptr=prefill_cu_seqlens_q,
        cu_seqlens_k_aligned_ptr=cu_seqlens_k_aligned,
        actual_kv_lens_ptr=prefill_seqused_k,
        scale=scale,
        topk_val=indices.shape[1],
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_om=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        stride_im=indices.stride(0),
        stride_in=indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_SIZE_M=block_m,
        **fixed_config,
    )
    return o
