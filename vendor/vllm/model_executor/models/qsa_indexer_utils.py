import torch

try:
    # Import flashinfer before tilelang to prevent flashinfer's CudaRTLibrary
    # from picking up the invalid libcudart_stub.so introduced by tilelang.
    import flashinfer.comm
    import tilelang
    from tilelang import language as T

    can_use_tl = True
except ImportError:
    can_use_tl = False


def bf16_mqa_logits_torch(
    q: torch.Tensor,  # (m=24, nqhead=16, h=256)  bf16
    kv: torch.Tensor,  # (m=24, nkvhead=2, h=256) bf16
    weights: torch.Tensor,  # (m=24,nqhead=16) bf16
    cu_seqlen_ks: torch.Tensor,  # (m,) each q's start k [0,0,0,0,0,...0]
    cu_seqlen_ke: torch.Tensor,  # (m,) each q's stop  k [1,2,3,4,5,...24]
    score_scale: torch.dtype,
) -> torch.Tensor:
    """Compute FP16 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H_Q, D]. bf16
        kv: k_fp16, [N, H_K, D]. bf16
        weights: weights of shape [M, H_Q], dtype bf16
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    num_heads = q.shape[1]  # (qlen,nheadsq,headdim)
    num_kv_heads = k.shape[1]  # (klen,nheadsk,headdim)
    # GQA Key Expansion if needed
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)  # (klen,nheadsq,headdim)
    # [S_q, H, D] × [S_k, H, D] -> [S_q, S_k, H]
    scores = torch.einsum("qhd,khd->qkh", q, k)
    # Apply ReLU activation (Lightning Indexer uses ReLU instead of softmax)
    scores = torch.relu(scores)
    # Weight per head: [S_q, H] -> [S_q, 1, H]
    # logits:[S_q, S_k]
    logits = (scores * weights.unsqueeze(1)).sum(dim=-1)
    # Apply scaling
    logits = logits / score_scale
    # Apply Mask
    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    logits = logits.masked_fill(~mask, float("-inf")).float()

    return logits


if can_use_tl:

    @tilelang.jit
    def clean_logits_(
        threads: int = 512,
        block_K: int = 4096,
    ):
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        dtype = T.float
        indices_dtype = T.int32

        # 全部置成-inf
        @T.prim_func
        def clean_logits_kernel(
            Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
            CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
        ):
            with T.Kernel(seq_len, threads=threads) as bx:  # 每个cudablock处理一行。
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                cu_k_s = CuSeqLenKS[bx]
                cu_k_e = CuSeqLenKE[bx]

                for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                    for k_i in T.serial(block_K // threads):
                        idx = n_i * block_K + k_i * threads + tx
                        if idx < cu_k_s or idx >= cu_k_e:
                            Logits[bx, idx] = -T.infinity(dtype)

        return clean_logits_kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def mqa_attn_fp16_return_logits(
        heads,  # nqheads
        index_dim,  # h
        block_N=64,
        num_stages=3,
        threads=512,
        block_Q=None,
    ):
        if block_Q is None:
            block_Q = 128 // heads

        # 这个算子里，qk都是fp8. 应该差不多
        dtype = T.bfloat16
        accum_dtype = T.float32
        index_dtype = T.int32

        # 动态注册seq_len，seq_len_kv。不是固定传进来的
        seq_len = T.dynamic("seq_len")  # 动态的qlen
        seq_len_kv = T.dynamic("seq_len_kv")  # # 动态的klen

        # 输入的shape. 用动态的绑定。每次不同输入进来，按实际IndexQ的tensor推算
        # q(qlen,nheads=32, h=64) -> [seq_len * nheads, h=64]
        index_q_shape = [seq_len * heads, index_dim]
        # k(klen, h)
        index_k_shape = [seq_len_kv, index_dim]
        # index_k_scale_shape = [seq_len_kv]
        # (qlen,klen) 都是动态的。
        logits_shape = [seq_len, seq_len_kv]

        # 可以用来代替prefill的算子

        @T.prim_func
        def mqa_attn_fp16_return_logits_kernel(
            # q[qlen * nqheads, h=64]
            IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore  # [seq_len * nqheads=32, h=64]
            # k[kv_len, h=64]
            IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore  # [seq_len_kv, h=64]
            # IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
            # logits(qlen,klen) 都是动态的。 accum_dtype = T.float32
            Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
            # W[qlen, nqheads=32]  fp32
            Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
            # (qlen,)  每个seq的kstart, kend. int32
            CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
                # 按qlen, 切blockQ.
                # 每个cudablock, 要处理的内容。 bx这里就是block_idx.x
                # T.kernel(*blocks, *threads)
                # 对应blockIndex.x, y, z
                # 一个cudablock内

                # #exactly __shared__ sm[][]
                # smem_q(blockm*qheads,h)
                index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
                # smem_k(blockN=blockk,h)   (k是N方向)
                index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
                # index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)

                # regs
                # s(block_k, blockm*nqheads)
                s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
                # s(block_k, blockm, nqheads))
                s_reshaped = T.reshape(s, (block_N, block_Q, heads))
                # logits(block_k, blockm) fp32
                logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
                # w(blockm, nqheads) fp32
                weights = T.alloc_fragment([block_Q, heads], accum_dtype)

                # 本cudablock，处理的start_m:(bx开始, block_idx.x * blockm)
                seq_len_i = bx * block_Q

                # 单个值。min/max
                # 本cudablock中的qtoken,其kv的范围。（在klen中）
                cu_k_s_min = T.alloc_var(index_dtype)
                cu_k_e_max = T.alloc_var(index_dtype)

                cu_k_s_min = 2147483647
                cu_k_e_max = -2147483648

                for bq_i in T.serial(block_Q):  # 串行的，非并行
                    # 遍历每个qtoken
                    # noqa: E501 CuSeqLenKS[seq_len_i + bq_i]: 本cudablock中的bqi个qtoken,其kvstart(不一定相同)
                    cu_k_s_min = T.min(
                        cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv)
                    )  # noqa: E501
                for bq_i in T.serial(block_Q):
                    cu_k_e_max = T.max(
                        cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv)
                    )  # noqa: E501

                # g->S
                # # noqa: E501 [qlen * nqheads, h=64] -> m方向[start_m,0], 复制一个blockm大小，到smem_q(blockm*qheads,h)
                T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
                # # noqa: E501 W[qlen, nqheads=32] -> m方向[start_m,0],到w_r(blockm, nqheads)
                T.copy(Weights[seq_len_i, 0], weights)

                for nbn_i in T.Pipelined(
                    T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages
                ):  # noqa: E501
                    # noqa: E501 迭代kv:本cudablock中的qtoken,其kv的范围固定。划成block_N大小，pipeline加载
                    # 第nbn_i个kv块，在kvlen中offset是startkv + i*blockN   g->S
                    # k[kv_len, h=64] -> smem_k(blockN=blockk,h)
                    T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                    T.gemm(
                        index_k_shared,  # smem_k(blockN=blockk,h)
                        index_q_shared,  # noqa: E501 smem_q(blockm*qheads,h) -> (h,blockm*qheads)
                        # smem_k(blockN=blockk,h) * smem_q(h,blockm*qheads)
                        s,  # s(block_k, blockm*nqheads)
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    # 多线程并行
                    for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                        # # noqa: E501 每个s[block_k,blockq, qheadi], 乘上w[blockq, qheadi].
                        # s取的是max(s,0). 相当于relu。
                        s_reshaped[bn_i, bq_i, h_i] = (
                            T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                        )  # noqa: E501

                    # 有reduce_sum
                    # s(block_k, blockm, nqheads) -> logits(block_k, blockm))
                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for bq_i, bn_i in T.Parallel(block_Q, block_N):
                        # r-> G
                        # # noqa: E501 Logits(start_m + qtoken_i, start_k+ blockk_i*blockN + ktokenj ) = logits(block_k, blockm))
                        # 其他地方是-inf,相当于mask了。
                        Logits[
                            seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i
                        ] = logits[bn_i, bq_i]

        return mqa_attn_fp16_return_logits_kernel

    def mqa_attn_fp16_return_logits_interface(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True
    ):
        # q:(qlen, nqheads, h)
        seq_len, heads, index_dim = q.shape
        # kv:(klen, h)
        seq_len_kv = kv.shape[0]

        clean_logits_kernel = clean_logits_()

        # here
        mqa_attn_return_logits_kernel = mqa_attn_fp16_return_logits(
            heads=heads, index_dim=index_dim
        )  # noqa: E501
        # logits(qlen,klen)
        logits = torch.empty(
            [seq_len, seq_len_kv], device=q.device, dtype=torch.float32
        )

        mqa_attn_return_logits_kernel(
            q.view(seq_len * heads, index_dim),
            kv,
            logits,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
        if clean_logits:
            clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
        return logits


@torch.compile
def fp16_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    score_scale: float,
    is_context_lens_2d=False,
    max_seq_len=0,
):
    """Compute FP16 MQA logits using paged KV-cache.
    Args:
        q: Query tensor of shape [B, next_n, H_Q, D]. dtype `torch.bfloat16`.
        kv_cache: Paged KV-cache with shape
            [num_blocks, block_size, H_KV, D], dtype `torch.bfloat16`
        weights: Tensor of shape [B * next_n, H_Q], dtype `torch.bfloat16`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, num_kv_heads, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )  # noqa: E501
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = (
            torch.full((next_n,), context_len, device="cuda", dtype=torch.int32)
            if is_context_lens_2d
            else torch.arange(context_len - next_n, context_len, device="cuda")
        )  # noqa: E501
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )

        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]  # noqa: E501 [num_blocks, block_size, kv_heads, dim]
        # noqa E501 [num_blocks, block_size, kv_heads, dim] -> [kv_heads, h, num_blocks, block_size]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(
            kv_slice.size(2), dim, -1
        )  # [kv_heads, dim, total_tokens]
        if num_kv_heads < heads:
            repeat_factor = heads // num_kv_heads
            kx = kx.repeat_interleave(repeat_factor, dim=0)
        #  (m,nqheads,h) ->  (nqheads,m,h)
        qx = q[i].transpose(0, 1)  # noqa: E501 q[i]: [next_n, heads, dim] -> [heads, next_n, dim]
        s = torch.matmul(qx, kx).to(logits.dtype)  # noqa: E501 [heads, next_n, dim] @ [nheads, dim, total_tokens] -> [heads, next_n, total_tokens]

        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (
            k_offsets[None, :] <= q_offsets[:, None]
        )
        s = torch.where(mask[None, :, :], s, float("-inf"))  # noqa: E501 mask shape: [1, next_n, total_tokens]
        s = torch.relu(s) * weight_slice[..., None]  # noqa: E501 weight_slice: [heads, next_n] -> [heads, next_n, 1]
        s = s.sum(dim=0)  # [next_n, total_tokens]
        logits[i * next_n : (i + 1) * next_n, :total_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
        )  # noqa: E501
    logits.div_(score_scale)
    return logits


@torch.compile
def fp16_paged_mqa_logits_torch_optimize(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    score_scale: float,
    is_context_lens_2d=False,
    max_seq_len=0,
):
    """Compute FP16 MQA logits using paged KV-cache.
    Args:
        q: Query tensor of shape [B, next_n, H_Q, D]. dtype `torch.bfloat16`.
        kv_cache: Paged KV-cache with shape [num_blocks,
            block_size, H_KV, D], dtype `torch.bfloat16`
        weights: Tensor of shape [B * next_n, H_Q], dtype `torch.bfloat16`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    batch_size, next_n, heads, dim = q.size()
    assert next_n == 1
    num_block, block_size, num_kv_heads, dim = kv_cache.size()

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )  # noqa: E501
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )  # noqa: E501

        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]  # [num_blocks, block_size, kv_heads, dim]
        # noqa: E501 [num_blocks, block_size, kv_heads, dim] -> [kv_heads, h, num_blocks, block_size]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(
            kv_slice.size(2), dim, -1
        )  # [kv_heads, dim, total_tokens]

        #  (m,nqheads,h) ->  (nqheads,m,h)
        qx = q[i].transpose(0, 1)  # noqa: E501 q[i]: [next_n, heads, dim] -> [heads, next_n, dim]
        s = torch.matmul(qx, kx).to(logits.dtype)  # noqa: E501 [heads, next_n, dim] @ [nheads, dim, total_tokens] -> [heads, next_n, total_tokens]

        total_len = num_blocks * block_size
        s = (
            torch.relu(s) * weight_slice[..., None]
        )  # weight_slice: [heads, next_n] -> [heads, next_n, 1]
        s = s.sum(dim=0)  # [next_n, total_tokens]
        logits[i * next_n : (i + 1) * next_n, :total_len] = s
    logits.div_(score_scale)
    return logits


if can_use_tl:

    def get_tilelang_nomtp_config(batch_size, max_context_len):
        """
        根据 Benchmark 数据硬编码的最优配置选择器
        """
        # 默认值
        threads = 128
        num_stages = 3
        block_per_cta = 8
        if batch_size >= 64:
            if max_context_len <= 256:
                num_stages = 4
                block_per_cta = 4
            elif max_context_len <= 512:
                num_stages = 3
                block_per_cta = 8
            elif max_context_len <= 1024:  # 1k 左右
                num_stages = 5
                block_per_cta = 16
            elif max_context_len <= 2048:  # 2k 左右
                num_stages = 4
                block_per_cta = 32
            elif max_context_len <= 4096:  # 4k 左右
                num_stages = 4
                block_per_cta = 64
            elif max_context_len <= 8192:  # 8k 左右
                num_stages = 5
                block_per_cta = 32
            elif max_context_len <= 16384:
                num_stages = 4
                block_per_cta = 64
            elif max_context_len <= 32768:
                num_stages = 4
                block_per_cta = 16
            else:  # 16k 及以上
                block_per_cta = 16
                num_stages = 4

        return block_per_cta, num_stages, threads

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def fp16_paged_mqa_logits_tilelang(
        heads,  # nqheads
        head_dim,  # h
        block_size=64,  # physical block_size
        block_per_cta=8,
        num_stages=3,
        threads=128,
    ):
        dtype = T.bfloat16
        accum_dtype = T.float32

        # 每个 CTA 处理的 KV Block 数量
        BLOCKS_PER_CTA = block_per_cta
        block_K = block_size

        batch_size = T.dynamic("batch_size")
        num_blocks = T.dynamic("num_blocks")
        max_blocks_per_seq = T.dynamic("max_blocks")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def paged_mqa_kernel(
            Q: T.Tensor((batch_size, 1, heads, head_dim), dtype),  # noqa: E501 type: ignore
            K_cache: T.Tensor((num_blocks, block_size, 1, head_dim), dtype),  # noqa: E501 type: ignore
            Weights: T.Tensor((batch_size, heads), dtype),  # type: ignore
            BlockTables: T.Tensor((batch_size, max_blocks_per_seq), T.int32),  # noqa: E501 type: ignore
            ContextLens: T.Tensor((batch_size,), T.int32),  # type: ignore
            Logits: T.Tensor((batch_size, max_model_len), accum_dtype),  # noqa: E501 type: ignore
            score_scale: T.float32,
        ):
            with T.Kernel(
                batch_size,
                T.ceildiv(max_blocks_per_seq, BLOCKS_PER_CTA),
                threads=threads,
            ) as (bx, b_group):  # noqa: E501
                # Shared Memory
                index_q_shared = T.alloc_shared([heads, head_dim], dtype)
                index_k_shared = T.alloc_shared([block_K, head_dim], dtype)

                # Fragments
                weights_fr = T.alloc_fragment([heads], dtype)
                s = T.alloc_fragment([block_K, heads], accum_dtype)
                logits_res = T.alloc_fragment([block_K], accum_dtype)

                # 1. G->S
                T.copy(Q[bx, 0, :, :], index_q_shared)
                T.copy(Weights[bx, :], weights_fr)

                cur_seqlen_k = ContextLens[bx]

                # 2. kv iter
                for k_idx in T.Pipelined(BLOCKS_PER_CTA, num_stages=num_stages):  # noqa: E501
                    # 计算当前在序列中的 block index
                    current_block_in_seq = b_group * BLOCKS_PER_CTA + k_idx
                    if current_block_in_seq * block_K < cur_seqlen_k:
                        # 3. Load K Block
                        physical_block_idx = BlockTables[bx, current_block_in_seq]
                        T.copy(K_cache[physical_block_idx, :, 0, :], index_k_shared)

                        # 4. GEMM
                        # noqa: E501 index_k_shared (64, h) @ index_q_shared (heads, h).T -> s (64, heads)
                        T.gemm(
                            index_k_shared,
                            index_q_shared,
                            s,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                        # 5. ReLU + Weight + ReduceSum
                        for ki, hi in T.Parallel(block_K, heads):
                            s[ki, hi] = T.max(s[ki, hi], 0.0) * weights_fr[hi]

                        T.reduce_sum(s, logits_res, dim=1, clear=True)

                        # 6. r-> Global Memory
                        for ki in T.Parallel(block_K):
                            global_pos = current_block_in_seq * block_K + ki
                            if global_pos < cur_seqlen_k:
                                Logits[bx, global_pos] = logits_res[ki] / score_scale

        return paged_mqa_kernel

    def tilelang_paged_mqa_logits_interface(
        q,  # [B, next_n, H_Q, D]
        kv_cache,  # [num_blocks, block_size, H_KV, D]
        weights,  # [B * next_n, H_Q]
        context_lens,  # [B]
        block_tables,  # [B, max_blocks]
        schedule_metadata,
        max_model_len,
        score_scale,
        max_seq_len,
    ):
        batch_size, next_n, heads, head_dim = q.shape
        num_blocks, block_size, kv_heads, _ = kv_cache.shape
        qlen = batch_size * next_n
        assert next_n == 1
        assert kv_heads == 1
        b_cta, stages, th = get_tilelang_nomtp_config(batch_size, max_seq_len)

        mqa_attn_return_logits_kernel = fp16_paged_mqa_logits_tilelang(
            heads=heads,
            head_dim=head_dim,
            block_size=block_size,
            block_per_cta=b_cta,
            num_stages=stages,
            threads=th,
        )  # noqa: E501

        logits = torch.full(
            (qlen, max_model_len), float("-inf"), device=q.device, dtype=torch.float32
        )

        weights = weights.reshape(weights.shape).contiguous()
        mqa_attn_return_logits_kernel(
            q,
            kv_cache.view(num_blocks, block_size, 1, head_dim),
            weights,
            block_tables,
            context_lens,
            logits,
            score_scale,
        )

        return logits

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def fp16_paged_mqa_mtp_v2_big_gemm(
        heads,
        head_dim,
        next_n,
        batch_size,
        num_blocks,
        max_blocks,
        max_model_len,
        block_size=64,
        block_per_cta=16,
        num_stages=3,
        threads=256,
    ):
        dtype = T.bfloat16
        accum_dtype = T.float32
        total_heads = next_n * heads  # 4 * 16 = 64
        block_K = block_size

        @T.prim_func
        def paged_mqa_mtp_kernel(
            Q: T.Tensor((batch_size, total_heads, head_dim), dtype),  # noqa: E501 type: ignore
            K_cache: T.Tensor((num_blocks, block_size, 1, head_dim), dtype),  # noqa: E501 type: ignore
            Weights: T.Tensor((batch_size, total_heads), dtype),  # noqa: E501 type: ignore
            BlockTables: T.Tensor((batch_size, max_blocks), T.int32),  # noqa: E501 type: ignore
            ContextLens: T.Tensor((batch_size,), T.int32),  # noqa: E501 type: ignore
            Logits: T.Tensor((batch_size, next_n, max_model_len), accum_dtype),  # noqa: E501 type: ignore
            score_scale: T.float32,
        ):
            with T.Kernel(
                batch_size, T.ceildiv(max_blocks, block_per_cta), threads=threads
            ) as (bx, b_group):  # noqa: E501
                # 1. 资源分配
                index_q_shared = T.alloc_shared([total_heads, head_dim], dtype)  # noqa: E501
                index_k_shared = T.alloc_shared([block_K, head_dim], dtype)  # noqa: E501

                # 将 Fragment 声明为 3D 形状 [64, 4, 16]
                # 这样可以在不改变 GEMM 逻辑的前提下，对最后一维进行 reduce_sum
                weights_fr = T.alloc_fragment([total_heads], dtype)
                # 使用 reshape 产生一个 2D 视图供计算使用
                weights_fr_res = T.reshape(weights_fr, (next_n, heads))

                # s(block_k, blockm*nqheads)
                s = T.alloc_fragment([block_K, total_heads], accum_dtype)
                # s(block_k, blockm, nqheads))
                s_reshaped = T.reshape(s, (block_K, next_n, heads))
                # 存储规约后的结果 [64, 4]
                logits_res = T.alloc_fragment([block_K, next_n], accum_dtype)  # noqa: E501

                # 预加载 (Block 级)
                T.copy(Q[bx, :, :], index_q_shared)
                # 权重同样使用 2D 视图
                T.copy(Weights[bx, :], weights_fr)

                cur_seqlen_k = ContextLens[bx]

                for k_tile_idx in T.Pipelined(block_per_cta, num_stages=num_stages):  # noqa: E501
                    current_block_in_seq = b_group * block_per_cta + k_tile_idx

                    if current_block_in_seq * block_K < cur_seqlen_k:
                        physical_block_idx = BlockTables[bx, current_block_in_seq]
                        T.copy(K_cache[physical_block_idx, :, 0, :], index_k_shared)

                        # 2. Big GEMM (64 x 64)
                        T.gemm(
                            index_k_shared,
                            index_q_shared,
                            s,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                        # 3. 寄存器内 ReLU + Weights
                        for ki, ni, hi in T.Parallel(block_K, next_n, heads):
                            s_reshaped[ki, ni, hi] = (
                                T.max(s_reshaped[ki, ni, hi], 0.0)
                                * weights_fr_res[ni, hi]
                            )  # noqa: E501

                        # 4. 核心：在寄存器级别调用 reduce_sum
                        # 规约 dim=2 (heads 维度)，得到 [64, 4] 的结果
                        T.reduce_sum(s_reshaped, logits_res, dim=2, clear=True)

                        # 5. 统一写回 Logits
                        for ni, ki in T.Parallel(next_n, block_K):
                            q_offset = cur_seqlen_k - next_n + ni + 1
                            global_k_pos = current_block_in_seq * block_K + ki
                            if global_k_pos < q_offset:
                                Logits[bx, ni, global_k_pos] = (
                                    logits_res[ki, ni] / score_scale
                                )  # noqa: E501

        return paged_mqa_mtp_kernel

    def get_tilelang_mtp_config(batch_size, max_context_len):
        """
        根据 Benchmark 数据硬编码的最优配置选择器
        """
        # 默认值
        threads = 256
        num_stages = 3
        block_per_cta = 16

        if batch_size <= 32:  # 低并发场景 (参考 Batch 16 数据)
            if max_context_len >= 8000:
                num_stages = 3
                block_per_cta = 32
        else:  # 高并发场景 (参考 Batch 128 数据)
            if max_context_len <= 2048:  # 2k 左右
                num_stages = 3
                block_per_cta = 16
            elif max_context_len <= 4096:  # 4k 左右
                num_stages = 5
                block_per_cta = 16
            elif max_context_len <= 8192:  # 8k 左右
                num_stages = 5
                block_per_cta = 32
            else:  # 8k 及以上,含16k
                block_per_cta = 64
                num_stages = 4

        return block_per_cta, num_stages, threads

    def tilelang_paged_mqa_logits_mtp_interface(
        q,  # [B, next_n, H_Q, D]
        kv_cache,  # [num_blocks, block_size, H_KV, D]
        weights,  # [B * next_n, H_Q]
        context_lens,  # [B]
        block_tables,  # [B, max_blocks]
        schedule_metadata,
        max_model_len,
        score_scale,
        max_seq_len,
    ):
        batch_size, next_n, heads, head_dim = q.shape
        num_blocks, block_size, _, _ = kv_cache.shape
        max_blocks = block_tables.shape[-1]

        # 在输入端平铺 Q 和 Weights
        q = q.view(batch_size, next_n * heads, head_dim).contiguous()
        weights = weights.reshape(batch_size, next_n * heads).contiguous()

        b_cta, stages, th = get_tilelang_mtp_config(batch_size, max_seq_len)

        kernel = fp16_paged_mqa_mtp_v2_big_gemm(
            heads=heads,
            head_dim=head_dim,
            next_n=next_n,
            batch_size=batch_size,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            max_model_len=max_model_len,  # noqa: E501
            block_size=block_size,
            num_stages=stages,  # tuned
            threads=th,
            block_per_cta=b_cta,
        )

        # weights = weights.view(batch_size, next_n, heads).contiguous()
        logits = torch.full(
            (batch_size, next_n, max_model_len),
            float("-inf"),  # noqa: E501
            device=q.device,
            dtype=torch.float32,
        )

        kernel(
            q, kv_cache, weights, block_tables, context_lens, logits, float(score_scale)
        )

        return logits.view(batch_size * next_n, max_model_len)
