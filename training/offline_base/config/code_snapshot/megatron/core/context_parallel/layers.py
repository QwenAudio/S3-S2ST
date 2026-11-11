try:
    from einops import rearrange
except ImportError:
    rearrange = None

from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
import torch

class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input, scatter_idx, gather_idx):

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = torch.distributed.get_world_size(group)
        
        if scatter_idx != 0:
          input_list = [t for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
          input_merged = torch.cat(input_list, dim=gather_idx)
          output = torch.empty_like(input_merged)
          torch.distributed.all_to_all_single(output, input_merged, group=group, async_op=False)
        else:
          output_merged = torch.empty_like(input).contiguous()
          torch.distributed.all_to_all_single(output_merged, input.contiguous(), group=group, async_op=False)
          output_list = [t for t in torch.tensor_split(output_merged, seq_world_size, scatter_idx)]
          output = torch.cat(output_list, dim=gather_idx)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        context_parallel_group (ProcessGroup): context parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention,
        context_parallel_group,
        scatter_idx=2,
        gather_idx=0,
    ):

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.cpg = context_parallel_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.world_size = torch.distributed.get_world_size(self.cpg)

    def forward(self, query, key, value, *args, **kwargs):
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        q_chunks = torch.chunk(query, self.world_size, self.scatter_idx)
        k_chunks = torch.chunk(key, self.world_size, self.scatter_idx)
        v_chunks = torch.chunk(value, self.world_size, self.scatter_idx)
        concated_chunk = []
        for q_c, k_c, v_c in zip(q_chunks, k_chunks, v_chunks):
            concated_c = torch.cat((q_c, k_c, v_c), dim=self.scatter_idx)
            concated_chunk.append(concated_c)
        qkv = torch.cat(concated_chunk, dim=self.scatter_idx)
        qkv_layer = _SeqAllToAll.apply(self.cpg, qkv, self.scatter_idx, self.gather_idx)
        query_layer, key_layer, value_layer = torch.split(qkv_layer, 
                                                          (query.shape[self.scatter_idx] // self.world_size, 
                                                           key.shape[self.scatter_idx] // self.world_size, 
                                                           value.shape[self.scatter_idx] // self.world_size), 
                                                           dim=self.scatter_idx)
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        output = _SeqAllToAll.apply(self.cpg, context_layer, self.gather_idx, self.scatter_idx)
        return output

class _DistFlashAttnVarlenWithKVGather(torch.autograd.Function):

    @staticmethod
    def prepare_cu_seqlens(cu_seqlens, cp_group):
        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)
        total_length = cu_seqlens[-1].item()
        assert total_length % cp_size == 0, \
            f'total length {total_length} can not be divided by cp size {cp_size}!'
        length_per_rank = total_length // cp_size
        # tokens for cur rank: [rank * length_per_rank, (rank + 1) * length_per_rank)
        left_token_boundary = cp_rank * length_per_rank
        right_token_boundary = (cp_rank + 1) * length_per_rank
        # [left, right] range in cu_seqlens which contains tokens for cur rank
        left = torch.searchsorted(cu_seqlens, left_token_boundary).item()
        left = left - 1 if cu_seqlens[left] != cp_rank * length_per_rank else left
        right = torch.searchsorted(cu_seqlens, right_token_boundary).item()

        # slice q into cp_size parts, only maintain valid tokens in range
        cu_seqlens_q = cu_seqlens[left: right + 1].clone()
        q_range = (left_token_boundary, right_token_boundary)
        cu_seqlens_q[0], cu_seqlens_q[-1] = q_range[0], q_range[1]
        cu_seqlens_q -= q_range[0]
        # the left boundary of kv need to maintain the full sequence
        cu_seqlens_k = cu_seqlens[left: right + 1].clone()
        k_range = (cu_seqlens_k[0].item(), right_token_boundary)
        cu_seqlens_k[0], cu_seqlens_k[-1] = k_range[0], k_range[1]
        cu_seqlens_k -= k_range[0]

        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

        return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, k_range

    @staticmethod
    def allgather(_tensor, comm_group):
        comm_size = torch.distributed.get_world_size(group=comm_group)
        full_tensor = torch.empty(
            size=(_tensor.shape[0] * comm_size, *_tensor.shape[1:]),
            dtype=_tensor.dtype,
            device=_tensor.device,
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor=full_tensor,
            input_tensor=_tensor,
            group=comm_group,
            async_op=False,
        )
        return full_tensor

    @staticmethod
    def reduce_scatter(_tensor, comm_group):
        comm_size = torch.distributed.get_world_size(group=comm_group)
        assert _tensor.shape[0] % comm_size == 0
        sub_tensor = torch.empty(
            size=(_tensor.shape[0] // comm_size, *_tensor.shape[1:]),
            dtype=_tensor.dtype,
            device=_tensor.device,
        )
        torch.distributed.reduce_scatter_tensor(
            output=sub_tensor,
            input=_tensor,
            group=comm_group,
            async_op=False,
        )
        return sub_tensor

    @staticmethod
    def forward(ctx, cp_group, q, k, v, cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, k_range, dropout_p, 
                softmax_scale, causal, window_size, deterministic,
                return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        num_head_k = k.shape[1]
        # assert num_head_k <= 2, \
        #     f'DistAttn with KVGather requires num_head for kv <= 2, but got {num_head_k}!'
        k = k.contiguous()
        v = v.contiguous()
        full_k = _DistFlashAttnVarlenWithKVGather.allgather(k, cp_group)
        full_v = _DistFlashAttnVarlenWithKVGather.allgather(v, cp_group)
        outputs = _flash_attn_varlen_forward(
            q=q,
            k=full_k[k_range[0]: k_range[1]],
            v=full_v[k_range[0]: k_range[1]],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            return_softmax=return_softmax and dropout_p > 0,
        )
        assert len(outputs) == 8, f'wrong flash-attn version!'
        # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
        out, softmax_lse, S_dmask = outputs[0], outputs[5], outputs[6]
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.cp_group = cp_group
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.k_range = k_range
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        cp_group = ctx.cp_group
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        k_range = ctx.k_range
        dropout_p = ctx.dropout_p
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal
        window_size = ctx.window_size
        deterministic = ctx.deterministic
        full_k = _DistFlashAttnVarlenWithKVGather.allgather(k, cp_group)
        full_v = _DistFlashAttnVarlenWithKVGather.allgather(v, cp_group)
        dq = torch.empty_like(q)
        full_dk = torch.zeros_like(full_k)
        full_dv = torch.zeros_like(full_v)
        _flash_attn_varlen_backward(
            dout=dout,
            q=q,
            k=full_k[k_range[0]: k_range[1]],
            v=full_v[k_range[0]: k_range[1]],
            out=out,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=full_dk[k_range[0]: k_range[1]],
            dv=full_dv[k_range[0]: k_range[1]],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            deterministic=deterministic,
        )
        dk = _DistFlashAttnVarlenWithKVGather.reduce_scatter(full_dk, cp_group)
        dv = _DistFlashAttnVarlenWithKVGather.reduce_scatter(full_dv, cp_group)
        return (None,) + (dq, dk, dv,) + (None,) * 11

class DistributedAttentionWithKVGather(torch.nn.Module):
    """
    Split query across CP ranks, while allgather key & value across CP ranks.
    This solution is specialized for GQA, and without head limited.
    q, k, v: The tensor containing the query, key, and value. (S, B, H, D)
    """

    def __init__(self, cp_group, dropout_p=0.0, softmax_scale=None, 
                 causal=True, window_size=(-1, -1), deterministic=False,
                 return_attn_probs=False):
        super().__init__()
        assert causal == True, f'Dist-Attn with KVGather only support causal mode!'
        self.cp_group = cp_group
        self.cp_size = torch.distributed.get_world_size(group=cp_group)
        self.cp_rank = torch.distributed.get_rank(group=cp_group)
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.deterministic = deterministic
        self.return_attn_probs = return_attn_probs

    def forward(self, q, k, v, cu_seqlens):
        is_rearrange = False
        if len(q.shape) == 4:
            # [seqlen//cp, batch_size, num_heads//tp, head_dim]
            is_rearrange = True
            seqlen, batch_size = q.shape[0], q.shape[1]
            q, k, v = [rearrange(x, 's b ... -> (b s) ...').contiguous()
                        for x in (q, k, v)]
        else:
            # [t, h, d]
            assert len(q.shape) == 3
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            k_range,
        ) = _DistFlashAttnVarlenWithKVGather.prepare_cu_seqlens(
            cu_seqlens,
            self.cp_group,
        )
        outputs = _DistFlashAttnVarlenWithKVGather.apply(
            self.cp_group, 
            q, 
            k, 
            v, 
            cu_seqlens_q, 
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k, 
            k_range,
            self.dropout_p, 
            self.softmax_scale, 
            self.causal, 
            self.window_size,
            self.deterministic,
            self.return_attn_probs,
        )
        output = outputs[0] if self.return_attn_probs else outputs
        if is_rearrange:
            output = rearrange(output, '(b s) ... -> s b ...', b=batch_size).contiguous()
        return output if not self.return_attn_probs else (output, *outputs[1:])