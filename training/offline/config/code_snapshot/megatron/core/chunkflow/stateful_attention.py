
"""Attention."""
import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from typing import Optional, Dict

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
    )
from .chunkflow_state_store import state_store_manager
import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
try:
    from einops import rearrange
except ImportError:
    rearrange = None

# FlashAttn introduce alibi param since v2.4.0.post1
flash_attn_with_alibi = flash_attn.__version__ >= "2.4.0.post1"

def flash_attn_varlen_with_state(q, k, v,  state_store: Dict[str, torch.Tensor], dropout_p, softmax_scale=None, causal=True, **_flash_attn_kwargs):
    """
    q: (batch_size, seqlen_q, dim)
    k: (batch_size, seqlen_k, dim)
    v: (batch_size, seqlen_k, dim)
    state_store: dict with keys 'k_cache' and 'v_cache'
    dropout_p: float
    softmax_scale: float
    causal: bool
    Returns:
        out: (batch_size, seqlen_q, dim)
    """
    out = FlashAttnVarlenWithStateFunc.apply(q, k, v, state_store, dropout_p, causal, softmax_scale, _flash_attn_kwargs)
    return out

class StatefulSelfAttention(TEDotProductAttention):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p"):
        super(StatefulSelfAttention, self).__init__(config, layer_number, attn_mask_type, \
        attention_type, attention_dropout=attention_dropout, softmax_scale=softmax_scale, k_channels=k_channels, \
        v_channels=v_channels, cp_comm_type=cp_comm_type)
        self.uuid = layer_number
        self.softmax_scale = softmax_scale

    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None):
        assert attn_mask_type == AttnMaskType.causal, "Only causal attention is supported in StatefulAttention"
        state_store_for_this_module = state_store_manager.get_state_store().get_or_create(self.uuid)
        assert packed_seq_params["qkv_format"] == "thd", "Only thd format is supported now in StatefulAttention"

        chunk_count = packed_seq_params["chunk_count"]
        if chunk_count == 1:
            chunk_idx = packed_seq_params.pop("chunk_idx")
            chunk_count = packed_seq_params.pop("chunk_count")
            chunk_size = packed_seq_params.pop("chunk_size")

            out = super().forward(query, key, value, attention_mask, attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params)

            packed_seq_params["chunk_count"] = chunk_count
            packed_seq_params["chunk_idx"] = chunk_idx
            packed_seq_params["chunk_size"] = chunk_size
        else:
             # Stateful Attention Calling
            query, key, value = [rearrange(x, '(b s) ... -> b s ...', b=1) for x in [query, key, value]]
            out = FlashAttnVarlenWithStateFunc.apply(
                query,
                key,
                value,
                state_store_for_this_module,
                self.attention_dropout,
                True,
                self.softmax_scale,
                packed_seq_params,
            )
            out = rearrange(out, 't h d -> t (h d)')

        return out

class FlashAttnVarlenWithStateFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        state_store: Dict[str, torch.Tensor],
        dropout_p,
        causal,
        softmax_scale,
        _flash_attn_kwargs,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        batch_size = q.shape[0]
        # If ChunkFlow is enabled
        if 'k_cache' in state_store:
            k_cache, v_cache = state_store['k_cache'], state_store['v_cache']
            offset = k_cache.shape[1]
            # During the first-time forward passes, we construct states
            if not state_store.get_reuse_state():
                k_whole = torch.cat([k_cache, k], dim=1).contiguous()
                v_whole = torch.cat([v_cache, v], dim=1).contiguous()
                state_store['k_cache'], state_store['v_cache'] = k_whole, v_whole
            else:
                # Second-time forward passes, we directly reuse the states
                offset = _flash_attn_kwargs["chunk_idx"] * _flash_attn_kwargs["chunk_size"]
                end = offset + _flash_attn_kwargs["chunk_size"]
                k_whole, v_whole = state_store['k_cache'][:, :end], state_store['v_cache'][:, :end]
        else:
            offset = 0
            k_whole = k
            v_whole = v
            state_store['k_cache'], state_store['v_cache'] = k_whole, v_whole
        seqlen_k = k_whole.shape[1]
        seqlen_q = q.shape[1]

        ctx._seqlen_k = seqlen_k
        ctx._seqlen_q = seqlen_q
        ctx._offset = offset
        q, k_whole, v_whole = [rearrange(x, 'b s ... -> (b s) ...').contiguous() for x in [q, k_whole, v_whole]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device="cuda")

        q = q.contiguous()
        k_whole = k_whole.contiguous()
        v_whole = v_whole.contiguous()
        if flash_attn_with_alibi:
            out, q, k_whole, v_whole, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q,
                k_whole,
                v_whole,
                cu_seqlens_q,
                cu_seqlens_k,
                seqlen_q,
                seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                return_softmax=False,
                alibi_slopes=None,
            )
        else:
            out, q, k_whole, v_whole, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q,
                k_whole,
                v_whole,
                cu_seqlens_q,
                cu_seqlens_k,
                seqlen_q,
                seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                return_softmax=False,
            )

        ctx._state_store = state_store
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state,

        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        k_whole = ctx._state_store['k_cache']
        v_whole = ctx._state_store['v_cache']
        batch_size = q.size(0) // ctx._seqlen_q
        assert batch_size == 1, "BatchSize must be 1 for StatefulSelfAttention"
        pk = k_whole[:, :ctx._seqlen_k]
        pv = v_whole[:, :ctx._seqlen_k].contiguous()
        ctx._state_store['k_cache'], ctx._state_store['v_cache'] = pk[:, :-ctx._seqlen_q], pv[:, :-ctx._seqlen_q]
        pk, pv = [rearrange(x, 'b s ... -> (b s) ...') for x in [pk, pv]]
        pk = pk.contiguous()
        pv = pv.contiguous()
        q = q.contiguous()
        last_idx = not 'k_grad' in ctx._state_store
        dq, dk, dv = torch.empty_like(q), torch.empty_like(pk), torch.empty_like(pv)
        if flash_attn_with_alibi:
            _flash_attn_varlen_backward(
                dout,
                q,
                pk,
                pv,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                ctx._seqlen_q,
                ctx._seqlen_k,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                (-1,-1),
                rng_state=rng_state,
                alibi_slopes=None,
                deterministic=False
            )
        else:
            _flash_attn_varlen_backward(
                dout,
                q,
                pk,
                pv,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                ctx._seqlen_q,
                ctx._seqlen_k,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                (-1,-1),
                rng_state=rng_state,
            )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        dq, dk, dv = [rearrange(x, '(b s) ... -> b s ...', b=batch_size) for x in [dq, dk, dv]]
        dk = dk.contiguous()
        dv = dv.contiguous()
        dq = dq.contiguous()
        if not last_idx:
            k_grad_p, v_grad_p = ctx._state_store['k_grad'], ctx._state_store['v_grad']
            dk += k_grad_p
            dv += v_grad_p
        ctx._state_store['k_grad'], ctx._state_store['v_grad'] = dk[:, :ctx._offset], dv[:, :ctx._offset]
        dk = dk[:, ctx._offset:ctx._seqlen_k]
        dv = dv[:, ctx._offset:ctx._seqlen_k]
        return dq, dk, dv, None, None, None, None, None, None, None
