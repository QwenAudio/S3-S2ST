# Copyright (c) 2025. Qwen3.5 GatedDeltaNet linear attention for Megatron-LM.
#
# Implements the Gated Delta Rule linear attention mechanism used in Qwen3.5-MoE.
# Reference: test/modeling_qwen3_5_moe.py (Qwen3_5MoeGatedDeltaNet)

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

logger = logging.getLogger(__name__)

import os
use_causal_conv1d = os.environ.get('GATED_DELTA_NET_USE_CAUSAL_CONV1D', '0') == '1'
# by default disabled, does not see benefits
if use_causal_conv1d:
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_fn = None
        causal_conv1d_update = None
else:
    causal_conv1d_fn = None
    causal_conv1d_update = None

use_fla = os.environ.get('GATED_DELTA_NET_USE_FLA', '1') == '1'
if use_fla:
    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError:
        chunk_gated_delta_rule = None
        fused_recurrent_gated_delta_rule = None
        FusedRMSNormGated = None
else:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None
    FusedRMSNormGated = None

_is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along a dimension."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def _cu_seqlens_to_seq_idx(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Convert cu_seqlens (cumulative sequence lengths) to seq_idx format.

    cu_seqlens: [N+1] tensor where N is the number of sequences
                e.g., [0, 5, 12, 20] means 3 sequences of lengths [5, 7, 8]

    Returns: seq_idx of shape [1, total_len] where each position indicates
             which sequence it belongs to (0, 1, 2, ...)
    """
    device = cu_seqlens.device
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    num_seqs = lengths.shape[0]
    seq_ids = torch.arange(num_seqs, device=device, dtype=torch.int32)
    seq_idx = torch.repeat_interleave(seq_ids, lengths).unsqueeze(0)

    return seq_idx

class RMSNormGated(nn.Module):
    """RMSNorm with gating: output = norm(x) * (1+weight) * silu(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def _torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Pure PyTorch fallback for chunked gated delta rule recurrence."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(MegatronModule):
    """GatedDeltaNet linear attention layer for Qwen3.5-MoE.

    Uses causal conv1d + gated delta rule recurrence as a replacement for
    standard softmax attention. This module supports tensor parallelism by
    splitting heads across TP ranks.

    Args:
        config: TransformerConfig with linear attention parameters.
        layer_number: Layer index (1-based as per Megatron convention).
    """

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.hidden_size = config.hidden_size

        # Linear attention dimensions from config
        self.num_v_heads = config.linear_attention_num_value_heads
        self.num_k_heads = config.linear_attention_num_key_heads
        self.head_k_dim = config.linear_attention_key_head_dim
        self.head_v_dim = config.linear_attention_value_head_dim
        self.conv_kernel_size = config.linear_attention_conv_kernel_dim

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        # TP handling: split value heads across TP ranks
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.num_v_heads_per_partition = divide(self.num_v_heads, world_size)
        self.num_k_heads_per_partition = divide(self.num_k_heads, world_size)
        self.key_dim_per_partition = self.head_k_dim * self.num_k_heads_per_partition
        self.value_dim_per_partition = self.head_v_dim * self.num_v_heads_per_partition

        # QKV projection (column-parallel)
        self.conv_dim_per_partition = self.key_dim_per_partition * 2 + self.value_dim_per_partition
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        # Gate projection (column-parallel, outputs value_dim)
        self.in_proj_z = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='z',
        )

        # Beta projection (not TP-split, small dimension = num_v_heads)
        # Can combine to reduce comm
        self.in_proj_b = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='b',
        )

        # Alpha/dt projection (not TP-split, small dimension = num_v_heads)
        # Can combine to reduce comm
        self.in_proj_a = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='a',
        )

        # Causal Conv1d (full-sized, TP partition sliced in forward)
        # Use ColumnParallelLinear as weight container
        self.conv1d = tensor_parallel.ColumnParallelLinear(
            self.conv_kernel_size,
            self.conv_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='conv',
        )

        # dt_bias and A_log (per-head, not TP-split since num_v_heads is typically small)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Output norm with gating
        if FusedRMSNormGated is not None:
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=config.layernorm_epsilon,
                activation='silu',
            )
        else:
            self.norm = RMSNormGated(self.head_v_dim, eps=config.layernorm_epsilon)

        # Output projection (row-parallel)
        self.out_proj = tensor_parallel.RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )

        # Select fast or fallback implementations
        self._causal_conv1d_fn = causal_conv1d_fn
        self._chunk_fn = chunk_gated_delta_rule or _torch_chunk_gated_delta_rule

        if not _is_fast_path_available:
            logger.warning_once(
                "GatedDeltaNet: fast path not available. Install causal-conv1d and "
                "flash-linear-attention for optimal performance. Falling back to "
                "pure PyTorch implementation."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [seq_len, batch_size, hidden_size] (Megatron sbhd format)
            attention_mask: Optional mask tensor.
            inference_params: Inference parameters (not used in linear attention).
            rotary_pos_emb: Rotary position embeddings (not used in linear attention).
            packed_seq_params: Packed sequence parameters (not used in linear attention).

        Returns:
            output: [seq_len, batch_size, hidden_size]
            bias: None (for API compatibility with Attention)
        """
        seq_len, batch_size, _ = hidden_states.shape
        if self.config.sequence_parallel:
            seq_len *= mpu.get_tensor_model_parallel_world_size()

        # Mask padding states
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        # QKV projection
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)  # [s, b, conv_dim_per_partition]
        mixed_qkv = mixed_qkv.permute(1, 2, 0)  # [b, conv_dim, s]

        # Gate projection
        z, _ = self.in_proj_z(hidden_states)  # [s, b, value_dim_per_partition]
        z = z.permute(1, 0, 2).reshape(batch_size, seq_len, -1, self.head_v_dim)

        # Beta and alpha projections
        b, _ = self.in_proj_b(hidden_states)  # [s, b, num_v_heads]
        b = b.permute(1, 0, 2)             # [b, s, num_v_heads]
        a, _ = self.in_proj_a(hidden_states)  # [s, b, num_v_heads]
        a = a.permute(1, 0, 2)             # [b, s, num_v_heads]

        # Causal convolution
        conv_weight_partition = self.conv1d.weight

        cu_seqlens = None
        if packed_seq_params is not None and "cu_seqlens_q" in packed_seq_params:
            cu_seqlens = packed_seq_params["cu_seqlens_q"]
        
        if self._causal_conv1d_fn is not None:
            seq_idx = None
            if cu_seqlens is not None:
                seq_idx = _cu_seqlens_to_seq_idx(cu_seqlens)
            mixed_qkv = self._causal_conv1d_fn(
                x=mixed_qkv,
                weight=conv_weight_partition,
                bias=self.conv1d.bias,
                activation='silu',
                seq_idx=seq_idx,
            )
        else:
            # Apply conv1d with sliced weights for this partition
            mixed_qkv = F.conv1d(
                mixed_qkv,
                conv_weight_partition.unsqueeze(1),
                bias=None,
                stride=1,
                padding=self.conv_kernel_size - 1,
                groups=self.conv_dim_per_partition,
            )
            mixed_qkv = F.silu(mixed_qkv[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [b, s, conv_dim_per_partition]

        # Split QKV
        group_size = self.head_k_dim + self.head_k_dim + (self.num_v_heads // self.num_k_heads) * self.head_v_dim
        mixed_qkv = mixed_qkv.reshape(batch_size, seq_len, self.num_k_heads_per_partition, group_size)

        query = mixed_qkv[..., :self.head_k_dim]
        key = mixed_qkv[..., self.head_k_dim:2 * self.head_k_dim]
        value = mixed_qkv[..., 2 * self.head_k_dim:]

        # query: [b, s, num_k_heads_per_partition, head_k_dim]
        # key:   [b, s, num_k_heads_per_partition, head_k_dim]
        # value: [b, s, num_k_heads_per_partition, v_per_k * head_v_dim]

        value = value.reshape(batch_size, seq_len, self.num_v_heads_per_partition, self.head_v_dim)

        # Compute gating parameters
        beta = b.sigmoid()
        g = -self.A_log.reshape(mpu.get_tensor_and_context_parallel_world_size(), -1)[mpu.get_tensor_model_parallel_rank()].float().exp() * \
            F.softplus(a.float() + self.dt_bias.reshape(mpu.get_tensor_and_context_parallel_world_size(), -1)[mpu.get_tensor_model_parallel_rank()])

        # Expand K heads to match V heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            expand_ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(expand_ratio, dim=2)
            key = key.repeat_interleave(expand_ratio, dim=2)

        # Run gated delta rule recurrence
        core_attn_out, _ = self._chunk_fn(
            query, key, value,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        # Reshape and apply gated norm
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        core_attn_out = core_attn_out.permute(1, 0, 2)  # [s, b, hidden_size]

        # Output projection (row-parallel)
        output, bias = self.out_proj(core_attn_out)

        return output, bias
