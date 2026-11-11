# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""LoRA (Low-Rank Adaptation) support for Megatron attention layers.

Provides in-place LoRA injection for ColumnParallelLinear and RowParallelLinear
layers used in both SelfAttention and GatedDeltaNet attention modules.

The monkey-patch approach attaches lora_A / lora_B as nn.Linear sub-modules
directly onto existing parallel linear layers, preserving state_dict key
compatibility with base model checkpoints.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.training import print_rank_0


def _get_linear_weight(layer):
    weight = getattr(layer, 'weight', None)
    if weight is None:
        raise RuntimeError(
            f'LoRA injection requires an allocated weight on {type(layer).__name__}.'
        )
    return weight


def _get_column_parallel_lora_dims(layer) -> Tuple[int, int]:
    """Return input size and local output size for Megatron or TE column-parallel linear."""
    weight = _get_linear_weight(layer)
    in_features = getattr(layer, 'input_size', weight.shape[1])
    out_features = getattr(layer, 'output_size_per_partition', weight.shape[0])
    return in_features, out_features


def _get_row_parallel_lora_dims(layer) -> Tuple[int, int]:
    """Return local input size and output size for Megatron or TE row-parallel linear."""
    weight = _get_linear_weight(layer)
    in_features = getattr(layer, 'input_size_per_partition', weight.shape[1])
    out_features = getattr(layer, 'output_size', weight.shape[0])
    return in_features, out_features


def _get_sequence_parallel(layer) -> bool:
    if hasattr(layer, 'sequence_parallel'):
        return layer.sequence_parallel
    config = getattr(layer, 'config', None)
    return bool(getattr(config, 'sequence_parallel', False))


def inject_lora_column_parallel(layer, rank: int, alpha: float, dropout: float = 0.0):
    """Attach LoRA adapters to a ColumnParallelLinear and patch its forward.

    For ColumnParallelLinear, the output dimension is already partitioned across
    TP ranks. LoRA B produces the same partition-sized output, so no extra
    communication is needed.

    Args:
        layer: A ColumnParallelLinear instance to inject LoRA into.
        rank: LoRA rank (r).
        alpha: LoRA scaling factor.
        dropout: Dropout rate applied to LoRA input.
    """
    in_features, out_features = _get_column_parallel_lora_dims(layer)

    layer.lora_A = nn.Linear(in_features, rank, bias=False)
    layer.lora_B = nn.Linear(rank, out_features, bias=False)
    layer.lora_scaling = alpha / rank
    layer.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # Standard LoRA init: A with kaiming, B with zeros (starts as identity)
    nn.init.kaiming_uniform_(layer.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(layer.lora_B.weight)

    # lora_A is duplicated across TP ranks (full input_size -> rank).
    # Its gradient must be all-reduced across TP to stay in sync.
    if _get_sequence_parallel(layer):
        layer.lora_A.weight.sequence_parallel = True

    # Move to same device/dtype as base weight
    layer.lora_A = layer.lora_A.to(device=layer.weight.device, dtype=layer.weight.dtype)
    layer.lora_B = layer.lora_B.to(device=layer.weight.device, dtype=layer.weight.dtype)

    original_forward = layer.forward

    def lora_forward(input_: torch.Tensor, *args, **kwargs):
        # Base forward: output shape [s, b, out_per_partition]
        output, bias = original_forward(input_, *args, **kwargs)
        # When sequence_parallel is enabled, the base ColumnParallelLinear
        # internally all-gathers the scattered input (seq/tp -> seq) before
        # matmul, producing full-sequence output.  We must do the same for
        # the LoRA path so that the shapes match for the addition.
        if _get_sequence_parallel(layer):
            lora_input = gather_from_sequence_parallel_region(input_)
        else:
            lora_input = input_
        # LoRA delta: lora_A maps hidden_size -> rank, lora_B maps rank -> out_per_partition
        lora_out = layer.lora_B(layer.lora_A(layer.lora_dropout(lora_input)))
        output = output + lora_out * layer.lora_scaling
        return output, bias

    layer.forward = lora_forward


def inject_lora_row_parallel(layer, rank: int, alpha: float, dropout: float = 0.0):
    """Attach LoRA adapters to a RowParallelLinear and patch its forward.

    For RowParallelLinear, the input dimension is partitioned across TP ranks.
    Each rank computes a partial LoRA delta that must be reduced (all-reduce or
    reduce-scatter) across TP ranks, matching the base layer's reduction.

    Args:
        layer: A RowParallelLinear instance to inject LoRA into.
        rank: LoRA rank (r).
        alpha: LoRA scaling factor.
        dropout: Dropout rate applied to LoRA input.
    """
    in_features, out_features = _get_row_parallel_lora_dims(layer)

    layer.lora_A = nn.Linear(in_features, rank, bias=False)
    layer.lora_B = nn.Linear(rank, out_features, bias=False)
    layer.lora_scaling = alpha / rank
    layer.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    nn.init.kaiming_uniform_(layer.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(layer.lora_B.weight)

    # lora_B is duplicated across TP ranks (rank -> full output_size).
    # Its gradient must be all-reduced across TP to stay in sync.
    if _get_sequence_parallel(layer):
        layer.lora_B.weight.sequence_parallel = True

    layer.lora_A = layer.lora_A.to(device=layer.weight.device, dtype=layer.weight.dtype)
    layer.lora_B = layer.lora_B.to(device=layer.weight.device, dtype=layer.weight.dtype)

    original_forward = layer.forward

    def lora_forward(input_: torch.Tensor):
        # Base forward handles input partitioning + all-reduce/reduce-scatter
        output, bias = original_forward(input_)

        # Compute LoRA delta with partitioned input
        if getattr(layer, 'input_is_parallel', True):
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        delta = layer.lora_B(layer.lora_A(layer.lora_dropout(input_parallel)))
        delta = delta * layer.lora_scaling

        # Reduce delta across TP ranks (same as base layer's reduction)
        if not getattr(layer, 'explicit_expert_comm', False):
            if _get_sequence_parallel(layer):
                delta = reduce_scatter_to_sequence_parallel_region(delta)
            else:
                delta = reduce_from_tensor_model_parallel_region(delta)

        output = output + delta
        return output, bias

    layer.forward = lora_forward


def inject_lora_into_model(
    model,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    skip_linear_attention_out_proj: bool = False,
):
    """Inject LoRA adapters into all attention projection layers of the decoder.

    Supports both SelfAttention (full attention) and GatedDeltaNet (linear attention)
    layer types in the Qwen3.5 heterogeneous architecture.

    For SelfAttention layers:
      - linear_qkv (ColumnParallelLinear): Q/K/V projection
      - linear_proj (RowParallelLinear): output projection

    For GatedDeltaNet layers:
      - in_proj_qkv (ColumnParallelLinear): Q/K/V projection
      - in_proj_z (ColumnParallelLinear): gate projection
      - out_proj (RowParallelLinear): output projection

    Args:
        model: The FunBaseModel (or GPTModel subclass) with a .decoder attribute.
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA scaling factor (alpha).
        lora_dropout: Dropout rate for LoRA input.
    """
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.gated_delta_net import GatedDeltaNet

    decoder = model.decoder  # TransformerBlock
    lora_param_count = 0
    full_attn_count = 0
    linear_attn_count = 0
    skipped_linear_out_proj_count = 0

    for layer_idx, layer in enumerate(decoder.layers):
        attn = layer.self_attention

        if isinstance(attn, SelfAttention):
            inject_lora_column_parallel(attn.linear_qkv, lora_rank, lora_alpha, lora_dropout)
            inject_lora_row_parallel(attn.linear_proj, lora_rank, lora_alpha, lora_dropout)
            full_attn_count += 1
        elif isinstance(attn, GatedDeltaNet):
            inject_lora_column_parallel(attn.in_proj_qkv, lora_rank, lora_alpha, lora_dropout)
            inject_lora_column_parallel(attn.in_proj_z, lora_rank, lora_alpha, lora_dropout)
            if skip_linear_attention_out_proj:
                skipped_linear_out_proj_count += 1
            else:
                inject_lora_row_parallel(attn.out_proj, lora_rank, lora_alpha, lora_dropout)
            linear_attn_count += 1
        else:
            continue

        # Count LoRA params for this layer
        for name, p in attn.named_parameters():
            if 'lora_' in name:
                lora_param_count += p.numel()

    print_rank_0(
        f'[LoRA] Injected LoRA adapters: rank={lora_rank}, alpha={lora_alpha}, '
        f'dropout={lora_dropout}'
    )
    print_rank_0(
        f'[LoRA] Full attention layers: {full_attn_count}, '
        f'Linear attention layers: {linear_attn_count}'
    )
    if skip_linear_attention_out_proj:
        print_rank_0(
            '[LoRA] Skipped GatedDeltaNet out_proj LoRA injection on '
            f'{skipped_linear_out_proj_count} linear-attention layer(s).'
        )
    print_rank_0(f'[LoRA] Total LoRA parameters: {lora_param_count:,}')


def get_lora_state_dict(model):
    """Extract state dict containing only LoRA parameters.

    Args:
        model: The model (possibly wrapped in Float16Module/DDP).

    Returns:
        Dict of LoRA parameter name -> tensor.
    """
    unwrapped = model
    while hasattr(unwrapped, 'module'):
        unwrapped = unwrapped.module
    return {k: v for k, v in unwrapped.state_dict().items() if 'lora_' in k}


def load_lora_checkpoint(model, path: str):
    """Load a LoRA-only checkpoint and apply to model.

    Args:
        model: List of model chunks or a single model.
        path: Path to the LoRA checkpoint file.
    """
    print_rank_0(f'[LoRA] Loading LoRA checkpoint from {path}')

    if os.path.isfile(path):
        lora_sd = torch.load(path, map_location='cpu')
        if 'model' in lora_sd:
            lora_sd = lora_sd['model']
    else:
        # For dist-ckpt format, the main load_checkpoint handles loading.
        # The LoRA keys will be loaded as part of the normal checkpoint flow
        # when using --load with a LoRA-only dist-ckpt directory.
        print_rank_0(f'[LoRA] Directory checkpoint detected at {path}. '
                     f'Use --load to load dist-ckpt LoRA checkpoints.')
        return

    unwrapped = model[0] if isinstance(model, list) else model
    while hasattr(unwrapped, 'module'):
        unwrapped = unwrapped.module

    missing, unexpected = unwrapped.load_state_dict(lora_sd, strict=False)
    loaded_lora_keys = [k for k in lora_sd.keys() if 'lora_' in k]
    print_rank_0(f'[LoRA] Loaded {len(loaded_lora_keys)} LoRA parameter tensors.')
    if unexpected:
        print_rank_0(f'[LoRA] Warning: {len(unexpected)} unexpected keys in LoRA checkpoint.')


# ---------------------------------------------------------------------------
# Delta assembly utilities for mcore-to-HF checkpoint conversion
# ---------------------------------------------------------------------------

def compute_lora_delta(
    lora_A_weight: torch.Tensor,
    lora_B_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Compute a single LoRA weight-space delta: scaling * B @ A.

    Args:
        lora_A_weight: (lora_rank, in_features) from lora_A nn.Linear.
        lora_B_weight: (out_features, lora_rank) from lora_B nn.Linear.
        scaling: lora_alpha / lora_rank.

    Returns:
        Delta tensor of shape (out_features, in_features).
    """
    return (lora_B_weight.float() @ lora_A_weight.float()) * scaling


# Target definitions per layer type:
#   (state_dict_attr, message_key, parallel_type)
_FULL_ATTN_TARGETS = [
    ("linear_qkv", "qkv", "col"),
    ("linear_proj", "dense", "row"),
]
_LINEAR_ATTN_TARGETS = [
    ("in_proj_qkv", "in_proj_qkv", "col"),
    ("in_proj_z", "in_proj_z", "col"),
    ("out_proj", "out_proj", "row"),
]


def assemble_lora_deltas(
    lora_rank_state_dicts: List[dict],
    num_layers: int,
    layer_types: List[str],
    lora_rank: int,
    lora_alpha: float,
    target_dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Pre-compute full-size LoRA weight deltas from per-TP-rank state dicts.

    Loads LoRA A/B weights from each TP rank's state dict, computes per-rank
    deltas, concatenates across TP (dim=0 for ColumnParallel, dim=1 for
    RowParallel), and returns full-size deltas keyed by layer index and
    message key name.

    Args:
        lora_rank_state_dicts: List of model state dicts, one per TP rank.
        num_layers: Total number of decoder layers.
        layer_types: Per-layer type list ("full_attention" or "linear_attention").
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA scaling factor.
        target_dtype: Cast final deltas to this dtype (default bf16).

    Returns:
        dict mapping layer_idx -> {message_key: full_delta_tensor}.
        Only layers that have LoRA keys in the state dict are included.
    """
    scaling = lora_alpha / lora_rank
    tp_size = len(lora_rank_state_dicts)
    deltas: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx in range(num_layers):
        if layer_types[layer_idx] == "full_attention":
            targets = _FULL_ATTN_TARGETS
        elif layer_types[layer_idx] == "linear_attention":
            targets = _LINEAR_ATTN_TARGETS
        else:
            continue

        layer_deltas: Dict[str, torch.Tensor] = {}
        for attr_name, delta_key, parallel_type in targets:
            prefix = f"decoder.layers.{layer_idx}.self_attention.{attr_name}"
            key_A = f"{prefix}.lora_A.weight"
            key_B = f"{prefix}.lora_B.weight"

            # Check if LoRA keys exist in rank-0 state dict
            if key_A not in lora_rank_state_dicts[0]:
                continue

            # Compute per-rank deltas and concatenate
            rank_deltas = []
            for tp_rank in range(tp_size):
                sd = lora_rank_state_dicts[tp_rank]
                delta_i = compute_lora_delta(sd[key_A], sd[key_B], scaling)
                rank_deltas.append(delta_i)

            # ColumnParallel: output is TP-split -> concat dim=0
            # RowParallel: input is TP-split -> concat dim=1
            cat_dim = 0 if parallel_type == "col" else 1
            full_delta = torch.cat(rank_deltas, dim=cat_dim).to(target_dtype)
            layer_deltas[delta_key] = full_delta

        if layer_deltas:
            deltas[layer_idx] = layer_deltas

    return deltas
