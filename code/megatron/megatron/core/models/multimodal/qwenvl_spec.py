# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TEColumnParallelLinear,
        TERowParallelLinear
    )
except ImportError:
    HAVE_TE = False


def get_qwen2vl_vision_model_with_transformer_engine_spec():
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_qwen2vl_vision_model_local_spec() -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add
        ),
    )

def get_qwen2vl_vision_projector_spec(use_te: bool = False) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    return ModuleSpec(
        module=MultimodalProjector,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear if not use_te else TEColumnParallelLinear,
            linear_fc2=RowParallelLinear if not use_te else TERowParallelLinear
        )
    )