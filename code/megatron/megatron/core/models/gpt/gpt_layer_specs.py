# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.diff_attention import DiffSelfAttention, DiffSelfAttentionSubmodules
from megatron.core.transformer.shared_layer import SharedParam
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, FlashMoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.chunkflow.stateful_attention import StatefulSelfAttention
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn('Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


# TODO：Reorg these bullshit.
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    diff_attention: Optional[bool] = False,
    flash_moe: Optional[bool] = False,
    fp8: Optional[str] = None,
    chunkflow: Optional[bool] = False,
    causal: Optional[bool] = True,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        flash_moe (bool, optional): To use flashmoe for moe layers. Defaults to False.
        fp8 (str, optional): Flag to decide the linear layer spec for MoE. Defaults to None.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm, flash_moe=flash_moe, fp8=fp8
    )
    if diff_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=DiffSelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=DiffSelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        # TODO: check convergence,
                        # TENorm significantly harms convergence when used
                        # for QKLayerNorm; we instead use the Apex implementation.
                        q_layernorm=TENorm if qk_layernorm else IdentityOp,
                        k_layernorm=TENorm if qk_layernorm else IdentityOp,
                        v_layernorm=TENorm,
                        lambda_module=SharedParam,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    elif multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TEColumnParallelLinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_down_proj=TEColumnParallelLinear,
                        linear_kv_up_proj=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=TENorm if qk_layernorm else IdentityOp,
                        kv_layernorm=TENorm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                input_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        core_attention = TEDotProductAttention
        # Use StatefulSelfAttention if chunkflow is enabled
        if chunkflow:
            core_attention = StatefulSelfAttention
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=core_attention,
                        linear_proj=TERowParallelLinear,
                        # TENorm significantly harms convergence when used
                        # for QKLayerNorm; we instead use the Apex implementation.
                        q_layernorm=TENorm if qk_layernorm else IdentityOp,
                        k_layernorm=TENorm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    diff_attention: Optional[bool] = False,
    flash_moe: Optional[bool] = False,
    causal: Optional[bool] = True,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        flash_moe (bool, optional): To use flashmoe for moe layers. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm, flash_moe=flash_moe,
    )
    if diff_attention:
        raise NotImplementedError

    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=ColumnParallelLinear,
                        linear_q_down_proj=ColumnParallelLinear,
                        linear_q_up_proj=ColumnParallelLinear,
                        linear_kv_down_proj=ColumnParallelLinear,
                        linear_kv_up_proj=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        kv_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl if num_experts else IdentityOp,
                input_layernorm=LNImpl if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=LNImpl,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                    'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
                },
            ),
        )


def get_qwen3_5_moe_layer_specs(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    flash_moe: Optional[bool] = False,
    fp8: Optional[str] = None,
    layer_types: Optional[list] = None,
    chunkflow: Optional[bool] = False,
    causal: Optional[bool] = True,
    attn_output_gate: Optional[bool] = False,
) -> list:
    """Generate per-layer specs for Qwen3.5-MoE with heterogeneous layer types.

    Returns a list of ModuleSpec, one per layer. Full attention layers use standard
    SelfAttention with output gating; linear attention layers use GatedDeltaNet.

    Args:
        num_experts: Number of MoE experts.
        moe_grouped_gemm: Use grouped GEMM for MoE.
        qk_layernorm: Use layernorm for Q/K.
        flash_moe: Use FlashMoE implementation.
        fp8: FP8 mode for TE.
        layer_types: Per-layer type list, e.g. ['linear_attention', 'full_attention', ...].
        chunkflow: Use chunkflow for attention.
        causal: Use causal attention mask.
        attn_output_gate: If True, use unfused input_layernorm + linear for full attention
            layers (the doubled Q projection is incompatible with TE fused LayerNormLinear).

    Returns:
        list[ModuleSpec]: One spec per layer.
    """
    from megatron.core.transformer.gated_delta_net import GatedDeltaNet

    if layer_types is None:
        raise ValueError("layer_types must be provided for Qwen3.5 specs")

    mlp = _get_mlp_module_spec(
        use_te=HAVE_TE,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        flash_moe=flash_moe,
        fp8=fp8,
    )

    core_attention = TEDotProductAttention if HAVE_TE else DotProductAttention
    if chunkflow:
        core_attention = StatefulSelfAttention

    NormImpl = TENorm if HAVE_TE else LNImpl
    LinearProj = TERowParallelLinear if HAVE_TE else RowParallelLinear

    if attn_output_gate:
        # With output gating the Q projection is doubled; use unfused layernorm + linear
        # so that input_layernorm is a separate module (same layout as linear attention layers).
        LinearQKV = TEColumnParallelLinear if HAVE_TE else ColumnParallelLinear
        full_attn_spec = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=NormImpl,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=LinearQKV,
                        core_attention=core_attention,
                        linear_proj=LinearProj,
                        q_layernorm=NormImpl if qk_layernorm else IdentityOp,
                        k_layernorm=NormImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=NormImpl if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        # Standard TE fused LayerNorm + QKV linear
        LinearQKV = TELayerNormColumnParallelLinear if HAVE_TE else ColumnParallelLinear
        full_attn_spec = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.padding},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=LinearQKV,
                        core_attention=core_attention,
                        linear_proj=LinearProj,
                        q_layernorm=NormImpl if qk_layernorm else IdentityOp,
                        k_layernorm=NormImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=NormImpl if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )

    # Linear attention spec (GatedDeltaNet) -- always uses explicit input_layernorm
    linear_attn_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=NormImpl,
            self_attention=ModuleSpec(
                module=GatedDeltaNet,
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=NormImpl if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

    specs = []
    for layer_type in layer_types:
        if layer_type == 'full_attention':
            specs.append(full_attn_spec)
        elif layer_type == 'linear_attention':
            specs.append(linear_attn_spec)
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}. "
                           f"Expected 'full_attention' or 'linear_attention'.")

    return specs


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    flash_moe: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        elif use_te and fp8:
            linear_fc1 = TEColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=FlashMoELayer if flash_moe else MoELayer,
            submodules=MoESubmodules(
                experts=(
                    MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                    if not moe_grouped_gemm or use_te_grouped_gemm
                    else None
                ),
                shared_experts=ModuleSpec(
                    module=SharedExpertMLP,
                    params={"gate": True},
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                    ),
                ),
            ),
        )
