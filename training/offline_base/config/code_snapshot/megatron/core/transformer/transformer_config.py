# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch.nn.functional as F

from ..model_parallel_config import ModelParallelConfig
from ..utils import (
    get_te_version,
    get_triton_version,
    init_method_normal,
    is_te_min_version,
    is_triton_min_version,
    scaled_init_method_normal,
)


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter,
    including those in ModelParallelConfig.
    """

    ####################
    # model architecture
    ####################
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    num_layers_with_padding: int = 0
    """Number of transformer layers in a transformer block that padding for uneven (v)pp."""

    num_layers_without_padding: int = 0
    """Number of transformer layers in a transformer block that not padding for uneven (v)pp."""

    first_pipeline_num_layers: int = None
    """Number of transformer layers on first pipeline stage.
    None implies equal layer division across PP ranks."""

    last_pipeline_num_layers: int = None
    """Number of transformer layers on last pipeline stage.
    None implies equal layer division across PP ranks."""

    pipeline_stages_num_layers_override: Optional[List[int]] = None
    """Override the number of transformer layers for each pipeline stage."""

    num_layers_per_pipeline_stage: int = None

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: int = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: int = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    kv_channels: int = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    add_bias_linear: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    recompute_mlp_activation_function: bool = False
    """Activation function recompute in MLP."""

    recompute_layernorm: bool = False
    """Input_layernorm and pre_mlp_layernorm recompute."""

    recompute_attn_layer: bool = False
    """recompute whole attention layer"""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory.
    The stored input is casted back to the original precision before backprop compuatation."""

    num_moe_experts: int = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[Tuple[int, int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    normalization: bool = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply LayerNorm to the query and key embeddings."""

    v_layernorm: bool = False
    """Whether to apply LayerNorm to the value embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    ####################
    # initialization
    ####################
    init_method: Callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: Callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    init_clip_std: bool = False
    """If true, clip to init range to [-3sigma, 3sigma] to improve training stability."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    bf16_memory_efficient_cross_entropy: bool = False
    """If True, uses memory efficient cross entropy in bfloat16."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where only the memory intensive part of attention is checkpointed.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: str = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: int = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: bool = None
    """If True, distribute recomputed activations across the model parallel group."""

    ####################
    # fp8 related
    ####################
    fp8: str = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
    Controls how often the scaling factor is recomputed.
    """

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation
    in higher precision."""

    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    tp_only_amax_red: bool = False
    """When set to True, reduce the FP8 AMAX only in the TP or TP-CP domain"""

    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: int = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared epxerts execute after the routed experts."""

    moe_router_load_balancing_type: str = "aux_loss"
    """Determines the load balancing strategy for the router. "aux_loss" corresponds to the load
    balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing
    algorithm used in S-BASE, and "none" implies no load balancing."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_aux_loss_fusion: bool = False
    """Fusion for router layer with aux loss load balancing and post softmax."""

    moe_router_pre_softmax: bool = False
    """Enable pre-softmax routing for MoE, which means softmax is before the top-k selection.
    By default, softmax is done after top-k."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_aux_loss_coeff: float = 0  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: float = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: float = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False  # TODO: Support token dropping.
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather' and 'alltoall'."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: float = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = 'probs'
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    moe_permute_fusion: bool = False
    """Fuse permute/unpermute operations for moe layer."""

    moe_mock_router_sigma: float = None
    """Force the sigma value for the Gaussian noise used in input jitter. This is useful for
    debugging and testing."""

    moe_expert_dropout: float = 0.0
    """Dropout probability for the MoE expert."""

    moe_drop_across_ep: bool = False
    """Drop tokens across experts in the same parallel group."""

    flash_moe_ga_fusion: bool = False
    """Use gradient accumulation fusion in grouped and shared mlp. """

    flash_moe_interleaving: bool = False
    """Use interleaving in moe. """

    flash_moe_mem_saving_mlp: bool = False
    """Use memory-efficient mlp in moe. """

    flash_moe_cpu_offload: bool = False
    """Use activation offloading in moe. """

    flash_moe_offload_attention_overlap_ratio: float = 0.3
    """The offload overlap ratio for self-attention. """

    moe_norm_topk_scores: bool = False
    """Whether to normalize topk scores in moe router."""

    ckpt_format: str = "torch_dist"
    """The format of ckpt. """

    moe_lbl_all_reduce_overlap: bool = False
    """Whether To Overlap the All Reduce in load balancing loss computation."""

    ##################
    # Context Parallel
    ##################
    cp_comm_type: Union[str, List[str]] = None
    """Inter-gpu communication type for context parallelism.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    cp_comm_type of each layer can be "p2p" or "all_gather" or "a2a".
    "p2p": Exchange KV chunks with P2P communications in ring topology. P2P is async and can be
    overlapped with attention compute.
    "all_gather": All-gather to get full sequence of KV before attention. The all-gather is not
    async, and cannot be overlapped.
    "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP group, and gather to get
    full sequence of QKV.
    """

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer layers are swapped with a CUDA graphed version."""

    external_cuda_graph: bool = False
    """When set to true, TransformerLayer layers are swapped with user provided CUDA graphs."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    use_flash_cross_entropy: bool = True
    """When set to true, enables flash cross entropy loss instead of fused_cross_entropy."""

    inplace_flash_cross_entropy: bool = True
    """When set to true, enables inplace flash cross entropy loss."""


    ################
    # flashmoe
    ################
    use_flash_moe: bool = False
    """Whether to use flashmoe layer."""

    use_flashmoe_router: bool = False
    """Whether to use flashmoe router."""

    use_flashmoe_hep: bool = False
    """Currently not supported. Just for compatibility reasons."""

    use_dynamic_collate: bool = False

    ########
    # SFT
    ########
    use_smart_batching: bool = False
    """using smart batching optimization for sft training"""
    use_chunkflow: bool = False
    """using chunkflow optimization for sft training"""
    dynamic_padding_full: bool = False
    """padding sequence to max sequence length to avoid some hang cases."""

    chunk_size: int = -1
    chunks_kept: int = -1

    ####################
    # Qwen3.5 / Hybrid attention
    ####################
    layer_types: Optional[List[str]] = None
    """Per-layer attention type list, e.g. ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', ...].
    If None, all layers use full attention. Used for hybrid architectures like Qwen3.5."""

    full_attention_interval: int = 0
    """If > 0 and layer_types is None, auto-generate layer_types such that every Nth layer is
    full_attention and the rest are linear_attention. 0 means disabled (all full attention)."""

    attn_output_gate: bool = False
    """If True, the Q projection outputs 2x size (query + gate). After attention,
    output is multiplied by sigmoid(gate) before the output projection. Used in Qwen3.5."""

    ####################
    # GatedDeltaNet (linear attention) config
    ####################
    linear_attention_conv_kernel_dim: int = 4
    """Kernel size for the causal Conv1d in GatedDeltaNet linear attention layers."""

    linear_attention_key_head_dim: int = 128
    """Key head dimension for GatedDeltaNet linear attention layers."""

    linear_attention_value_head_dim: int = 128
    """Value head dimension for GatedDeltaNet linear attention layers."""

    linear_attention_num_key_heads: int = 16
    """Number of key heads for GatedDeltaNet linear attention layers."""

    linear_attention_num_value_heads: int = 64
    """Number of value heads for GatedDeltaNet linear attention layers."""

    mrope_interleaved: bool = False
    """If True, use interleaved M-RoPE layout instead of chunked layout.
    Used by Qwen3.5 for multimodal rotary position embeddings."""

    mrope_section: Optional[List[int]] = None
    """Section sizes for M-RoPE interleaving, e.g. [11, 11, 10] for Qwen3.5."""


    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError('num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError('num_moe_experts must be non-negative.')

        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f'moe_shared_expert_intermediate_size must be '
                    f'num_shared_experts * ffn_size_of_each_shared_expert, '
                    f'but got {self.moe_shared_expert_intermediate_size}'
                )
            if self.moe_shared_expert_overlap and self.moe_token_dispatcher_type not in [
                "alltoall"
            ]:
                raise ValueError(
                    f'moe_shared_expert_overlap only works with alltoall token dispatcher.'
                )

        if self.moe_expert_capacity_factor is not None:
            if self.moe_token_dispatcher_type not in ["alltoall", "alltoall_seq"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with alltoall token dispatcher'
                )
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["aux_loss", "none"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with aux_loss or none load balancing'
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        # Auto-generate layer_types from full_attention_interval if not explicitly set
        if self.layer_types is None and self.full_attention_interval > 0 and self.num_layers > 0:
            self.layer_types = []
            for i in range(self.num_layers):
                if (i + 1) % self.full_attention_interval == 0:
                    self.layer_types.append('full_attention')
                else:
                    self.layer_types.append('linear_attention')

        if self.layer_types is not None and len(self.layer_types) != self.num_layers:
            raise ValueError(
                f'layer_types length ({len(self.layer_types)}) must match '
                f'num_layers ({self.num_layers}).'
            )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f'CPU offloading can be done only for layers less than {self.num_layers}'
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                'Currently there is no support for Pipeline parallelism with CPU offloading'
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                'CPU offloading does not work when activation recomputation is enabled'
            )

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full"'
                    'or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so '
                    'recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} '
                    'recompute_num_layers must be between '
                    '1 and num_layers_per_pipeline_rank: '
                    f'{self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} '
                    'recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be '
                    f'false when sequence parallel is enabled: {self.sequence_parallel}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {self.num_layers} must be divisible by '
                        f'virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:
            if self.activation_func not in [F.gelu, F.silu]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either "
                    "gelu or swiglu"
                )
            if (
                self.activation_func == F.gelu
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False, "
                    "and activation function is gelu, add_bias_linear must also be True."
                )
        if self.activation_func_fp8_input_store:
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")
        if self.apply_rope_fusion and self.rotary_interleaved:
            raise ValueError('rotary_interleaved does not work with apply_rope_fusion.')

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )

        if (
            self.moe_token_dispatcher_type == "alltoall_seq"
            and self.tensor_model_parallel_size != self.expert_tensor_parallel_size
        ):
            raise ValueError(
                "alltoall_seq dispatcher not support different TP size for MoE and Dense layer."
            )

        if self.num_moe_experts and self.fp8:
            # TE version below 1.7.0 will raise Error when handle zeros tokens for expert
            if not is_te_min_version("1.7.0.dev0"):
                raise ValueError(
                    "Only transformer-engine>=1.7.0 supports MoE FP8 training, "
                    f"but your version is {get_te_version()}."
                )

            if self.moe_grouped_gemm:
                raise ValueError("Grouped GEMM of MoE not support fp8 for now.")

        if self.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']:
            if self.variable_seq_lengths is True:
                raise ValueError(
                    f"Token dispatcher type: {self.moe_token_dispatcher_type} does not support "
                    f"variable sequence length, please use alltoall dispatcher instead."
                )

        if self.cp_comm_type is not None:
            if isinstance(self.cp_comm_type, list):
                assert len(self.cp_comm_type) == self.num_layers, (
                    f"Length of cp_comm_type ({len(self.cp_comm_type)}) should equal to "
                    f"the total number of transformer layers ({self.num_layers})!"
                )
            else:
                assert isinstance(
                    self.cp_comm_type, str
                ), "Unsupported communication type for context parallelism!"

            if self.moe_permute_fusion:
                if not is_te_min_version("1.11.0.dev0"):
                    raise ValueError(
                        "Only transformer-engine>=1.11.0 supports MoE permute fusion, "
                        f"but your version is {get_te_version()}."
                    )
            if self.moe_grouped_gemm and not is_te_min_version("1.11.0.dev0"):
                raise ValueError(
                    "Only transformer-engine>=1.11.0 supports FP8 grouped gemm, "
                    f"but your version is {get_te_version()}."
                )

        if self.moe_router_aux_loss_fusion:
            if self.moe_router_pre_softmax:
                raise ValueError(
                    "moe_router_aux_loss_fusion doesn't work with moe_router_pre_softmax."
                )
            if not is_triton_min_version("2.1.0", check_equality=False):
                raise ValueError(
                    "Only triton>=2.2.0 supports MoE aux loss fusion, "
                    f"but your version is {get_triton_version()}."
                )
        if self.use_chunkflow:
            if self.chunks_kept == -1 or self.chunk_size == -1:
                raise ValueError(
                    "when chunkflow enables chunk_size and chunk_kept should be set"
                    f"but get chunk_sie {self.chunk_size} chunk_kept {self.chunks_kept}"
                )

@dataclass
class MLATransformerConfig(TransformerConfig):
    """Configuration object for megatron-core Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig. Included YaRN RoPE parameters that is fused in MLA.
    """

    multi_latent_attention: bool = True
    """Whether to use Multi-Latent Attention."""

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_head_dim: int = 128
    """Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim"""

    qk_pos_emb_head_dim: int = 64
    """Dimension of the position embedding in the QK projection."""

    v_head_dim: int = 128
    """Dimension of the head in the V projection."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings."""

    normalization: str = "RMSNorm"
    """Default normalization layer for MLA models is RMSNorm."""

    max_position_embeddings: int = 163840
    """Maximum position embeddings for the original model."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE."""

    mscale: float = 0.707
    """Mscale for YaRN RoPE in Multi-Latent Attention."""

    mscale_all_dim: float = 0.707
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention."""



@dataclass
class DiffTransformerConfig(TransformerConfig):
    """Configuration object for megatron-core Diff Attention transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig.
    """

    diff_attention: bool = True
    """Whether to use diff attention. """

    diff_lambda_init: float = 0.5
    """The lambda parameter for the diff attention."""

    use_vectorized_lambda: bool = True
    """Whether to use vectorized lambda for the diff attention."""


@dataclass
class VLVisionConfig(TransformerConfig):
    """Configuration object for qwen vl Vision transformers.

    """
    spatial_patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    temporal_merge_size: int = 2
    in_channels: int = 3
    # qwen2.5vl specific
    vision_window_size: int = None
    fullatt_block_indexes: list[int] = None

@dataclass
class AudioConfig(TransformerConfig):
    """Configuration object for audio processing transformers.
    
    Args:
        input_size (int): Input feature dimension (e.g., mel-spectrogram features)
        linear_units (int): Hidden dimension for audio encoder
        attention_heads (int): Number of attention heads
        num_blocks (int): Number of encoder blocks
        downsample_rate (int): Downsampling rate for audio features
        use_padmask (bool): Whether to use padding mask
        tokenizer_version (str): Version of the audio tokenizer
        tokenizer_frame_rate (float): Frame rate for audio tokenizer
        encoder_dim (int): Output dimension of audio encoder
        llm_dim (int): Input dimension for language model
        ffn_dim (int): Hidden dimension for audio adaptor
        n_layer (int): Number of layers in audio adaptor
        attention_dropout_rate (float): Dropout rate for attention
        dropout_rate (float): General dropout rate
        att_type (str): Attention type to use
        kernel_size (int): Kernel size for FSMN
        sanm_shfit (int): Shift parameter for SANM
        max_position_embeddings (int): Maximum position embeddings for RoPE
        rope_theta (float): Theta parameter for RoPE
    """
    
    # Audio processing parameters
    encoder_type: str = "sensevoice" # "sensevoice" or "qwen3"
    input_size: int = 128       # encoder input
    linear_units: int = 1280    # encoder hidden
    attention_heads: int = 20
    num_blocks: int = 32
    downsample_rate: int = 4
    use_padmask: bool = True
    encoder_no_use_fused_attention: bool = False
    encoder_use_packed_attention: bool = False

    # Tokenizer parameters
    tokenizer_version: str = "25hz"
    tokenizer_frame_rate: float = 25
    tokenizer_code_book_size: int = 6761
    
    # Audio adaptor parameters
    encoder_dim: int = 1280     # adaptor input
    llm_dim: int = 4096         # adaptor output
    ffn_dim: int = 2048         # adaptor hidden
    adaptor_downsample_rate: int = 2
    adaptor_num_layers: int = 2
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.0

    # Freeze parameters
    freeze_encoder: bool = True
    encoder_lr_scale: float = None
    freeze_adaptor: bool = False
    adaptor_lr_scale: float = None
    freeze_decoder: bool = False
    decoder_lr_scale: float = None
    audio_input_output_lr_scale: float = None
    recompute_encoder: bool = False

    encoder_adaptor_sliding_window_size: int = 10000000     # default value of 10000000 effectively disables this
    encoder_adaptor_sliding_window_left_context: int = 0
    encoder_adaptor_bucketing_threshold: int = 10000000     # default value of 10000000 effectively disables this
    
    # Attention configuration
    att_type: str = "self_att_fsmn_sdpa"
    kernel_size: int = 31
    sanm_shfit: int = 0
    max_position_embeddings: int = 2048
    rope_theta: float = 10000

    # used when encoder_type == "qwen3"
    encoder_transformer_config: TransformerConfig = None
    encoder_transformer_layer_spec = None # ModuleSpec

    is_dual_channel: bool = False
    audio_output_head: bool = True
    """Whether to initialize audio embedding and audio output head.
    Set to False when only text output is needed (e.g., ASR-only models)."""