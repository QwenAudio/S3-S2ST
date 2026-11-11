# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import math
import os
import platform
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm.attention.backends.abstract import AttentionBackend

try:
    import bladnn
except ImportError:
    bladnn = None

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fused_moe import (
    SharedFusedMoE,
    constant_experts_compute_triton,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.models.qsa_indexer_utils import (
    bf16_mqa_logits_torch,
    can_use_tl,
    fp16_paged_mqa_logits_torch,
    fp16_paged_mqa_logits_torch_optimize,
)  # noqa E501
from vllm.v1.attention.backends.mla.indexer import (
    Qwen3NextIndexerBackend,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    Qwen3NextIndexerAttentionSpec,
)

if can_use_tl:
    from vllm.model_executor.models.qsa_indexer_utils import (
        mqa_attn_fp16_return_logits_interface,
        tilelang_paged_mqa_logits_interface,
        tilelang_paged_mqa_logits_mtp_interface,
    )



KVCache = tuple[torch.Tensor, torch.Tensor]

logger = init_logger(__name__)

bladnn_workspace = None


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        # [dynamic_topk_moe] num_const_experts
        self.moe_plus_plus_constant = (
            getattr(config, "moe_plus_plus_constant", None) or 0
        )
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts + self.moe_plus_plus_constant,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        # add const_expert
        if self.moe_plus_plus_constant > 0:
            self.constant = torch.nn.Parameter(
                torch.zeros(config.moe_plus_plus_constant, config.hidden_size)
            )
            self.num_experts = config.num_experts  # num_route
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob

        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            routing_method_type=RoutingMethodType.Renormalize,
            max_top_k=config.num_experts if self.moe_plus_plus_constant > 0 else -1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            if envs.VLLM_MOE_RANDOM_GATING:
                router_logits = torch.randn_like(router_logits)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        # add const experts result
        if self.moe_plus_plus_constant > 0:
            final_hidden_states = constant_experts_compute_triton(
                expert_indices=self.experts.topk_ids_constant,
                expert_weights=self.experts.topk_weights_constant,
                num_experts=self.num_experts,
                constant_weights=self.constant,
                hidden_states=hidden_states,
                final_hidden_states=final_hidden_states,
            )

        return final_hidden_states.view(orig_shape)


class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
        # Qwen3-Next and Qwen3.5 has a different ba layout as well.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
            dtype=config.dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # The optimized version failed to be compiled with
        # pytorch 2.7.0a (ngc 25.03) / 2.8.0 (ppu 1.7.0).
        self.use_compat_impl = not is_torch_equal_or_newer("2.9.0")

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Use the fused triton kernel only on x86_64 platforms as it fails on aarch64
        self.use_fused_qkvzba_split_reshape_cat = (
            platform.machine() == "x86_64" 
            and self.num_k_heads // self.tp_size > 0
            and self.num_v_heads // self.tp_size > 0
            and is_power_of_two(self.num_v_heads // self.num_k_heads)
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[sum((key_dim, key_dim, value_dim, value_dim))],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads * 2],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """

        if self.use_compat_impl:
            torch.ops.vllm.gdn_attention_compat(
                hidden_states,
                output,
                self.prefix,
            )
            return

        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        if self.use_fused_qkvzba_split_reshape_cat:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_compat(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        if self.use_fused_qkvzba_split_reshape_cat:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        global bladnn_workspace
        if bladnn_workspace is None:
            bladnn_workspace = torch.zeros(
                (1000,), dtype=torch.int32, device=mixed_qkv.device
            )
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            num_tokens = mixed_qkv.shape[0]
            q = torch.zeros(
                [1, num_tokens, self.num_k_heads, self.head_k_dim],
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )  # noqa: E501
            v = torch.zeros(
                [1, num_tokens, self.num_v_heads, self.head_v_dim],
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )  # noqa: E501
            g = torch.zeros(
                [1, num_tokens, self.num_v_heads],
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )  # noqa: E501
            initial_state = torch.zeros(
                [1, self.num_v_heads, self.head_k_dim, self.head_v_dim],
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )  # noqa: E501
            cu_seqlens = torch.tensor(
                [0, num_tokens], device=mixed_qkv.device, dtype=torch.int32
            )  # noqa: E501
            chunk_gated_delta_rule(
                q,
                q,
                v,
                g,
                g,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        use_bladnn_gdn_recur = envs.VLLM_GDN_USE_BLADNN

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                retrieve_parent_token=attn_metadata.retrieve_parent_token,
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        g, beta = fused_gdn_gating(self.A_log, a.contiguous(), b, self.dt_bias)

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            if not use_bladnn_gdn_recur:
                core_attn_out_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        g=g_spec,
                        beta=beta_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[
                            : attn_metadata.num_spec_decodes + 1
                        ],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        retrieve_parent_token=attn_metadata.retrieve_parent_token,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            else:
                core_attn_out_spec = torch.empty(
                    value_spec.shape, dtype=value_spec.dtype, device=value_spec.device
                )  # (1, (B S), H, D)
                query_spec, key_spec = map(
                    lambda x: rearrange(x, "1 l h d -> l h d"), (query_spec, key_spec)
                )
                value_spec = rearrange(value_spec, "1 l h d -> l h d")
                g_spec, beta_spec = map(
                    lambda x: rearrange(x, "1 l d -> l d"), (g_spec, beta_spec)
                )
                bladnn.gdn_main_recur(
                    query_spec,
                    key_spec,
                    value_spec,
                    g_spec,
                    beta_spec,
                    core_attn_out_spec,
                    ssm_state,
                    ssm_state,
                    bladnn_workspace,
                    scale=None,
                    req_ids=spec_state_indices_tensor,
                    token_verify_ids=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    use_qk_l2_norm=True,
                )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q=query_non_spec.contiguous(),
                k=key_non_spec.contiguous(),
                v=value_non_spec.contiguous(),
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            if not use_bladnn_gdn_recur:
                core_attn_out_non_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[
                            : attn_metadata.num_decodes + 1
                        ],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            else:
                batch = attn_metadata.num_decodes
                query_non_spec, key_non_spec = map(
                    lambda x: rearrange(
                        x, "1 (b s) h d -> b s h d", b=batch, s=1, d=self.head_k_dim
                    ),
                    (query_non_spec, key_non_spec),
                )
                value_non_spec = rearrange(
                    value_non_spec,
                    "1 (b s) h d -> b s h d",
                    b=batch,
                    s=1,
                    d=self.head_v_dim,
                )
                g_non_spec, beta_non_spec = map(
                    lambda x: rearrange(
                        x,
                        "1 (b s) d -> b s d",
                        b=batch,
                        s=1,
                    ),
                    (g_non_spec, beta_non_spec),
                )

                core_attn_out_non_spec = torch.empty(
                    value_non_spec.shape,
                    dtype=value_non_spec.dtype,
                    device=value_non_spec.device,
                )
                bladnn.gdn_main_recur(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                    core_attn_out_non_spec,
                    ssm_state,
                    ssm_state,
                    bladnn_workspace,
                    scale=None,
                    req_ids=non_spec_state_indices_tensor,
                    use_qk_l2_norm=True,
                )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            if use_bladnn_gdn_recur and attn_metadata.num_prefills == 0:
                core_attn_out_non_spec = core_attn_out_non_spec.reshape(
                    1,
                    core_attn_out_non_spec.shape[0],
                    *core_attn_out_non_spec.shape[2:],
                )
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            if use_bladnn_gdn_recur:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(1)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


class Qwen3NextIndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype  # we use bf16 here for now
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self.dtype == torch.uint8:
            head_size_padded = 2 ** math.ceil(math.log2(self.head_dim))
            return MLAAttentionSpec(  # Only has one vector instead of K + V
                block_size=self.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                dtype=self.dtype,
                head_size_padded=head_size_padded,
            )
        else:
            return (
                Qwen3NextIndexerAttentionSpec(  # Only has one vector instead of K + V
                    block_size=self.cache_config.block_size,
                    num_kv_heads=self.num_kv_heads,  # diff from dpsk, num_kvhead!=1
                    head_size=self.head_dim,
                    dtype=self.dtype,
                )
            )

    def forward(self): ...
    def get_attn_backend(self) -> AttentionBackend:
        if self.dtype == torch.uint8:
            return DeepseekV32IndexerBackend
        else:
            return Qwen3NextIndexerBackend


def qwen_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    score_scale: float,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, k.shape[1], head_dim), torch.bfloat16),
        )

        return qwen_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q,
            k,
            weights,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            score_scale,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    # TODO:写torch版本
    #  ---1 写入kcache---
    nkvhead = k.shape[1]
    head_dim = k.shape[2]
    # （tokens, nkvheads, h）
    cache_view = kv_cache.view(-1, nkvhead, head_dim)
    cache_view[slot_mapping] = k[: slot_mapping.shape[0]]

    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = attn_metadata.prefill

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp16_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, nkvhead, head_dim), torch.bfloat16),
        )
        for chunk in prefill_metadata.chunks:
            k_fp16 = k_fp16_full[0][: chunk.total_seq_lens]
            # use cuda kernel
            ops.cp_gather_indexer_k_noquant_cache(
                kv_cache,
                k_fp16,
                chunk.block_table,
                chunk.cu_seq_lens,
            )
            use_tl = can_use_tl and envs.VLLM_QSA_PREFILL_USE_TL_INDEXER
            if use_tl:
                assert k_fp16.shape[1] == 1
                k_fp16 = k_fp16.squeeze(1)
                fp16_mqa_logits_func = mqa_attn_fp16_return_logits_interface  # tl
                logits = fp16_mqa_logits_func(
                    q=q[chunk.token_start : chunk.token_end],
                    kv=k_fp16,
                    weights=weights[chunk.token_start : chunk.token_end].to(
                        torch.float32
                    ),
                    cu_seqlen_ks=chunk.cu_seqlen_ks,
                    cu_seqlen_ke=chunk.cu_seqlen_ke,
                )  # noqa: E501
                logits /= score_scale
            else:
                fp16_mqa_logits_func = bf16_mqa_logits_torch
                logits = fp16_mqa_logits_func(
                    q[chunk.token_start : chunk.token_end],
                    k_fp16,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    score_scale,
                )

            num_rows = logits.shape[0]
            topk_indices = topk_indices_buffer[  # (m,topk)
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        decode_lens = (
            decode_metadata.decode_lens
        )  # tensor([4], device='cuda:0', dtype=torch.int32) (sps=3)

        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_decode_tokens = pack_seq_triton(q[:num_decode_tokens], decode_lens)
            decode_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens
            ).view(-1, *weights.shape[1:])
        else:
            # q:(m, nqheads, h) -> q(bs,m,nqheads, h)
            padded_q_decode_tokens = q[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
            # w:(m, h)
            decode_weights = weights[:num_decode_tokens]
        batch_size = padded_q_decode_tokens.shape[0]
        next_n = padded_q_decode_tokens.shape[1]  # sps_nextn
        assert batch_size == decode_metadata.seq_lens.shape[0]
        fp16_paged_mqa_logits_func = (
            fp16_paged_mqa_logits_torch
            if next_n != 1
            else fp16_paged_mqa_logits_torch_optimize
        )  # noqa: E501

        use_tl = can_use_tl and envs.VLLM_QSA_DECODE_USE_TL_INDEXER
        if use_tl:
            assert kv_cache.shape[2] == 1, "currently, tl only support index_kv_heads=1"
            if next_n == 1:
                assert next_n == 1, "currently, tl only support next_n=1"
                assert kv_cache.shape[1] == 64, (
                    "currently, we only support block_size=64"
                )
                fp16_paged_mqa_logits_func = tilelang_paged_mqa_logits_interface
            else:
                fp16_paged_mqa_logits_func = tilelang_paged_mqa_logits_mtp_interface
        logits = fp16_paged_mqa_logits_func(
            padded_q_decode_tokens,  # q(bs,nextn,nqheads, h)
            kv_cache,  # [num_blocks, block_size, H_KV, D]
            decode_weights,  # [B * next_n, H_Q]
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            score_scale=score_scale,
            max_seq_len=attn_metadata.max_seq_len,
        )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]

        if decode_metadata.requires_padding:
            topk_indices = pack_seq_triton(topk_indices, decode_lens)

        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def qwen_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    score_scale: float,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="qwen_sparse_attn_indexer",
    op_func=qwen_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=qwen_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Qwen3NextConfig,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        norm_class=Qwen3NextRMSNorm,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        # Indexer dimensions from config
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_kv_heads = config.index_kv_heads
        self.index_head_dim = config.index_head_dim

        # Projection sizes
        self.index_q_dim = self.index_n_heads * self.index_head_dim
        self.index_k_dim = self.index_kv_heads * self.index_head_dim
        # W projection: one weight per query head per token
        self.index_w_dim = self.index_n_heads
        # Combined QKW projection for efficiency
        self.index_qkw_dim = self.index_q_dim + self.index_k_dim + self.index_w_dim

        # no tensor parallel, just replicated
        self.index_qkw_proj = ReplicatedLinear(
            config.hidden_size,
            self.index_qkw_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qkw_proj",
        )

        # Q/K LayerNorm for stable scale across layers
        # This is critical because input hidden_states norm varies greatly across layers
        self.index_q_layernorm = norm_class(
            self.index_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_layernorm = norm_class(
            self.index_head_dim, eps=config.rms_norm_eps
        )

        # Score scale factor
        self.score_scale = self.index_head_dim**0.5
        self.topk_indices_buffer = topk_indices_buffer

        use_indexter_fp8 = envs.VLLM_QSA_USE_FP8_INDEXER
        self.use_indexter_fp8 = use_indexter_fp8
        logger.info_once(
            "Initialize Qwen3-Next Indexer with fp8 = %s",
            self.use_indexter_fp8,
        )
        if self.use_indexter_fp8:
            # try fp8
            self.scale_fmt = "ue8m0"
            self.quant_block_size = self.index_head_dim
            self.k_cache = Qwen3NextIndexerCache(
                # NOTE:can's use in hybrid model for different page size:
                # dsa_pagesize: block_size = 544 headdim=260 -> pagesize=141440
                # gdn_pagesize(max_pagesize): blocksize=544,
                # headdim=2048->max_page_size=1114112.
                # not compatable with dsa_pagesize
                head_dim=self.index_head_dim
                + self.index_head_dim // self.quant_block_size * 4,
                num_kv_heads=self.index_kv_heads,
                dtype=torch.uint8,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
            )
        else:
            # bf16 kv
            # (maxblock, block_size=64, numheadskv,  hdim)  each element bf16.
            self.k_cache = Qwen3NextIndexerCache(
                head_dim=self.index_head_dim,
                num_kv_heads=self.index_kv_heads,
                dtype=torch.torch.bfloat16,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
            )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size,  # noqa: E501
        )

        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = get_tensor_model_parallel_world_size()
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)

    def _align_heads_for_kernel(self, q, w, current_heads):
        pad_heads = 32 - current_heads
        if pad_heads > 0:
            # 1. 对 Q 进行 Padding (在 heads 维度)
            # noqa E501 q shape: [seq_len, current_heads, head_dim] -> [seq_len, target_heads, head_dim]
            q = F.pad(q, (0, 0, 0, pad_heads)).contiguous()
            # 2. 对 Weight 进行 Padding
            # w shape: [current_heads, ...] -> [target_heads, ...]
            w = F.pad(w, (0, pad_heads), value=0.0).contiguous()
        return q, w

    def forward(
        self, hidden_states: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        # ========== Stage 1: QKW Projection ==========
        qkw_mixed = self.index_qkw_proj(hidden_states)
        # Split Q, K, W
        q, k, w = torch.split(
            qkw_mixed, [self.index_q_dim, self.index_k_dim, self.index_w_dim], dim=-1
        )
        # ========== Stage 2: Apply LayerNorm ==========
        q = self.index_q_layernorm(
            q.view(-1, self.index_n_heads, self.index_head_dim)
        ).view(  # noqa: E501
            -1, self.index_n_heads * self.index_head_dim
        )
        k = self.index_k_layernorm(
            k.view(-1, self.index_kv_heads, self.index_head_dim)
        ).view(  # noqa: E501
            -1, self.index_kv_heads * self.index_head_dim
        )

        # ========== Stage 3: Apply Rotary Position Embeddings ==========
        q = q.view(-1, self.index_n_heads, self.index_head_dim)  # noqa: E501 for infer index_head_dim
        q, k = rotary_emb(positions, q, k)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        k = k.view(-1, self.index_kv_heads, self.index_head_dim)

        if self.use_indexter_fp8:
            # -----try fp8 quant ----
            assert self.index_kv_heads == 1
            k = k.view(-1, self.index_head_dim)  # (m, headdim)
            q = q.view(-1, self.index_head_dim)  # (m*nheads, headdim)
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
            # (m, nqheads, headdim)
            q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)
            # (m,nqheads,1)
            q_scale = q_scale.view(-1, self.index_n_heads, 1)
            # (m,nqheads,1) * scale / logits's score_scale
            w = w.unsqueeze(-1) * q_scale / self.score_scale
            # (m,nqheads)
            w = w.squeeze(-1)
            # TODO:opt, use a buffer
            q_fp8_padded, w_padded = self._align_heads_for_kernel(
                q_fp8, w, self.index_n_heads
            )  # noqa: E501
            # final_heads 将会是 32
            return torch.ops.vllm.sparse_attn_indexer(
                self.tp_rank,
                self.tp_size,
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8_padded,
                k,
                w_padded.to(torch.float32),
                self.quant_block_size,
                self.scale_fmt,
                self.index_topk,
                self.index_head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            return torch.ops.vllm.qwen_sparse_attn_indexer(
                hidden_states,  # (m,k)
                self.k_cache.prefix,  # 'model.layers.0.self_attn.indexer.k_cache'
                self.k_cache.kv_cache[0],  # DeepseekV32IndexerCache()   []
                q,  # (m=2048, nhead=64, h=128)  fp8  我们现在：（m,qnheads,h） bf16
                k,  # (m=2048, h=128)  bf16           我们现在：（m,qkvheads,h） bf16
                w,  # (m=2048, nhead=64) fp32         我们现在：（m,qheads） bf16
                self.index_topk,  # topk
                self.index_head_dim,
                self.max_model_len,  # 163840k.shape
                self.max_total_seq_len,  # 327680
                self.topk_indices_buffer,  # (maxchunk=2048,topk) int32
                self.score_scale,
            )


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rotary_emb: RotaryEmbedding | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
            self.per_layer_rotary_emb = False
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                dual_chunk_attention_config=self.dual_chunk_attention_config,
            )
            self.per_layer_rotary_emb = True

        # 普通att. 可以支持indexer
        # 如果用indexer,直接在这里实现。
        self.use_dsa = topk_indices_buffer is not None  # hasattr(config, "index_topk")
        if self.use_dsa:
            if envs.VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD >= 0:
                assert (
                    envs.VLLM_QSA_PREFILL_USE_CP 
                    or envs.VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP
                ), (
                    
                        "When VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD >= 0, "
                        "either VLLM_QSA_PREFILL_USE_CP or "
                        "VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP must be set to True "
                        "to enable context parallelism for QSA prefill."
                    
                )

            self.index_head_dim = config.index_head_dim
            # Create separate rope_parameters for indexer with partial_rotary_factor=1.0
            # Indexer always uses full rotation regardless of main attention's setting
            indexer_rope_parameters = None
            if config.rope_parameters is not None:
                indexer_rope_parameters = config.rope_parameters.copy()
                # TODO: read from config
                indexer_rope_parameters["partial_rotary_factor"] = 0.5
            self.indexer_rope_emb = get_rope(  # 传进去，传进indexer
                head_size=self.index_head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=indexer_rope_parameters,
                dual_chunk_attention_config=self.dual_chunk_attention_config,
            )
            self.indexer = Indexer(
                vllm_config=vllm_config,
                config=config,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            indexer=self.indexer,
            total_num_kv_heads=self.total_num_kv_heads,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        if self.use_dsa:
            # noqa E501 text-only pretrain use independent indexer_rope_emb, while vl-hybrid-pretrain use same rope_emb with main model 
            # noqa E501 here we use same rope_emb with main model, which align with pretrain (but dpsk use independent indexer_rope_emb)
            # noqa E501 main-model: 256 * partial_rotary_factor(0.25) just eqaul to index_head_dim(64)
            self.indexer(hidden_states, positions, self.rotary_emb)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        rotary_emb: RotaryEmbedding | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            topk_indices_buffer = None  # only use qsa in full-attn
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                vllm_config=vllm_config,
                config=config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                rotary_emb=rotary_emb,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
            # print(f"gangouyu linear_attentio output shape: {self_attention_output.shape}, linear_attentio output: {self_attention_output}")
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
            # print(f"gangouyu full_attentio output shape: {self_attention_output.shape}, full_attentio output: {self_attention_output}")
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # print(f"gangouyu after mlp hidden_states shape: {hidden_states.shape}, hidden_states: {hidden_states}")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        logger.info("Using rope scaling: %r", rope_parameters)
        if rope_parameters is not None:
            self.rotary_emb_dynamic_yarn = rope_parameters.get(
                "type", ""
            ) == "yarn" and rope_parameters.get("dynamic", False)
            self.rotary_emb_semi_dynamic_yarn = rope_parameters.get(
                "type", ""
            ) == "yarn" and rope_parameters.get("semi_dynamic", False)
            if self.rotary_emb_dynamic_yarn or self.rotary_emb_semi_dynamic_yarn:
                # no static scaling anymore
                rope_parameters["dynamic_scaling_factor"] = rope_parameters.get(
                    "factor", 1.0
                )
                rope_parameters["factor"] = 1.0
            rope_parameters["dynamic_yarn"] = self.rotary_emb_dynamic_yarn
            rope_parameters["semi_dynamic_yarn"] = self.rotary_emb_semi_dynamic_yarn
            config.rope_parameters = rope_parameters
        else:
            self.rotary_emb_dynamic_yarn = False
            self.rotary_emb_semi_dynamic_yarn = False
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_parameters = config.rope_parameters
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        logger.info(
            "Initialize Qwen3NextModel: rope_parameters = %s, "
            "dual_chunk_attention_config = %s",
            rope_parameters,
            dual_chunk_attention_config,
        )

        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        self.device = current_platform.device_type
        self.use_qsa = hasattr(config, "index_topk") and config.index_topk > 0
        if self.use_qsa:
            topk_tokens = config.index_topk
            logger.info(
                "Initialize Qwen3NextModel with QSA attention: "
                "topk_tokens = %s ",
                topk_tokens,
            )
            topk_indices_buffer = torch.empty(
                # (chunklen, topk)
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                rotary_emb=self.rotary_emb,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        layer_idx = self.start_layer  # 从起始层开始计数
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            # 1. 打印输入信息 (可选：打印 shape 而不是具体数值，避免刷屏)
            # print(f"gangouyu [Layer {layer_idx}] Input hidden_states shape: {hidden_states.shape}, hidden_states: {hidden_states}")
            
            # 2. 执行前向传播
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            
            # 3. 打印输出信息
            # print(f"gangouyu [Layer {layer_idx}] Output hidden_states shape: {hidden_states.shape}, hidden_states: {hidden_states}")
            
            # 4. (可选) 如果你想看具体的数值统计（均值/方差），防止 NaN
            # print(f"  -> HS Mean: {hidden_states.mean().item():.6f}, Std: {hidden_states.std().item():.6f}")

            layer_idx += 1

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=getattr(self.config, "num_experts", 0),
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue

                    if "index_" in name:
                        name = name.replace(
                            "self_attn.index_", "self_attn.indexer.index_"
                        )

                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        if getattr(self.model.config, "num_experts", 0) == 0:
            return

        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        assert cache_config.mamba_cache_mode in ("none", "light"), (
            "Qwen3Next currently only supports 'light' prefix caching, "
            "please use '--mamba-cache-mode=light'"
        )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."]
            + (["lm_head."] if self.config.tie_word_embeddings else []),
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


@maybe_transfer_kv_layer
def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@maybe_transfer_kv_layer
def gdn_attention_compat(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_compat(
        hidden_states=hidden_states,
        output=output,
    )


def gdn_attention_compat_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_compat",
    op_func=gdn_attention_compat,
    mutates_args=["output"],
    fake_impl=gdn_attention_compat_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    qkvz_bs_stride,
    ba_bs_stride,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZDIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BADIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V

    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (
        mixed_qkvz + i_bs * qkvz_bs_stride + i_qk * QKVZDIM_T + tl.arange(0, q_end)
    )
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (
        mixed_qkvz + i_bs * qkvz_bs_stride + i_qk * QKVZDIM_T + tl.arange(q_end, k_end)
    )
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (
        mixed_qkvz + i_bs * qkvz_bs_stride + i_qk * QKVZDIM_T + tl.arange(k_end, v_end)
    )
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (
        mixed_qkvz + i_bs * qkvz_bs_stride + i_qk * QKVZDIM_T + tl.arange(v_end, z_end)
    )

    blk_q_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )

    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )

    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )

    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )

    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))

    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(0, b_end):
        blk_b_ptr = mixed_ba + i_bs * ba_bs_stride + i_qk * BADIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * ba_bs_stride + i_qk * BADIM_T + i
        blk_a_st_ptr = (
            a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end)
        )
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    T = mixed_qkvz.shape[0]
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [T, qkv_dim_t], dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    z = torch.empty(
        [T, num_heads_v, head_v], dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    b = torch.empty([T, num_heads_v], dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    grid = (T, num_heads_qk)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        mixed_qkvz.stride(0),
        mixed_ba.stride(0),
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)
