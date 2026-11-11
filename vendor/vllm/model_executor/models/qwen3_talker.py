# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, List, Optional, Tuple, Union

try:
    import deep_gemm
    from deep_ep import Buffer
except (ImportError, ModuleNotFoundError):
    pass
import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.expert_parallel import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

GLOBAL_EXTRA_EXPERT_STREAM = None

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert_gate = torch.nn.Linear(
                config.hidden_size,
                1,
                bias=False,
            )
            self.shared_expert = Qwen2MoeMLP(
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
            self.shared_expert_gate = None

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=hasattr(config, "n_group")
            and hasattr(config, "topk_group"),
            num_expert_group=getattr(config, "n_group", None),
            topk_group=getattr(config, "topk_group", None),
            prefix=f"{prefix}.experts",
        )

    def _shared_expert(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_logits, _ = self.shared_expert_gate(hidden_states)
                shared_output = F.sigmoid(shared_logits) * shared_output
        return shared_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        if envs.VLLM_MOE_RANDOM_GATING:
            router_logits = torch.randn_like(router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if final_hidden_states[0] is None:
            final_hidden_states = final_hidden_states[1]
        else:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)

class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rotary_emb: RotaryEmbedding | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        target_head_size: int | None = None,
        use_reduce_scatter: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=not use_reduce_scatter,
            prefix=f"{prefix}.o_proj",
        )

        if rotary_emb:
            self.rotary_emb = rotary_emb
            self.per_layer_rotary_emb = False
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=rope_parameters,
                dual_chunk_attention_config=dual_chunk_attention_config,
            )
        if rope_parameters is not None:
            self.rotary_emb_dynamic_yarn = rope_parameters.get("dynamic_yarn", False)
            self.rotary_emb_semi_dynamic_yarn = rope_parameters.get(
                "semi_dynamic_yarn", False
            )
        else:
            self.rotary_emb_dynamic_yarn = False
            self.rotary_emb_semi_dynamic_yarn = False

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
            **{
                "target_head_size": target_head_size,
            }
            if target_head_size and target_head_size != self.head_dim
            else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Determine if we can use fused QK norm + RoPE. And fused_qk_norm_rope
        # must be True when fused_with_kv_cache is True.
        self.fused_qk_norm_rope = envs.VLLM_FUSE_QKNORM_AND_ROPE and isinstance(
            self.rotary_emb, RotaryEmbedding
        )
        if self.fused_qk_norm_rope:
            logger.info_once("Using fused QK norm + RoPE kernel for Qwen3MoeAttention")

        self.fused_qk_norm_rope_with_kvcache_write = (
            envs.VLLM_FUSE_QKNORM_ROPE_AND_KVCACHE_WRITE
            and isinstance(self.rotary_emb, RotaryEmbedding)
            and cache_config.cache_dtype == "auto"
            and getattr(self.attn.impl, "support_fuse_kvcache_write", False)
        )
        if self.fused_qk_norm_rope_with_kvcache_write:
            logger.info_once(
                "Using fused QK norm + RoPE + KV cache write kernel for "
                "Qwen3MoeAttention"
            )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    def apply_fused_qk_norm_rope(self, qkv, positions):
        if self.fused_qk_norm_rope_with_kvcache_write:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is not None:
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.attn.layer_name]
                attn_metadata.skip_write_kvcache = True
                kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
                key_cache, value_cache = kv_cache.unbind(0)
                ops.fused_qk_norm_rope_and_kvcache_write_flash(
                    qkv,
                    self.num_heads,
                    self.num_kv_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.q_norm.variance_epsilon,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    self.rotary_emb.cos_sin_cache,
                    self.rotary_emb.is_neox_style,
                    positions.view(-1),
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                )

                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                return q, k, v

        # Fallback to fused QK norm + RoPE without KV write
        ops.trtllm_fused_qk_norm_rope(
            qkv,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_norm.variance_epsilon,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            self.rotary_emb.is_neox_style,
            positions.view(-1),
        )

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def apply_qk_norm_rope(self, qkv, positions):
        if self.fused_qk_norm_rope:
            q, k, v = torch.ops.vllm.qwen3_moe_apply_fused_qk_norm_rope(
                self.layer_name, qkv, positions
            )
        else:
            # Fallback to non-fused QK Norm & RoPE implementation
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

            if q.size(0) > 0 and k.size(0) > 0:
                q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self.apply_qk_norm_rope(qkv, positions)
        del qkv

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)

        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rotary_emb: Optional[RotaryEmbedding] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        self.quant_config = quant_config
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.use_reduce_scatter = (
            envs.VLLM_TPEP_REDUCE_SCATTER
            and self.tp_size > 1
            and get_expert_parallel_world_size() > 1
        )

        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rotary_emb=rotary_emb,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
            target_head_size=getattr(config, "target_head_size", None),
            use_reduce_scatter=self.use_reduce_scatter,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        self.is_final_layer = layer_idx >= config.num_hidden_layers - 1

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                vllm_config=vllm_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None or self.use_reduce_scatter:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if self.use_reduce_scatter:
            start_index = hidden_states.shape[0] // self.tp_size * self.tp_rank
            end_index = hidden_states.shape[0] // self.tp_size * (self.tp_rank + 1)
            hidden_states = get_tp_group().reduce_scatter(hidden_states, dim=0)
            residual = residual[start_index:end_index]

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if not envs.VLLM_TEST_SKIP_MLP:
            hidden_states = self.mlp(hidden_states)

            if self.use_reduce_scatter:
                if not self.is_final_layer:
                    hidden_states = hidden_states + residual
                    hidden_states = get_tp_group().all_gather(hidden_states, dim=0)
                    residual = None
                else:
                    hidden_states = get_tp_group().all_gather(hidden_states, dim=0)
                    residual = get_tp_group().all_gather(residual, dim=0)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3MoeTalker(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

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
        rope_parameters = getattr(config, "rope_parameters", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        logger.info(
            "Initialize Qwen3MoEMoel model: rope_parameters = %s, "
            "dual_chunk_attention_config = %s",
            rope_parameters,
            dual_chunk_attention_config,
        )

        set_default_rope_theta(config, default_theta=1000000)
        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        # for tts model
        self.text_embedding = None
        if vllm_config.model_config.is_qwen3_tts_talker_model():
            self.text_embedding = VocabParallelEmbedding(
                config.text_vocab_size,
                config.text_hidden_size,
                prefix=f"{prefix}.text_embedding")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(
                vllm_config=vllm_config,
                rotary_emb=self.rotary_emb,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        self.layer_name = prefix
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.need_add_residual = not (
            envs.VLLM_TPEP_REDUCE_SCATTER
            and get_tensor_model_parallel_world_size() > 1
            and get_expert_parallel_world_size() > 1
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (
            self.text_embedding(input_ids) if self.text_embedding is not None else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # Collect auxiliary hidden states if specified
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual
                    if (self.need_add_residual and residual is not None)
                    else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
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

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeTalkerForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen3MoeMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3MoeTalker(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.aux_hidden_state_layers: tuple[int] = tuple()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

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
            if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
