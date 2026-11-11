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
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.utils.torch_utils import direct_register_custom_op

from .adapters import as_sparse_embedding_model
from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rotary_emb: RotaryEmbedding | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        target_head_size: int | None = None,
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
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
            prefix=f"{prefix}.o_proj",
        )

        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
            self.per_layer_rotary_emb = False
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position,
                rope_parameters=rope_parameters,
                dual_chunk_attention_config=dual_chunk_attention_config,
            )
            self.per_layer_rotary_emb = True
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
            attn_type=attn_type,
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
        ops.fused_qk_norm_rope(
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
            q, k, v = torch.ops.vllm.qwen3_apply_fused_qk_norm_rope(
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


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rotary_emb: RotaryEmbedding | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            rotary_emb=rotary_emb,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            target_head_size=getattr(config, "target_head_size", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
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
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )

        # Accept hidden layer of Omni 3.0 Talker,
        # Thinker will output the hidden_states of this layer
        self.accept_hidden_layer = None


class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def set_accept_hidden_layer(self, layer: int) -> None:
        self.model.accept_hidden_layer = layer

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
        logits_as_input: bool = False,
        prefill_callback: Callable | None = None,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            logits_as_input=logits_as_input,
            prefill_callback=prefill_callback,
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


def apply_fused_qk_norm_rope(
    layer_name: str,
    qkv: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self: Qwen3Attention = forward_context.no_compile_layers[layer_name]
    q, k, v = self.apply_fused_qk_norm_rope(qkv, positions)
    return q, k, v


def apply_fused_qk_norm_rope_fake(
    layer_name: str,
    qkv: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self: Qwen3Attention = forward_context.no_compile_layers[layer_name]
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    return q, k, v


direct_register_custom_op(
    op_name="qwen3_apply_fused_qk_norm_rope",
    op_func=apply_fused_qk_norm_rope,
    mutates_args=[],
    fake_impl=apply_fused_qk_norm_rope_fake,
    dispatch_key=current_platform.dispatch_key,
)


Qwen3ForSparseEmbedding = as_sparse_embedding_model(Qwen3ForCausalLM)
