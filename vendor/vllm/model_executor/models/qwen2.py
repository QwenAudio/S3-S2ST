# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from transformers import Qwen2Config

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionMetadata, AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_broadcast,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedColumnReplicatedLinear,
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
    load_weight_for_bladnn_kernels,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved, set_default_rope_theta
from vllm.utils import is_abort_request_id
from vllm.utils.torch_utils import direct_register_custom_op

from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
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


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        tensor_parallel: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if tensor_parallel:
            GateUpProjectLinear = MergedColumnParallelLinear
            DownProjectLinear = RowParallelLinear
        else:
            GateUpProjectLinear = MergedColumnReplicatedLinear
            DownProjectLinear = ReplicatedLinear
        self.gate_up_proj = GateUpProjectLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = DownProjectLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        del gate_up  # release memory
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-06,
        head_dim: int | None = None,
        qkv_bias: bool = True,
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
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
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

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
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
            **(
                {"target_head_size": target_head_size}
                if target_head_size and target_head_size != self.head_dim
                else {}
            ),
        )

        self.qk_norm = qk_norm
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if self.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if self.qk_norm else None

        # Determine if we can use fused QK norm + RoPE. And fused_qk_norm_rope
        # must be True when fused_with_kv_cache is True.
        self.fused_qk_norm_rope = (
            envs.VLLM_FUSE_QKNORM_AND_ROPE
            and isinstance(self.rotary_emb, RotaryEmbedding)
            and self.qk_norm
        )
        if self.fused_qk_norm_rope:
            logger.info_once("Using fused QK norm + RoPE kernel for Qwen2Attention")

        self.fused_qk_norm_rope_with_kvcache_write = (
            envs.VLLM_FUSE_QKNORM_ROPE_AND_KVCACHE_WRITE
            and isinstance(self.rotary_emb, RotaryEmbedding)
            and self.qk_norm
            and cache_config.cache_dtype == "auto"
            and getattr(self.attn.impl, "support_fuse_kvcache_write", False)
        )
        if self.fused_qk_norm_rope_with_kvcache_write:
            logger.info_once(
                "Using fused QK norm + RoPE + KV cache write kernel for Qwen2Attention"
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

        # Fallback to fused kernel without KV write
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
            q, k, v = torch.ops.vllm.qwen2_apply_fused_qk_norm_rope(
                self.layer_name, qkv, positions
            )
        else:
            # Fallback to non-fused QK Norm & RoPE implementation
            if qkv.size(0) > 0:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            else:
                q, k, v = qkv, qkv, qkv

            if self.qk_norm:
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
        forward_context: ForwardContext = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.attn.layer_name]

        if (
            (self.rotary_emb_dynamic_yarn or self.rotary_emb_semi_dynamic_yarn)
            and attn_metadata
            and attn_metadata.query_start_loc is not None
        ):
            assert not self.per_layer_rotary_emb, (
                "Per-layer rotary embeddings are not supported "
                "with dynamic yarn rotary embedding."
            )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self.apply_qk_norm_rope(qkv, positions)
        del qkv

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
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
        self.tensor_parallel_mlp = getattr(config, "tensor_parallel_mlp", True)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            rope_parameters=config.rope_parameters,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            qk_norm=getattr(config, "use_qk_norm", False),
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "qkv_bias", True),
            cache_config=cache_config,
            quant_config=quant_config,
            rotary_emb=rotary_emb,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            tensor_parallel=self.tensor_parallel_mlp,
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
        if not envs.VLLM_TEST_SKIP_MLP:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def qwen_2_model_invariants(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    """Shape invariants for Qwen2Model Model, those are translated to
    runtime assertions for unbacked dynamic shapes and are compiled away for
    backed"""
    # All these should be equal.
    # input_ids.size()[0]
    # positions.size()[-1]
    # intermediate_tensors["hidden_states"].size()[0]
    # inputs_embeds.size()[0]
    torch._check(input_ids.size()[0] == positions.size()[-1])
    if intermediate_tensors is not None:
        torch._check(
            input_ids.size()[0] == intermediate_tensors["hidden_states"].size()[0]
        )

    if inputs_embeds is not None:
        torch._check(input_ids.size()[0] == inputs_embeds.size()[0])

    # Hidden dimensions should match (hidden_size)
    # intermediate_tensors["hidden_states"].size()[1]
    # inputs_embeds.size()[1]
    if inputs_embeds is not None and intermediate_tensors is not None:
        torch._check(
            inputs_embeds.size()[1] == intermediate_tensors["hidden_states"].size()[1]
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    shape_invariants=qwen_2_model_invariants,
)
class Qwen2Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if is_interleaved(vllm_config.model_config.hf_text_config):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                )
            )

        self.config = config
        self.quant_config = quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.current_device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
        )

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
        rope_parameters = config.rope_parameters
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        logger.info(
            "Initialize Qwen2Model: rope_parameters = %s, "
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

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                getattr(config, "embedding_size", config.hidden_size),
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                rotary_emb=self.rotary_emb,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
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

        forward_context: ForwardContext = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        request_ids = kwargs.pop("request_ids", [])

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if (
                self.tp_size > 1
                and attn_metadata is not None
                and getattr(attn_metadata, "num_prefills", None) == 1
                and isinstance(request_ids, list)
                and len(request_ids) == 1
                and isinstance(input_ids, torch.Tensor)
                and input_ids.size(0) > 8192
            ):
                forward_stop_flag = torch.zeros(
                    1, dtype=torch.float16, device=self.current_device
                )

                if self.tp_rank == 0 and is_abort_request_id(request_ids[0]):
                    forward_stop_flag = torch.ones(
                        1, dtype=torch.float16, device=self.current_device
                    )

                forward_stop_flag_broadcast = tensor_model_parallel_broadcast(
                    forward_stop_flag, src=0
                )
                if forward_stop_flag_broadcast.gt(0.5):
                    break

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name, loaded_weight = load_weight_for_bladnn_kernels(
                self.quant_config, name, loaded_weight
            )

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".g_idx") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".g_idx") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
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
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config
        self.model = Qwen2Model(
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
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
        # masking the padded parts of the hidden states
        unpadded_vocab_size = getattr(
            self.config, "unpadded_vocab_size", self.config.vocab_size
        )
        if logits is not None and unpadded_vocab_size < self.config.vocab_size:
            logits[:, unpadded_vocab_size:].fill_(float("-inf"))
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self: Qwen2Attention = forward_context.no_compile_layers[layer_name]
    q, k, v = self.apply_fused_qk_norm_rope(qkv, positions)
    return q, k, v


def apply_fused_qk_norm_rope_fake(
    layer_name: str,
    qkv: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self: Qwen2Attention = forward_context.no_compile_layers[layer_name]
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    return q, k, v


direct_register_custom_op(
    op_name="qwen2_apply_fused_qk_norm_rope",
    op_func=apply_fused_qk_norm_rope,
    mutates_args=[],
    fake_impl=apply_fused_qk_norm_rope_fake,
    dispatch_key=current_platform.dispatch_key,
)
