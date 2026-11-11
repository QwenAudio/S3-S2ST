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

import argparse
from typing import Iterable, Optional

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import tensor_model_parallel_broadcast
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2MLP
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class Qwen2EagleDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = {
                "type": "dynamic"
                if getattr(config, "use_dynamic_ntk", False)
                else "linear",
                "factor": 1.0,
            }
        self.self_attn = Qwen2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            rotary_emb=rotary_emb,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        if layer_idx != 0:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states, None


class Qwen2EagleForCausalLM(nn.Module):
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

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        del lora_config
        # TODO (@robertgshaw2): see if this can be moved out
        if cache_config.sliding_window is not None and hasattr(
            config, "max_window_layers"
        ):
            raise ValueError(
                "Sliding window for some but all layers is not "
                "supported. This model uses sliding window "
                "but `max_window_layers` = {} is less than "
                "`num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                )
            )

        super().__init__()
        self.config = config
        self.quant_config = quant_config

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
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_parameters = getattr(config, "rope_parameters", None)
        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_parameters=rope_parameters,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.fc = ColumnParallelLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=getattr(config, "bias", True),
            gather_output=True,
            skip_bias_add=False,
            quant_config=quant_config,
        )
        self.layers = nn.ModuleList(
            [
                Qwen2EagleDecoderLayer(
                    layer_idx,
                    config,
                    cache_config,
                    quant_config,
                    rotary_emb=self.rotary_emb,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        if config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        previous_hidden_states: Optional[
            torch.Tensor
        ] = None,  # no such argument during warmup
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if previous_hidden_states is None:
            previous_hidden_states = torch.empty_like(hidden_states)
        tensor_model_parallel_broadcast(previous_hidden_states)
        hidden_states = torch.cat(
            [hidden_states, previous_hidden_states],
            dim=-1,
        )
        hidden_states, residual = self.fc(hidden_states)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        logger.warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded."
                        )
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


def apply_lmhead_weights(
    target_model: str,
    draft_model: str,
):
    """Apply the LM head weights from the draft model to the target model."""

    def _load_model(model_path):
        if model_path.endswith(".safetensors"):
            import safetensors.torch

            return safetensors.torch.load_file(model_path)
        return torch.load(model_path)

    def _save_model(model, model_path):
        if model_path.endswith(".safetensors"):
            import safetensors.torch

            safetensors.torch.save_file(model, model_path)
        else:
            torch.save(model, model_path)

    target_model_weights = _load_model(target_model)
    draft_model_weights = _load_model(draft_model)

    if "lm_head.weight" in draft_model_weights:
        print("Draft model already has LM head weights.")
        return

    if "lm_head.weight" in draft_model_weights:
        print("Draft model does not have LM head weights.")
        raise ValueError("Draft model does not have LM head weights.")

    draft_model_weights["lm_head.weight"] = target_model_weights["lm_head.weight"]
    _save_model(draft_model_weights, draft_model)
    print("Applied LM head weights from target model to draft model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_model", type=str, required=True)
    parser.add_argument("-d", "--draft_model", type=str, required=True)
    args = parser.parse_args()

    apply_lmhead_weights(
        target_model=args.target_model,
        draft_model=args.draft_model,
    )
