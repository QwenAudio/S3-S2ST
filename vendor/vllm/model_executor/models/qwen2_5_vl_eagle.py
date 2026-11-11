# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
)
from vllm.v1.sample.metadata import SamplingMetadata

from .utils import AutoWeightsLoader, is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "hidden_states": 0,
        "input_embeds": 0,
    }
)
class Qwen2_5_VLEagleModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        rope_scaling = getattr(config, "rope_scaling", None)
        logger.info("Using rope scaling in Qwen2_5VLEagle model: %r", rope_scaling)
        if rope_scaling is not None:
            self.rotary_emb_dynamic_yarn = rope_scaling.get(
                "type", ""
            ) == "yarn" and rope_scaling.get("dynamic", False)
            self.rotary_emb_semi_dynamic_yarn = rope_scaling.get(
                "type", ""
            ) == "yarn" and rope_scaling.get("semi_dynamic", False)
            if self.rotary_emb_dynamic_yarn or self.rotary_emb_semi_dynamic_yarn:
                # no static scaling anymore
                rope_scaling["dynamic_scaling_factor"] = rope_scaling.get("factor", 1.0)
                rope_scaling["factor"] = 1.0
            rope_scaling["dynamic_yarn"] = self.rotary_emb_dynamic_yarn
            rope_scaling["semi_dynamic_yarn"] = self.rotary_emb_semi_dynamic_yarn
            config.rope_scaling = rope_scaling
        else:
            self.rotary_emb_dynamic_yarn = False
            self.rotary_emb_semi_dynamic_yarn = False
        # not use head dim from config
        head_dim = config.hidden_size // config.num_attention_heads
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_parameters = getattr(config, "rope_parameters", None)
        logger.info(
            "Initialize Qwen2_5VLEagleModel model: rope_theta = %s, "
            "rope_parameters = %s",
            rope_theta,
            rope_parameters,
        )

        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_parameters=rope_parameters,
        )
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # the quant config of eagle model not share with target model
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    self.config,
                    cache_config=cache_config,
                    rotary_emb=self.rotary_emb,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                self.config.target_hidden_size * 2, self.config.hidden_size, bias=True
            )
        else:
            self.fc = torch.nn.Linear(
                self.config.hidden_size * 2, self.config.hidden_size, bias=True
            )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states = hidden_states + residual
        return hidden_states, hidden_states

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
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
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


class EagleQwen2_5_VLForConditionalGeneration(Qwen2ForCausalLM):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen2_5_VLEagleModel(
            vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
        )
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_lm_head_weight = False
        includes_norm_weight = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
                includes_lm_head_weight = True
            elif "model.norm.weight" in name:
                includes_norm_weight = True
            model_weights[name] = loaded_weight

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_lm_head_weight:
            skip_substrs.append("lm_head.weight")
        if not includes_norm_weight:
            skip_substrs.append("model.norm.weight")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
