# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2MLP
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.transformers_utils.config import set_default_rope_theta

from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class Qwen3MoeEagleDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        rotary_emb=None,
    ) -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = get_draft_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        target_head_size = getattr(config, "target_head_size", None)

        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=getattr(config, "rope_parameters", None),
            max_position=config.max_position_embeddings,
            qk_norm=False,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=head_dim,
            qkv_bias=getattr(config, "qkv_bias", True),
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            rotary_emb=rotary_emb,
            prefix=f"{prefix}.self_attn",
            target_head_size=target_head_size,
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + self attention
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        # Pre-norm + MLP
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen3MoeEagleModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # EAGLE fusion: norm embeddings + norm target hidden → concat → project
        self.e_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.h_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False,
        )

        head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        set_default_rope_theta(config, default_theta=1000000)
        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
        )

        self.layers = nn.ModuleList(
            [
                Qwen3MoeEagleDecoderLayer(
                    vllm_config=vllm_config,
                    rotary_emb=self.rotary_emb,
                    prefix=maybe_prefix(
                        prefix, f"layers.{start_layer_id}",
                    ),
                )
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        e_feat = self.e_norm(input_embeds)
        h_feat = self.h_norm(hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([e_feat, h_feat], dim=-1),
        )
        # RMSNorm returns a single tensor when residual is None, which
        # breaks tuple unpacking under torch.compile. Use a zero residual.
        residual = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip bias tensors not present in the model
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader,
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class EagleQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """EAGLE draft model for Qwen3 MoE targets.

    Inherits Qwen3MoeForCausalLM for isinstance() compatibility and
    compute_logits() (lm_head shared from target at runtime via
    process_eagle_weight).
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config,
        )
        self.model = Qwen3MoeEagleModel(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs.",
            )
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def transform(inputs):
            name, loaded_weight = inputs
            if "lm_head" not in name:
                name = "model." + name
            process_eagle_weight(self, name)
            return name, loaded_weight

        loader = AutoWeightsLoader(self, skip_prefixes=None)
        loader.load_weights(map(transform, weights))
