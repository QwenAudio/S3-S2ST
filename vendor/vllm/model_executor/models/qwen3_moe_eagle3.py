# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Iterable
from typing import Generator, Optional

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
)
from vllm.transformers_utils.config import set_default_rope_theta

from .utils import AutoWeightsLoader, is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


@contextlib.contextmanager
def with_draft_model_config(
    vllm_config: VllmConfig,
) -> Generator[VllmConfig, None, None]:
    """Ensure that the draft model config is set in the vLLM config."""
    if (
        vllm_config.speculative_config is None
        or vllm_config.speculative_config.draft_model_config is None
    ):
        raise ValueError("speculative_config.draft_model_config must be set")
    model_config = vllm_config.model_config
    quant_config = vllm_config.quant_config
    vllm_config.model_config = vllm_config.speculative_config.draft_model_config
    vllm_config.quant_config = vllm_config.draft_quant_config
    try:
        yield vllm_config
    finally:
        vllm_config.model_config = model_config
        vllm_config.quant_config = quant_config


class Qwen3MoeEagle3DecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rotary_emb: Optional[RotaryEmbedding] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, rotary_emb=rotary_emb, prefix=prefix)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "qkv_proj"),
        )
        self.mlp = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class Qwen3MoeEagle3ModelWithDense(nn.Module):
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
        # not use head dim from config
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_parameters = config.rope_parameters
        logger.info(
            "Initialize Qwen3MoEEagle3Model model: rope_parameters = %s, ",
            rope_parameters,
        )

        set_default_rope_theta(config, default_theta=1000000)
        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # the quant config of eagle model not share with target model
        with with_draft_model_config(vllm_config):
            self.layers = nn.ModuleList(
                [
                    Qwen3MoeEagle3DecoderLayer(
                        vllm_config=vllm_config,
                        rotary_emb=self.rotary_emb,
                        prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
                    )
                ]
            )
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                self.config.target_hidden_size * 3, self.config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                self.config.hidden_size * 3, self.config.hidden_size, bias=False
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]

        residual = None
        hidden_states, residual = self.layers[0](
            positions,
            inputs_embeds,
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
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # draft will share embed with target
            if "embed_tokens" in name:
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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
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
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@support_torch_compile
class Qwen3MoeEagle3ModelWithMoe(nn.Module):
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
        # not use head dim from config
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_parameters = config.rope_parameters
        logger.info(
            "Initialize Qwen3MoEEagle3Model model: rope_parameters = %s, ",
            rope_parameters,
        )

        set_default_rope_theta(config, default_theta=1000000)
        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # the quant config of eagle model not share with target model
        with with_draft_model_config(vllm_config):
            self.layers = nn.ModuleList(
                [
                    Qwen3MoeDecoderLayer(
                        vllm_config=vllm_config,
                        rotary_emb=self.rotary_emb,
                        prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
                    )
                ]
            )
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
        )

        self.pre_fc_base_norm = RMSNorm(
            self.config.hidden_size * 3,
            eps=self.config.rms_norm_eps,
        )
        if hasattr(self.config, "target_hidden_size"):
            self.fc_base = torch.nn.Linear(
                self.config.target_hidden_size * 3, self.config.hidden_size, bias=False
            )
        else:
            self.fc_base = torch.nn.Linear(
                self.config.hidden_size * 3, self.config.hidden_size, bias=False
            )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]

        residual = None
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        hidden_states, residual = self.layers[0](
            positions,
            hidden_states,
            residual,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
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
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # draft will share embed with target
            if "embed_tokens" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
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
                    # # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
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
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Eagle3Qwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.ffn_dense = (
            not hasattr(self.config, "num_experts") or self.config.num_experts == 0
        )
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        if self.ffn_dense:
            self.model = Qwen3MoeEagle3ModelWithDense(
                vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
            )
        else:
            self.model = Qwen3MoeEagle3ModelWithMoe(
                vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
            )
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros((self.config.vocab_size), dtype=torch.long),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds=inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        base = torch.arange(self.config.vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # combine multiple auxiliary hidden states returned by eagle3
        if self.ffn_dense:
            return self.model.fc(hidden_states)
        hidden_states = self.model.pre_fc_base_norm(hidden_states)
        return self.model.fc_base(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def remap_weight_names(name_and_weight):
            name, weight = name_and_weight
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
            if "lm_head" not in name:
                name = "model." + name
            return name, weight

        model_weights = map(remap_weight_names, weights)

        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights)
