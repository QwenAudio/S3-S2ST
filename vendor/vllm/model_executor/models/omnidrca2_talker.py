# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable
from itertools import islice
from typing import Callable

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OmniDRCA2Config

from .interfaces import SupportsPP
from .interfaces_base import VllmModelForTextGeneration
from .qwen2 import Qwen2Model
from .qwen3 import Qwen3Model
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3LLMModel(Qwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        matching_hidden_size: int,
        prefix: str = "",
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.input_matching = nn.Linear(
            matching_hidden_size, self.config.hidden_size, bias=False
        )
        self.output_matching = nn.Linear(
            self.config.hidden_size, matching_hidden_size, bias=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
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

        hidden_states = self.input_matching(hidden_states)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = self.output_matching(hidden_states)

        return hidden_states


class OmniDRCA2TalkerForConditionalGeneration(
    nn.Module,
    SupportsPP,
    VllmModelForTextGeneration,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: OmniDRCA2Config = vllm_config.model_config.hf_config
        self.omnidrca2_config = config
        self.cosyvoice_codex_config = getattr(config, "cosyvoice_codex", None)
        if isinstance(self.cosyvoice_codex_config, dict):
            self.has_cosyvoice_codex = bool(
                self.cosyvoice_codex_config.get("enabled", False)
            )
        else:
            self.has_cosyvoice_codex = bool(
                getattr(self.cosyvoice_codex_config, "enabled", False)
            )
        self.config = config.audio_config
        quant_config = vllm_config.quant_config

        logger.info(
            "init OmniDRCA2TalkerForConditionalGeneration vllm_config %s prefix %s",
            vllm_config,
            prefix,
        )
        if self.has_cosyvoice_codex:
            self._init_cosyvoice_codex_talker(vllm_config, prefix)
            return

        # rewrite lm_head shape of talker
        assert self.config.crq_transformer_config.tie_word_embeddings
        self.config.crq_transformer_config.vocab_size = self.config.codebook_size
        self.config.crq_transformer_config.embedding_size = self.config.hidden_size

        self.model = Qwen3LLMModel(
            vllm_config=vllm_config.with_hf_config(
                self.config.crq_transformer_config,
            ),
            matching_hidden_size=self.config.hidden_size,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            if self.config.crq_transformer_config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.config.codebook_size,
                    self.config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.config.codebook_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def _codex_cfg_get(self, key: str, default=None):
        if isinstance(self.cosyvoice_codex_config, dict):
            return self.cosyvoice_codex_config.get(key, default)
        return getattr(self.cosyvoice_codex_config, key, default)

    def _init_cosyvoice_codex_talker(
        self, vllm_config: VllmConfig, prefix: str = ""
    ) -> None:
        model_dir = vllm_config.model_config.model
        llm_init_param_path = self._codex_cfg_get("llm_init_param_path")
        if not llm_init_param_path:
            raise ValueError("cosyvoice_codex.llm_init_param_path is required")

        local_config = os.path.join(model_dir, llm_init_param_path)
        if not os.path.isabs(llm_init_param_path) and os.path.isdir(local_config):
            llm_init_param_path = local_config
        llm_config = Qwen2Config.from_pretrained(llm_init_param_path)
        if getattr(llm_config, "rope_parameters", None) is None:
            llm_config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": getattr(llm_config, "rope_theta", 1000000.0),
            }
        llm_config.vocab_size = int(llm_config.vocab_size)

        self.config.hidden_size = int(self._codex_cfg_get("llm_hidden_size", 896))
        self.config.group_size = int(getattr(self.config, "group_size", 1))
        self.config.codebook_size = int(self._codex_cfg_get("codec_vocab_size"))
        self.config.codec_unit = int(
            self._codex_cfg_get("codec_unit", self.config.codebook_size)
        )
        if not hasattr(self.config, "bos_token_id"):
            self.config.bos_token_id = 0
        if not hasattr(self.config, "eos_token_id"):
            self.config.eos_token_id = self.config.codec_unit

        self.cosyvoice_model_dir = model_dir
        self.cosyvoice_llm_file = self._codex_cfg_get("llm_file", "talker.pt")
        self.cosyvoice_input_proj_file = self._codex_cfg_get(
            "input_proj_file", "talker_input_proj.pt"
        )
        self.cosyvoice_codec_embed_file = self._codex_cfg_get(
            "codec_embed_file", "talker_codec_embed.pt"
        )
        self.cosyvoice_codec_head_file = self._codex_cfg_get(
            "codec_head_file", "talker_codec_head.pt"
        )
        self._cosyvoice_weights_loaded = False

        self.model = Qwen2Model(
            vllm_config=vllm_config.with_hf_config(llm_config),
            prefix=maybe_prefix(prefix, "model"),
        )
        self.input_proj = nn.Linear(
            int(self.omnidrca2_config.text_config.hidden_size),
            self.config.hidden_size,
            bias=False,
        )
        self.codec_embed = nn.Embedding(
            self.config.codebook_size,
            self.config.hidden_size,
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.codebook_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.codebook_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        logger.info(
            "init CosyVoice codex talker: model_dir=%s codec_vocab_size=%s "
            "codec_unit=%s hidden_size=%s llm_init_param_path=%s",
            model_dir,
            self.config.codebook_size,
            self.config.codec_unit,
            self.config.hidden_size,
            llm_init_param_path,
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.has_cosyvoice_codex:
            input_ids = torch.clamp(input_ids, min=0, max=self.config.codebook_size - 1)
            return self.codec_embed(input_ids)
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if self.has_cosyvoice_codex:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            elif inputs_embeds.shape[-1] != self.config.hidden_size:
                inputs_embeds = self.input_proj(inputs_embeds)
            if positions.dim() > 1:
                positions = positions[-1]
            if positions.numel() != inputs_embeds.shape[0]:
                if positions.numel() > inputs_embeds.shape[0]:
                    positions = positions[: inputs_embeds.shape[0]]
                else:
                    raise ValueError(
                        "CosyVoice codex talker positions length "
                        f"{positions.numel()} does not match inputs_embeds length "
                        f"{inputs_embeds.shape[0]}"
                    )
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
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
        if self.has_cosyvoice_codex:
            return self._load_cosyvoice_codex_weights()

        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "audio_invert_tower.": "",
                "crq_transformer.": "model.",
                "input_matching.": "model.input_matching.",
                "output_matching.": "model.output_matching.",
                "audio_tower.embed_tokens": "model.embed_tokens",
            }
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "audio_tower.",
                "continuous_audio_tower.",
                "language_model.",
                "pre_matching.",
            ],
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)

    def _load_cosyvoice_codex_weights(self) -> set[str]:
        if self._cosyvoice_weights_loaded:
            return set()

        def _load_state(filename: str) -> dict[str, torch.Tensor]:
            path = os.path.join(self.cosyvoice_model_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"CosyVoice codex weight file not found: {path}")
            state = torch.load(path, map_location="cpu")
            if not isinstance(state, dict):
                raise TypeError(f"Expected state dict in {path}, got {type(state)}")
            return state

        loaded = set()
        llm_state = _load_state(self.cosyvoice_llm_file)
        loaded.update(self.model.load_weights(llm_state.items()))

        input_proj_state = _load_state(self.cosyvoice_input_proj_file)
        if input_proj_state:
            self.input_proj.load_state_dict(input_proj_state, strict=True)
            loaded.update(f"input_proj.{name}" for name in input_proj_state)

        codec_embed_state = _load_state(self.cosyvoice_codec_embed_file)
        self.codec_embed.load_state_dict(codec_embed_state, strict=True)
        loaded.update(f"codec_embed.{name}" for name in codec_embed_state)

        if get_pp_group().is_last_rank:
            codec_head_state = _load_state(self.cosyvoice_codec_head_file)
            params_dict = dict(self.lm_head.named_parameters(remove_duplicate=False))
            for name, loaded_weight in codec_head_state.items():
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(f"lm_head.{name}")

        self._cosyvoice_weights_loaded = True
        logger.info("Loaded CosyVoice codex talker weights from %s", self.cosyvoice_model_dir)
        return set(dict(self.named_parameters(remove_duplicate=False)))
