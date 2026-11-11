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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

import glob
import os
from typing import Iterable, Optional, Set, Union

import torch
import torch.nn as nn
from transformers.models.qwen3_tts_moe.configuration_qwen3_tts_moe import (
    Qwen3TTSMoeTalkerConfig,
)
from transformers.models.qwen3_tts_moe.modeling_qwen3_tts_moe import (
    Qwen3TTSMoeTalkerResizeMLP,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger

# from vllm.model_executor.layers.linear import ColumnParallelLinear
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .interfaces_base import VllmModelForTextGeneration
from .qwen3_subtalker import Qwen3TTSMoeTalkerCodePredictorModelForConditionalGeneration
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3TTSMoeForConditionalGeneration(
    # nn.Module, SupportsMultiModal, SupportsPP):
    nn.Module,
    SupportsPP,
    VllmModelForTextGeneration,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen3TTSMoeTalkerConfig = vllm_config.model_config.hf_config
        self.pad_embed = self.get_pad_embedding(config, vllm_config.model_config.model)
        if hasattr(config, "talker_config"):
            self.config = config.talker_config
        else:
            self.config = config
        logger.info(
            "init Qwen3TTSMoeForConditionalGeneration vllm_config %s prefix %s",
            vllm_config,
            prefix,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=self.config,
            architectures=["Qwen3MoeTalkerForCausalLM"],
        )
        self.code_predictor = (
            Qwen3TTSMoeTalkerCodePredictorModelForConditionalGeneration(
                config=self.config.code_predictor_config, talker_config=self.config
            )
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def get_pad_embedding(
        self,
        config,
        model_path,
    ) -> torch.Tensor:
        if hasattr(config, "talker_config"):
            talker_hf_config = config.talker_config

        text_projection = Qwen3TTSMoeTalkerResizeMLP(
            talker_hf_config.text_hidden_size,
            talker_hf_config.intermediate_size,
            talker_hf_config.hidden_size,
            talker_hf_config.hidden_act,
            bias=True,
        )

        safetensors = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for key, value in safetensors_weights_iterator(
            safetensors, use_tqdm_on_load=True
        ):
            if key.startswith("talker.text_projection."):
                text_projection.state_dict()[
                    key.replace("talker.text_projection.", "")
                ].copy_(value).cuda()
            if key in ["talker.model.text_embedding.weight"]:
                text_embedding = torch.nn.Embedding.from_pretrained(value).cuda()
        pad_embedding = text_projection(
            text_embedding(
                torch.tensor(
                    [config.tts_pad_token_id],
                )
            )
        )
        return pad_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # subtalker_sampling_tensors: Optional[SubtalkerSamplingTensors] = None,
        # rand_buf: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            # for profile_run
            inputs_embeds = self.pad_embed.clone().repeat(input_ids.shape[0], 1)

        input_ids = None
        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        if isinstance(hidden_states, tuple):
            (hidden_states, _) = hidden_states
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # TODO: use bad word
        logits = self.language_model.compute_logits(hidden_states)
        batch_size, vocab_size = logits.shape
        allowed_ids = list(range(0, self.config.code_predictor_config.vocab_size)) + [
            self.config.codec_eos_token_id
        ]
        mask = torch.ones(vocab_size, dtype=torch.bool).to(hidden_states.device)
        mask[allowed_ids] = False
        mask = mask.unsqueeze(0).expand(batch_size, self.config.vocab_size)
        logits = logits.masked_fill(mask, -1e8)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "talker.codec_head.": "language_model.lm_head.",
                "talker.model.layers": "language_model.model.layers",
                "talker.model.text_embedding": "language_model.model.text_embedding",
                "talker.model.codec_embedding": "language_model.model.embed_tokens",
                "talker.model.norm": "language_model.model.norm",
                "talker.code_predictor": "code_predictor",
            }
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.text_projection"],
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)
