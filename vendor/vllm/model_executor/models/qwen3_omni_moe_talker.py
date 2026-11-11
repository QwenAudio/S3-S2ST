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
from functools import cached_property
from typing import Iterable, Optional, Set, Union

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerResizeMLP,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP

# yapf: disable
from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    _get_feat_extract_output_lengths,
)

# yapf: enable
from .qwen3_subtalker import (
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_llm_pos_ids_for_vision

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsMRoPE,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen3OmniMoeTalkerConfig = vllm_config.model_config.hf_config
        self.pad_embed = self.get_pad_embedding(config, vllm_config.model_config.model)
        if hasattr(config, "talker_config"):
            self.config = config.talker_config
        else:
            self.config = config
        if hasattr(config, "thinker_config"):
            self.thinker_config = config.thinker_config
        logger.info(
            "init Qwen3OmniMoeTalkerForConditionalGeneration vllm_config %s prefix %s",
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
            Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(
                config=self.config.code_predictor_config
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

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_pad_embedding(
        self,
        config,
        model_path,
    ) -> torch.Tensor:
        if hasattr(config, "talker_config"):
            talker_hf_config = config.talker_config

        text_projection = Qwen3OmniMoeTalkerResizeMLP(
            talker_hf_config.thinker_hidden_size,
            talker_hf_config.intermediate_size,
            talker_hf_config.hidden_size,
            talker_hf_config.hidden_act,
            bias=True,
        ).to(dtype=torch.bfloat16)

        safetensors = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for key, value in safetensors_weights_iterator(
            safetensors, use_tqdm_on_load=True
        ):
            if key.startswith("talker.text_projection."):
                text_projection.state_dict()[
                    key.replace("talker.text_projection.", "")
                ].copy_(value)
            if key in ["thinker.model.embed_tokens.weight"]:
                thinker_embedding = torch.nn.Embedding.from_pretrained(value)
                thinker_embedding.to(torch.device("cuda"))

        pad_embedding = text_projection(
            thinker_embedding(
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
        logger.info("start talker forward")

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            # for profile_run and thinker finished:
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
                "talker.model.codec_embedding": "language_model.model.embed_tokens",
                "talker.model.norm": "language_model.model.norm",
                "talker.code_predictor": "code_predictor",
            }
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "thinker.",
                "token2wav.",
                "talker.hidden_projection",
                "talker.text_projection",
            ],
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        image_grid_thw = []
        video_grid_thw = []
        second_per_grid_ts = []
        audio_feature_lengths = []
        use_audio_in_video = False
        for mm_feature in mm_features:
            mm_item = mm_feature.data
            if mm_item is None:
                continue
            mm_input = mm_item.get_data()
            if (t := mm_input.get("image_grid_thw")) is not None:
                image_grid_thw.append(t.tolist())
            if (t := mm_input.get("video_grid_thw")) is not None:
                video_grid_thw.append(t.tolist())
            if (t := mm_input.get("second_per_grid_ts")) is not None:
                second_per_grid_ts.append(t)
            if (t := mm_input.get("audio_feature_lengths")) is not None:
                audio_feature_lengths.append(t)
            # Check for use_audio_in_video
            use_audio_in_video_value = mm_input.get("use_audio_in_video")
            if use_audio_in_video_value is not None:
                use_audio_in_video = bool(use_audio_in_video_value.item())

        input_ids = torch.tensor(input_tokens)
        if input_ids is None or input_ids.ndim != 1:
            raise ValueError("_omni3_get_input_positions_tensor expects 1D input_ids")

        thinker_config = self.thinker_config

        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        vision_start_token_id = thinker_config.vision_start_token_id
        vision_end_token_id = thinker_config.vision_end_token_id

        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)

        src_item = input_tokens
        if not second_per_grid_ts:
            second_per_grid_ts = [1] * video_grid_thw.shape[0]
        new_src_item: list[int] = []
        llm_pos_ids_list: list[torch.Tensor] = []

        idx = 0
        while idx < len(src_item):
            new_src_item_len = len(new_src_item)
            start_idx = (
                llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            )
            if use_audio_in_video and idx > 0:
                if (
                    src_item[idx] == vision_end_token_id
                    and src_item[idx - 1] == audio_end_token_id
                ):
                    # processing the <|audio_eos|> before <|vision_eos|>
                    start_idx -= 1
                elif (
                    src_item[idx] == audio_start_token_id
                    and src_item[idx - 1] == vision_start_token_id
                ):
                    # processing the <|audio_bos|> after <|vision_eos|>
                    start_idx -= 1
            new_src_item.append(src_item[idx])
            llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
            llm_pos_ids_list.append(llm_pos_ids)

            # move to the next token
            idx += len(new_src_item) - new_src_item_len

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        mrope_position_delta = (
            torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item)
        )

        return llm_positions, mrope_position_delta
