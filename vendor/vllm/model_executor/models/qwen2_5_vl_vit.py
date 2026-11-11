# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""Inference-only Qwen2.5-VL-ViT model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping
from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import MultiModalEmbeddingPooler
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLVideoInputs,
)
from .qwen2_vl import Qwen2VLDummyInputsBuilder as Qwen2_5_VLDummyInputsBuilder
from .utils import AutoWeightsLoader

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class Qwen2_5_VL_ViT(Qwen2_5_VLForConditionalGeneration, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.language_model = None

        self._pooler = MultiModalEmbeddingPooler()

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        return image_embeds

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)
        return video_embeds

    def get_multimodal_embeddings_v0(
        self,
        image_input: Optional[Qwen2_5_VLImageInputs] = None,
        video_input: Optional[Qwen2_5_VLVideoInputs] = None,
    ) -> Optional[MultiModalEmbeddings]:
        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            multimodal_embeddings += (image_embeds,)

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            multimodal_embeddings += (video_embeds,)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Optional[MultiModalEmbeddings]:
        """Run forward pass for Qwen2.5-VL-ViT.

        Args:
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
            second_per_grid_ts: Tensor `(num_videos)` of video time interval (
                in seconds) for each grid along the temporal dimension in the
                3D position IDs. `None` if no videos are passed.
        """

        # NOTE: In v1, mm_embeds is always generated at model runner from
        # `get_multimodal_embeddings`
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)

        if image_input is None and video_input is None:
            mm_embeds = None
        else:
            mm_embeds = self.get_multimodal_embeddings_v0(
                image_input=image_input, video_input=video_input
            )

        if image_input is not None:
            mm_embeds += (image_input["image_grid_thw"],)
        if video_input is not None:
            mm_embeds += (video_input["video_grid_thw"],)
        return mm_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["language_model"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
