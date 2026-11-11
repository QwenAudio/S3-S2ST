# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Inference-only Qwen3-Next-VL model compatible with HuggingFace weights."""
import copy
from collections.abc import Iterable
from typing import Optional, Union

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next_vl import (
    Qwen3NextVLConfig,
    Qwen3NextVLTextConfig,
)

from .interfaces import IsHybrid
from .qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextModel,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)

class Qwen3NextVLProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3NextVLConfig)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # the same shape as input_embeds
        "deepstack_input_embeds": 0,
    }
)
class Qwen3NextVLModel(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not get_pp_group().is_first_rank:
            assert self.start_layer >= len(
                vllm_config.model_config.hf_config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_input_embeds: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_shape = hidden_states.shape
        if residual is not None:
            hidden_states, _ = self.norm(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                residual.reshape(-1, residual.shape[-1]),
            )
        else:
            hidden_states, _ = self.norm(
                hidden_states.reshape(-1, hidden_states.shape[-1])
            )
        return hidden_states.view(hidden_shape)


class Qwen3NextVLForCausalLM(Qwen3NextForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextForCausalLM, self).__init__()

        config: Qwen3NextVLTextConfig = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert cache_config.mamba_cache_mode in ("none", "light"), (
            "Qwen3NextVL currently only support 'light' prefix caching, "
            "please use '--mamba-cache-mode=light'"
        )
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextVLModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config
                    else lora_config.lora_vocab_padding_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3NextVLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3NextVLForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3VLForConditionalGeneration, self).__init__()
        config: Qwen3NextVLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if not multimodal_config.get_limit_per_prompt(
            "image"
        ) and not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        self.language_model = Qwen3NextVLForCausalLM(
            vllm_config=temp_vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.use_deepstack = (
            hasattr(config.vision_config, "deepstack_visual_indexes")
            and len(config.vision_config.deepstack_visual_indexes) > 0
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        # register buffer for deepstack
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["mtp."]
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        if self.config.text_config.tie_word_embeddings:
            skip_prefixes.extend(["language_model.lm_head."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return Qwen3NextForCausalLM.get_mamba_state_shape_from_config(temp_vllm_config)
