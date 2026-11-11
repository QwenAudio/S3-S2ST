# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""

import copy
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
)
from vllm.model_executor.models.qwen3_next_vl import Qwen3NextVLForConditionalGeneration
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextVLTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)

logger = init_logger(__name__)
KVCache = tuple[torch.Tensor, torch.Tensor]


class Qwen3NextVLMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        config: Qwen3NextVLTextConfig = model_config.hf_text_config
        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
        )
        text_config = vllm_config.model_config.hf_config.text_config
        temp_model_config = copy.deepcopy(vllm_config.model_config)
        temp_model_config.hf_config = text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

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
            "Initialize Qwen3NextVLMTPModel: rope_parameters = %s, "
            "dual_chunk_attention_config = %s",
            rope_parameters,
            dual_chunk_attention_config,
        )

        self.rotary_emb = get_rope(
            head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        temp_vllm_config = VllmConfig(
            model_config=temp_model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
        )
        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                temp_vllm_config,
                layer_type="full_attention",
                rotary_emb=self.rotary_emb,
                prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            # for vl model, inputs_embeds is not None
            # assert inputs_embeds is not None
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
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
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
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
                    # Skip layers on other devices.
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
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3NextVLMTP(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        vision_config = vllm_config.model_config.hf_config.vision_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        assert cache_config.mamba_cache_mode in ("none", "light"), (
            "Qwen3NextVLMTP currently only support 'light' prefix caching, "
            "please use '--mamba-cache-mode=light'"
        )
        self.quant_config = vllm_config.quant_config
        super().__init__()
        self.config = config
        self.model = Qwen3NextVLMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )
        self.unpadded_vocab_size = config.vocab_size
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
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
        self.use_deepstack = (
            hasattr(vision_config, "deepstack_visual_indexes")
            and len(vision_config.deepstack_visual_indexes) > 0
        )
        self.deepstack_num_level = (
            len(vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )
        self.visual_dim = vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def _remove_deepstack_embeds(
        self, multimodal_embeddings: MultiModalEmbeddings
    ) -> MultiModalEmbeddings:
        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)
        (
            multimodal_embeddings_main,
            _,
        ) = torch.split(
            multimodal_embeddings_cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )
        multimodal_embeddings = torch.split(
            multimodal_embeddings_main, visual_lens, dim=0
        )
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._get_text_embeddings(
            input_ids,
            self.model.get_input_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError(
                "`get_input_embeddings` now requires `is_multimodal` arg, "
                "please update your model runner according to "
                "https://github.com/vllm-project/vllm/pull/16229."
            )
        if self.use_deepstack:
            multimodal_embeddings = self._remove_deepstack_embeds(
                multimodal_embeddings=multimodal_embeddings,
            )
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mrope_input_positions(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        second_per_grid_ts: Optional[list[float]] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""
        video_grid_thw = [[1, h, w] for t, h, w in video_grid_thw for _ in range(t)]
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))
