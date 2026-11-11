# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OmniDRCA2Config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .omnidrca2_encoder import OmniDRCA2AudioEncoder, OmniDRCA2DiscreteEncoder
from .omnidrca2_processor import OmniDRCA2Processor
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)


# # === Audio Inputs === #
class OmniDRCA2FeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
        - nsi: Number of speech indexs
    """

    type: Literal["audio_features"]
    input_features: torch.Tensor

    feature_attention_mask: torch.Tensor

    speech_ids: torch.Tensor

    speech_attention_mask: torch.Tensor

    feature_exist_mask: torch.Tensor


class OmniDRCA2EmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs"),
    ]


OmniDRCA2Inputs = Union[OmniDRCA2FeatureInputs, OmniDRCA2EmbeddingInputs]

# === Audio Encoder === #


# From OmniDRCA2Encoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    feat_lengths = (input_lengths + 4) // 5
    return feat_lengths


class OmniDRCA2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OmniDRCA2Config)

    def get_hf_processor(self, **kwargs: object) -> OmniDRCA2Processor:
        return self.ctx.get_hf_processor(OmniDRCA2Processor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class OmniDRCA2DummyInputsBuilder(BaseDummyInputsBuilder[OmniDRCA2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token

        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        # audio_len = feature_extractor.chunk_length * sampling_rate
        audio_len = 10 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


def _omniDrca2_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        speech_ids=MultiModalFieldConfig.batched("audio"),
        speech_attention_mask=MultiModalFieldConfig.batched("audio"),
        feature_exist_mask=MultiModalFieldConfig.batched("audio"),
    )


class OmniDRCA2MultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[AudioItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_omniDrca2_field_config,
            )

        return super()._parse_audio_data(data)


class OmniDRCA2MultiModalProcessor(BaseMultiModalProcessor[OmniDRCA2ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return OmniDRCA2MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # NOTE - we rename audios -> audio in mm data because transformers has
        # deprecated audios for the omniDrca2 processor and will remove
        # support for it in transformers 4.54.
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios
        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            ret_val = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
            return ret_val
        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _omniDrca2_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")

        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        speech_attention_mask = out_mm_data.get("speech_attention_mask")
        if speech_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(speech_attention_mask, torch.Tensor)
            audio_output_lens = _get_feat_extract_output_lengths(
                speech_attention_mask.sum(-1)
            )

            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_omnidrca2(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(
                    f"The audio (len={audio_len}) is too short "
                    "to be represented inside the model"
                )

            audio_tokens = [audio_token_id] * num_features
            return audio_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_omnidrca2,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    OmniDRCA2MultiModalProcessor,
    info=OmniDRCA2ProcessingInfo,
    dummy_inputs=OmniDRCA2DummyInputsBuilder,
)
class OmniDRCA2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            config.audio_config._attn_implementation_autoset = True
            config.audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        self.audio_tower = OmniDRCA2DiscreteEncoder(config.audio_config)
        self.continuous_audio_tower = OmniDRCA2AudioEncoder(config.audio_config)
        self.audio_pad_token_id = (
            config.audio_config.pad_token_id
            if config.audio_config.pad_token_id is not None
            else -1
        )

        self.pre_matching = nn.Linear(
            config.audio_config.hidden_size,
            config.audio_config.hidden_size * config.audio_config.group_size,
            bias=True,
        )

        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3MoeForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return mm_input.reshape(-1, *mm_input.shape[2:])
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[OmniDRCA2Inputs]:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        speech_ids = kwargs.pop("speech_ids", None)
        speech_attention_mask = kwargs.pop("speech_attention_mask", None)
        feature_exist_mask = kwargs.pop("feature_exist_mask", None)
        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of audio embeds. Got type: {type(audio_embeds)}"
                )
            audio_embeds = self._validate_and_reshape_mm_tensor(
                audio_embeds, "audio_embeds"
            )
            return OmniDRCA2EmbeddingInputs(
                type="audio_embeds", audio_embeds=audio_embeds
            )

        if input_features is not None:
            # input_features = self._validate_and_reshape_mm_tensor(
            #     input_features, 'input_features')
            # feature_attention_mask = self._validate_and_reshape_mm_tensor(
            #     feature_attention_mask, 'feature_attention_mask')
            return OmniDRCA2FeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                speech_ids=speech_ids,
                speech_attention_mask=speech_attention_mask,
                feature_exist_mask=feature_exist_mask,
            )

        raise AssertionError("This line should be unreachable.")

    def _get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        speech_maxlen: Optional[int] = None,
    ):
        """
        Encodes audios into continuous embeddings that
        can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices.
                Mask values selected in `[0, 1]`:
            audio_feature_lengths
                (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            # audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=-1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = (
            self.continuous_audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths
                if audio_feature_lengths is not None
                else feature_attention_mask.sum(-1)
            )
        )

        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.continuous_audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
            speech_maxlen=speech_maxlen,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features, audio_output_lengths

    def _process_batch_audio(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor,
        speech_ids,
        speech_attention_mask,
        feature_exist_mask,
    ):
        # 将长度扩展到5的倍数
        speech_padding_target_length = (
            (speech_ids.shape[-1] + self.audio_tower.group_size - 1)
            // self.audio_tower.group_size
        ) * self.audio_tower.group_size
        if speech_padding_target_length > speech_ids.shape[-1]:
            padding_length = speech_padding_target_length - speech_ids.shape[-1]
            speech_ids = torch.nn.functional.pad(
                speech_ids, (0, padding_length), value=self.audio_pad_token_id
            )
        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                speech_attention_mask.sum(-1)
            )
        )

        continuous_audio_features, continuous_audio_output_lengths = (
            self._get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                speech_maxlen=speech_ids.shape[-1],
            )
        )

        audio_features, *_ = self.audio_tower(
            speech_ids,
            continuous_audio_features=continuous_audio_features,
            continuous_audio_output_lengths=continuous_audio_output_lengths,
            feature_exist_mask=feature_exist_mask,
        )

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_output_lengths = audio_output_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens)
            .expand(num_audios, max_audio_tokens)
            .to(audio_output_lengths.device)
            < audio_output_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        return masked_audio_features, audio_output_lengths.flatten()

    def _process_audio_input(
        self, audio_input: OmniDRCA2Inputs
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]
        speech_ids = audio_input["speech_ids"]
        speech_attention_mask = audio_input["speech_attention_mask"]
        feature_exist_mask = audio_input["feature_exist_mask"]
        if isinstance(speech_ids, torch.Tensor):
            masked_audio_features, audio_output_lengths = self._process_batch_audio(
                input_features,
                feature_attention_mask,
                speech_ids,
                speech_attention_mask,
                feature_exist_mask,
            )
        else:
            masked_audio_features, audio_output_lengths = [], []
            for i, cur_speech_ids in enumerate(speech_ids):
                cur_speech_ids = cur_speech_ids.unsqueeze(dim=0)
                cur_input_feature = input_features[i : i + 1]
                cur_feature_attention_mask = feature_attention_mask[i : i + 1]
                cur_speech_attention_mask = speech_attention_mask[i].unsqueeze(dim=0)
                cur_feature_exist_mask = feature_exist_mask[i : i + 1]
                cur_masked_audio_features, cur_audio_output_lengths = (
                    self._process_batch_audio(
                        cur_input_feature,
                        cur_feature_attention_mask,
                        cur_speech_ids,
                        cur_speech_attention_mask,
                        cur_feature_exist_mask,
                    )
                )
                masked_audio_features.append(cur_masked_audio_features)
                audio_output_lengths.append(cur_audio_output_lengths)
            masked_audio_features = torch.cat(masked_audio_features, dim=0)
            audio_output_lengths = torch.cat(audio_output_lengths, dim=0)

        # Split to tuple of embeddings for individual audio input.
        return torch.split(masked_audio_features, audio_output_lengths.tolist())

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = _merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.audio_token_index,
            )
        return inputs_embeds

    @torch.inference_mode
    @torch.compile
    def thinker_to_talker_projection(
        self, hidden_states: torch.Tensor, accept_hidden_states: torch.Tensor
    ):
        accept_hidden_states = self.pre_matching(hidden_states + accept_hidden_states)
        accept_hidden_states = accept_hidden_states.reshape(
            hidden_states.size(0),
            self.config.audio_config.group_size,
            self.config.audio_config.hidden_size,
        )
        return hidden_states, accept_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # incase inplace modify input_embeding when inference thinker model
        if envs.VLLM_OMNI_ENABLE_TALKER:
            assert inputs_embeds is not None
            accept_hidden_states = inputs_embeds.clone()
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if envs.VLLM_OMNI_ENABLE_TALKER:
            return self.thinker_to_talker_projection(
                hidden_states, accept_hidden_states
            )
        else:
            return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["audio_invert_tower."]
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "audio_invert_tower.pre_matching.": "pre_matching.",
            }
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)
