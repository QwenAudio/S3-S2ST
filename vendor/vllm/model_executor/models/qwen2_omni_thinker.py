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
import math
import operator
import time
from functools import cached_property, partial
from itertools import accumulate
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import EncoderDecoderCache, StaticCache
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.envs import VLLM_OMNI_THINKER_AVENCODER_CACHE
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioInputs,
    Qwen2AudioProcessingInfo,
    _get_feat_extract_output_lengths,
)

# yapf: disable
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionTransformer,
    Qwen2VLImageEmbeddingInputs,
    Qwen2VLImageInputs,
    Qwen2VLImagePixelInputs,
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    Qwen2VLVideoEmbeddingInputs,
    Qwen2VLVideoInputs,
    Qwen2VLVideoPixelInputs,
)

# yapf: enable
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
    NestedTensors,
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
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen2_omni import (
    Qwen2OmniAudioEncoderConfig,
    Qwen2OmniNaViTAudioEncoderConfig,
    Qwen2OmniTalkerConfig,
    Qwen2OmniThinkerConfig,
)
from vllm.transformers_utils.processor import cached_feature_extractor_from_config
from vllm.transformers_utils.processors.qwen2_omni import (
    Qwen2OmniNaViTThinkerProcessor,
    Qwen2OmniProcessor,
)
from vllm.utils.cache import LRUCache

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

try:
    from flash_attn import flash_attn_varlen_func
except (ImportError, ModuleNotFoundError):
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError):
        from vllm_flash_attn import flash_attn_varlen_func

logger = init_logger(__name__)


# Copied from transformers.models.whisper.modeling_whisper.WhisperAttention
# with Whisper->Qwen2Omni
class Qwen2OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[Qwen2OmniThinkerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing "
                "`layer_idx` is not recommended and will lead to errors during the "
                "forward call, if caching is used. Please make sure to provide a "
                "`layer_idx` when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape
    # with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        seq_length, _ = hidden_states.size()

        # get query proj
        # query_states = self.q_proj(hidden_states)
        query_states = (hidden_states @ self.q_proj.weight.t()) + self.q_proj.bias

        query_states = query_states.reshape(seq_length, self.num_heads, -1)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use
                # all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).reshape(
                seq_length, self.num_heads, -1
            )
            value_states = self.v_proj(current_states).reshape(
                seq_length, self.num_heads, -1
            )
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast
                # auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(1, 2)
        ) / math.sqrt(self.head_dim)

        attention_mask = torch.full(
            [1, seq_length, key_states.shape[1]],
            torch.finfo(query_states.dtype).min,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(
            query_states.dtype
        )

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size "
                    f"{(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        # attn_probs = nn.functional.dropout(
        #     attn_weights, p=self.dropout, training=self.training)
        attn_output = (
            torch.matmul(attn_weights, value_states)
            .transpose(0, 1)
            .reshape(seq_length, self.embed_dim)
        )

        # Use the `embed_dim` from the config (stored in the class)
        # rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.whisper.modeling_whisper.WhisperFlashAttention2
# with Whisper->Qwen2Omni
class Qwen2OmniAudioFlashAttention2(Qwen2OmniAudioAttention):
    """
    Qwen2Omni flash attention module. This module inherits from `Qwen2OmniAttention`
    as the weights of the module stays untouched. The only required change would be on
    the forward pass where it needs to correctly call the public API of flash attention
    and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed
        # here is bottom-right alignement, that was made default for flash_attn>=2.1.
        # This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen
        # (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "The `static` cache implementation is not compatible with "
                "`attn_implementation='flash_attention_2'`. "
                "Use `attn_implementation='sdpa'` in the meantime, and open an issue at https://github.com/huggingface/transformers"
            )
        # Qwen2OmniFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError(
                "Qwen2OmniFlashAttention2 attention does not support output_attentions"
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        # bsz, tgt_len, _ = hidden_states.size()
        seq_length, all_dim = hidden_states.size()

        # query_states = self.q_proj(hidden_states).reshape(
        #     seq_length, self.num_heads, -1)
        query_states = (hidden_states @ self.q_proj.weight.t()) + (
            self.q_proj.bias if self.q_proj.bias is not None else 0
        )

        query_states = query_states.reshape(seq_length, self.num_heads, -1)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use
                # all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            # key_states = self.k_proj(current_states).reshape(
            #     seq_length, self.num_heads, -1)
            # value_states = self.v_proj(current_states).reshape(
            #     seq_length, self.num_heads, -1)
            key_states = (current_states @ self.k_proj.weight.t()) + (
                self.k_proj.bias if self.k_proj.bias is not None else 0
            )
            key_states = key_states.reshape(seq_length, self.num_heads, -1)
            value_states = (current_states @ self.v_proj.weight.t()) + (
                self.v_proj.bias if self.v_proj.bias is not None else 0
            )
            value_states = value_states.reshape(seq_length, self.num_heads, -1)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast
                # auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()  ## self.n_window

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
        )
        attn_output = attn_output.reshape(seq_length, all_dim)

        # attn_output = self.out_proj(attn_output)
        attn_output = (attn_output @ self.out_proj.weight.t()) + (
            self.out_proj.bias if self.out_proj.bias is not None else 0
        )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2OmniAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Union[Qwen2OmniAudioEncoderConfig, Qwen2OmniNaViTAudioEncoderConfig],
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Qwen2OmniAudioFlashAttention2(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        added by malinhan , 11,16
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding
                elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`):
                mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # hidden_states = nn.functional.dropout(
        #     hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = (hidden_states @ self.fc1.weight.t()) + (
            self.fc1.bias if self.fc1.bias is not None else 0
        )
        hidden_states = self.activation_fn(hidden_states)

        # hidden_states = nn.functional.dropout(
        #     hidden_states, p=self.activation_dropout, training=self.training)

        # hidden_states = self.fc2(hidden_states)
        hidden_states = (hidden_states @ self.fc2.weight.t()) + (
            self.fc2.bias if self.fc2.bias is not None else 0
        )

        # hidden_states = nn.functional.dropout(
        #     hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copy from modeling_qwen2_omni_thinker.py, with minor changes
# to the "positional_embedding" part.
class Qwen2OmniAudioEncoder(PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers.
    Each layer is a [`Qwen2OmniAudioEncoderLayer`].

    Args:
        config: Qwen2OmniAudioEncoderConfig
    """

    # Ignore copy
    config_class = Union[Qwen2OmniAudioEncoderConfig, Qwen2OmniNaViTAudioEncoderConfig]
    main_input_name = "input_audio_features"
    _no_split_modules = ["Qwen2OmniAudioEncoderLayer"]

    def __init__(
        self,
        config: Union[Qwen2OmniAudioEncoderConfig, Qwen2OmniNaViTAudioEncoderConfig],
    ):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.n_window = config.n_window

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList(
            [Qwen2OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)  ## TODO layernorm
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        with torch.no_grad():
            self.proj.bias.zero_()
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape
                `(batch_size, feature_size, sequence_length)`):

                Float values of mel features extracted from the raw speech waveform.
                Raw speech waveform can be obtained by loading a `.flac` or `.wav` audio
                file into an array of type `list[float]` or a `numpy.ndarray`, *e.g.*
                via the soundfile library (`pip install soundfile`). To prepare the
                array into `input_features`, the [`AutoFeatureExtractor`] should be used
                for extracting the mel features, padding and conversion into a tensor of
                type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

            feature_lens: [B], torch.LongTensor , mel length

            aftercnn_lens : [B], torch.LongTensor , mel length after cnn

            head_mask (`torch.Tensor` of shape
                `(encoder_layers, encoder_attention_heads)`, *optional*):

                Mask to nullify selected heads of the attention modules.
                Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of
                a plain tuple.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Ignore copy
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )

        feature_lens_accum = list(
            accumulate(feature_lens.tolist(), func=operator.add, initial=0)
        )

        all_feature_chunks = []  # list of [seq, dim]
        all_chunk_lengths = []
        for index_ in range(feature_lens.shape[0]):
            # feature_length = feature_lens[index_]
            feature_length = feature_lens_accum[index_ + 1] - feature_lens_accum[index_]
            # each_audio_split_list = \
            #     input_features[index_][ : , : feature_length].split(
            #         self.n_window*2, dim=1)
            # each_audio_split_list = input_features[index_].split(
            #     self.n_window*2, dim=1)
            each_audio_split_list = input_features[
                :, feature_lens_accum[index_] : feature_lens_accum[index_ + 1]
            ].split(self.n_window * 2, dim=1)

            for each_audio_split in each_audio_split_list:
                each_split_embed = nn.functional.gelu(self.conv1(each_audio_split))
                each_split_embed = nn.functional.gelu(
                    self.conv2(each_split_embed)
                ).transpose_(
                    0, 1
                )  # [len, dim]

                embed_pos = self.positional_embedding.weight[
                    : each_split_embed.shape[0], :
                ].to(each_split_embed.dtype)
                all_feature_chunks.append(each_split_embed + embed_pos)
                all_chunk_lengths.append(each_split_embed.shape[0])
                feature_length -= self.n_window * 2

            assert feature_length <= 0 and feature_length > -2 * self.n_window

        hidden_states = torch.cat(all_feature_chunks, dim=0)  # [seqlen , dim]
        cu_seqlens = list(accumulate(all_chunk_lengths, func=operator.add, initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens).to(
            device=hidden_states.device, dtype=torch.int32
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        tmp_hidden_states = []
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, "
                f"but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        cu_seqlens,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        cu_seqlens,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                tmp_hidden_states.append(hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        ################################################### :
        ### following megatron , malinhan
        assert hidden_states.shape[0] == sum(aftercnn_lens.tolist()), (
            hidden_states.shape,
            aftercnn_lens.tolist(),
        )
        hidden_states_list = hidden_states.split(
            aftercnn_lens.tolist(), dim=0
        )  ## list of [len, dim]
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(
                each_audio_states.transpose_(0, 1)
            ).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)

            token_audio_list.append(each_audio_states)

        token_audio = torch.cat(token_audio_list, dim=0)

        if output_hidden_states:
            encoder_states = encoder_states + (token_audio,)

        if not return_dict:
            return tuple(
                v
                for v in [token_audio, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=token_audio,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of
        the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def _qwen2_omni_thinker_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    audio_input_features = hf_inputs.get(  # noqa: F841
        "input_audio_features", torch.empty((0, 0, 0))
    )
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths", torch.empty((0, 0)))

    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    return dict(
        input_audio_features=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_feature_lengths, dim=1
        ),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
        pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes
        ),
        video_embeds=MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
        second_per_grid_ts=MultiModalFieldConfig.batched("video"),
    )


class Qwen2OmniThinkerMultiModalDataParser(Qwen2VLMultiModalDataParser):
    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"input_audio_features", "audio_feature_lengths"},
                fields_factory=_qwen2_omni_thinker_field_config,
            )

        return super()._parse_audio_data(data)


class Qwen2OmniThinkerProcessingInfo(Qwen2AudioProcessingInfo, Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config((Qwen2OmniThinkerConfig, Qwen2OmniTalkerConfig))

    def get_hf_processor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, list[float]]] = None,
        **kwargs: object,
    ) -> Union[Qwen2OmniProcessor, Qwen2OmniNaViTThinkerProcessor]:
        if fps is not None:
            kwargs["fps"] = fps
        processor = self.ctx.get_hf_processor(
            (Qwen2OmniProcessor, Qwen2OmniNaViTThinkerProcessor),
            feature_extractor=self.get_feature_extractor(sampling_rate=sampling_rate),
            image_processor=self.get_image_processor(
                min_pixels=min_pixels, max_pixels=max_pixels, size=size
            ),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|AUDIO|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|IMAGE|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|VIDEO|>"
        return processor

    def get_feature_extractor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        return cached_feature_extractor_from_config(
            self.ctx.model_config,
            **self._get_feature_extractor_kwargs(sampling_rate=sampling_rate, **kwargs),
        )

    def _get_feature_extractor_kwargs(
        self,
        *,
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        if sampling_rate is not None:
            kwargs["sampling_rate"] = sampling_rate
        return kwargs

    def get_max_audio_tokens(self) -> int:
        hf_config = self.ctx.get_hf_config(
            (Qwen2OmniThinkerConfig, Qwen2OmniTalkerConfig)
        )
        max_source_position = hf_config.audio_config.max_source_positions
        output_lengths = (max_source_position - 2) // 2 + 1
        return output_lengths

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "audio": self.get_max_audio_tokens(),
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len, mm_counts),
        }


class Qwen2OmniThinkerDummyInputsBuilder(
    BaseDummyInputsBuilder[Qwen2OmniThinkerProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()

        audio_token: str = hf_processor.audio_token
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return (
            audio_token * num_audios
            + image_token * num_images
            + video_token * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        feature_extractor = self.info.get_feature_extractor()

        target_audio_length = (
            min(
                feature_extractor.chunk_length,
                30,
            )
            * feature_extractor.sampling_rate
        )
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        mm_data = {
            "audio": self._get_dummy_audios(
                length=target_audio_length, num_audios=num_audios
            ),
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

        return mm_data


class Qwen2OmniThinkerMultiModalProcessor(
    BaseMultiModalProcessor[Qwen2OmniThinkerProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2OmniThinkerMultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        if audios:
            mm_data["audios"] = audios
            mm_kwargs = dict(
                **mm_kwargs,
            )
        else:
            # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
            pass

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        audio_config = self.info.ctx.get_hf_config().audio_config

        input_features = hf_inputs.pop("input_features", None)
        feature_attention_mask = hf_inputs.get("feature_attention_mask", None)
        if "input_audio_features" not in hf_inputs and input_features is not None:
            if (
                isinstance(audio_config, Qwen2OmniNaViTAudioEncoderConfig)
                and feature_attention_mask is not None
            ):
                input_features = input_features.permute(0, 2, 1)[
                    feature_attention_mask.bool()
                ].permute(1, 0)
            hf_inputs["input_audio_features"] = input_features
        if (
            "audio_feature_lengths" not in hf_inputs
            and feature_attention_mask is not None
        ):
            hf_inputs["audio_feature_lengths"] = feature_attention_mask.sum(-1)
        return hf_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen2_omni_thinker_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(
            processor, "audio_token", getattr(tokenizer, "audio_token", "<|AUDIO|>")
        )
        image_token = getattr(
            processor, "image_token", getattr(tokenizer, "image_token", "<|IMAGE|>")
        )
        video_token = getattr(
            processor, "video_token", getattr(tokenizer, "video_token", "<|VIDEO|>")
        )
        audio_bos_token = getattr(
            processor,
            "audio_bos_token",
            getattr(tokenizer, "audio_bos_token", "<|audio_bos|>"),
        )
        audio_eos_token = getattr(
            processor,
            "audio_eos_token",
            getattr(tokenizer, "audio_eos_token", "<|audio_eos|>"),
        )
        vision_bos_token = getattr(
            processor,
            "vision_bos_token",
            getattr(tokenizer, "vision_bos_token", "<|vision_bos|>"),
        )
        vision_eos_token = getattr(
            processor,
            "vision_eos_token",
            getattr(tokenizer, "vision_eos_token", "<|vision_eos|>"),
        )

        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]
        audio_bos_id = vocab[audio_bos_token]  # noqa: F841
        audio_eos_id = vocab[audio_eos_token]  # noqa: F841
        vision_bos_id = vocab[vision_bos_token]  # noqa: F841
        vision_eos_id = vocab[vision_eos_token]  # noqa: F841

        audio_feature_lengths = out_mm_kwargs.get("audio_feature_lengths")
        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths
            )
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_qwen2_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=partial(get_replacement_qwen2_vision, modality="video"),
            ),
        ]


class Qwen2OmniConditionalGenerationMixin:
    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[Qwen2AudioInputs]:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None
        input_audio_features = self._validate_and_reshape_mm_tensor(
            input_audio_features, "input_audio_features"
        )
        if feature_attention_mask is not None:
            feature_attention_mask = self._validate_and_reshape_mm_tensor(
                feature_attention_mask, "feature_attention_mask"
            )
        if not isinstance(input_audio_features, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of audio input features. "
                f"Got type: {type(input_audio_features)}"
            )
        return Qwen2AudioInputs(
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _parse_and_validate_image_input(
        self,
        **kwargs: dict[str, Any],
    ) -> Optional[Qwen2VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image pixel values. "
                    f"Got type: {type(pixel_values)}"
                )

            return Qwen2VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of image embeddings. "
                    f"Got type: {type(image_embeds)}"
                )
            return Qwen2VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self,
        **kwargs: dict[str, Any],
    ) -> Optional[Qwen2VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            return Qwen2VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of video embeddings. "
                    f"Got type: {type(video_embeds)}"
                )
            return Qwen2VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioInputs,
        audio_hashes: list[str] = None,
        cached_audio_features: torch.Tensor = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)
        if audio_feature_lengths.ndim == 2:
            assert audio_feature_lengths.shape[0] == 1
            audio_feature_lengths = audio_feature_lengths.squeeze(0)

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)
        )

        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        return audio_features

    def _process_image_input(self, image_input: Qwen2VLImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_input["image_grid_thw"])
        return image_embeds

    def _process_video_input(
        self,
        video_input: Qwen2VLVideoInputs,
        video_hashes: list[str] = None,
        cached_video_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        video_grid_thw = video_input["video_grid_thw"]
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds


class AVEncoderCache:

    def __init__(self):
        AV_ENCODER_CACHE_SIZE = 100

        self.audio_cache_queue_size = AV_ENCODER_CACHE_SIZE
        self.audio_cache_ptr = 0
        self.cached_audio_hash_table = [None] * self.audio_cache_queue_size
        self.cached_audio_features = [None] * self.audio_cache_queue_size

        self.video_cache_queue_size = AV_ENCODER_CACHE_SIZE
        self.video_cache_ptr = 0
        self.cached_video_hash_table = [None] * self.video_cache_queue_size
        self.cached_video_features = [None] * self.video_cache_queue_size

    def get_audio_hash_index(self, new_hash):
        for idx, hash in enumerate(self.cached_audio_hash_table):
            if hash == new_hash:
                return idx
        return -1

    def get_video_hash_index(self, new_hash):
        for idx, hash in enumerate(self.cached_video_hash_table):
            if hash == new_hash:
                return idx
        return -1

    def get_input_no_cache(self, kwargs) -> dict:

        def hash_with_stride(tensor, w_stride=20, h_stride=20):
            sampled = tensor[::w_stride, ::h_stride]  # 每隔stride个元素取一个
            return hash(str(sampled.tolist()))

        audio_chunk_with_hash = []
        video_chunk_with_hash = []

        if "audio_feature_lengths" in kwargs and "input_audio_features" in kwargs:
            audio_feature_lengths = kwargs["audio_feature_lengths"][0]
            start_idx = 0
            for i, chunk_size in enumerate(audio_feature_lengths):
                end_idx = start_idx + chunk_size
                fea = kwargs["input_audio_features"][:, :, start_idx:end_idx]
                cur_hash = hash(str(fea[:, ::8, ::10].tolist()))
                audio_chunk_with_hash.append(
                    [
                        i,
                        start_idx,
                        end_idx,
                        cur_hash,
                        self.get_audio_hash_index(cur_hash),
                    ]
                )
                start_idx = end_idx
        if "video_grid_thw" in kwargs and "pixel_values_videos" in kwargs:
            video_grid_thw = kwargs["video_grid_thw"][0]
            start_idx = 0
            for i, chunk_shape in enumerate(video_grid_thw):
                end_idx = start_idx + chunk_shape[0] * chunk_shape[1] * chunk_shape[2]
                fea = kwargs["pixel_values_videos"][:, start_idx:end_idx, :]
                cur_hash = hash(str(fea[:, ::128, ::32].tolist()))
                video_chunk_with_hash.append(
                    [
                        i,
                        start_idx,
                        end_idx,
                        cur_hash,
                        self.get_video_hash_index(cur_hash),
                    ]
                )
                start_idx = end_idx

        new_kwargs = {}

        # only use inputs not cached
        for key in kwargs:
            if key not in [
                "audio_feature_lengths",
                "input_audio_features",
                "feature_attention_mask",
                "pixel_values_videos",
                "video_grid_thw",
            ]:
                new_kwargs[key] = kwargs[key]
            elif key == "audio_feature_lengths":
                # [1, 10]
                no_cached_audio_chunk_index = [
                    x[0] for x in audio_chunk_with_hash if x[-1] == -1
                ]
                new_kwargs[key] = kwargs[key][0][no_cached_audio_chunk_index]
            elif key == "input_audio_features":
                # [1, 128, 2000]
                total_length = sum(
                    [x[2] - x[1] for x in audio_chunk_with_hash if x[-1] == -1]
                )
                new_fea = torch.empty(
                    (kwargs[key].size(0), kwargs[key].size(1), total_length),
                    dtype=kwargs[key].dtype,
                    device=kwargs[key].device,
                )
                cur_pos = 0
                for chunk in audio_chunk_with_hash:
                    if chunk[-1] == -1:
                        fea_length = chunk[2] - chunk[1]
                        new_fea[:, :, cur_pos : cur_pos + fea_length] = kwargs[key][
                            :, :, chunk[1] : chunk[2]
                        ]
                        cur_pos += fea_length
                new_kwargs[key] = new_fea
            elif key == "feature_attention_mask":
                # [1, 10, 30000]
                no_cached_audio_chunk_index = [
                    x[0] for x in audio_chunk_with_hash if x[-1] == -1
                ]
                new_kwargs[key] = kwargs[key][:, no_cached_audio_chunk_index, :]
            elif key == "pixel_values_videos":
                # [1, 57600, 1176]
                total_length = sum(
                    [x[2] - x[1] for x in video_chunk_with_hash if x[-1] == -1]
                )
                new_fea = torch.empty(
                    (kwargs[key].size(0), total_length, kwargs[key].size(2)),
                    dtype=kwargs[key].dtype,
                    device=kwargs[key].device,
                )
                cur_pos = 0
                for chunk in video_chunk_with_hash:
                    if chunk[-1] == -1:
                        fea_length = chunk[2] - chunk[1]
                        new_fea[:, cur_pos : cur_pos + fea_length, :] = kwargs[key][
                            :, chunk[1] : chunk[2], :
                        ]
                        cur_pos += fea_length
                new_kwargs[key] = new_fea
            elif key == "video_grid_thw":
                # [1, 10, 3]
                no_cached_video_chunk_index = [
                    x[0] for x in video_chunk_with_hash if x[-1] == -1
                ]
                new_kwargs[key] = kwargs[key][:, no_cached_video_chunk_index, :]
        # audio_chunk_with_hash: [chunk_idx, mel start, mel end, hash, cache_idx]
        return new_kwargs, audio_chunk_with_hash, video_chunk_with_hash

    def add_back_audio_cached_embeds(self, audio_embeds, audio_chunk_with_hash):
        if not audio_chunk_with_hash:
            # no audio embeds
            return audio_embeds
        if sum([1 for x in audio_chunk_with_hash if x[-1] != -1]) == 0:
            # no cache hit
            return audio_embeds
        new_audio_embeds_list = []
        audio_embeds_pos = 0
        for chunk in audio_chunk_with_hash:
            if chunk[-1] == -1:
                # add audio embeds
                fea_length = (chunk[2] - chunk[1]) // 4
                new_audio_embeds_list.append(
                    audio_embeds[audio_embeds_pos : audio_embeds_pos + fea_length]
                )
                audio_embeds_pos = int(audio_embeds_pos + fea_length)
            else:
                # add back cache fea
                fea_length = (chunk[2] - chunk[1]) // 4
                cached_fea = self.cached_audio_features[chunk[-1]]
                new_audio_embeds_list.append(cached_fea)
        new_audio_embeds = torch.cat(new_audio_embeds_list, dim=0)
        return new_audio_embeds

    def add_back_video_cached_embeds(self, video_embeds, video_chunk_with_hash):
        if not video_chunk_with_hash:
            # no video embeds
            return video_embeds
        if sum([1 for x in video_chunk_with_hash if x[-1] != -1]) == 0:
            # no cache hit
            return video_embeds
        new_video_embeds_list = []
        video_embeds_pos = 0
        for chunk in video_chunk_with_hash:
            if chunk[-1] == -1:
                # add video embeds
                fea_length = (chunk[2] - chunk[1]) // 4
                new_video_embeds_list.append(
                    video_embeds[video_embeds_pos : video_embeds_pos + fea_length]
                )
                video_embeds_pos = int(video_embeds_pos + fea_length)
            else:
                # add back cache fea
                fea_length = (chunk[2] - chunk[1]) // 4
                cached_fea = self.cached_video_features[chunk[-1]]
                new_video_embeds_list.append(cached_fea)
        new_video_embeds = torch.cat(new_video_embeds_list, dim=0)
        return new_video_embeds

    def update_cache(
        self, audio_embeds, video_embeds, audio_chunk_with_hash, video_chunk_with_hash
    ):

        # chunk embedding infos
        no_cached_audio_embeds_chunk = []
        no_cached_video_embeds_chunk = []
        cur_pos = 0
        for chunk in audio_chunk_with_hash:
            if chunk[-1] == -1:
                embeds_length = (chunk[2] - chunk[1]) // 4
                new_embeds_chunk = [
                    chunk[0],
                    cur_pos,
                    int(cur_pos + embeds_length),
                    chunk[3],
                ]
                no_cached_audio_embeds_chunk.append(new_embeds_chunk)
                cur_pos = int(cur_pos + embeds_length)
        idx = 0
        cur_pos = 0
        for chunk in video_chunk_with_hash:
            if chunk[-1] == -1:
                embeds_length = (chunk[2] - chunk[1]) // 4
                new_embeds_chunk = [
                    chunk[0],
                    cur_pos,
                    int(cur_pos + embeds_length),
                    chunk[3],
                ]
                no_cached_video_embeds_chunk.append(new_embeds_chunk)
                cur_pos = int(cur_pos + embeds_length)
        for chunk in no_cached_audio_embeds_chunk:
            idx, start, end, hash = chunk
            self.cached_audio_hash_table[self.audio_cache_ptr] = hash
            # [500, 3584]
            self.cached_audio_features[self.audio_cache_ptr] = audio_embeds[
                start:end, :
            ].clone()
            self.audio_cache_ptr = (
                self.audio_cache_ptr + 1
            ) % self.audio_cache_queue_size
        for chunk in no_cached_video_embeds_chunk:
            idx, start, end, hash = chunk
            self.cached_video_hash_table[self.video_cache_ptr] = hash
            # [14400, 3584]
            self.cached_video_features[self.video_cache_ptr] = video_embeds[
                start:end, :
            ].clone()
            self.video_cache_ptr = (
                self.video_cache_ptr + 1
            ) % self.video_cache_queue_size


if VLLM_OMNI_THINKER_AVENCODER_CACHE:
    av_encoder_cache = AVEncoderCache()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2OmniThinkerMultiModalProcessor,
    info=Qwen2OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2OmniThinkerDummyInputsBuilder,
)
class Qwen2OmniThinkerForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, Qwen2OmniConditionalGenerationMixin
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio, image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2OmniThinkerConfig = vllm_config.model_config.hf_config
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

        self.audio_tower = Qwen2OmniAudioEncoder(config.audio_config)
        if config.vision_config.model_type == "qwen2_omni_navit_vision_encoder":
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                # quant_config=self._maybe_ignore_quant_config(quant_config),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )
        else:
            self.visual = Qwen2VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                # quant_config=self._maybe_ignore_quant_config(quant_config),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )
        self.quant_config = quant_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        new_kwargs = None
        audio_chunk_with_hash = None
        video_chunk_with_hash = None
        audio_input = None
        image_input = None
        video_input = None
        start_time = time.time() * 1000
        if VLLM_OMNI_THINKER_AVENCODER_CACHE:
            new_kwargs, audio_chunk_with_hash, video_chunk_with_hash = (
                av_encoder_cache.get_input_no_cache(kwargs)
            )
            audio_input = self._parse_and_validate_audio_input(**new_kwargs)
            image_input = self._parse_and_validate_image_input(**new_kwargs)
            video_input = self._parse_and_validate_video_input(**new_kwargs)

            audio_hit_rate = 0
            video_hit_rate = 0
            if len(audio_chunk_with_hash) > 0:
                audio_hit_rate = sum(
                    [1 for x in audio_chunk_with_hash if x[-1] != -1]
                ) / len(audio_chunk_with_hash)
            if len(video_chunk_with_hash) > 0:
                video_hit_rate = sum(
                    [1 for x in video_chunk_with_hash if x[-1] != -1]
                ) / len(video_chunk_with_hash)
            logger.info(
                "Thinker AV Encoder Cache Hit Rate: "
                "audio: %f(all: %d), video: %f(all: %d)",
                audio_hit_rate * 100,
                len(audio_chunk_with_hash),
                video_hit_rate * 100,
                len(video_chunk_with_hash),
            )
        else:
            audio_input = self._parse_and_validate_audio_input(**kwargs)
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            logger.info("Thinker AV Encoder Cache OFF")

        if audio_input is None and image_input is None and video_input is None:
            return None

        multimodal_embeddings: List[Tuple[NestedTensors, str]] = []
        logger.debug("start get multimodal embeddings")
        if VLLM_OMNI_THINKER_AVENCODER_CACHE:
            audio_embeds = None
            image_embeds = None
            video_embeds = None
            if audio_input is not None:
                if audio_input["audio_feature_lengths"].numel() > 0:
                    audio_embeds = self._process_audio_input(audio_input)
                else:
                    audio_embeds = None
                logger.debug("got audio_embeds")
            if image_input is not None:
                image_embeds = self._process_image_input(image_input)
                logger.debug("got image_embeds")
            if video_input is not None:
                if video_input["video_grid_thw"].numel() > 0:
                    video_embeds = self._process_video_input(video_input)
                else:
                    video_embeds = None
                logger.debug("got video_embeds")

            av_encoder_cache.update_cache(
                audio_embeds=audio_embeds,
                video_embeds=video_embeds,
                audio_chunk_with_hash=audio_chunk_with_hash,
                video_chunk_with_hash=video_chunk_with_hash,
            )
            logger.debug("av encoder cache updated")

            if audio_input is not None:
                audio_embeds = av_encoder_cache.add_back_audio_cached_embeds(
                    audio_embeds=audio_embeds,
                    audio_chunk_with_hash=audio_chunk_with_hash,
                )
                logger.debug("add back cached audio embeds")
                multimodal_embeddings.append((audio_embeds, "audio"))
            if image_input is not None:
                multimodal_embeddings.append((image_embeds, "image"))
            if video_input is not None:
                video_embeds = av_encoder_cache.add_back_video_cached_embeds(
                    video_embeds=video_embeds,
                    video_chunk_with_hash=video_chunk_with_hash,
                )
                logger.debug("add back cached video embeds")
                multimodal_embeddings.append((video_embeds, "video"))
        else:
            if audio_input is not None:
                audio_embeds = self._process_audio_input(audio_input)
                multimodal_embeddings.append((audio_embeds, "audio"))
                logger.debug("got audio_embeds")
            if image_input is not None:
                image_embeds = self._process_image_input(image_input)
                multimodal_embeddings.append((image_embeds, "image"))
                logger.debug("got image_embeds")
            if video_input is not None:
                video_embeds = self._process_video_input(video_input)
                multimodal_embeddings.append((video_embeds, "video"))
                logger.debug("got video_embeds")
        logger.info(
            "Thinker get_multimodal_embeddings Done, cost: %f ms",
            time.time() * 1000 - start_time,
        )
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        for embeddings, modality in multimodal_embeddings:
            if modality == "audio":
                placeholder_token_id = self.config.audio_token_index
            if modality == "image":
                placeholder_token_id = self.config.image_token_index
            if modality == "video":
                placeholder_token_id = self.config.video_token_index
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                embeddings,
                placeholder_token_id,
                modality=modality,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            # Return the embeddings for the input text before merging the
            # multimodal data, align with "modeling_qwen2_omni_thinker.py".
            text_inputs_embeds = self.get_input_embeddings(
                input_ids,
                (
                    [
                        (torch.zeros_like(embeddings), modality)
                        for embeddings, modality in multimodal_embeddings
                    ]
                    if multimodal_embeddings is not None
                    else None
                ),
            )

            input_ids = None
        else:
            text_inputs_embeds = inputs_embeds

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return text_inputs_embeds, hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "audio_tower.positional_embedding": 
                    "audio_tower.positional_embedding.weight",
                "lm_head.": "language_model.lm_head.",
                "model.": "language_model.model.",
            }
        )

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)
