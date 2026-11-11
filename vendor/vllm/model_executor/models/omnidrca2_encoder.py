# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E501
# Adapted from https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/funaudiochat/modeling_funaudiochat.py

# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    auto_docstring,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)

from vllm.transformers_utils.configs import OmniDRCA2AudioEncoderConfig, OmniDRCA2Config

logger = logging.getLogger(__name__)

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func,
    )
else:
    flash_attn_varlen_func = None


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@auto_docstring
class OmniDRCA2PreTrainedModel(PreTrainedModel):
    config_class = OmniDRCA2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OmniDRCA2DecoderLayer", "OmniDRCA2VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = False
    _supports_static_cache = False

    def _init_weights(self, module):
        # important: this ported version of Qwen2.5OmniThinker isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02
        )
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)


class OmniDRCA2AudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OmniDRCA2AudioEncoderConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        seq_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
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
        attn_output = (
            torch.matmul(attn_weights, value_states)
            .transpose(0, 1)
            .reshape(seq_length, self.embed_dim)
        )
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = self.out_proj(attn_output)
        return attn_output


class OmniDRCA2AudioFlashAttention2(OmniDRCA2AudioAttention):
    """
    Qwen2.5OmniThinker flash attention module. This module inherits from `OmniDRCA2AudioAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        seq_length, all_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states)
        key_states = key_states.reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states)
        value_states = value_states.reshape(seq_length, self.num_heads, -1)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
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
        attn_output = self.out_proj(attn_output)
        return attn_output


class OmniDRCA2AudioSdpaAttention(OmniDRCA2AudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        seq_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        attention_mask = torch.zeros(
            [1, seq_length, key_states.shape[0]],
            device=query_states.device,
            dtype=torch.bool,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = True
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(0, 1)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


OMNIDRCA2_AUDIO_ATTENTION_CLASSES = {
    "eager": OmniDRCA2AudioAttention,
    "flash_attention_2": OmniDRCA2AudioFlashAttention2,
    "sdpa": OmniDRCA2AudioSdpaAttention,
}


class OmniDRCA2AudioEncoderLayer(nn.Module):
    def __init__(self, config: OmniDRCA2AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = OMNIDRCA2_AUDIO_ATTENTION_CLASSES[config._attn_implementation](
            config
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
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        outputs = (hidden_states,)
        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class OmniDRCA2AudioEncoder(OmniDRCA2PreTrainedModel):
    config_class = OmniDRCA2AudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["OmniDRCA2AudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: OmniDRCA2AudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [OmniDRCA2AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        self.positional_embedding_type = getattr(
            config, "positional_embedding_type", "sinusoids"
        )
        if self.positional_embedding_type == "learnable":  # sinusoids, learnable
            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        else:
            self.positional_embedding = SinusoidsPositionEmbedding(
                self.max_source_positions, embed_dim
            )
            self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
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

    @auto_docstring
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        speech_maxlen=None,
    ):
        r"""
        input_features (`torch.float` of shape `(features_dim, N)`): N = sum(feature_lens)
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        # 1. 识别有效和无效的样本
        original_batch_size = feature_lens.size(0)
        device = input_features.device
        valid_mask = feature_lens > 0
        valid_indices = torch.where(valid_mask)[0]
        # 如果所有输入的长度都为0，直接返回一个全零的张量，保持正确的形状
        if valid_indices.numel() == 0:
            output_dim = self.proj.out_features
            final_shape = (original_batch_size, speech_maxlen, output_dim)
            return BaseModelOutput(
                last_hidden_state=torch.zeros(
                    final_shape, device=device, dtype=self.proj.weight.dtype
                )
            )
        # 2. 仅对有效样本进行数据准备
        # 从扁平化的 input_features 中分离出有效样本的特征
        input_features_list = input_features.split(feature_lens.tolist(), dim=1)
        valid_input_features_list = [input_features_list[i] for i in valid_indices]
        valid_input_features = torch.cat(valid_input_features_list, dim=1)
        # 筛选出有效样本的长度
        valid_feature_lens = feature_lens[valid_mask]
        valid_aftercnn_lens = aftercnn_lens[valid_mask]
        # 3. 对有效样本执行原有的处理逻辑
        # --- START: Processing logic for valid samples only ---
        chunk_num = torch.ceil(valid_feature_lens / (self.n_window * 2)).long()
        # 使用更鲁棒的方式计算 chunk_lengths
        chunk_lengths_list = []
        full_chunk_len = self.n_window * 2
        for i, length in enumerate(valid_feature_lens):
            num_chunks_for_sample = chunk_num[i].item()
            if num_chunks_for_sample == 0:
                continue
            chunk_lengths_list.extend([full_chunk_len] * (num_chunks_for_sample - 1))
            last_chunk_len = length % full_chunk_len
            if last_chunk_len == 0:
                last_chunk_len = full_chunk_len
            chunk_lengths_list.append(last_chunk_len)
        chunk_lengths = torch.tensor(
            chunk_lengths_list, dtype=torch.long, device=device
        )
        chunk_list = valid_input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = (
            self.padded_and_mask_function(
                chunk_list, chunk_lengths, padding_value=0, padding_side="right"
            )
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)
        if self.positional_embedding_type == "learnable":
            embed_pos = self.embed_positions.weight
            padded_embed = padded_embed + embed_pos[
                : padded_embed.shape[1], :
            ].unsqueeze(0).to(padded_embed.dtype)
        else:
            padded_embed = (
                padded_embed
                + self.positional_embedding.positional_embedding[
                    : padded_embed.shape[1], :
                ]
                .unsqueeze(0)
                .to(padded_embed.dtype)
            )
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        for idx, encoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    cu_seqlens,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    cu_seqlens,
                )
            hidden_states = layer_outputs[0]
        hidden_states_list = hidden_states.split(valid_aftercnn_lens.tolist(), dim=0)
        padded_states = nn.utils.rnn.pad_sequence(
            hidden_states_list, batch_first=True, padding_value=0
        )
        # padded_states 的 batch_size 现在是有效样本的数量 (num_valid_samples)
        num_valid_samples, max_seq_len, hidden_dim = padded_states.shape
        padded_states = padded_states.transpose(1, 2)
        pooled_states = self.avg_pooler(padded_states)
        pooled_states = pooled_states.transpose(1, 2)
        processed_states = self.ln_post(pooled_states)
        valid_processed_states = self.proj(
            processed_states
        )  # 形状为 (num_valid_samples, seq_len, hidden_dim)
        # --- END: Processing logic for valid samples only ---
        # 4. 重建完整的批次输出
        # 创建一个全零的输出张量，其 batch_size 是原始的 batch_size
        output_hidden_states = torch.zeros(
            (original_batch_size, speech_maxlen, valid_processed_states.shape[-1]),
            dtype=valid_processed_states.dtype,
            device=device,
        )
        # 将有效样本的处理结果填充到输出张量的相应位置
        # 使用 valid_indices 进行高级索引，这会自动将结果放置到正确的位置
        max_copy_len = min(valid_processed_states.shape[1], speech_maxlen)
        output_hidden_states[valid_indices, :max_copy_len] = valid_processed_states[
            :, :max_copy_len
        ]
        return BaseModelOutput(last_hidden_state=output_hidden_states)

    def padded_and_mask_function(
        self, tensor_list, tensor_len, padding_value=0, padding_side="right"
    ):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )
        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]
        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

# # encoder: SenseVoiceEncoder
# encoder_conf = {
#     "input_size": 128,
#     "attention_heads": 20,
#     "linear_units": 1280,
#     "num_blocks": 32,
#     "dropout_rate": 0.1,
#     "positional_dropout_rate": 0.1,
#     "attention_dropout_rate": 0.1,
#     "kernel_size": 31,
#     "sanm_shfit": 0,
#     "att_type": "self_att_fsmn_sdpa",
#     "downsample_rate": 4,
#     "use_padmask": True,
#     "max_position_embeddings": 2048,
#     "rope_theta": 10000,
#     "freeze": False,
#     "freeze_exclude_key": "none",
# }


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, is_pad_mask=is_pad_mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            if not is_pad_mask:
                qk = qk + mask[:n_ctx, :n_ctx]
            else:
                mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
                min_value = -float(
                    "inf"
                )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
                qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None and is_pad_mask:
            w = w.masked_fill(mask, 0.0)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionSdpa(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, is_pad_mask=is_pad_mask, is_causal=False)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        is_causal = kwargs.get("is_causal", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.5
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if mask is not None:
            if not is_pad_mask:
                mask = None
                is_causal = True
            else:
                mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        qk = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
            min_value = -float(
                "inf"
            )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
            qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionSdpaRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        is_causal = kwargs.get("is_causal", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.5
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if mask is not None:
            mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


class MultiHeadAttentionFSMNRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

        self.fsmn_block = nn.Conv1d(
            linear_units,
            linear_units,
            kwargs.get("kernel_size", 15),
            stride=1,
            padding=0,
            groups=linear_units,
            bias=False,
        )
        # padding
        left_padding = (kwargs.get("kernel_size", 15) - 1) // 2
        left_padding = left_padding + kwargs.get("sanm_shfit", 0)
        right_padding = kwargs.get("kernel_size", 15) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))

    def fsmn(self, inputs, mask):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2) + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        memory = self.fsmn(v, mask=mask)
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        qk = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
            min_value = -float(
                "inf"
            )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
            qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)

        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionFSMNSdpaRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

        self.fsmn_block = nn.Conv1d(
            linear_units,
            linear_units,
            kwargs.get("kernel_size", 15),
            stride=1,
            padding=0,
            groups=linear_units,
            bias=False,
        )
        # padding
        left_padding = (kwargs.get("kernel_size", 15) - 1) // 2
        left_padding = left_padding + kwargs.get("sanm_shfit", 0)
        right_padding = kwargs.get("kernel_size", 15) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))

    def fsmn(self, inputs, mask):
        b, t, d = inputs.size()  # b, t, d
        if mask is not None:
            # A streaming attention mask is [B, T_query, T_key].  FSMN only
            # needs the per-frame validity mask, which is its diagonal.
            frame_mask = (
                torch.diagonal(mask, dim1=-2, dim2=-1)
                if mask.dim() == 3
                else torch.reshape(mask, (b, -1))
            )
            frame_mask = frame_mask.unsqueeze(-1)  # b, t, 1
            inputs = inputs * frame_mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2) + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * frame_mask
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        memory = self.fsmn(v, mask=mask)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_causal = kwargs.get("is_causal", False)
        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        query_mask = None
        if mask is not None:
            if mask.dim() == 3:
                query_mask = torch.diagonal(mask, dim1=-2, dim2=-1)
            else:
                query_mask = mask.reshape(b, -1)
            mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, *, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if query_mask is not None:
            attn_output = attn_output.masked_fill(
                query_mask[:, None, :, None].logical_not(), 0.0
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


att_type_dict = {
    "default": MultiHeadAttention,
    "sdpa": MultiHeadAttentionSdpa,
    "self_att": MultiHeadAttentionRoPE,
    "self_att_sdpa": MultiHeadAttentionSdpaRoPE,
    "self_att_fsmn": MultiHeadAttentionFSMNRoPE,
    "self_att_fsmn_sdpa": MultiHeadAttentionFSMNSdpaRoPE,
}


class EncoderLayerSANMLarge(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        att_type = kwargs.get("att_type", "self_att_fsmn_sdpa")
        self.attn = att_type_dict[att_type](linear_units, attention_heads, **kwargs)
        self.attn_ln = LayerNorm(linear_units)

        n_mlp = linear_units * 4
        self.mlp = nn.Sequential(
            Linear(linear_units, n_mlp), nn.GELU(), Linear(n_mlp, linear_units)
        )
        self.mlp_ln = LayerNorm(linear_units)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        x = x + self.attn(self.attn_ln(x), mask=mask, **kwargs)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
        
class SenseVoiceEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        linear_units: int,
        attention_heads: int,
        num_blocks: int,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = Conv1d(input_size, linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(linear_units, linear_units, kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList(
            [
                EncoderLayerSANMLarge(linear_units, attention_heads, **kwargs)
                for _ in range(num_blocks)
            ]
        )
        self.ln_post = LayerNorm(linear_units)
        self.use_padmask = kwargs.get("use_padmask", True)
        self.downsample_rate = kwargs.get("downsample_rate", 4)
        self.streaming = os.environ.get("AUDIO_ENCODER_STREAMING", "false").lower() in (
            "1", "true", "yes", "on"
        )
        self.streaming_chunk_ms = float(
            os.environ.get("AUDIO_ENCODER_STREAMING_CHUNK_MS", "640")
        )
        if self.streaming_chunk_ms <= 0:
            raise ValueError(
                "AUDIO_ENCODER_STREAMING_CHUNK_MS must be positive, "
                f"got {self.streaming_chunk_ms}"
            )
        # Fbank frame shift is 10 ms and the two convolutions downsample time.
        conv_downsample = self.conv1.stride[0] * self.conv2.stride[0]
        self.streaming_chunk_frames = max(
            1, round(self.streaming_chunk_ms / (10.0 * conv_downsample))
        )
        if self.streaming:
            logger.info(
                "Using streaming audio encoder attention: chunk=%s ms "
                "(%s encoder frames)",
                self.streaming_chunk_ms,
                self.streaming_chunk_frames,
            )
        # debug

    def _streaming_attention_mask(self, lengths, sequence_length, device):
        """Return [B, T, T] bool mask; True entries may be attended."""
        positions = torch.arange(sequence_length, device=device)
        chunk_ids = torch.div(
            positions, self.streaming_chunk_frames, rounding_mode="floor"
        )
        block_causal = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)
        valid_keys = positions.unsqueeze(0) < lengths.to(device=device).unsqueeze(1)
        return block_causal.unsqueeze(0) & valid_keys[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        **kwargs,
    ):
        use_padmask = self.use_padmask
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        if self.streaming:
            if olens is None:
                olens = torch.full(
                    (x.shape[0],), max_pos, dtype=torch.long, device=x.device
                )
            padding_mask = self._streaming_attention_mask(
                olens, max_pos, x.device
            )
        elif use_padmask and olens is not None:
            padding_mask = (~make_pad_mask(olens)[:, None, :]).to(torch.bool).to(x.device)
        else:
            padding_mask = None

        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        for layer, block in enumerate(self.blocks):
            x = block(x, mask=padding_mask, position_ids=position_ids)

        x = self.ln_post(x)

        if ilens is None:
            return x
        else:
            return x, olens



class OmniDRCA2DiscreteEncoder(OmniDRCA2PreTrainedModel):
    config_class = OmniDRCA2AudioEncoderConfig

    def __init__(self, config: OmniDRCA2AudioEncoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.group_size = config.group_size
        self.hidden_size = config.hidden_size
        self.group_type = getattr(
            config, "group_type", "concat"
        )  # "concat", "mean", "concat_mean"
        self.output_matching_bias = getattr(config, "output_matching_bias", False)
        self.continuous_features_mode = getattr(
            config, "continuous_features_mode", "add"
        )  # "add", "replace"
        self.embed_tokens = nn.Embedding(
            config.codebook_size, self.hidden_size, self.padding_idx
        )
        if self.group_type == "concat":
            self.output_matching = nn.Linear(
                self.hidden_size * self.group_size,
                self.hidden_size,
                bias=self.output_matching_bias,
            )
        elif self.group_type == "mean":
            self.output_matching = nn.Linear(
                self.hidden_size, self.hidden_size, bias=self.output_matching_bias
            )
        elif self.group_type == "concat_mean":
            self.output_matching = nn.Linear(
                self.hidden_size * self.group_size,
                self.hidden_size,
                bias=self.output_matching_bias,
            )
            self.continual_output_matching = nn.Linear(
                self.hidden_size, self.hidden_size, bias=self.output_matching_bias
            )
        elif self.group_type == "dual_mean":
            self.output_matching = nn.Linear(
                self.hidden_size, self.hidden_size, bias=self.output_matching_bias
            )
            self.continual_output_matching = nn.Linear(
                self.hidden_size, self.hidden_size, bias=self.output_matching_bias
            )
        else:
            raise ValueError(f"Group Type {self.group_type} not supported")
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        audio_ids,
        continuous_audio_features=None,
        continuous_audio_output_lengths=None,
        feature_exist_mask=None,
        return_dict=None,
    ):
        inputs_embeds = self.embed_tokens(audio_ids)
        # Replace continuous_audio_features
        if continuous_audio_features is not None and self.group_type in [
            "concat",
            "mean",
        ]:
            if self.continuous_features_mode == "add":
                inputs_embeds[feature_exist_mask] += continuous_audio_features
            else:
                inputs_embeds[feature_exist_mask] = continuous_audio_features
        inputs_embeds = inputs_embeds.reshape(
            inputs_embeds.shape[0], -1, self.group_size * self.hidden_size
        )
        if self.group_type == "concat":
            hidden_states = self.output_matching(inputs_embeds)
        elif self.group_type == "mean":
            inputs_embeds_mean_value = inputs_embeds.reshape(
                inputs_embeds.shape[0], -1, self.group_size, self.hidden_size
            ).mean(dim=2)
            hidden_states = self.output_matching(inputs_embeds_mean_value)
        elif self.group_type == "concat_mean":
            hidden_states = self.output_matching(inputs_embeds)
            if continuous_audio_features is not None:
                continuous_audio_features = continuous_audio_features.reshape(
                    continuous_audio_features.shape[0],
                    -1,
                    self.group_size,
                    self.hidden_size,
                )
                continuous_audio_features = continuous_audio_features.mean(dim=2)
                continuous_audio_hidden_states = self.continual_output_matching(
                    continuous_audio_features
                )
                if self.continuous_features_mode == "add":
                    hidden_states[feature_exist_mask] += continuous_audio_hidden_states
                else:
                    hidden_states[feature_exist_mask] = continuous_audio_hidden_states
        elif self.group_type == "dual_mean":
            inputs_embeds_mean_value = inputs_embeds.reshape(
                inputs_embeds.shape[0], -1, self.group_size, self.hidden_size
            ).mean(dim=2)
            hidden_states = self.output_matching(inputs_embeds_mean_value)
            if continuous_audio_features is not None:
                continuous_audio_features = continuous_audio_features.reshape(
                    continuous_audio_features.shape[0],
                    -1,
                    self.group_size,
                    self.hidden_size,
                )
                continuous_audio_features = continuous_audio_features.mean(dim=2)
                continuous_audio_hidden_states = self.continual_output_matching(
                    continuous_audio_features
                )
                if self.continuous_features_mode == "add":
                    hidden_states[feature_exist_mask] += continuous_audio_hidden_states
                else:
                    hidden_states[feature_exist_mask] = continuous_audio_hidden_states
        encoder_states = (inputs_embeds, hidden_states)
        all_attentions = None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = input_lengths
        output_lengths = (input_lengths + self.group_size - 1) // self.group_size
        return input_lengths, output_lengths
