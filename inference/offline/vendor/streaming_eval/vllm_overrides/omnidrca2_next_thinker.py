# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Union, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    EmbeddingItems,
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
from .interfaces import IsHybrid, MultiModalEmbeddings, SupportsMultiModal, SupportsPP, SupportsMRoPE
from .omnidrca2_encoder import SenseVoiceEncoder
from .omnidrca2_next_processor import OmniDRCA2NextProcessor
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
)

from .qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5_MoeMixtureOfExperts,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(self.dim, -1)).transpose(self.dim, -1)


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask


class sequence_mask(nn.Module):
    def __init__(self, max_seq_len=512, flip=True):
        super().__init__()

    def forward(self, lengths, max_seq_len=None, dtype=torch.float32, device=None):
        if max_seq_len is None:
            max_seq_len = lengths.max()
        row_vector = torch.arange(0, max_seq_len, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        return mask.type(dtype).to(device) if device is not None else mask.type(dtype)

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class Transformer(nn.Module):
    def __init__(
        self, downsample_rate=2, encoder_dim=1280, llm_dim=4096, ffn_dim: int = 2048, **kwargs
    ):
        super().__init__()
        self.k = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, self.llm_dim)
        self.streaming_chunk_size = 0

        self.make_pad_mask = sequence_mask(flip=False)

        self.blocks = None
        if kwargs.get("n_layer", 2) > 0:
            self.blocks = nn.ModuleList(
                [
                    EncoderLayer(
                        llm_dim,
                        MultiHeadedAttention(
                            kwargs.get("attention_heads", 8),
                            llm_dim,
                            kwargs.get("attention_dropout_rate", 0.0),
                        ),
                        PositionwiseFeedForward(
                            llm_dim,
                            llm_dim // 4,
                            kwargs.get("dropout_rate", 0.0),
                        ),
                        kwargs.get("dropout_rate", 0.0),
                    )
                    for i in range(kwargs.get("n_layer", 2))
                ]
            )

    def forward(self, x, ilens=None):

        batch_size, seq_len, dim = x.size()
        # num_frames_to_discard = seq_len % self.k
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        # if num_frames_to_discard > 0:
        #     x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, chunk_num, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        olens = None
        olens = (ilens - 1) // self.k + 1
        # masks = (~make_pad_mask(olens)[:, None, :]).to(x.device)
        valid = self.make_pad_mask(olens).to(device=x.device, dtype=torch.bool)
        if self.streaming_chunk_size > 0:
            positions = torch.arange(x.shape[1], device=x.device)
            chunk_ids = torch.div(
                positions, self.streaming_chunk_size, rounding_mode="floor"
            )
            block_causal = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)
            masks = (
                block_causal.unsqueeze(0)
                & valid.unsqueeze(1)
                & valid.unsqueeze(2)
            )
        else:
            masks = valid[:, None, :]

        if self.blocks is not None:
            for layer, block in enumerate(self.blocks):
                x, masks = block(x, masks)
        return x, olens

# encoder: SenseVoiceEncoder
encoder_conf = {
    "input_size": 128,
    "attention_heads": 20,
    "linear_units": 1280,
    "num_blocks": 32,
    "dropout_rate": 0.1,
    "positional_dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "kernel_size": 31,
    "sanm_shfit": 0,
    "att_type": "self_att_fsmn_sdpa",
    "downsample_rate": 4,
    "use_padmask": True,
    "max_position_embeddings": 2048,
    "rope_theta": 10000,
    "freeze": False,
    "freeze_exclude_key": "none",
}

# audio_adaptor: Transformer
audio_adaptor_conf = {
    "freeze": False,
    "downsample_rate": 2,
    "ffn_dim": 2048,
    "llm_dim": 2048, # 35BA3B 2048, 397B 4096
    "encoder_dim": 1280,
    "n_layer": 2,
}

def load_pretrained_module(
    path: str,
    module: torch.nn.Module,
    ignore_init_mismatch: bool = False,
    map_location: str = "cpu",
    scope_map=None,
    excludes=None,
    strict: bool = True,
    remove_prefix=None,
    check_no_extra_keys: bool = True,
):
    """
    从 CKPT 路径加载 state_dict，按 key 映射（scope_map/excludes）后加载到给定的 module
    （如随机初始化后的 audio_encoder 或 audio_adaptor），实现预训练参数赋值。

    参考 funasr/train_utils/load_pretrained_model.py 的逻辑：
    - 支持 ckpt 内层 key：state_dict / model_state_dict / model
    - scope_map: 成对 [src_prefix, dst_prefix]，用于 key 映射（如 "module." -> ""）
    - excludes: 不加载的 key 前缀列表
    - ignore_init_mismatch: 若 True，shape 不一致时跳过该 key 不报错
    - strict: 若 False，load_state_dict 允许 module 中多/少 key
    - check_no_extra_keys: 若 True，CKPT 中存在 module 不需要的 key 时会报错，保证结构完全一致
    """
    ckpt = torch.load(path, map_location=map_location)
    src_state = ckpt.get("state_dict", ckpt)
    src_state = src_state.get("model_state_dict", src_state)
    src_state = src_state.get("model", src_state)
    if not isinstance(src_state, dict):
        raise ValueError(f"CKPT 中未找到 state_dict/model_state_dict/model: {path}")

    if remove_prefix is not None:
        src_state = {k.replace(remove_prefix, ""): v for k, v in src_state.items()}

    if scope_map is None:
        scope_map = ["module.", "None"]
    if isinstance(scope_map, str):
        scope_map = scope_map.split(",")
    if len(scope_map) % 2 != 0:
        scope_map = list(scope_map) + ["None"]

    if excludes is not None:
        excludes = excludes.split(",") if isinstance(excludes, str) else list(excludes)

    dst_state = module.state_dict()
    loaded = 0
    skipped_mismatch = 0
    used_src_keys = set()  # 记录 CKPT 中实际用到的 key，用于 check_no_extra_keys
    for k in list(dst_state.keys()):
        if excludes:
            if any(k.startswith(ex) for ex in excludes):
                continue
        k_src = k
        for i in range(0, len(scope_map), 2):
            src_prefix = scope_map[i] if scope_map[i].lower() != "none" else ""
            dst_prefix = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""
            if dst_prefix == "" and (src_prefix + k) in src_state:
                k_src = src_prefix + k
                break
            if dst_prefix and k.startswith(dst_prefix):
                cand = k.replace(dst_prefix, src_prefix, 1)
                if cand in src_state:
                    k_src = cand
                    break
        if k_src not in src_state:
            raise RuntimeError(f"key not found: {k_src}")
            # continue
        if dst_state[k].shape != src_state[k_src].shape:
            if ignore_init_mismatch:
                skipped_mismatch += 1
                continue
            raise RuntimeError(
                f"shape mismatch: dst {k} {dst_state[k].shape} vs src {k_src} {src_state[k_src].shape}"
            )
        dst_state[k] = src_state[k_src].clone()
        used_src_keys.add(k_src)
        loaded += 1
    if check_no_extra_keys:
        extra = set(src_state.keys()) - used_src_keys
        if extra:
            raise RuntimeError(
                f"CKPT has extra keys not in module (structure mismatch): {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}"
            )
    module.load_state_dict(dst_state, strict=strict)
    return loaded, skipped_mismatch


# # === Audio Inputs === #
class OmniDRCA2NextFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
        - nsi: Number of speech indexs
    """

    type: Literal["audio_features"]
    input_features: torch.Tensor

    audio_feature_lengths: torch.Tensor

    # feature_attention_mask: torch.Tensor

    # speech_ids: torch.Tensor

    # speech_attention_mask: torch.Tensor

    # feature_exist_mask: torch.Tensor


class OmniDRCA2NextEmbeddingInputs(TensorSchema):
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

    audio_output_lengths: Optional[torch.Tensor] = None


OmniDRCA2NextInputs = Union[OmniDRCA2NextFeatureInputs, OmniDRCA2NextEmbeddingInputs]

# === Audio Encoder === #


# From OmniDRCA2Encoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    feat_lengths = (input_lengths + 4) // 5
    return feat_lengths


class OmniDRCA2NextProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(OmniDRCA2Config)

    def get_hf_processor(self, **kwargs: object) -> OmniDRCA2NextProcessor:
        return self.ctx.get_hf_processor(OmniDRCA2NextProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class OmniDRCA2NextDummyInputsBuilder(BaseDummyInputsBuilder[OmniDRCA2NextProcessingInfo]):
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
        # feature_extractor = self.info.get_feature_extractor()

        # sampling_rate = feature_extractor.sampling_rate
        # audio_len = feature_extractor.chunk_length * sampling_rate
        sampling_rate = 16000
        audio_len = 10 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


def _omniDrca2_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
        audio_output_lengths=MultiModalFieldConfig.batched("audio"),
    )


class OmniDRCA2NextMultiModalDataParser(MultiModalDataParser):
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


class OmniDRCA2NextMultiModalProcessor(BaseMultiModalProcessor[OmniDRCA2NextProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return OmniDRCA2NextMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _cached_apply_hf_processor(
        self,
        prompt,
        mm_data_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
        *,
        mm_uuids=None,
    ):
        # Bypass caching to ensure _call_hf_processor handles audio
        # placeholder expansion on the real prompt (not a dummy text).
        return self._apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Raw waveform inputs are expanded by the HF processor below.  External
        # speech embeddings bypass that processor, so vLLM itself must expand
        # the one <|audio_pad|> placeholder to one token per embedding frame.
        return not any(
            isinstance(items, (EmbeddingItems, DictEmbeddingItems))
            for items in mm_items.values()
        )

    def _call_hf_processor( # 这里拿到 prompt 对其进行 tokenizer 以及占位符操作
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # print(f"gangouyu prompt: {prompt}")
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        if mm_data.get("audio_embeds") is not None:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(
                dict(
                    input_ids=[prompt_ids],
                    audio_embeds=mm_data["audio_embeds"],
                ),
                tensor_type="pt",
            )

        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs_with_sr = dict(**mm_kwargs, sampling_rate=feature_extractor.sampling_rate)
        result = super()._call_hf_processor(
            prompt=prompt, mm_data=mm_data, mm_kwargs=mm_kwargs_with_sr,
            tok_kwargs=tok_kwargs,
            skip_dtype_keys={"input_features"}, # 保证是 fp32 
        )

        # Expand the single audio placeholder token to N copies so that
        # _find_mm_placeholders can locate it.  All paths (profiling and
        # real inference) now use <|audio_pad|> as the placeholder.
        audio_output_lengths = result.get("audio_output_lengths")
        if audio_output_lengths is not None:
            processor = self.info.get_hf_processor(**mm_kwargs)
            audio_token = getattr(processor, "audio_token", "<|audio_pad|>")
            audio_token_id = processor.tokenizer.convert_tokens_to_ids(audio_token)

            # placeholder_id is what forward() uses to locate audio positions
            hf_config = self.info.get_hf_config()
            placeholder_id = getattr(hf_config, "audio_token_index", audio_token_id)

            input_ids = result["input_ids"][0].tolist()
            if isinstance(audio_output_lengths, torch.Tensor):
                audio_output_lengths = audio_output_lengths.tolist()

            new_input_ids: list[int] = []
            audio_idx = 0
            for tid in input_ids:
                if tid == audio_token_id and audio_idx < len(audio_output_lengths):
                    n = audio_output_lengths[audio_idx]
                    new_input_ids.extend([placeholder_id] * n)
                    audio_idx += 1
                else:
                    new_input_ids.append(tid)

            result["input_ids"] = torch.tensor([new_input_ids])
            # print(f"gangouyu _call_hf_processor: "
            #       f"original_ids({len(input_ids)})={input_ids}..., "
            #       f"audio_token_id={audio_token_id}, placeholder_id={placeholder_id}, "
            #       f"audio_output_lengths={audio_output_lengths}, "
            #       f"new_ids({len(new_input_ids)})={new_input_ids}...", flush=True)

        return result


    # def _postprocess_output(self, output):
    #     """Override: keep input_features as fp32, skip bf16 cast entirely."""
    #     from vllm.multimodal.processing import json_map_leaves

    #     model_dtype = self.info.ctx.model_config.dtype
    #     print(f"gangouyu postprocess_output model_dtype: {model_dtype}")
    #     def _postprocess_one(x):
    #         if isinstance(x, torch.Tensor) and x.is_floating_point():
    #             x = x.to(dtype=model_dtype)
    #         return x

    #     # Selectively cast: skip input_features
    #     for key, val in output.items():
    #         if key == "input_features":
    #             continue  # keep original fp32, no roundtrip
    #         output[key] = json_map_leaves(_postprocess_one, val)

    #     return output


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
        audio_token = getattr(processor, "audio_token", "<|audio_pad|>")

        # Use audio_token_index from model config — must match
        # _call_hf_processor expansion and forward().
        hf_config = self.info.get_hf_config()
        audio_token_id = getattr(
            hf_config, "audio_token_index",
            processor.tokenizer.convert_tokens_to_ids(audio_token),
        )

        out_mm_data = out_mm_kwargs.get_data()
        audio_output_lens = out_mm_data.get("audio_output_lengths")
        if audio_output_lens is not None:
            assert isinstance(audio_output_lens, torch.Tensor)
            audio_output_lengths = audio_output_lens.tolist()
        else:
            audio_output_lengths = []

        def get_replacement_omnidrca2(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data.get("audio_embeds")
                if audio_embeds is not None:
                    audio_embeds_i = audio_embeds[item_idx]
                    assert len(audio_embeds_i.shape) == 2, "audio_embeds must be a 2D tensor"
                    num_features = audio_embeds_i.shape[0]
                else:
                    raise ValueError(
                        "Neither audio_output_lengths nor audio_embeds found in mm data"
                    )

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
    OmniDRCA2NextMultiModalProcessor,
    info=OmniDRCA2NextProcessingInfo,
    dummy_inputs=OmniDRCA2NextDummyInputsBuilder,

)


class OmniDRCA2NextForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    IsHybrid,
    SupportsMRoPE,
    Qwen3_5_MoeMixtureOfExperts,
):
    
    is_hybrid: ClassVar[Literal[True]] = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

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

        object.__setattr__(self, "audio_encoder", SenseVoiceEncoder(**encoder_conf))
        self.audio_encoder.float()  # ensure fp32 BEFORE loading weights
        self.audio_encoder.eval()
        encoder_output_dim = encoder_conf["linear_units"]
        model_path = vllm_config.model_config.model
        encoder_ckpt_path = os.path.join(model_path, "audio_encoder.pt")
        #FIXME(gangouyu.yp):  现在每个卡都会加载 forward，切实现为 torch naive，后面优化
        if os.path.exists(encoder_ckpt_path):
            loaded, skipped_mismatch = load_pretrained_module(encoder_ckpt_path, self.audio_encoder, remove_prefix="audio_encoder.")
            print(f"audio_encoder 加载完成: loaded={loaded}, encoder_ckpt_path: {encoder_ckpt_path}, skipped_mismatch={skipped_mismatch}")
        # Dense 4B uses 2560, while the MoE model uses 2048.  Derive this from
        # the converted checkpoint instead of the legacy global default.
        llm_dim = config.text_config.hidden_size
        adp_cfg = {**audio_adaptor_conf, "encoder_dim": encoder_output_dim, "llm_dim": llm_dim}
  
        object.__setattr__(self, "audio_adaptor", Transformer(**adp_cfg))
        self.audio_adaptor.float()  # ensure fp32 BEFORE loading weights
        self.audio_adaptor.eval()
        adaptor_ckpt_path = os.path.join(model_path, "audio_adaptor.pt")
        if os.path.exists(adaptor_ckpt_path):
            loaded, skipped_mismatch = load_pretrained_module(adaptor_ckpt_path, self.audio_adaptor, remove_prefix="audio_adaptor.")
            print(f"   audio_adaptor 加载完成: loaded={loaded}, audio_adaptor_ckpt_path: {adaptor_ckpt_path}, skipped_mismatch={skipped_mismatch}")

        is_moe = config.text_config.model_type == "qwen3_5_moe_text"
        language_model_arch = (
            "Qwen3_5MoeForCausalLM" if is_moe else "Qwen3_5ForCausalLM"
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=[language_model_arch],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        if is_moe:
            self.set_moe_parameters()

        gate_path = os.environ.get("VLLM_STREAM_GATE_PATH")
        if gate_path:
            hidden_size = int(config.text_config.hidden_size)
            gate = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2),
            )
            saved_gate = torch.load(gate_path, map_location="cpu")
            gate_state = {
                k.removeprefix("net."): v
                for k, v in saved_gate["state_dict"].items()
            }
            gate.load_state_dict(gate_state, strict=True)
            gate.eval()
            # Keep this inference-only head outside vLLM's checkpoint weight
            # inventory; it is loaded independently from streaming_gate.pt.
            object.__setattr__(self, "stream_gate", gate)
        else:
            object.__setattr__(self, "stream_gate", None)



    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|audio_bos|><|audio_pad|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

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
    ) -> Optional[OmniDRCA2NextInputs]:
        input_features = kwargs.pop("input_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if input_features is not None:
            # input_features: list of [T_i, C] -> stack to [B, max_T, C] with padding
            # or already tensor [B, T, C]
            if isinstance(input_features, list):
                from torch.nn.utils.rnn import pad_sequence
                # 每个 element 是 [T_i, C]，需要 pad 到 max_T
                input_features = pad_sequence(
                    input_features, batch_first=True, padding_value=0.0
                )
            elif isinstance(input_features, torch.Tensor) and input_features.dim() == 3:
                # 已经是 [B, T, C] 格式，保持原样
                pass
            else:
                raise ValueError(
                    f"input_features must be list of [T, C] or tensor [B, T, C], "
                    f"got {type(input_features)} with shape {getattr(input_features, 'shape', None)}"
                )

            # audio_feature_lengths: list of scalars or tensor [B]
            if isinstance(audio_feature_lengths, list):
                audio_feature_lengths = torch.tensor(audio_feature_lengths, dtype=torch.long)
            elif isinstance(audio_feature_lengths, torch.Tensor):
                audio_feature_lengths = audio_feature_lengths.reshape(-1)

            return OmniDRCA2NextFeatureInputs(
                type="audio_features",
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
            )

        if audio_embeds is not None:
            # Streaming evaluation supplies all adaptor outputs from one
            # full-record encoder pass.  Keep the item dimension here; the
            # offline implementation flattened it and therefore could not
            # represent several variable-length interleaved audio turns.
            audio_output_lengths = kwargs.pop("audio_output_lengths", None)
            if isinstance(audio_output_lengths, list):
                audio_output_lengths = torch.as_tensor(
                    audio_output_lengths, dtype=torch.long
                ).reshape(-1)
            elif isinstance(audio_output_lengths, torch.Tensor):
                audio_output_lengths = audio_output_lengths.reshape(-1)
            return OmniDRCA2NextEmbeddingInputs(
                type="audio_embeds",
                audio_embeds=audio_embeds,
                audio_output_lengths=audio_output_lengths,
            )

        return None

    def _get_audio_features(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            acoustic_audio_emb: [B, T_out, dim]
            audio_output_lengths: [B] - actual output token count per audio
        """
        # input_features: [B, T, C] -> [B, C, T] for Conv1d in SenseVoiceEncoder
        assert input_features.dtype == torch.float32, (
            f"input_features should be fp32 but got {input_features.dtype}, "
            f"check _postprocess_output override"
        )
        batch_size = input_features.size(0)

        # 逐条处理每条音频：trim 到实际长度避免 padded tensor 与 mask shape 不匹配
        all_embs = []
        all_lens = []
        for i in range(batch_size):
            actual_len = audio_feature_lengths[i].item()
            feat_i = input_features[i:i+1, :actual_len, :]  # [1, T_i, C]
            feat_i = feat_i.permute(0, 2, 1)  # [1, C, T_i]
            len_i = audio_feature_lengths[i:i+1]  # [1]
            with torch.no_grad():
                enc_out, enc_lens = self.audio_encoder(feat_i, len_i)
                emb, out_lens = self.audio_adaptor(enc_out, enc_lens)
            all_embs.append(emb.squeeze(0))  # [T_out_i, dim]
            all_lens.append(out_lens)

        if batch_size == 1:
            # 单条直接 unsqueeze 返回，无需 pad
            return all_embs[0].unsqueeze(0), all_lens[0]

        # 多条：pad 到 max output length
        max_out_len = max(e.size(0) for e in all_embs)
        dim = all_embs[0].size(1)
        device, dtype = all_embs[0].device, all_embs[0].dtype
        padded = torch.zeros(batch_size, max_out_len, dim, device=device, dtype=dtype)
        for i, emb in enumerate(all_embs):
            padded[i, :emb.size(0)] = emb

        audio_output_lengths = torch.cat(all_lens)
        return padded, audio_output_lengths


    def _process_audio_input(
        self, audio_input: OmniDRCA2NextInputs
    ) -> tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            lengths = audio_input.get("audio_output_lengths")
            if isinstance(audio_embeds, torch.Tensor):
                if audio_embeds.ndim == 2:
                    audio_embeds = audio_embeds.unsqueeze(0)
                items = list(audio_embeds)
            else:
                items = list(audio_embeds)
            if lengths is None:
                return tuple(items)
            return tuple(
                item[: int(lengths[i].item())]
                for i, item in enumerate(items)
            )

        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        acoustic_audio_emb, audio_output_lengths = self._get_audio_features(
            input_features, audio_feature_lengths
        )

        # acoustic_audio_emb: [B, T_out_padded, dim] (padded to same length)
        # Slice each item to its actual output length, returning tuple of [T_i, dim]
        return tuple(
            acoustic_audio_emb[i, :audio_output_lengths[i].item(), :]
            for i in range(acoustic_audio_emb.size(0))
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    # def get_input_embeddings(
    #     self,
    #     input_ids: torch.Tensor,
    #     multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    # ) -> torch.Tensor:
    #     inputs_embeds = self.language_model.get_input_embeddings(input_ids)
    #     if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
    #         is_multimodal = (input_ids == self.config.audio_token_index)
    #         inputs_embeds = _merge_multimodal_embeddings(
    #             inputs_embeds,
    #             multimodal_embeddings,
    #             is_multimodal,
    #         )
    #     return inputs_embeds

    @torch.inference_mode
    @torch.compile
    def thinker_to_talker_projection(
        self, hidden_states: torch.Tensor, accept_hidden_states: torch.Tensor
    ):
        import os

        # OMNIDRCA2_PRE_MATCHING_LAZY_LOAD_PATCH
        if not hasattr(self, "pre_matching"):
            import torch
            model_dir = os.environ.get("OMNIDRCA2_MODEL_DIR")
            if model_dir is None:
                model_dir = getattr(self.config, "_name_or_path", None)
            if model_dir is None:
                model_dir = getattr(self.config, "name_or_path", None)
            if model_dir is None:
                raise AttributeError(
                    "pre_matching is missing and OMNIDRCA2_MODEL_DIR/config path is unavailable"
                )
            pre_matching_path = os.path.join(model_dir, "pre_matching.pt")
            state = torch.load(pre_matching_path, map_location=hidden_states.device)
            weight = state["weight"] if isinstance(state, dict) and "weight" in state else state
            self.pre_matching = torch.nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=False,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self.pre_matching.weight.data.copy_(
                weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
            )
        # Megatron S2ST training feeds the final Thinker hidden state directly
        # into cosyvoice_llm_input_proj.  Keep inference identical by default.
        # The legacy residual input can still be selected for old checkpoints
        # whose pre_matching layer was trained with hidden + input_embedding.
        use_legacy_residual = os.environ.get(
            "OMNIDRCA2_TALKER_ADD_INPUT_EMBEDDING", "0"
        ).lower() in ("1", "true", "yes", "on")
        projection_input = (
            hidden_states + accept_hidden_states
            if use_legacy_residual
            else hidden_states
        )
        accept_hidden_states = self.pre_matching(projection_input)
        group_size = getattr(
            self.config.audio_config,
            "group_size",
            int(__import__("os").environ.get("OMNIDRCA2_AUDIO_GROUP_SIZE", "1")),
        )
        audio_hidden_size = getattr(self.config.audio_config, "hidden_size", None)
        if audio_hidden_size is None:
            audio_hidden_size = accept_hidden_states.shape[-1] // group_size
        accept_hidden_states = accept_hidden_states.reshape(
            hidden_states.size(0),
            group_size,
            audio_hidden_size,
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
        
        # if input_ids is not None:
        #     print(f"gangouyu input_ids shape: {input_ids.shape}, input_ids: {input_ids}", flush=True)

        # if positions is not None:
        #     print(f"gangouyu positions shape: {positions.shape}, positions: {positions}", flush=True)

        # if inputs_embeds is not None:
        # #     # print(f"gangouyu inputs_embeds shape: {inputs_embeds.shape}", flush=True)
        #     if inputs_embeds.shape[0] == 87:
        # # #         # torch.save(inputs_embeds, "inputs_embeds_87_pai_vllm.pt")
        
            # print(f"gangouyu llm forward input inputs_embeds shape: {inputs_embeds.shape}, inputs_embeds: {inputs_embeds}", flush=True)
        
        
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # print(f"gangouyu llm out hidden_states shape: {hidden_states.shape}, hidden_states: {hidden_states}", flush=True)

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
        if self.stream_gate is not None:
            gate = self.stream_gate.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            logits = gate(hidden_states)
            threshold = float(os.environ.get("VLLM_STREAM_GATE_THRESHOLD", "0.5"))
            threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
            # Greedy sampling of ids 0/1 now implements p(WRITE) >= threshold.
            logits[:, 1] -= math.log(threshold / (1.0 - threshold))
            return logits
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = [
            "audio_invert_tower.",
            "continuous_audio_tower.",
            "audio_tower.",
            "pre_matching.",
            "audio_encoder.",
            "audio_adaptor.",
        ]

        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "language_model.model.",
                "model.lm_head.": "language_model.lm_head.",
                "lm_head.": "language_model.lm_head.",
            }
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)
    

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> tuple[torch.Tensor, int]:
        """Get M-RoPE input positions for text-only and audio inputs.

        For audio inputs, we use sequential positions like text.
        This is a simplified implementation for audio multimodal model.
        """
        import numpy as np

        # For audio model, use simple 1D position encoding
        # The mrope_section from config is [11, 11, 10] = 32 for head_dim
        # We replicate the same position across all 3 dimensions
        num_tokens = len(input_tokens)
        positions = np.arange(num_tokens)
        # Broadcast to shape [3, num_tokens] for T/H/W dimensions
        llm_positions = np.broadcast_to(positions, (3, num_tokens))

        # mrope_position_delta is 0 for simple sequential positions
        mrope_position_delta = 0

        return torch.from_numpy(llm_positions), mrope_position_delta
    
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

# Offline run-local extension: same trained Hadamard layer fusion as streaming.
import os as _fusion_os
if _fusion_os.environ.get("OFFLINE_LAYER_FUSION") == "1":
    from profiles.layerfusion.vllm_layer_fusion import install as _install_fusion
    _install_fusion(OmniDRCA2NextForConditionalGeneration)
