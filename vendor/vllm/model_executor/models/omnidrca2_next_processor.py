# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

# Copyright 2025 The HuggingFace Inc. team.
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
"""
Processor class for OmniDRCA2Next.

与 OmniDRCA2Processor 的区别：
  音频特征提取不使用 HuggingFace WhisperFeatureExtractor，
  而是使用自定义 WhisperFrontend + extract_fbank 生成 fbank 特征，
  并按 12.5Hz 下采样公式计算 audio_output_lengths（token 数）。

输出 BatchFeature 包含：
  input_ids             : [1, seq_len]
  input_features        : [num_audios, max_T, d]   fbank 特征（pad 对齐）
  audio_feature_lengths : [num_audios]              各条音频的 fbank 帧长
  audio_output_lengths  : [num_audios]              12.5Hz 下采样后的 token 数
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import (
    AudioKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class OmniDRCA2NextAudioKwargs(AudioKwargs, total=False):
    speech_kwargs: TextKwargs = {
        **TextKwargs.__annotations__,
    }


class OmniDRCA2NextProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: OmniDRCA2NextAudioKwargs = {
        **AudioKwargs.__annotations__,
    }
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {"speech_kwargs": {}},
    }


def _compute_audio_output_length(speech_lengths_frames: int) -> int:
    """
    计算经过音频编码器两层 stride-2 Conv + 12.5Hz 下采样后的 token 数量。

    公式：
        after_conv1 = (L - 3 + 2*1) // 2 + 1
        after_conv2 = (after_conv1 - 3 + 2*1) // 2 + 1
        output      = (after_conv2 - 1) // 2 + 1

    其中 L = speech_lengths_frames（fbank 时间帧数，对应 25Hz 前端输出）。
    """
    olens = 1 + (speech_lengths_frames - 3 + 2 * 1) // 2
    olens = 1 + (olens - 3 + 2 * 1) // 2
    return (olens - 1) // 2 + 1


class OmniDRCA2NextProcessor(ProcessorMixin):
    r"""
    Constructs a OmniDRCA2Next processor which wraps a WhisperFrontend-based
    feature extractor and a tokenizer into a single processor.

    与 OmniDRCA2Processor 不同，本处理器使用 WhisperFrontend + extract_fbank
    进行音频特征提取，输出包含 audio_feature_lengths 和 audio_output_lengths，
    与 vLLM OmniDRCA2NextMultiModalProcessor 的 _call_hf_processor 完全对齐。

    Args:
        feature_extractor: 保留占位，兼容 ProcessorMixin 基类（可传 None）。
        tokenizer: OmniDRCA2 / Qwen 系列 tokenizer。
        chat_template: Jinja2 chat template 字符串（可选）。
        audio_token: 音频占位 token，默认 ``"<|AUDIO|>"``。
        audio_bos_token: 音频起始 token，默认 ``"<|audio_bos|>"``。
        audio_eos_token: 音频结束 token，默认 ``"<|audio_eos|>"``。
        audio_pad_token: 音频 pad token，默认 ``"<|audio_pad|>"``。
        frontend_fs: WhisperFrontend 采样率，默认 16000。
        frontend_n_mels: 梅尔滤波器数量，默认 128。
        frontend_do_pad_trim: 是否对音频做 pad/trim，默认 False。
        frontend_filters_path: 梅尔滤波器文件路径（.npz），None 时由 WhisperFrontend 自动查找。
    """

    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "audio_token",
        "audio_bos_token",
        "audio_eos_token",
        "audio_pad_token",
        "audio_group_size",
        "frontend_fs",
        "frontend_n_mels",
        "frontend_do_pad_trim",
        "frontend_filters_path",
    ]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_pad_token="<|audio_pad|>",
        audio_group_size: int = 5,
        frontend_fs: int = 16000,
        frontend_n_mels: int = 128,
        frontend_do_pad_trim: bool = False,
        frontend_filters_path: Optional[str] = None,
    ):
        if chat_template is None:
            chat_template = self.default_chat_template

        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        self.audio_bos_token = (
            tokenizer.audio_bos_token
            if hasattr(tokenizer, "audio_bos_token")
            else audio_bos_token
        )
        self.audio_eos_token = (
            tokenizer.audio_eos_token
            if hasattr(tokenizer, "audio_eos_token")
            else audio_eos_token
        )
        self.audio_pad_token = (
            tokenizer.audio_pad_token
            if hasattr(tokenizer, "audio_pad_token")
            else audio_pad_token
        )
        self.audio_group_size = audio_group_size

        # WhisperFrontend 配置（懒加载）
        self._frontend_fs = frontend_fs
        self._frontend_n_mels = frontend_n_mels
        self._frontend_do_pad_trim = frontend_do_pad_trim
        self._frontend_filters_path = frontend_filters_path
        self._frontend = None

        # feature_extractor 仅用于兼容 ProcessorMixin.save_pretrained 等基础方法
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    # ------------------------------------------------------------------
    # WhisperFrontend 懒加载
    # ------------------------------------------------------------------

    def get_frontend(self):
        """懒加载 WhisperFrontend，避免 import 时强制依赖 funasr。"""
        if self._frontend is None:
            from vllm.model_executor.models.whisper import WhisperFrontend  # type: ignore
            kwargs: dict = dict(
                fs=self._frontend_fs,
                n_mels=self._frontend_n_mels,
                do_pad_trim=self._frontend_do_pad_trim,
            )
            if self._frontend_filters_path is not None:
                kwargs["filters_path"] = self._frontend_filters_path
            self._frontend = WhisperFrontend(**kwargs)
        return self._frontend

    # ------------------------------------------------------------------
    # 主调用入口
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audio: Union[np.ndarray, List[np.ndarray]] = None,
        audios=None,  # kept for BC，对齐 OmniDRCA2Processor 接口
        **kwargs: Unpack[OmniDRCA2NextProcessorKwargs],
    ) -> BatchFeature:
        """
        主方法：tokenize 文本 + 提取音频特征，返回合并后的 BatchFeature。

        Args:
            text: 待 tokenize 的文本（含 ``<|AUDIO|>`` 占位符）。
            audio: 原始音频波形（np.ndarray 或其列表），采样率应与
                ``frontend_fs`` 一致（默认 16000 Hz）。
            audios: 已废弃，请使用 ``audio``（保留向后兼容）。

        Returns:
            BatchFeature，包含：
                input_ids             : [1, seq_len]
                input_features        : [num_audios, max_T, d]
                audio_feature_lengths : [num_audios]
                audio_output_lengths  : [num_audios]
        """
        # 向后兼容：audios -> audio
        if audios is not None and audio is None:
            audio = audios
            warnings.warn(
                "Keyword argument `audios` is deprecated, please use `audio` instead.",
                FutureWarning,
                stacklevel=2,
            )

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        if isinstance(text, str):
            text = [text]

        output_kwargs = self._merge_kwargs(
            OmniDRCA2NextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # ----------------------------------------------------------------
        # 文本 tokenize
        # ----------------------------------------------------------------
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", "pt")
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # ----------------------------------------------------------------
        # 音频特征提取（WhisperFrontend + extract_fbank） 替换了 hugging faca whisper extract
        # ----------------------------------------------------------------
        if audio is not None:
            from vllm.utils.load_audio_utils import (  # type: ignore
                load_audio_text_image_video,
                extract_fbank,
            )

            audio_list: list = audio if isinstance(audio, (list, tuple)) else [audio]
            frontend = self.get_frontend()

            fbank_list: List[torch.Tensor] = []
            fbank_lens_list: List[int] = []
            token_lens_list: List[int] = []

            for aud in audio_list:
                # 数据类型归一化
                if isinstance(aud, str):
                    data_src = load_audio_text_image_video(
                        aud, fs=self._frontend_fs, use_ffmpeg=True
                    )
                elif isinstance(aud, torch.Tensor):
                    data_src = aud.squeeze()
                elif isinstance(aud, np.ndarray):
                    data_src = torch.from_numpy(aud).squeeze()
                else:
                    raise ValueError(
                        f"Unsupported audio type: {type(aud)}, "
                        "expected str / np.ndarray / torch.Tensor."
                    )

                # 提取 fbank：speech [1, d, T]，speech_lengths [1]
                speech, speech_lengths = extract_fbank(
                    data_src,
                    data_type="sound",
                    frontend=frontend,
                    is_final=True,
                )
                speech = speech.permute(0, 2, 1)  # [1, T, d]

                T = speech_lengths[0].item()  # fbank 帧数（25Hz）
                token_len = _compute_audio_output_length(T)

                fbank_list.append(speech[0])   # [T, d]
                fbank_lens_list.append(T)
                token_lens_list.append(token_len)

            # Pad 到最大帧长
            input_features = pad_sequence(
                fbank_list, batch_first=True, padding_value=0.0
            )  # [num_audios, max_T, d]

            # print(f"gangouyu next procrss after extract_fbank input_features dtype: {input_features.dtype}, input_features shape: {input_features.shape}, input_features: {input_features}")
            # torch.save(input_features, "pai_vllm_processor_input_features.pt")

            inputs["input_features"] = input_features
            inputs["audio_feature_lengths"] = torch.tensor(
                fbank_lens_list, dtype=torch.int32
            )
            inputs["audio_output_lengths"] = torch.tensor(
                token_lens_list, dtype=torch.int32
            )

        return BatchFeature(data=dict(inputs), tensor_type=return_tensors)

    # ------------------------------------------------------------------
    # 工具方法（对齐 OmniDRCA2Processor 接口）
    # ------------------------------------------------------------------

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict, **kwargs):
        """
        从模型目录加载子组件（feature_extractor、tokenizer）。
        由 ProcessorMixin.from_pretrained 调用。
        """
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(
                    cls.get_possibly_dynamic_module(n) if n is not None else None
                    for n in class_name
                )
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = cls.get_possibly_dynamic_module(class_name)
            # 这里会加载 whipser feature_extractor（但是实际没有用到）, 读取 preprocessor_config，但是实际上面 call 调用的是自己实现的 feature_extractor
            args.append(
                attribute_class.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
            )
        return args

    def batch_decode(self, *args, **kwargs):
        """转发至 tokenizer.batch_decode。"""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """转发至 tokenizer.decode。"""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = (
            self.feature_extractor.model_input_names
            if self.feature_extractor is not None
            else []
        )
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["audio_feature_lengths", "audio_output_lengths"]
            )
        )

    @property
    def default_chat_template(self):
        """默认 chat template（与 OmniDRCA2Processor 保持一致）。"""
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on


__all__ = ["OmniDRCA2NextProcessor", "_compute_audio_output_length"]
