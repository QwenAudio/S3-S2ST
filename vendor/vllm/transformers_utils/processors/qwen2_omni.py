# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for Qwen2Omni.
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np

from functools import lru_cache
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
)
from transformers.image_utils import ImageInput

try:
    from transformers.image_utils import VideoInput
except ImportError:
    from transformers.video_utils import VideoInput  # type: ignore[import-untyped]
from vllm.envs import VLLM_OMNI_PREPROCESS_ON_GPU
from vllm.transformers_utils.processors.fbank import FastAudioFeature
from vllm.transformers_utils.processors.image_process import FastVideoFeature


def exact_div(x, y):
    assert x % y == 0
    return x // y


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


class Qwen2OmniProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2OmniNaViTThinkerProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            # audio = load_audio(audio)
            pass
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    audio = audio.to(torch.float32)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class Qwen2OmniProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2Omni processor.

    [`Qwen2OmniProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2OmniProcessor.__call__`] and [`~Qwen2OmniProcessor.decode`] for more information.

    Args:

        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
    """

    attributes = ["omni_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    omni_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        if chat_template is None:
            chat_template = self.default_chat_template
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs: Unpack[Qwen2OmniProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `list[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        """

        output_kwargs = self._merge_kwargs(
            Qwen2OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audios is not None:
            audio_mels = []
            mel_lengths = []
            for audio in audios:
                audio_inputs = log_mel_spectrogram(audio, n_mels=128)
                audio_mels.append(audio_inputs)
                mel_lengths.append(audio_inputs.shape[1])  # [n, ]
            audio_mels = torch.cat(audio_mels, dim=1)  ##  [C, T]

            audios_inputs = {
                "input_audio_features": audio_mels,
                "audio_feature_lengths": torch.LongTensor(mel_lengths),
            }
            audios_lengths = audios_inputs["audio_feature_lengths"]
        else:
            audios_inputs = {}
            audios_lengths = None

        if images is not None:
            image_inputs = self.omni_processor(
                images=images, videos=None, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            video_durations = [x[1] for x in videos]
            video_nframes = [x[2] for x in videos]
            video_second_per_grid = [x[3] for x in videos]
            videos = [x[0] for x in videos]
            videos_inputs = self.omni_processor(
                images=None, videos=videos, **output_kwargs["videos_kwargs"]
            )
            # videos_inputs['video_durations'] = torch.FloatTensor(video_durations)
            # videos_inputs['video_nframes'] = torch.LongTensor(video_nframes)
            videos_inputs["video_second_per_grid"] = torch.FloatTensor(
                video_second_per_grid
            )

            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.omni_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|IMAGE|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|IMAGE|>",
                        "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|IMAGE|>")

        if video_grid_thw is not None:
            merge_length = self.omni_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|VIDEO|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|VIDEO|>",
                        "<|placeholder|>"
                        * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|VIDEO|>")

        if audios_lengths is not None:
            index = 0
            for i in range(len(text)):
                while "<|AUDIO|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|AUDIO|>",
                        "<|placeholder|>"
                        * (((audios_lengths[index] - 1) // 2 + 1 - 2) // 2 + 1),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|AUDIO|>")

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **audios_inputs}
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )

    @property
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of <|AUDIO|> tokens

        Example:

        ```python
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav",
                    },
                    {"type": "text", "text": "How about this one?"},
                ],
            },
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """

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
            "{% if 'audio' in content or 'audio_url' in content %}"
            "{% set audio_count.value = audio_count.value + 1 %}"
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
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


AutoProcessor.register("Qwen2OmniProcessor", Qwen2OmniProcessor)


class Qwen2OmniNaViTThinkerProcessor(ProcessorMixin):
    r"""
    Constructs a QwenOmniThinker processor.
    [`Qwen2OmniNaViTThinkerProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2OmniNaViTThinkerProcessor.__call__`] and [`~Qwen2OmniNaViTThinkerProcessor.decode`] for more information.

    Args:
        omni_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["omni_processor", "feature_extractor", "tokenizer"]
    omni_processor_class = "Qwen2VLImageProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template"]

    def __init__(
        self,
        omni_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
    ):
        super().__init__(
            omni_processor, feature_extractor, tokenizer, chat_template=chat_template
        )
        if VLLM_OMNI_PREPROCESS_ON_GPU:
            pre_device = os.getenv('PREPROCESS_DEVICE_T', 'cpu')
            self.get_audio_feature = FastAudioFeature(feature_size=self.feature_extractor.feature_size, device=pre_device)
            self.get_video_feature = FastVideoFeature(
                                        min_pixels=self.omni_processor.min_pixels,
                                        max_pixels=self.omni_processor.max_pixels,
                                        patch_size=self.omni_processor.patch_size,
                                        merge_size=self.omni_processor.merge_size,
                                        temporal_patch_size=self.omni_processor.temporal_patch_size,
                                        device=pre_device)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        sampling_rate: Optional[int] = 16000,
        fps: Optional[list[float]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        **kwargs: Unpack[Qwen2OmniNaViTThinkerProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwrags` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audios (`np.ndarray`, `list[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
            fsp (`int`, defaults to 2):
                The frames per second of video input.
        """

        output_kwargs = self._merge_kwargs(
            Qwen2OmniNaViTThinkerProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            padding=padding,
            **kwargs,
        )

        if audios is not None:
            if not VLLM_OMNI_PREPROCESS_ON_GPU:
                audios_inputs = self.feature_extractor(
                    audios, sampling_rate=sampling_rate, return_attention_mask=True, padding="max_length", **kwargs
                )
                audios_inputs["feature_attention_mask"] = audios_inputs.pop(
                    "attention_mask"
                )  # rename feature_attention_mask to prevent conflicts later on
                audios_inputs["input_features"] = audios_inputs.pop(
                    "input_features"
                )  # rename input_features to prevent conflicts later on
            else:
                fbank, mask = self.get_audio_feature.get_fbanks(audios, self.feature_extractor.n_samples)
                audios_inputs = {}
                audios_inputs["feature_attention_mask"] = mask # rename feature_attention_mask to prevent conflicts later on
                audios_inputs["input_features"] = fbank # rename input_features to prevent conflicts later on

            if (
                "audio_feature_lengths" not in audios_inputs
                and "feature_attention_mask" in audios_inputs
            ):
                feature_attention_mask = audios_inputs["feature_attention_mask"]
                audio_feature_lengths = feature_attention_mask.sum(-1)
                audios_inputs["audio_feature_lengths"] = audio_feature_lengths
            audios_lengths = audios_inputs["audio_feature_lengths"]
        else:
            audios_inputs = {}
            audios_lengths = None

        if images is not None:
            if not VLLM_OMNI_PREPROCESS_ON_GPU:
                images_inputs = self.omni_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            else:
                images_inputs = self.get_video_feature.image_process(
                                    images=images,
                                    videos=None,
                                    do_resize=self.omni_processor.do_resize,
                                    size=self.omni_processor.size,
                                    do_rescale=self.omni_processor.do_rescale,
                                    rescale_factor=self.omni_processor.rescale_factor,
                                    do_normalize=self.omni_processor.do_normalize,
                                    image_mean=self.omni_processor.image_mean,
                                    image_std=self.omni_processor.image_std,
                                    do_convert_rgb=self.omni_processor.do_convert_rgb,
                                    **output_kwargs["images_kwargs"])
            image_grid_thw = images_inputs["image_grid_thw"]
        else:
            images_inputs = {}
            image_grid_thw = None

        if videos is not None:
            # make it compatible with qwen2-omni processor.
            if isinstance(videos[0], list):
                videos = [x[0] for x in videos]
            if not VLLM_OMNI_PREPROCESS_ON_GPU:
                videos_inputs = self.omni_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            else:
                videos_inputs = self.get_video_feature.image_process(
                                    images=None,
                                    videos=videos,
                                    do_resize=self.omni_processor.do_resize,
                                    size=self.omni_processor.size,
                                    do_rescale=self.omni_processor.do_rescale,
                                    rescale_factor=self.omni_processor.rescale_factor,
                                    do_normalize=self.omni_processor.do_normalize,
                                    image_mean=self.omni_processor.image_mean,
                                    image_std=self.omni_processor.image_std,
                                    do_convert_rgb=self.omni_processor.do_convert_rgb,
                                    **output_kwargs["videos_kwargs"])
            if fps is None:
                fps = [2.0] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                fps[i] / self.omni_processor.temporal_patch_size
                for i in range(len(fps))
            ]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.omni_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|IMAGE|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|IMAGE|>",
                        "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|IMAGE|>")

        if video_grid_thw is not None:
            merge_length = self.omni_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|VIDEO|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|VIDEO|>",
                        "<|placeholder|>"
                        * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|VIDEO|>")

        if audios_lengths is not None:
            index = 0
            for i in range(len(text)):
                while "<|AUDIO|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|AUDIO|>",
                        "<|placeholder|>"
                        * (((audios_lengths[index] - 1) // 2 + 1 - 2) // 2 + 1),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|AUDIO|>")

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audios_inputs}
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        omni_processor_input_names = self.omni_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + omni_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


AutoProcessor.register("Qwen2OmniNaViTThinkerProcessor", Qwen2OmniNaViTThinkerProcessor)
