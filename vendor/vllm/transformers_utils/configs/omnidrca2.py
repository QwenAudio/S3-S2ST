# ruff: noqa: E501
# Adapted from https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/funaudiochat/configuration_funaudiochat.py

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

"""OmniDRCA2 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OmniDRCA2AudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniDRCA2Encoder`]. It is used to instantiate a
    Qwen2-Audio audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.
    e.g. [chtan/OmniDRCA2-7B](https://huggingface.co/chtan/OmniDRCA2-7B)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `OmniDRCA2Processor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    Example:
    ```python
    >>> from transformers import OmniDRCA2EncoderConfig, OmniDRCA2Encoder
    >>> # Initializing a OmniDRCA2EncoderConfig
    >>> configuration = OmniDRCA2EncoderConfig()
    >>> # Initializing a OmniDRCA2Encoder (with random weights)
    >>> model = OmniDRCA2Encoder(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omnidrca2_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim


class OmniDRCA2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniDRCA2ForConditionalGeneration`]. It is used to instantiate an
    Qwen2-Audio model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2-Audio.
    e.g. [chtan/OmniDRCA2-7B](https://huggingface.co/chtan/OmniDRCA2-7B)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
    Example:
    ```python
    >>> from transformers import OmniDRCA2ForConditionalGeneration, OmniDRCA2Config, OmniDRCA2EncoderConfig, Qwen2Config
    >>> # Initializing a OmniDRCA2Encoder config
    >>> audio_config = OmniDRCA2EncoderConfig()
    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()
    >>> # Initializing a OmniDRCA2 configuration
    >>> configuration = OmniDRCA2Config(audio_config, text_config)
    >>> # Initializing a model from the qwen2-audio style configuration
    >>> model = OmniDRCA2ForConditionalGeneration(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omnidrca2"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151646,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "omnidrca2_audio_encoder")
            audio_config = OmniDRCA2AudioEncoderConfig(**audio_config)
            if audio_config.crq_transformer_name_or_path is not None:
                from vllm.transformers_utils.config import get_config

                audio_config.crq_transformer_config = get_config(
                    audio_config.crq_transformer_name_or_path, trust_remote_code=False
                )

        elif audio_config is None:
            audio_config = OmniDRCA2AudioEncoderConfig(
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )
        self.audio_config = audio_config
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()
        self.text_config = text_config
        super().__init__(**kwargs)


__all__ = ["OmniDRCA2Config", "OmniDRCA2AudioEncoderConfig"]
