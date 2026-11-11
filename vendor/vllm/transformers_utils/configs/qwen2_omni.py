# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen2Omni model configuration"""

import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen2OmniVisionEncoderConfig(PretrainedConfig):
    model_type = "qwen2_omni_vision_encoder"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if config_dict.get("model_type") == "qwen2_omni":
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Qwen2OmniAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2OmniAudioEncoder`]. It is used to instantiate a
    Qwen2-Omni audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.

    e.g. [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Qwen2OmniProcessor` class.
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
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        n_window (int defaults to 100)
            The chunk for conv and flash attn in AudioEncoder

    Example:

    ```python
    >>> from transformers import Qwen2OmniAudioEncoderConfig, Qwen2OmniAudioEncoder

    >>> # Initializing a Qwen2OmniAudioEncoderConfig
    >>> configuration = Qwen2OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2OmniAudioEncoder (with random weights)
    >>> model = Qwen2OmniAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_omni_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
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
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim


class Qwen2OmniNaViTVisionEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`QwenOmniThinkerVision`]. It is used to instantiate a
    Qwen2-VL vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-VL
    architecture.

    e.g. [Qwen/Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of layers (depth) in the model.
        embed_dim (`int`, *optional*, defaults to 1280):
            Dimensionality of the embeddings.
        hidden_size (`int`, *optional*, defaults to 3584):
            The size of the hidden layers.
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function used in the model. Supported options include `"quick_gelu"` and others as applicable.
        mlp_ratio (`float`, *optional*, defaults to 4):
            The ratio used to determine the size of the MLP (Multi-Layer Perceptron) hidden layer.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        patch_size (`int`, *optional*, defaults to 14):
            The size of the patches extracted from the input.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The size used for patches along the temporal dimension.
        _attn_implementation (`str`, *optional*, defaults to `"flash_attention_2"`):
            The attention implementation strategy used in the model.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the initializer used for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen2OmniNaViTVisionEncoderConfig, Qwen2OmniNaViTVisionEncoder

    >>> # Initializing a Qwen2OmniNaViTVisionEncoderConfig
    >>> configuration = Qwen2OmniNaViTVisionEncoderConfig()

    >>> # Initializing a Qwen2OmniNaViTVisionEncoder (with random weights)
    >>> model = Qwen2OmniNaViTVisionEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_omni_navit_vision_encoder"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        embed_dim=1280,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        _attn_implementation="flash_attention_2",
        init_std=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self._attn_implementation = _attn_implementation
        self.init_std = init_std

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if config_dict.get("model_type") == "qwen2_omni_navit_thinker":
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Qwen2OmniNaViTAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2OmniNaViTAudioEncoder`]. It is used to instantiate a
    Qwen-Omni-Thinker audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.

    e.g. [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Qwen2OmniNaViTThinkerProcessor` class.
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
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        n_window (`int`, *optional*, defaults to 100):
            The chunk for conv and flash attn in AudioEncoder.
        output_dim (`int`, *optional*, defaults to 3584):
            The output dimention of AudioEncoder.

    Example:

    ```python
    >>> from transformers import Qwen2OmniNaViTAudioEncoderConfig, Qwen2OmniNaViTAudioEncoder

    >>> # Initializing a Qwen2OmniNaViTAudioEncoderConfig
    >>> configuration = Qwen2OmniNaViTAudioEncoderConfig()

    >>> # Initializing a Qwen2OmniNaViTAudioEncoder (with random weights)
    >>> model = Qwen2OmniNaViTAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_omni_navit_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        _attn_implementation="flash_attention_2",
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
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self._attn_implementation = _attn_implementation


class Qwen2OmniThinkerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2OmniThinkerForConditionalGeneration`]. It is used to instantiate an
    Qwen2-Omni model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2-Omni.

    e.g. [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        vision_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the vision backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.
        image_token_index (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_index (`int`, *optional*, defaults to 151656):
            The video token index to encode the video prompt.
        vocab_size (int )

    Example:

    ```python
    >>> from transformers import Qwen2OmniThinkerForConditionalGeneration, Qwen2OmniThinkerConfig, Qwen2OmniAudioEncoderConfig, Qwen2OmniVisionEncoderConfig

    >>> # Initializing a Qwen2OmniAudioEncoder config
    >>> audio_config = Qwen2OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing a Qwen2Omni configuration
    >>> configuration = Qwen2OmniThinkerConfig(audio_config, text_config)

    >>> # Initializing a model from the qwen2-audio style configuration
    >>> model = Qwen2OmniThinkerForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_omni_thinker"
    is_composition = False

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        audio_token_index=151646,
        image_token_index=151655,
        video_token_index=151656,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        attention_dropout=0.0,
        rope_scaling=None,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
        user_token_id=872,
        init_std=0.02,
        _attn_implementation="flash_attention_2",
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.user_token_id = user_token_id

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.position_id_per_seconds = position_id_per_seconds  # zf
        self.seconds_per_chunk = seconds_per_chunk  # zf
        self.audio_start_token_id = audio_start_token_id  # zf
        self.audio_end_token_id = audio_end_token_id  # zf

        self._attn_implementation = _attn_implementation
        self.init_std = init_std

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does defeault RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        if isinstance(vision_config, dict):
            if vision_config["model_type"] == "qwen2_omni_navit_vision_encoder":
                self.vision_config = Qwen2OmniNaViTVisionEncoderConfig(**vision_config)
            else:
                self.vision_config = Qwen2OmniVisionEncoderConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2OmniVisionEncoderConfig()

        if isinstance(audio_config, dict):
            if audio_config["model_type"] == "qwen2_omni_navit_audio_encoder":
                self.audio_config = Qwen2OmniNaViTAudioEncoderConfig(**audio_config)
            else:
                self.audio_config = Qwen2OmniAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = Qwen2OmniAudioEncoderConfig(
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

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen2OmniTalkerConfig(PretrainedConfig):
    r"""


    ```python
    >>> from transformers import Qwen2OmniTalkerForConditionalGeneration, Qwen2OmniThinkerConfig, Qwen2OmniAudioEncoderConfig, Qwen2OmniVisionEncoderConfig

    >>> # Initializing a Qwen2OmniAudioEncoder config
    >>> audio_config = Qwen2OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing a Qwen2Omni configuration
    >>> configuration = Qwen2OmniThinkerConfig(audio_config, text_config)

    >>> # Initializing a model from the qwen2-audio style configuration
    >>> model = Qwen2OmniTalkerForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_omni_talker"
    is_composition = False

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        audio_token_index=151646,
        image_token_index=151655,
        video_token_index=151656,
        vocab_size=999,
        tts_text_spk_0_token_id=151869,
        tts_text_spk_1_token_id=151870,
        tts_text_spk_2_token_id=151871,
        tts_text_spk_3_token_id=151872,
        tts_text_start_token_id=999999,
        tts_text_end_token_id=999999,
        tts_text_pad_token_id=999999,
        tts_codec_start_token_id=999999,
        tts_codec_end_token_id=999999,
        tts_codec_pad_token_id=999999,
        tts_codec_mask_token_id=999999,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        embedding_size=3584,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        head_dim=128,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        attention_dropout=0.0,
        rope_scaling=None,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
        init_std=0.02,
        spatial_merge_size=2,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index

        self.tts_text_spk_0_token_id = tts_text_spk_0_token_id
        self.tts_text_spk_1_token_id = tts_text_spk_1_token_id
        self.tts_text_spk_2_token_id = tts_text_spk_2_token_id
        self.tts_text_spk_3_token_id = tts_text_spk_3_token_id

        self.tts_text_start_token_id = tts_text_start_token_id
        self.tts_text_end_token_id = tts_text_end_token_id
        self.tts_text_pad_token_id = tts_text_pad_token_id
        self.tts_codec_start_token_id = tts_codec_start_token_id
        self.tts_codec_end_token_id = tts_codec_end_token_id
        self.tts_codec_pad_token_id = tts_codec_pad_token_id

        self.tts_codec_mask_token_id = tts_codec_mask_token_id

        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.embedding_size = embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.position_id_per_seconds = position_id_per_seconds  # zf
        self.seconds_per_chunk = seconds_per_chunk  # zf
        self.audio_start_token_id = audio_start_token_id  # zf
        self.audio_end_token_id = audio_end_token_id  # zf

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does defeault RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        self.init_std = init_std
        self.spatial_merge_size = spatial_merge_size

        if isinstance(vision_config, dict):
            if vision_config["model_type"] == "qwen2_omni_navit_vision_encoder":
                self.vision_config = Qwen2OmniNaViTVisionEncoderConfig(**vision_config)
            else:
                self.vision_config = Qwen2OmniVisionEncoderConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2OmniVisionEncoderConfig()

        if isinstance(audio_config, dict):
            if audio_config["model_type"] == "qwen2_omni_navit_audio_encoder":
                self.audio_config = Qwen2OmniNaViTAudioEncoderConfig(**audio_config)
            else:
                self.audio_config = Qwen2OmniAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = Qwen2OmniAudioEncoderConfig(
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

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
