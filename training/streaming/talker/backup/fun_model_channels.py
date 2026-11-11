# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Literal, Optional, List
from collections import OrderedDict
from contextlib import contextmanager
import os
import torch
from torch import Tensor
import torch.nn as nn
import logging
import traceback

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
# from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig, AudioConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import get_tensor_model_parallel_group

from megatron.core import mpu

from .audio_encoder import SenseVoiceEncoder, Qwen3Encoder
from .audio_adaptor import AudioAdaptor
from .speech_tokenizer import SpeechTokenizer
from .fun_utils import pad_cu_seqlens, unpad_cu_seqlens, freeze_module, set_module_lr_scale, process_with_sliding_window_and_bucketing
from .fun_model import FunBaseModel, dtype_map, compute_accuracy, force_gatherable


@contextmanager
def _unpatch_modelopt_load_pretrained():
    """Temporarily restore HF loading if NVIDIA modelopt patched it incompatibly."""
    from transformers import modeling_utils

    pre_trained_model = modeling_utils.PreTrainedModel
    current = getattr(pre_trained_model, "_load_pretrained_model", None)
    original = None
    if current is not None and getattr(current, "__name__", "").startswith("_new"):
        original = getattr(current, "__wrapped__", None)
        if original is None and getattr(current, "__closure__", None):
            for cell in current.__closure__:
                try:
                    value = cell.cell_contents
                    if callable(value) and getattr(value, "__name__", "") == "_load_pretrained_model":
                        original = value
                        break
                except Exception:
                    pass
        if original is None and hasattr(pre_trained_model, "_modelopt_cache"):
            original = pre_trained_model._modelopt_cache.get("_load_pretrained_model")
    if original is not None:
        pre_trained_model._load_pretrained_model = original
    try:
        yield
    finally:
        if original is not None:
            pre_trained_model._load_pretrained_model = current


# class FunBaseModel(LanguageModule):
class FunBaseModelChannels(FunBaseModel):
    """
    Args:
        config (TransformerConfig): Language model configuration
        audio_config (AudioConfig): Audio processing configuration
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): Maximum sequence length
        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)
        fp16_lm_cross_entropy (bool): Use fp16 for cross entropy
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks
        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are shared
        position_embedding_type (Literal): Position embedding type
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings
        rotary_base (int): Base period for rotary position embeddings
        rope_scaling (bool): Whether to use RoPE scaling
        seq_len_interpolation_factor (Optional[float]): Scale of linearly interpolating RoPE for longer sequences
        length_normalized_loss (bool): Whether to use length normalized loss
        audio_encoder_activation_checkpoint (bool): Whether to use activation checkpointing for audio encoder
        feat_permute (bool): Whether to permute features for audio encoder
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        audio_config: AudioConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,   # position embedding
        seq_length: int,            # actual training sequence length
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        length_normalized_loss: bool = False,
        audio_encoder_activation_checkpoint: bool = False,
        feat_permute: bool = True,
    ) -> None:
        # super().__init__(config=config)
        super().__init__(
            config=config,
            audio_config=audio_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            seq_length=seq_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            length_normalized_loss=length_normalized_loss,
            audio_encoder_activation_checkpoint=audio_encoder_activation_checkpoint,
            feat_permute=feat_permute,
        )

        self.audio_code_book_size = self.audio_config.tokenizer_code_book_size
        self.mask_token_id = -60
        self.special_tokens_audio_count = 463
        self.audio_loss_weight = float(getattr(self.audio_config, "audio_loss_weight", 1.0))
        self.has_audio_output_head = (
            bool(self.audio_config.audio_output_head)
            and self.audio_loss_weight != 0.0
        )
        self.has_cosyvoice_codex_head = bool(getattr(self.audio_config, "cosyvoice_codex_enabled", False))
        self.has_cosyvoice_fm = bool(getattr(self.audio_config, "cosyvoice_fm_enabled", False)) and self.post_process
        self._cosyvoice_codex_label_debug_logged = False
        self._cosyvoice_codex_forward_debug_logged = False
        self._audio_token_mismatch_log_count = 0
        self.cosyvoice_codec_unit = getattr(self.audio_config, "cosyvoice_codex_codec_unit", None)
        self.cosyvoice_llm_init_param_path = getattr(
            self.audio_config, "cosyvoice_codex_llm_init_param_path", None
        )
        self.cosyvoice_llm_dtype = getattr(self.audio_config, "cosyvoice_codex_llm_dtype", "fp32")
        self.cosyvoice_llm_freeze = bool(getattr(self.audio_config, "cosyvoice_codex_llm_freeze", True))
        self.cosyvoice_codex_use_text_token_input = bool(
            getattr(self.audio_config, "cosyvoice_codex_use_text_token_input", False)
        )
        self._cosyvoice_text_token_debug_logged = False
        if self.has_audio_output_head:
            logging.info(
                f'[MODEL INIT] init audio codec head, code_book_size={self.audio_code_book_size}, '
                f'special_tokens_audio_count={self.special_tokens_audio_count}'
            )
            self.init_audio_head()
        elif bool(self.audio_config.audio_output_head):
            logging.info(
                '[MODEL INIT] skip audio codec head because audio_loss_weight=0.0'
            )
        if self.has_cosyvoice_codex_head:
            self.init_cosyvoice_codex_head()
        if self.has_cosyvoice_fm:
            if not self.has_cosyvoice_codex_head:
                raise ValueError("CosyVoice FM requires the codec LLM branch")
            from funasr.models.cosyvoice.flow_matching_model import CosyVoiceFlowMatching
            fm_conf = dict(getattr(self.audio_config, "cosyvoice_fm_model_conf", {}))
            expected_dim = int(self.cosyvoice_llm_hidden_size)
            configured_dim = int(fm_conf.get("embed_dim", expected_dim))
            if configured_dim != expected_dim:
                raise ValueError(
                    f"FM embed_dim={configured_dim} must match codec LLM hidden size={expected_dim}"
                )
            self.cosyvoice_fm = CosyVoiceFlowMatching(**fm_conf)
            # FM is replicated rather than tensor-sharded.  Mark its weights as
            # shared so Megatron all-reduces their gradients across TP ranks.
            if mpu.get_tensor_model_parallel_world_size() > 1:
                for param in self.cosyvoice_fm.parameters():
                    param.sequence_parallel = True
            logging.info("[MODEL INIT] enabled joint CosyVoice FM training: %s", fm_conf)


    def init_audio_head(self):
        # audio embedding and head
        if self.pre_process:
            # self.audio_embed = nn.Embedding(self.audio_code_book_size + self.special_tokens_audio_count, config.hidden_size)
            self.embedding_audio = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.audio_code_book_size + self.special_tokens_audio_count,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type='none',
            )

        if self.post_process:
            # self.audio_head = nn.Linear(config.hidden_size, self.audio_code_book_size + self.special_tokens_audio_count)
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer_audio = []
                self.grad_output_buffer_audio = []
            else:
                self.embedding_activation_buffer_audio = None
                self.grad_output_buffer_audio = None

            self.output_layer_audio = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.audio_code_book_size + self.special_tokens_audio_count,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and False, # not share audio embeddings and output weights
                embedding_activation_buffer=self.embedding_activation_buffer_audio,
                grad_output_buffer=self.grad_output_buffer_audio,
            )

            if self.freeze_decoder:
                logging.info(f'[MODEL INIT] freeze audio input and output layers')
                if self.pre_process:
                    freeze_module(self.embedding_audio)
                if self.post_process:
                    freeze_module(self.output_layer_audio)
            audio_input_output_lr_scale = None
            if self.audio_config.audio_input_output_lr_scale is not None:
                logging.info(f'[MODEL INIT] set audio input and output layers lr scale to audio_input_output_lr_scale {self.audio_config.audio_input_output_lr_scale}')
                audio_input_output_lr_scale = self.audio_config.audio_input_output_lr_scale
            elif self.audio_config.decoder_lr_scale is not None and self.audio_config.decoder_lr_scale != 1.0:
                logging.info(f'[MODEL INIT] set audio input and output layers lr scale to decoder_lr_scale {self.audio_config.decoder_lr_scale}')
                audio_input_output_lr_scale = self.audio_config.decoder_lr_scale
            if audio_input_output_lr_scale is not None and audio_input_output_lr_scale != 1.0:
                if self.pre_process:
                    set_module_lr_scale(self.embedding_audio, audio_input_output_lr_scale)
                if self.post_process:
                    set_module_lr_scale(self.output_layer_audio, audio_input_output_lr_scale)
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer_audio()

    def init_cosyvoice_codex_head(self):
        if not self.post_process:
            return

        if self.cosyvoice_codec_unit is None:
            self.cosyvoice_codec_unit = self.audio_code_book_size + self.special_tokens_audio_count

        tp_world_size = int(
            getattr(
                self.audio_config,
                "cosyvoice_codex_vocab_padding_multiple",
                0,
            )
            or mpu.get_tensor_model_parallel_world_size()
        )
        self.cosyvoice_codec_vocab_size = (
            (int(self.cosyvoice_codec_unit) + tp_world_size - 1) // tp_world_size
        ) * tp_world_size
        if not self.cosyvoice_llm_init_param_path:
            raise ValueError(
                "CosyVoiceMixedLMSimpleSysV2 codex LLM is enabled but "
                "cosyvoice_codex_llm_init_param_path is empty."
            )

        from transformers import AutoModelForCausalLM

        logging.info(
            "[MODEL INIT] load CosyVoiceMixedLMSimpleSysV2 codex LLM from "
            f"{self.cosyvoice_llm_init_param_path}, dtype={self.cosyvoice_llm_dtype}, "
            f"freeze={self.cosyvoice_llm_freeze}"
        )
        with _unpatch_modelopt_load_pretrained():
            cosyvoice_llm = AutoModelForCausalLM.from_pretrained(
                self.cosyvoice_llm_init_param_path,
                device_map=None,
            )
        self.cosyvoice_llm = cosyvoice_llm.model.to(dtype_map[self.cosyvoice_llm_dtype])
        self.cosyvoice_llm_hidden_size = int(cosyvoice_llm.config.hidden_size)
        if self.cosyvoice_llm_freeze:
            freeze_module(self.cosyvoice_llm)
            self.cosyvoice_llm.eval()
        else:
            # The codex LLM receives projected main-decoder hidden states through
            # inputs_embeds, so its token embedding table is not used in forward.
            # Keep it frozen to avoid DDP waiting for a gradient that will not exist.
            if (
                not self.cosyvoice_codex_use_text_token_input
                and hasattr(self.cosyvoice_llm, "embed_tokens")
            ):
                freeze_module(self.cosyvoice_llm.embed_tokens)

        if self.config.hidden_size == self.cosyvoice_llm_hidden_size:
            self.cosyvoice_llm_input_proj = nn.Identity()
        else:
            self.cosyvoice_llm_input_proj = nn.Linear(
                self.config.hidden_size,
                self.cosyvoice_llm_hidden_size,
                bias=False,
            )
        if self.cosyvoice_codex_use_text_token_input:
            # This branch replaces the projection rather than adding to it.
            # Freezing avoids an unused trainable parameter under DDP and keeps
            # the module/state-dict layout compatible with old checkpoints.
            freeze_module(self.cosyvoice_llm_input_proj)
        self.cosyvoice_codec_embed = nn.Embedding(
            self.cosyvoice_codec_vocab_size,
            self.cosyvoice_llm_hidden_size,
        )

        logging.info(
            "[MODEL INIT] init CosyVoiceMixedLMSimpleSysV2 codex LLM and head, "
            f"main_hidden_size={self.config.hidden_size}, "
            f"codex_llm_hidden_size={self.cosyvoice_llm_hidden_size}, "
            f"codec_unit={self.cosyvoice_codec_unit}, "
            f"parallel_vocab_size={self.cosyvoice_codec_vocab_size}, "
            f"codec_embedding_size={self.cosyvoice_codec_embed.weight.shape}"
            f", use_text_token_input={self.cosyvoice_codex_use_text_token_input}"
        )

        self.cosyvoice_llm_codec_head = tensor_parallel.ColumnParallelLinear(
            self.cosyvoice_llm_hidden_size,
            self.cosyvoice_codec_vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
        )
        cosyvoice_llm_trainable = sum(
            p.numel() for p in self.cosyvoice_llm.parameters() if p.requires_grad
        )
        cosyvoice_llm_frozen = sum(
            p.numel() for p in self.cosyvoice_llm.parameters() if not p.requires_grad
        )
        cosyvoice_proj_trainable = sum(
            p.numel() for p in self.cosyvoice_llm_input_proj.parameters() if p.requires_grad
        )
        cosyvoice_codec_embed_trainable = sum(
            p.numel() for p in self.cosyvoice_codec_embed.parameters() if p.requires_grad
        )
        cosyvoice_head_trainable = sum(
            p.numel() for p in self.cosyvoice_llm_codec_head.parameters() if p.requires_grad
        )
        logging.info(
            "[MODEL INIT] CosyVoice codex trainable params: "
            f"llm={cosyvoice_llm_trainable}, "
            f"llm_frozen={cosyvoice_llm_frozen}, "
            f"input_proj={cosyvoice_proj_trainable}, "
            f"codec_embed={cosyvoice_codec_embed_trainable}, "
            f"codec_head={cosyvoice_head_trainable}"
        )

    def init_extra_output_specs(self):
        if mpu.get_context_parallel_world_size() <= 0:
            return
        seq_length = self.seq_length // mpu.get_context_parallel_world_size()

        # extra output specs
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            self.extra_output_specs = [
                (["b", seq_length], torch.int64),  # labels_ids, only gets splitted by cp
                (["b", seq_length], torch.int64),  # labels_ids_audio, only gets splitted by cp
            ]
            if bool(getattr(self.audio_config, "cosyvoice_codex_use_text_token_input", False)):
                self.extra_output_specs.append(
                    (["b", seq_length], torch.int64)  # input_ids_codec_text
                )
            self.extra_output_specs.extend([
                (["b", seq_length], torch.int32),  # loss_mask, only gets splitted by cp
                (["b", seq_length], torch.int32),  # loss_mask_audio, only gets splitted by cp
            ])
            if mpu.get_context_parallel_world_size() > 1:
                self.extra_output_specs.extend([
                    (["b", seq_length], torch.int32),  # packed_seq_params.cu_seqlens, padded to same shape as labels_ids
                    ([1], torch.int32),  # packed_seq_params.max_seqlen
                ])
        else:
            self.extra_output_specs = []

    def set_input_tensor(self, input_tensor) -> None:
        """Sets input tensor to the model.
        
        Args:
            input_tensor: Input tensor for the model
              0: decoder input
              1: labels_ids
              2: labels_ids_audio
              3: input_ids_codec_text
              4: loss_mask
              5: loss_mask_audio
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1 + len(self.extra_output_specs), f'input_tensor should only be length 1 + {len(self.extra_output_specs)} for FunBaseModel'

        self.decoder.set_input_tensor(input_tensor[0])

        self.labels_ids = None
        self.labels_ids_audio = None
        self.input_ids_codec_text = None
        self.loss_mask = None
        self.loss_mask_audio = None
        self.packed_seq_params = None

        if len(self.extra_output_specs) > 0:
            tensor_idx = 1
            self.labels_ids = input_tensor[tensor_idx]
            tensor_idx += 1
            self.labels_ids_audio = input_tensor[tensor_idx]
            tensor_idx += 1
            if bool(getattr(self.audio_config, "cosyvoice_codex_use_text_token_input", False)):
                self.input_ids_codec_text = input_tensor[tensor_idx]
                tensor_idx += 1
            self.loss_mask = input_tensor[tensor_idx]
            tensor_idx += 1
            self.loss_mask_audio = input_tensor[tensor_idx]
            tensor_idx += 1

            if mpu.get_context_parallel_world_size() > 1:
                cu_seqlens_padded = input_tensor[tensor_idx]
                max_seqlen = input_tensor[tensor_idx + 1]
                if cu_seqlens_padded is not None and max_seqlen is not None:
                    cu_seqlens = unpad_cu_seqlens(cu_seqlens_padded)
                    self.packed_seq_params = {
                        "cu_seqlens_q": cu_seqlens,
                        "cu_seqlens_kv": cu_seqlens,
                        "max_seqlen_q": max_seqlen,
                        "max_seqlen_kv": max_seqlen,
                        "qkv_format": "thd",
                    }
        
        # if self.pre_process:
        #     self.audio_hidden_state = input_tensor[0]
        # else:
        #     # self.lm_model.set_input_tensor(input_tensor[0])
        #     self.set_input_tensor(input_tensor[0])

    def get_passthrough_input_tensors(self, labels_ids, labels_ids_audio, input_ids_codec_text, loss_mask, loss_mask_audio, packed_seq_params) -> tuple[Tensor]:
        # passed from previous stage
        if self.labels_ids is not None:
            labels_ids = self.labels_ids
        if self.labels_ids_audio is not None:
            labels_ids_audio = self.labels_ids_audio
        if self.input_ids_codec_text is not None:
            input_ids_codec_text = self.input_ids_codec_text
        if self.loss_mask is not None:
            loss_mask = self.loss_mask
        if self.loss_mask_audio is not None:
            loss_mask_audio = self.loss_mask_audio
        if self.packed_seq_params is not None:
            packed_seq_params = self.packed_seq_params
        return labels_ids, labels_ids_audio, input_ids_codec_text, loss_mask, loss_mask_audio, packed_seq_params

    def _shift_cosyvoice_audio_labels(
        self,
        input_ids_audio: torch.Tensor,
        labels_ids_audio: torch.Tensor,
    ) -> torch.Tensor:
        placeholder_mask = labels_ids_audio == -50
        input_id_shift = torch.zeros_like(input_ids_audio)
        input_id_shift[:, :-1] = input_ids_audio[:, 1:]

        first_codec_label_mask = torch.zeros_like(placeholder_mask)
        first_codec_label_mask[:, :-1] = placeholder_mask[:, 1:]
        first_codec_label_mask = first_codec_label_mask & (labels_ids_audio == -100)

        labels_ids_audio = torch.where(
            placeholder_mask | first_codec_label_mask,
            input_id_shift,
            labels_ids_audio,
        )

        previous_placeholder_mask = torch.zeros_like(placeholder_mask)
        previous_placeholder_mask[:, 1:] = placeholder_mask[:, :-1]
        eos_self_label_mask = (
            previous_placeholder_mask
            & (~placeholder_mask)
            & (labels_ids_audio == input_ids_audio)
            & (labels_ids_audio >= 0)
        )
        labels_ids_audio = torch.where(
            eos_self_label_mask,
            torch.tensor(-100, device=labels_ids_audio.device, dtype=labels_ids_audio.dtype),
            labels_ids_audio,
        )
        return labels_ids_audio
    
    def forward(
        self,

        segment_fbank: torch.Tensor = None,
        segment_fbank_lengths: torch.Tensor = None,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        feature_beg: torch.Tensor = None,
        feature_type: torch.Tensor = None,
        fake_token_len: torch.Tensor = None,

        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        position_ids: Tensor = None,
        loss_mask: torch.Tensor = None,
        loss_mask_audio: torch.Tensor = None,

        input_ids_audio: torch.Tensor = None,
        labels_ids_audio: torch.Tensor = None,
        input_ids_codec_text: torch.Tensor = None,
        target_audio: torch.Tensor = None,
        target_audio_lengths: torch.Tensor = None,
        target_spk_embedding: torch.Tensor = None,

        items: list = None,
        speech_token_id_offset: list = None,

        sequence_chunks: list = None,
        speech_token_nums_per_sequence_chunk: list = None,  # shape: [batch, cp_size * cp_split_size]

        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: dict = None,
        **kwargs,
    ) -> tuple[Tensor, dict, int]:

        inputs_embeds = None
        loss_fake = None

        labels_ids, labels_ids_audio, input_ids_codec_text, loss_mask, loss_mask_audio, packed_seq_params = self.get_passthrough_input_tensors(
            labels_ids, labels_ids_audio, input_ids_codec_text, loss_mask, loss_mask_audio, packed_seq_params
        )

        # speech tokenize 
        if segment_fbank is not None:
            if self.config.timers is not None:
                self.config.timers('forward-audio-tokenize-extract', log_level=2).start()
            segment_speech_token, segment_speech_token_lengths = self.speech_tokenizer._extract_with_automodel(segment_fbank, segment_fbank_lengths)
            if self.config.timers is not None:
                self.config.timers('forward-audio-tokenize-extract', log_level=2).stop()
                self.config.timers('forward-audio-tokenize-insert', log_level=2).start()
            idx = 0
            frame_rate = self.audio_config.tokenizer_frame_rate
            slice_speech_tokens = []
            # Keep the per-slice token boundaries as well as the historical
            # concatenated view.  The true-streaming S2ST layout contains one
            # independent -50 run per source/target audio slice; replacing the
            # concatenation against all -50 positions at once lets a small
            # tokenizer length delta in one slice move a later PAUSE/EOS into
            # the wrong Talker envelope.
            slice_speech_token_chunks = []
            for i, item in enumerate(items):
                segments = {}
                for segment_id, segment in item["segments_dict"].items():
                    if not segment["onlie_discrete_speech_token"]:
                        continue
                    segment_speech_token_length = segment_speech_token_lengths[idx].item()
                    segment["tokens"] = segment_speech_token[idx][:segment_speech_token_length]
                    idx += 1
                    segments[segment["segment_id"]] = segment

                slice_speech_tokens.append([])
                user = item["user"]
                assistant = item["assistant"]
                for user_item, assistant_item in zip(user, assistant):
                    for _slice in [user_item, assistant_item]:
                        if _slice is None:
                            continue
                        if "onlie_discrete_speech_token" not in _slice:
                            raise ValueError(f"onlie_discrete_speech_token not in _slice: {_slice}")
                        if not _slice["onlie_discrete_speech_token"]:
                            continue
                        segment = segments[_slice["segment_id"]]
                        token_offset = int((_slice["start_time"] - segment["start_time"]) * frame_rate)
                        token_duration = int((_slice["end_time"] - _slice["start_time"]) * frame_rate)
                        speech_tokens = segment["tokens"][token_offset:token_offset + token_duration]
                        # assert len(speech_tokens) == token_duration, f'[speech_tokens length not match], {len(speech_tokens)} vs {token_duration}'
                        if len(speech_tokens) < token_duration:
                            print (f'[MODEL FORWARD] speech_tokens length not match], {len(speech_tokens)} vs {token_duration}')
                            print (f'[MODEL FORWARD] segment tokens length: {len(segment["tokens"])}')
                            print (f'[MODEL FORWARD] token_offset: {token_offset}, token_duration: {token_duration}')
                            print (f'[MODEL FORWARD] segment time start: {segment["start_time"]}, end: {segment["end_time"]}')
                            print (f'[MODEL FORWARD] _slice time start: {_slice["start_time"]}, end: {_slice["end_time"]}')
                            print (f'[MODEL FORWARD] item: {item}')

                            if len(segment["tokens"]) < token_duration:
                                print (f'[MODEL FORWARD] segment tokens less than token_duration: {len(segment["tokens"])} vs {token_duration}; keep actual tokens and ignore the remaining placeholders')
                            else:
                                speech_tokens = segment["tokens"][-token_duration:]
                                print (f'[MODEL FORWARD] pick last token duration: {len(segment["tokens"])} vs {token_duration}')

                        slice_speech_tokens[-1].append(speech_tokens)

                slice_speech_token_chunks.append(list(slice_speech_tokens[-1]))
                if len(slice_speech_tokens[-1]) == 0:
                    # 纯文本，或者audio to text的形式
                    assert len(input_ids_audio[i][input_ids_audio[i] == -50]) == 0, f'[MODEL FORWARD] input_ids_audio[i][input_ids_audio[i] == -50] is not empty, {len(input_ids_audio[i][input_ids_audio[i] == -50])}'
                    slice_speech_tokens[-1] = torch.tensor([], dtype=torch.long)
                else:
                    slice_speech_tokens[-1] = torch.cat(slice_speech_tokens[-1])
                # not need offset anymore
                # slice_speech_tokens[i] += speech_token_id_offset[0]

            def _replace_audio_placeholders(row_idx, replacement_tokens):
                placeholder_positions = torch.nonzero(
                    input_ids_audio[row_idx] == -50, as_tuple=False
                ).view(-1)
                num_placeholders = int(placeholder_positions.numel())
                replacement_tokens = replacement_tokens.long().to(input_ids_audio.device)
                num_tokens = int(replacement_tokens.numel())
                if num_placeholders == 0:
                    return
                if num_tokens != num_placeholders:
                    diff = num_placeholders - num_tokens
                    self._audio_token_mismatch_log_count += 1
                    if (
                        self._audio_token_mismatch_log_count <= 20
                        or abs(diff) >= 16
                        or self._audio_token_mismatch_log_count % 1000 == 0
                    ):
                        print(
                            '[MODEL FORWARD] input_ids_audio -50/token length mismatch, '
                            f'row={row_idx}, placeholders={num_placeholders}, tokens={num_tokens}, '
                            f'diff={diff}, count={self._audio_token_mismatch_log_count}'
                        )
                if num_tokens > num_placeholders:
                    last_placeholder_pos = int(placeholder_positions[-1].item())
                    eos_source_pos = last_placeholder_pos
                    eos_token = None
                    if labels_ids_audio is not None and eos_source_pos < input_ids_audio.size(1):
                        eos_candidate = labels_ids_audio[row_idx, eos_source_pos]
                        if int(eos_candidate.item()) >= 0:
                            eos_token = eos_candidate.clone()

                    extra_start_pos = last_placeholder_pos + 1
                    eos_reserved = 1 if eos_token is not None else 0
                    extra_capacity = max(0, input_ids_audio.size(1) - extra_start_pos - eos_reserved)
                    extra_needed = num_tokens - num_placeholders
                    extra_usable = min(extra_needed, extra_capacity)
                    if extra_usable > 0:
                        extra_positions = torch.arange(
                            extra_start_pos,
                            extra_start_pos + extra_usable,
                            device=input_ids_audio.device,
                            dtype=placeholder_positions.dtype,
                        )
                        token_positions = torch.cat([placeholder_positions, extra_positions], dim=0)
                    else:
                        token_positions = placeholder_positions

                    usable_tokens = min(num_tokens, int(token_positions.numel()))
                    if usable_tokens > 0:
                        input_ids_audio[row_idx, token_positions[:usable_tokens]] = replacement_tokens[:usable_tokens]
                        if labels_ids_audio is not None:
                            labels_ids_audio[row_idx, token_positions[:usable_tokens]] = -50

                    eos_target_pos = last_placeholder_pos + usable_tokens - num_placeholders + 1
                    if eos_token is not None and eos_target_pos < input_ids_audio.size(1):
                        input_ids_audio[row_idx, eos_target_pos] = eos_token
                        labels_ids_audio[row_idx, eos_target_pos] = eos_token

                    if usable_tokens < num_tokens:
                        print(
                            '[MODEL FORWARD] input_ids_audio extra tokens exceed sequence capacity, '
                            f'row={row_idx}, kept={usable_tokens}, tokens={num_tokens}, '
                            f'dropped={num_tokens - usable_tokens}'
                        )
                    return

                usable_tokens = min(num_placeholders, num_tokens)
                if usable_tokens > 0:
                    input_ids_audio[row_idx, placeholder_positions[:usable_tokens]] = replacement_tokens[:usable_tokens]
                if usable_tokens < num_placeholders:
                    ignore_start = usable_tokens
                    if labels_ids_audio is not None and usable_tokens > 0:
                        eos_source_pos = int(placeholder_positions[-1].item())
                        if eos_source_pos < input_ids_audio.size(1):
                            eos_token = labels_ids_audio[row_idx, eos_source_pos]
                            if int(eos_token.item()) >= 0:
                                eos_target_pos = placeholder_positions[usable_tokens]
                                input_ids_audio[row_idx, eos_target_pos] = eos_token
                                labels_ids_audio[row_idx, eos_target_pos] = eos_token
                                if eos_source_pos != int(eos_target_pos.item()):
                                    labels_ids_audio[row_idx, eos_source_pos] = -100
                                    input_ids_audio[row_idx, eos_source_pos] = self.mask_token_id
                                ignore_start = usable_tokens + 1
                    input_ids_audio[row_idx, placeholder_positions[ignore_start:]] = self.mask_token_id
                    if labels_ids_audio is not None:
                        labels_ids_audio[row_idx, placeholder_positions[ignore_start:]] = -100

            def _replace_streaming_audio_placeholders_by_slice(
                row_idx, replacement_chunks
            ):
                """Replace each online-audio placeholder run independently.

                Training examples reserve each online slice from its expected
                duration, while the runtime tokenizer can return a few fewer
                codec tokens.  A target run therefore moves only *its own*
                PAUSE/EOS forward; it must never consume positions belonging
                to another envelope.  Source-audio runs have no boundary and
                simply zero their unused tail.
                """
                placeholder_positions = torch.nonzero(
                    input_ids_audio[row_idx] == -50, as_tuple=False
                ).view(-1)
                runs = []
                if int(placeholder_positions.numel()) > 0:
                    run_start = 0
                    for split_idx in range(1, int(placeholder_positions.numel())):
                        if int(placeholder_positions[split_idx].item()) != int(
                            placeholder_positions[split_idx - 1].item()
                        ) + 1:
                            runs.append(placeholder_positions[run_start:split_idx])
                            run_start = split_idx
                    runs.append(placeholder_positions[run_start:])

                if len(runs) != len(replacement_chunks):
                    raise RuntimeError(
                        "streaming per-slice placeholder mismatch: "
                        f"row={row_idx} runs={len(runs)} "
                        f"token_chunks={len(replacement_chunks)} "
                        f"run_lengths={[int(run.numel()) for run in runs]} "
                        f"token_lengths={[int(chunk.numel()) for chunk in replacement_chunks]}"
                    )

                codec_pause = int(
                    os.environ.get("STREAMING_S2ST_CODEC_PAUSE_TOKEN_ID", "59051")
                )
                codec_eos = int(
                    os.environ.get("STREAMING_S2ST_CODEC_EOS_TOKEN_ID", "59050")
                )
                for run, chunk in zip(runs, replacement_chunks):
                    chunk = chunk.long().to(input_ids_audio.device)
                    reserved = int(run.numel())
                    actual = int(chunk.numel())
                    run_begin = int(run[0].item())
                    run_end = int(run[-1].item())
                    if actual > reserved:
                        raise RuntimeError(
                            "streaming audio tokenizer exceeded its per-slice reservation: "
                            f"row={row_idx} begin={run_begin} "
                            f"reserved={reserved} actual={actual}"
                        )

                    boundary = None
                    if labels_ids_audio is not None:
                        candidate = int(labels_ids_audio[row_idx, run_end].item())
                        if candidate in (codec_pause, codec_eos):
                            boundary = candidate
                    if boundary is not None and actual == 0:
                        raise RuntimeError(
                            "target codec tokenizer returned an empty segment: "
                            f"row={row_idx} begin={run_begin} boundary={boundary}"
                        )

                    if actual > 0:
                        input_ids_audio[row_idx, run[:actual]] = chunk
                    if actual == reserved:
                        continue

                    # Unused positions are alignment gaps: no codec embedding
                    # and no codec loss.  For a target run, move the original
                    # boundary immediately after the last real codec token.
                    input_ids_audio[row_idx, run[actual:]] = self.mask_token_id
                    if labels_ids_audio is not None:
                        labels_ids_audio[row_idx, run[actual:]] = -100
                    if boundary is None:
                        continue

                    boundary_target = run_begin + actual
                    original_boundary_input = run_end + 1
                    if original_boundary_input >= input_ids_audio.size(1):
                        raise RuntimeError(
                            "streaming target boundary falls outside sequence: "
                            f"row={row_idx} position={original_boundary_input}"
                        )
                    original_token = int(
                        input_ids_audio[row_idx, original_boundary_input].item()
                    )
                    if original_token != boundary:
                        raise RuntimeError(
                            "streaming target boundary input mismatch: "
                            f"row={row_idx} position={original_boundary_input} "
                            f"input={original_token} label={boundary}"
                        )
                    input_ids_audio[row_idx, boundary_target] = boundary
                    labels_ids_audio[row_idx, boundary_target] = boundary
                    input_ids_audio[row_idx, original_boundary_input] = self.mask_token_id
                    labels_ids_audio[row_idx, original_boundary_input] = -100

            for i in range(input_ids_audio.size(0)):
                # Slice isolation is required for both the historical single-BOS
                # envelope and dual-BOS training. Flattening a packed row lets
                # a short extraction shift every later slice/sample's codec.
                streaming_per_slice_replace = any(
                    os.environ.get(flag, "0").lower() in ("1", "true", "yes", "on")
                    for flag in (
                        "STREAMING_S2ST_CODEC_ENVELOPE",
                        "STREAMING_S2ST_INJECTED_DUAL_BOS",
                    )
                )
                if streaming_per_slice_replace:
                    if speech_token_nums_per_sequence_chunk is not None and len(
                        speech_token_nums_per_sequence_chunk[i]
                    ) != 1:
                        raise RuntimeError(
                            "streaming per-slice codec replacement requires CP=1"
                        )
                    _replace_streaming_audio_placeholders_by_slice(
                        i, slice_speech_token_chunks[i]
                    )
                elif speech_token_nums_per_sequence_chunk is not None:
                    # for cp, calculate the mask for speech tokens for this rank
                    cp_indexes_on_this_rank = []
                    for chunk in sequence_chunks:
                        cp_indexes_on_this_rank.append(chunk['index'])
                    slice_speech_tokens_mask = torch.zeros_like(slice_speech_tokens[i], dtype=torch.bool)
                    offset = 0
                    speech_token_nums = speech_token_nums_per_sequence_chunk[i]
                    for cp_index in range(len(speech_token_nums)):
                        token_num = speech_token_nums[cp_index]
                        if cp_index in cp_indexes_on_this_rank:
                            slice_speech_tokens_mask[offset:offset+token_num] = True
                        offset += token_num
                    _replace_audio_placeholders(i, slice_speech_tokens[i][slice_speech_tokens_mask])
                # # audio_placeholder
                # if len(input_ids[i][input_ids[i] == -50]) != len(slice_speech_tokens[i]):
                #     print (f'[MODEL FORWARD] input_ids[i][input_ids[i] == -50] length not match, {len(input_ids[i][input_ids[i] == -50])} vs {len(slice_speech_tokens[i])}, total input length: {len(input_ids[i])}')
                #     print (f'[MODEL FORWARD] item: {items[i]}')
                #     input_ids[i][input_ids[i] == -50] = 0
                else:
                    _replace_audio_placeholders(i, slice_speech_tokens[i])
            if self.config.timers is not None:
                self.config.timers('forward-audio-tokenize-insert', log_level=2).stop()
        else:
            self.config.timers('forward-audio-tokenize-extract', log_level=2).start()
            self.config.timers('forward-audio-tokenize-extract', log_level=2).stop()
            self.config.timers('forward-audio-tokenize-insert', log_level=2).start()
            self.config.timers('forward-audio-tokenize-insert', log_level=2).stop()

        if input_ids_audio is not None and labels_ids_audio is not None:
            labels_ids_audio = self._shift_cosyvoice_audio_labels(input_ids_audio, labels_ids_audio)
            
        # input_ids_audio_mask = input_ids_audio == self.mask_token_id
        # input_ids_audio[input_ids_audio < 0] = 0
        # inputs_embeds_audio = self.embedding_audio(input_ids=input_ids_audio, position_ids=None)
        # inputs_embeds_audio = inputs_embeds_audio.permute(1, 0, 2)
        # mask = input_ids_audio_mask.unsqueeze(-1)
        # inputs_embeds_audio = inputs_embeds_audio.masked_fill(mask, 0)
        input_ids_audio_for_cosyvoice = (
            input_ids_audio.clone()
            if self.has_cosyvoice_codex_head and input_ids_audio is not None
            else None
        )

        if self.pre_process:
            # Match original implementation exactly
            batch_size, token_num = input_ids.shape

            # input_ids_mask = input_ids == self.mask_token_id
            # input_ids[input_ids < 0] = 0
            # inputs_embeds = self.embedding(input_ids=input_ids, position_ids=position_ids)
            # inputs_embeds = inputs_embeds.permute(1, 0, 2)
            # batch_size, _, dims = inputs_embeds.shape
            # mask = input_ids_mask.unsqueeze(-1)
            # inputs_embeds = inputs_embeds.masked_fill(mask, 0)

            input_ids_mask = input_ids == self.mask_token_id
            input_ids[input_ids < 0] = 0
            inputs_embeds = self.embedding(input_ids=input_ids, position_ids=position_ids)
            inputs_embeds = inputs_embeds.permute(1, 0, 2)
            # Preserve the non-zero embedding at alignment placeholders.  Some
            # text-channel -60 positions are used only to align target-audio
            # tokens and are therefore not covered by source-speech features.
            inputs_embeds_unmasked = inputs_embeds
            for chunk in sequence_chunks:
                chunk_begin = chunk['begin']
                chunk_length = chunk['length']
                chunk_offset = chunk['offset']
                chunk_end = chunk_offset + chunk_length
                input_ids_mask_chunk = input_ids_mask[:, chunk_offset:chunk_end]
                # inputs_embeds_chunk = inputs_embeds[:, chunk_begin-chunk_offset:chunk_end-chunk_offset, :]
                inputs_embeds_chunk = inputs_embeds
                mask = input_ids_mask_chunk.unsqueeze(-1)
                inputs_embeds_chunk = inputs_embeds_chunk.masked_fill(mask, 0)
                # inputs_embeds[:, chunk_begin-chunk_offset:chunk_end-chunk_offset, :] = inputs_embeds_chunk
                inputs_embeds = inputs_embeds_chunk

            # b, s — audio token embedding (only when audio output head is enabled)
            if self.has_audio_output_head:
                input_ids_audio_mask = input_ids_audio == self.mask_token_id
                input_ids_audio[input_ids_audio < 0] = 0
                inputs_embeds_audio = self.embedding_audio(input_ids=input_ids_audio, position_ids=None)
                inputs_embeds_audio = inputs_embeds_audio.permute(1, 0, 2)
                for chunk in sequence_chunks:
                    chunk_begin = chunk['begin']
                    chunk_length = chunk['length']
                    chunk_offset = chunk['offset']
                    chunk_end = chunk_offset + chunk_length
                    input_ids_mask_chunk = input_ids_audio_mask[:, chunk_offset:chunk_end]
                    # inputs_embeds_audio_chunk = inputs_embeds_audio[:,chunk_begin-chunk_offset:chunk_end-chunk_offset, :]
                    inputs_embeds_audio_chunk = inputs_embeds_audio
                    mask = input_ids_mask_chunk.unsqueeze(-1)
                    inputs_embeds_audio_chunk = inputs_embeds_audio_chunk.masked_fill(mask, 0)
                    # inputs_embeds_audio[:, chunk_begin-chunk_offset:chunk_end-chunk_offset, :] = inputs_embeds_audio_chunk
                    inputs_embeds_audio = inputs_embeds_audio_chunk

            # Process audio if provided - matching original logic exactly
            if self.audio_encoder is not None and self.audio_adaptor is not None:
                if speech is not None:
                    if self.config.timers is not None:
                        self.config.timers('forward-audio-embedding-encode-real', log_level=2).start()
                    if len(speech_lengths.size()) > 1:
                        speech_lengths = speech_lengths[:, 0]

                    batch_size_speech, frames, _ = speech.shape
                    encoder_out, encoder_out_lens = self.forward_encoder_and_adaptor(speech, speech_lengths)
                    acoustic_audio_emb = encoder_out # slices, tokens, features
                    loss_fake = encoder_out.mean()
                    if self.config.timers is not None:
                        self.config.timers('forward-audio-embedding-encode-real', log_level=2).stop()
                        self.config.timers('forward-audio-embedding-encode-fake', log_level=2).start()
                        self.config.timers('forward-audio-embedding-encode-fake', log_level=2).stop()
                else:
                    if self.config.timers is not None:
                        self.config.timers('forward-audio-embedding-encode-fake', log_level=2).start()
                    batch_size_speech = 1
                    frames = 64
                    feature_dim = 560 if self.audio_config.encoder_type == "sensevoicesmall" else 128
                    speech = torch.zeros(batch_size_speech, frames, feature_dim).to(inputs_embeds.device)
                    speech_lengths = torch.tensor([frames], dtype=torch.int32).to(inputs_embeds.device)
                    encoder_out, encoder_out_lens = self.forward_encoder_and_adaptor(speech, speech_lengths)
                    acoustic_audio_emb = encoder_out
                    loss_fake = encoder_out.mean()
                    if self.config.timers is not None:
                        self.config.timers('forward-audio-embedding-encode-fake', log_level=2).stop()
                        self.config.timers('forward-audio-embedding-encode-real', log_level=2).start()
                        self.config.timers('forward-audio-embedding-encode-real', log_level=2).stop()
            else:
                loss_fake = 0.0

            if self.config.timers is not None:
                self.config.timers('forward-audio-embedding-insert', log_level=2).start()
            # tmp permute for audio feature insertion
            # inputs_embeds = inputs_embeds.permute(1, 0, 2)
            # batch_size, _, dims = inputs_embeds.shape
            speech_idx = 0
            speech_token_full = torch.zeros_like(inputs_embeds)
            speech_token_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
            for batch_idx in range(batch_size):
                for feature_idx in range(feature_beg.shape[1]):
                    feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                    if feature_beg_id > 0:
                        feature_length = fake_token_len[batch_idx, feature_idx].item()
                        # audio
                        if feature_type[batch_idx, feature_idx] == 1:
                            for chunk in sequence_chunks:
                                chunk_begin = chunk['begin']
                                chunk_length = chunk['length']
                                chunk_offset = chunk['offset']
                                chunk_end = chunk_begin + chunk_length

                                if feature_beg_id > chunk_end or feature_beg_id + feature_length < chunk_begin:
                                    # no intersection
                                    continue
                                feature_begin_to_chunk = feature_beg_id - chunk_begin
                                feature_end_to_chunk = feature_begin_to_chunk + feature_length
                                if feature_begin_to_chunk < 0:
                                    feature_begin_to_chunk = 0
                                    speech_token_begin = -feature_begin_to_chunk
                                else:
                                    speech_token_begin = 0
                                feature_end_to_chunk = min(feature_end_to_chunk, chunk_length)
                                feature_length_clipped = feature_end_to_chunk - feature_begin_to_chunk

                                speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]

                                input_embeds_begin = feature_begin_to_chunk + chunk_offset
                                assert input_embeds_begin >= 0, f'input_embeds_begin < 0, {input_embeds_begin}'
                                input_embeds_end = input_embeds_begin + feature_length_clipped
                                assert input_embeds_end >= 0, f'input_embeds_end < 0, {input_embeds_end}'

                                try:
                                    # inputs_embeds[
                                    # batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                                    # ] = speech_token
                                    
                                    # fix inplace error
                                    speech_token_full[batch_idx, input_embeds_begin:input_embeds_end, :] = speech_token[speech_token_begin:speech_token_begin+feature_length_clipped, :]
                                    speech_token_mask[batch_idx, input_embeds_begin:input_embeds_end, :] = True
                                except Exception as e:
                                    logging.error(f"{str(e)}, {traceback.format_exc()}")
                                    logging.info(
                                        f"assign speech token failed. batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                                    )
                                    feature_length = encoder_out_lens[speech_idx].item()
                                    speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]
                                    inputs_embeds[
                                        batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                                    ] = speech_token
                            speech_idx += 1
            # Bounded diagnostics around the speech-embedding splice. Set
            # SPEECH_WHERE_DEBUG_STEPS=0 to disable (default: first 4 forwards).
            speech_where_debug_limit = int(os.environ.get("SPEECH_WHERE_DEBUG_STEPS", "4"))
            speech_where_debug_step = getattr(self, "_speech_where_debug_step", 0)
            speech_where_debug = speech_where_debug_step < speech_where_debug_limit

            def _speech_where_tensor_stats(tensor):
                detached = tensor.detach().float()
                finite_values = detached[torch.isfinite(detached)]
                return {
                    "shape": tuple(detached.shape),
                    "dtype": str(tensor.dtype),
                    "numel": detached.numel(),
                    "nan": int(torch.isnan(detached).sum().item()),
                    "inf": int(torch.isinf(detached).sum().item()),
                    "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
                    "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
                    "finite_mean": float(finite_values.mean().item()) if finite_values.numel() else None,
                    "finite_abs_max": float(finite_values.abs().max().item()) if finite_values.numel() else None,
                }

            if speech_where_debug:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                # Each selected [batch, sequence] position should select all H
                # dimensions. A partial position indicates a malformed mask.
                speech_position_mask = speech_token_mask.any(dim=-1)
                speech_position_mask_all = speech_token_mask.all(dim=-1)
                partial_hidden_mask_positions = int(
                    (speech_position_mask != speech_position_mask_all).sum().item()
                )
                selected_speech = speech_token_full[speech_position_mask]
                if selected_speech.numel():
                    selected_f32 = selected_speech.detach().float()
                    selected_norms = torch.linalg.vector_norm(selected_f32, dim=-1)
                    norm_finite = selected_norms[torch.isfinite(selected_norms)]
                    selected_summary = {
                        "vectors": int(selected_f32.shape[0]),
                        "all_zero_vectors": int((selected_f32 == 0).all(dim=-1).sum().item()),
                        "near_zero_norm_vectors": int((selected_norms < 1.0e-8).sum().item()),
                        "nonfinite_vectors": int((~torch.isfinite(selected_f32).all(dim=-1)).sum().item()),
                        "norm_min": float(norm_finite.min().item()) if norm_finite.numel() else None,
                        "norm_max": float(norm_finite.max().item()) if norm_finite.numel() else None,
                        "norm_mean": float(norm_finite.mean().item()) if norm_finite.numel() else None,
                    }
                else:
                    selected_summary = {
                        "vectors": 0, "all_zero_vectors": 0,
                        "near_zero_norm_vectors": 0, "nonfinite_vectors": 0,
                        "norm_min": None, "norm_max": None, "norm_mean": None,
                    }
                logging.info(
                    "[SPEECH WHERE FORWARD] rank=%s step=%s mask_true_elements=%s "
                    "mask_positions=%s partial_hidden_mask_positions=%s selected=%s "
                    "speech_token_full=%s original_inputs_embeds=%s",
                    rank, speech_where_debug_step,
                    int(speech_token_mask.sum().item()),
                    int(speech_position_mask.sum().item()),
                    partial_hidden_mask_positions, selected_summary,
                    _speech_where_tensor_stats(speech_token_full),
                    _speech_where_tensor_stats(inputs_embeds),
                )

            inputs_embeds = torch.where(speech_token_mask, speech_token_full, inputs_embeds)

            if speech_where_debug:
                # Diagnose positions cleared by input_ids == mask_token_id but not
                # refilled by a source-speech feature.  These exact-zero rows are
                # numerically dangerous for RMSNorm backward.
                speech_position_mask = speech_token_mask.any(dim=-1)
                zero_position_mask = (inputs_embeds.detach().float() == 0).all(dim=-1)
                # input_ids/input_ids_mask keep the global packed sequence, while
                # the embedding on this TP/SP rank is local.  Reuse the exact
                # local mask applied by masked_fill above.
                local_input_ids_mask = input_ids_mask_chunk
                if local_input_ids_mask.shape != speech_position_mask.shape:
                    logging.warning(
                        "[SPEECH MASK GAP] rank=%s step=%s skip incompatible masks: "
                        "global_input_mask=%s local_input_mask=%s speech_mask=%s",
                        rank, speech_where_debug_step, tuple(input_ids_mask.shape),
                        tuple(local_input_ids_mask.shape), tuple(speech_position_mask.shape),
                    )
                    local_input_ids_mask = torch.zeros_like(speech_position_mask)
                masked_unfilled = local_input_ids_mask & ~speech_position_mask
                zero_from_mask = zero_position_mask & local_input_ids_mask
                zero_without_mask = zero_position_mask & ~local_input_ids_mask

                def _true_ranges(row_mask, limit=16):
                    positions = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
                    if not positions:
                        return []
                    ranges = []
                    begin = previous = positions[0]
                    for position in positions[1:]:
                        if position != previous + 1:
                            ranges.append((begin, previous + 1))
                            if len(ranges) >= limit:
                                return ranges
                            begin = position
                        previous = position
                    ranges.append((begin, previous + 1))
                    return ranges[:limit]

                per_batch = []
                for debug_batch_idx in range(batch_size):
                    supervised_zero = 0
                    if labels_ids is not None and labels_ids.shape[:2] == zero_position_mask.shape:
                        supervised_zero = int(
                            (zero_position_mask[debug_batch_idx]
                             & (labels_ids[debug_batch_idx] >= 0)).sum().item()
                        )
                    per_batch.append({
                        "batch": debug_batch_idx,
                        "input_mask": int(local_input_ids_mask[debug_batch_idx].sum().item()),
                        "speech_filled": int(speech_position_mask[debug_batch_idx].sum().item()),
                        "masked_unfilled": int(masked_unfilled[debug_batch_idx].sum().item()),
                        "zero_total": int(zero_position_mask[debug_batch_idx].sum().item()),
                        "zero_from_input_mask": int(zero_from_mask[debug_batch_idx].sum().item()),
                        "zero_without_input_mask": int(zero_without_mask[debug_batch_idx].sum().item()),
                        "supervised_zero": supervised_zero,
                        "masked_unfilled_ranges": _true_ranges(masked_unfilled[debug_batch_idx]),
                        "zero_ranges": _true_ranges(zero_position_mask[debug_batch_idx]),
                    })

                feature_rows = []
                encoder_idx = 0
                for debug_batch_idx in range(batch_size):
                    for debug_feature_idx in range(feature_beg.shape[1]):
                        debug_beg = int(feature_beg[debug_batch_idx, debug_feature_idx].item())
                        debug_type = int(feature_type[debug_batch_idx, debug_feature_idx].item())
                        debug_fake_len = int(fake_token_len[debug_batch_idx, debug_feature_idx].item())
                        if debug_beg <= 0:
                            continue
                        debug_encoder_len = None
                        if debug_type == 1:
                            if 'encoder_out_lens' in locals() and encoder_idx < int(encoder_out_lens.numel()):
                                debug_encoder_len = int(encoder_out_lens[encoder_idx].item())
                            encoder_idx += 1
                        feature_rows.append({
                            "batch": debug_batch_idx,
                            "feature": debug_feature_idx,
                            "type": debug_type,
                            "beg": debug_beg,
                            "fake_len": debug_fake_len,
                            "encoder_len": debug_encoder_len,
                            "length_delta": (
                                None if debug_encoder_len is None
                                else debug_fake_len - debug_encoder_len
                            ),
                        })

                audio_mask_count = None
                if input_ids_audio is not None:
                    audio_mask_count = int((input_ids_audio == self.mask_token_id).sum().item())
                logging.info(
                    "[SPEECH MASK GAP] rank=%s step=%s has_audio_output_head=%s "
                    "audio_channel_mask_positions=%s chunks=%s per_batch=%s features=%s",
                    rank, speech_where_debug_step, self.has_audio_output_head,
                    audio_mask_count, sequence_chunks, per_batch, feature_rows[:32],
                )
                logging.info(
                    "[SPEECH WHERE OUTPUT] rank=%s step=%s output=%s",
                    rank, speech_where_debug_step,
                    _speech_where_tensor_stats(inputs_embeds),
                )

                def _speech_where_grad_hook(name, step, hook_rank):
                    def hook(grad):
                        logging.info(
                            "[SPEECH WHERE BACKWARD] rank=%s step=%s tensor=%s grad=%s",
                            hook_rank, step, name, _speech_where_tensor_stats(grad),
                        )
                        return grad
                    return hook

                if inputs_embeds.requires_grad:
                    inputs_embeds.register_hook(_speech_where_grad_hook(
                        "where_output_inputs_embeds", speech_where_debug_step, rank
                    ))
                if speech_token_full.requires_grad:
                    speech_token_full.register_hook(_speech_where_grad_hook(
                        "speech_token_full", speech_where_debug_step, rank
                    ))
                if acoustic_audio_emb.requires_grad:
                    acoustic_audio_emb.register_hook(_speech_where_grad_hook(
                        "acoustic_audio_emb", speech_where_debug_step, rank
                    ))
                self._speech_where_debug_step = speech_where_debug_step + 1
            if self.has_audio_output_head:
                inputs_embeds = inputs_embeds + inputs_embeds_audio

            # Positions masked out in the text channel and not covered by a
            # speech/audio embedding otherwise remain exact zero vectors.  Such
            # rows make RMSNorm backward numerically unstable.  Refill only the
            # final exact-zero rows with their original non-zero embedding.
            zero_pos = inputs_embeds.norm(dim=-1) == 0
            if zero_pos.any():
                inputs_embeds = torch.where(
                    zero_pos.unsqueeze(-1), inputs_embeds_unmasked, inputs_embeds
                )
                if speech_where_debug:
                    logging.info(
                        "[ZERO POS REFILL] rank=%s step=%s refilled=%s emb_shape=%s",
                        rank, speech_where_debug_step, int(zero_pos.sum().item()),
                        tuple(inputs_embeds.shape),
                    )
            inputs_embeds = inputs_embeds.permute(1, 0, 2)
            if self.config.timers is not None:
                self.config.timers('forward-audio-embedding-insert', log_level=2).stop()

            if labels_ids is not None:
                labels_ids = torch.where(labels_ids == -1, torch.tensor(-100, device=labels_ids.device, dtype=labels_ids.dtype), labels_ids)
            if attention_mask is not None:
                attention_mask = torch.where(attention_mask < 0, torch.tensor(0, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask)
            
            # # shift inputs_embeds, labels_ids, attention_mask, position_ids
            # inputs_embeds = inputs_embeds[:, :-1, :]
            # labels_ids = labels_ids[:, 1:]
            # attention_mask = attention_mask[:, :-1]
            # position_ids = position_ids[:, :-1]

            # model_outputs = self.lm_model(
            #     input_ids=None,
            #     position_ids=position_ids,
            #     attention_mask=attention_mask,
            #     decoder_input=inputs_embeds.to(dtype_map[self.llm_dtype]) if inputs_embeds is not None else None,
            #     labels=labels_ids,
            #     inference_params=inference_params,
            #     packed_seq_params=packed_seq_params,
            #     **(extra_block_kwargs or {}),
            # )
        else:
            self.config.timers('forward-audio-embedding-encode-real', log_level=2).start()
            self.config.timers('forward-audio-embedding-encode-real', log_level=2).stop()
            self.config.timers('forward-audio-embedding-encode-fake', log_level=2).start()
            self.config.timers('forward-audio-embedding-encode-fake', log_level=2).stop()
            self.config.timers('forward-audio-embedding-insert', log_level=2).start()
            self.config.timers('forward-audio-embedding-insert', log_level=2).stop()

        # GPTModel forward
        if self.config.timers is not None:
            self.config.timers('forward-gpt-model-forward', log_level=2).start()
        # model_outputs = super().forward(
        #     input_ids=None,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     decoder_input=inputs_embeds.to(dtype_map[self.llm_dtype]) if inputs_embeds is not None else None,
        #     labels=None,
        #     inference_params=inference_params,
        #     packed_seq_params=packed_seq_params,
        #     **(extra_block_kwargs or {}),
        # )

        # copy GPTModel forward
        runtime_gather_output = None
        decoder_input = inputs_embeds.to(dtype_map[self.llm_dtype]) if inputs_embeds is not None else None
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            if self.config.mrope_interleaved:
                # Qwen3.5-style interleaved M-RoPE: requires position_ids tensor
                rotary_pos_emb = self.rotary_pos_emb(
                    position_ids, mrope_section=self.mrope_section
                )
            elif self.config.use_chunkflow or self.config.use_smart_batching:
                # ChunkFlow: calculate rotary_pos_emb on packed sequences
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, position_ids=position_ids.flatten())
            else:
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        if loss_fake is not None:
            decoder_input += loss_fake * 0.0

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        if self.config.timers is not None:
            self.config.timers('forward-gpt-model-forward', log_level=2).stop()

        if not self.post_process:
            # pp non-last stage, pass extra_outputs to next stage
            extra_outputs = [labels_ids, labels_ids_audio]
            if self.cosyvoice_codex_use_text_token_input:
                extra_outputs.append(input_ids_codec_text)
            extra_outputs.extend([loss_mask, loss_mask_audio])
            if mpu.get_context_parallel_world_size() > 1:
                cu_seqlens_padded = pad_cu_seqlens(packed_seq_params["cu_seqlens_q"], labels_ids.shape[0], labels_ids.shape[1])
                extra_outputs.extend([cu_seqlens_padded, packed_seq_params["max_seqlen_q"]])
            extra_outputs = tuple(extra_outputs)
            assert len(extra_outputs) == len(self.extra_output_specs)
            return hidden_states, extra_outputs

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_text, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        logits_audio = None
        if self.has_audio_output_head and self.audio_loss_weight != 0.0:
            output_weight_audio = None
            # if self.share_embeddings_and_output_weights:
            if False:
                output_weight_audio = self.shared_embedding_or_output_weight_audio()
            logits_audio, _ = self.output_layer_audio(hidden_states, weight=output_weight_audio, runtime_gather_output=runtime_gather_output)

        if self.has_cosyvoice_codex_head and self.post_process:
            if self.cosyvoice_codex_use_text_token_input:
                if input_ids_codec_text is None:
                    raise ValueError(
                        "--cosyvoice-codex-use-text-token-input requires "
                        "input_ids_codec_text from the training data pipeline"
                    )
                input_ids_codec_text_local = input_ids_codec_text
                if input_ids_codec_text_local.shape[1] != hidden_states.shape[0]:
                    if (
                        self.config.sequence_parallel
                        and mpu.get_tensor_model_parallel_world_size() > 1
                    ):
                        input_ids_codec_text_local = tensor_parallel.scatter_to_sequence_parallel_region(
                            input_ids_codec_text_local.transpose(0, 1).contiguous()
                        ).transpose(0, 1).contiguous()
                    if input_ids_codec_text_local.shape[1] != hidden_states.shape[0]:
                        input_ids_codec_text_local = input_ids_codec_text_local[
                            :, : hidden_states.shape[0]
                        ]
                text_embedding = self.cosyvoice_llm.embed_tokens
                text_vocab_size = int(text_embedding.num_embeddings)
                cosyvoice_text_valid = (
                    (input_ids_codec_text_local >= 0)
                    & (input_ids_codec_text_local < text_vocab_size)
                )
                cosyvoice_text_input_ids = torch.where(
                    cosyvoice_text_valid,
                    input_ids_codec_text_local,
                    torch.zeros_like(input_ids_codec_text_local),
                )
                cosyvoice_llm_inputs = text_embedding(cosyvoice_text_input_ids.long())
                cosyvoice_llm_inputs = cosyvoice_llm_inputs * cosyvoice_text_valid.unsqueeze(
                    -1
                ).to(cosyvoice_llm_inputs.dtype)
                cosyvoice_llm_inputs = cosyvoice_llm_inputs.transpose(0, 1).contiguous()
                if not self._cosyvoice_text_token_debug_logged:
                    self._cosyvoice_text_token_debug_logged = True
                    supervised_text = input_ids_codec_text_local >= 0
                    logging.info(
                        "[CODEX TEXT TOKEN INPUT] labels_shape=%s text_vocab_size=%s "
                        "supervised_tokens=%s valid_tokens=%s out_of_vocab_tokens=%s",
                        tuple(input_ids_codec_text_local.shape),
                        text_vocab_size,
                        int(supervised_text.sum().detach().cpu()),
                        int(cosyvoice_text_valid.sum().detach().cpu()),
                        int((supervised_text & ~cosyvoice_text_valid).sum().detach().cpu()),
                    )
            else:
                cosyvoice_llm_inputs = self.cosyvoice_llm_input_proj(hidden_states)
            if input_ids_audio_for_cosyvoice is not None:
                if input_ids_audio_for_cosyvoice.shape[1] != cosyvoice_llm_inputs.shape[0]:
                    if (
                        self.config.sequence_parallel
                        and mpu.get_tensor_model_parallel_world_size() > 1
                    ):
                        input_ids_audio_for_cosyvoice = tensor_parallel.scatter_to_sequence_parallel_region(
                            input_ids_audio_for_cosyvoice.transpose(0, 1).contiguous()
                        ).transpose(0, 1).contiguous()
                    if input_ids_audio_for_cosyvoice.shape[1] != cosyvoice_llm_inputs.shape[0]:
                        input_ids_audio_for_cosyvoice = input_ids_audio_for_cosyvoice[
                            :, : cosyvoice_llm_inputs.shape[0]
                        ]
                cosyvoice_codec_valid = (
                    (input_ids_audio_for_cosyvoice >= 0)
                    & (input_ids_audio_for_cosyvoice < int(self.cosyvoice_codec_unit))
                )
                cosyvoice_codec_input_ids = torch.where(
                    cosyvoice_codec_valid,
                    input_ids_audio_for_cosyvoice,
                    torch.zeros_like(input_ids_audio_for_cosyvoice),
                )
                cosyvoice_codec_embeds = self.cosyvoice_codec_embed(
                    cosyvoice_codec_input_ids.long()
                )
                cosyvoice_codec_embeds = cosyvoice_codec_embeds * cosyvoice_codec_valid.unsqueeze(
                    -1
                ).to(cosyvoice_codec_embeds.dtype)
                cosyvoice_llm_inputs = cosyvoice_llm_inputs + cosyvoice_codec_embeds.transpose(
                    0, 1
                ).to(cosyvoice_llm_inputs.dtype)
            cosyvoice_llm_inputs = cosyvoice_llm_inputs + self.cosyvoice_codec_embed.weight.sum() * 0.0
            # HF Qwen2 expects [batch, sequence, hidden], while Megatron uses [sequence, batch, hidden].
            cosyvoice_llm_inputs = cosyvoice_llm_inputs.transpose(0, 1).contiguous()
            cosyvoice_attention_mask = None
            if attention_mask is not None and attention_mask.dim() == 2:
                cosyvoice_attention_mask = attention_mask
            cosyvoice_llm_outputs = self.cosyvoice_llm(
                inputs_embeds=cosyvoice_llm_inputs.to(dtype_map[self.cosyvoice_llm_dtype]),
                attention_mask=cosyvoice_attention_mask,
            )
            cosyvoice_hidden_states = cosyvoice_llm_outputs[0].transpose(0, 1).contiguous()
            logits_cosyvoice_codex, _ = self.cosyvoice_llm_codec_head(
                cosyvoice_hidden_states, runtime_gather_output=runtime_gather_output
            )
            if not self._cosyvoice_codex_forward_debug_logged:
                self._cosyvoice_codex_forward_debug_logged = True
                try:
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

                    def _tensor_stats(name, tensor):
                        tensor_f = tensor.detach().float()
                        finite = torch.isfinite(tensor_f)
                        finite_tensor = torch.nan_to_num(
                            tensor_f, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        return (
                            f"{name}:shape={tuple(tensor.shape)},dtype={tensor.dtype},"
                            f"finite={bool(finite.all().detach().cpu())},"
                            f"min={float(finite_tensor.min().detach().cpu())},"
                            f"max={float(finite_tensor.max().detach().cpu())},"
                            f"mean={float(finite_tensor.mean().detach().cpu())}"
                        )

                    logging.info(
                        "[CODEX FORWARD DEBUG] rank=%s %s | %s | %s | %s",
                        rank,
                        _tensor_stats("main_hidden", hidden_states),
                        _tensor_stats("cosyvoice_inputs", cosyvoice_llm_inputs),
                        _tensor_stats("cosyvoice_hidden", cosyvoice_hidden_states),
                        _tensor_stats("cosyvoice_logits", logits_cosyvoice_codex),
                    )
                except Exception as exc:
                    logging.info("[CODEX FORWARD DEBUG] failed to log tensor stats: %s", exc)

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits_text,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        is_all_minus_100 = (labels_ids == -100).all() 
        if not is_all_minus_100:
            text_loss = self.compute_language_model_loss(labels_ids, logits_text)
        else:
            logits_text_mean = torch.nan_to_num(
                logits_text.mean(dim=-1), 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
            )
            text_loss = logits_text_mean * 0.0

        if self.has_audio_output_head and logits_audio is not None:
            is_all_minus_100_audio = (labels_ids_audio == -100).all()
            if not is_all_minus_100_audio:
                audio_loss = self.compute_language_model_loss(labels_ids_audio, logits_audio)
            else:
                logits_audio_mean = torch.nan_to_num(
                    logits_audio.mean(dim=-1), 
                    nan=0.0, 
                    posinf=0.0, 
                    neginf=0.0
                )
                audio_loss = logits_audio_mean * 0.0
        elif self.has_audio_output_head:
            audio_loss = text_loss * 0.0 + self.output_layer_audio.weight.sum() * 0.0
        else:
            audio_loss = text_loss * 0.0

        cosyvoice_codex_loss = None
        cosyvoice_codex_loss_mask = None
        cosyvoice_fm_loss = None
        if self.has_cosyvoice_codex_head and self.post_process:
            labels_ids_cosyvoice = labels_ids_audio.clone()
            if labels_ids_cosyvoice.shape[1] != logits_cosyvoice_codex.shape[0]:
                if (
                    self.config.sequence_parallel
                    and mpu.get_tensor_model_parallel_world_size() > 1
                ):
                    labels_ids_cosyvoice = tensor_parallel.scatter_to_sequence_parallel_region(
                        labels_ids_cosyvoice.transpose(0, 1).contiguous()
                    ).transpose(0, 1).contiguous()
                if labels_ids_cosyvoice.shape[1] != logits_cosyvoice_codex.shape[0]:
                    labels_ids_cosyvoice = labels_ids_cosyvoice[
                        :, : logits_cosyvoice_codex.shape[0]
                    ]
            invalid_cosyvoice_labels = (
                (labels_ids_cosyvoice < 0)
                | (labels_ids_cosyvoice >= int(self.cosyvoice_codec_unit))
            )
            if not self._cosyvoice_codex_label_debug_logged:
                self._cosyvoice_codex_label_debug_logged = True
                try:
                    valid_cosyvoice_labels = ~invalid_cosyvoice_labels
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    logging.info(
                        "[CODEX LABEL DEBUG] rank=%s labels_shape=%s raw_min=%s raw_max=%s "
                        "num_neg100=%s num_neg50=%s num_neg1=%s num_valid=%s num_ge_codec_unit=%s",
                        rank,
                        tuple(labels_ids_cosyvoice.shape),
                        int(labels_ids_cosyvoice.min().detach().cpu()),
                        int(labels_ids_cosyvoice.max().detach().cpu()),
                        int((labels_ids_cosyvoice == -100).sum().detach().cpu()),
                        int((labels_ids_cosyvoice == -50).sum().detach().cpu()),
                        int((labels_ids_cosyvoice == -1).sum().detach().cpu()),
                        int(valid_cosyvoice_labels.sum().detach().cpu()),
                        int((labels_ids_cosyvoice >= int(self.cosyvoice_codec_unit)).sum().detach().cpu()),
                    )
                except Exception as exc:
                    logging.info("[CODEX LABEL DEBUG] failed to log label stats: %s", exc)
            labels_ids_cosyvoice = torch.where(
                invalid_cosyvoice_labels,
                torch.tensor(-100, device=labels_ids_cosyvoice.device, dtype=labels_ids_cosyvoice.dtype),
                labels_ids_cosyvoice,
            )
            cosyvoice_codex_loss_mask = (labels_ids_cosyvoice != -100).to(torch.int32)
            is_all_ignore_cosyvoice = (labels_ids_cosyvoice == -100).all()
            if not is_all_ignore_cosyvoice:
                cosyvoice_codex_loss = self.compute_language_model_loss(
                    labels_ids_cosyvoice, logits_cosyvoice_codex
                )
            else:
                logits_cosyvoice_mean = torch.nan_to_num(
                    logits_cosyvoice_codex.mean(dim=-1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cosyvoice_codex_loss = logits_cosyvoice_mean * 0.0

            if self.has_cosyvoice_fm:
                if target_audio is None or target_audio_lengths is None:
                    raise ValueError("FM is enabled but target_audio/target_audio_lengths are missing")
                if target_spk_embedding is None:
                    raise ValueError("FM is enabled but target_spk_embedding is missing")
                if target_audio.shape[0] != labels_ids_cosyvoice.shape[0]:
                    raise ValueError(
                        f"FM target batch size {target_audio.shape[0]} does not match token batch "
                        f"size {labels_ids_cosyvoice.shape[0]}"
                    )
                # A codec hidden at position t predicts labels[t].  Select only
                # real FSQ tokens (special EOS/SOS tokens are outside codebook).
                codebook_size = int(self.cosyvoice_fm.codebook_size)
                fm_hidden = cosyvoice_hidden_states
                fm_labels = labels_ids_cosyvoice
                if self.config.sequence_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
                    local_hidden_seq_len = fm_hidden.shape[0]
                    fm_hidden = tensor_parallel.gather_from_sequence_parallel_region(
                        fm_hidden, tensor_parallel_output_grad=True
                    )
                    # The codec output head may already have gathered its sequence-parallel
                    # output, in which case labels_ids_cosyvoice is global while the codec
                    # hidden states are still local.  Gather labels only when they are local
                    # too; otherwise a second gather would duplicate their sequence length.
                    if fm_labels.shape[1] == local_hidden_seq_len:
                        fm_labels = tensor_parallel.gather_from_sequence_parallel_region(
                            fm_labels.transpose(0, 1).contiguous(),
                            tensor_parallel_output_grad=False,
                        ).transpose(0, 1).contiguous()
                if fm_labels.shape[1] != fm_hidden.shape[0]:
                    raise ValueError(
                        "FM codec labels and hidden states have different sequence lengths after "
                        f"sequence-parallel alignment: labels={fm_labels.shape[1]}, "
                        f"hidden={fm_hidden.shape[0]}"
                    )
                fm_mask = (fm_labels >= 0) & (fm_labels < codebook_size)
                hidden_btd = fm_hidden.transpose(0, 1)
                hidden_list = [hidden_btd[b, fm_mask[b]] for b in range(hidden_btd.shape[0])]
                codec_hidden = torch.nn.utils.rnn.pad_sequence(hidden_list, batch_first=True)
                codec_len = torch.tensor(
                    [x.shape[0] for x in hidden_list], device=codec_hidden.device, dtype=torch.long
                ).unsqueeze(1)
                if (codec_len == 0).any():
                    raise ValueError("FM batch contains a sample without supervised codec tokens")
                spk_emb = target_spk_embedding.to(
                    device=codec_hidden.device, dtype=codec_hidden.dtype
                )
                if spk_emb.shape != (codec_hidden.shape[0], 192):
                    raise ValueError(
                        f"FM speaker embedding must have shape [B, 192], got {tuple(spk_emb.shape)}"
                    )
                spk_emb_len = torch.ones(codec_hidden.shape[0], device=codec_hidden.device, dtype=torch.long)
                samples_per_codec = int(self.cosyvoice_fm.mel_extractor.hop_length) * int(
                    self.cosyvoice_fm.feat_token_ratio
                )
                aligned_audio = []
                aligned_lengths = []
                for b in range(codec_hidden.shape[0]):
                    wanted = int(codec_len[b, 0].item()) * samples_per_codec
                    available = int(target_audio_lengths[b].item())
                    wav = target_audio[b, :min(wanted, available)]
                    if wav.numel() < wanted:
                        wav = torch.nn.functional.pad(wav, (0, wanted - wav.numel()))
                    aligned_audio.append(wav)
                    aligned_lengths.append(wanted)
                target_audio = torch.nn.utils.rnn.pad_sequence(aligned_audio, batch_first=True)
                target_audio_lengths = torch.tensor(
                    aligned_lengths, device=target_audio.device, dtype=torch.long
                )
                cosyvoice_fm_loss, _, _ = self.cosyvoice_fm(
                    target_audio,
                    target_audio_lengths.unsqueeze(1),
                    spk_emb,
                    spk_emb_len,
                    codec_hidden=codec_hidden,
                    codec_len=codec_len,
                )

        # pp last stage, only pass loss_mask to loss_func
        extra_outputs = (loss_mask, loss_mask_audio)
        assert len(extra_outputs) == 2
        if cosyvoice_fm_loss is not None:
            return (text_loss, audio_loss, cosyvoice_codex_loss, cosyvoice_codex_loss_mask, cosyvoice_fm_loss), extra_outputs
        if cosyvoice_codex_loss is not None:
            return (text_loss, audio_loss, cosyvoice_codex_loss, cosyvoice_codex_loss_mask), extra_outputs
        return (text_loss, audio_loss), extra_outputs

    def setup_embeddings_and_output_layer_audio(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initalizes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.
        """

        # Set `is_embedding_or_output_parameter` attribute.
        if self.pre_process:
            self.embedding_audio.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.output_layer_audio.weight is not None:
            self.output_layer_audio.weight.is_embedding_or_output_parameter = True

        # if not self.share_embeddings_and_output_weights:
        if not False:
            return

        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_or_output_weight_audio().zero_out_wgrad = True
            return

        if parallel_state.is_pipeline_first_stage() and self.pre_process and not self.post_process:
            self.shared_embedding_or_output_weight_audio().shared_embedding = True

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.output_layer_audio.weight.data.fill_(0)
            self.output_layer_audio.weight.shared = True
            self.output_layer_audio.weight.shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.shared_embedding_or_output_weight_audio()
                weight.data = weight.data.cuda()
                torch.distributed.all_reduce(
                    weight.data, group=parallel_state.get_embedding_group()
                )

        elif not getattr(LanguageModule, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageModule.embedding_warning_printed = True

    def shared_embedding_or_output_weight_audio(self) -> Tensor:
        """Gets the emedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding_audio.word_embeddings.weight
        elif self.post_process:
            return self.output_layer_audio.weight
        return None
        
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for MinMoS2TModel.
        
        Args:
            prefix (str): Module name prefix
            sharded_offsets (tuple): PP related offsets
            metadata (Optional[Dict]): metadata controlling sharded state dict creation
            
        Returns:
            ShardedStateDict: sharded state dict for the model
        """
        metadata = metadata or {}
        if getattr(self.config, 'layer_types', None) and len(set(self.config.layer_types)) > 1:
            metadata['non_homogeneous_layers'] = True

        sharded_state_dict = MegatronModule.sharded_state_dict(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        
        return sharded_state_dict
