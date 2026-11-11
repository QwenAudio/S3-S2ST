# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Literal, Optional, List
import torch
from torch import Tensor
import logging
import traceback
import os

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
# from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
# from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig, AudioConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule

from megatron.core import mpu

from .audio_encoder import SenseVoiceEncoder, SensevoiceSmallEncoder, Qwen3Encoder
from .audio_adaptor import AudioAdaptor
from .speech_tokenizer import SpeechTokenizer
from .fun_utils import pad_cu_seqlens, unpad_cu_seqlens, get_nvte_fused_attn, set_nvte_fused_attn, freeze_module, set_module_lr_scale, process_with_sliding_window_and_bucketing

# Utility functions matching the original implementation
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy matching the original implementation."""
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_outputs.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()


def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively."""
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (int, float)):
        return torch.tensor(data, device=device)
    else:
        return data


# class FunBaseModel(LanguageModule):
class FunBaseModel(GPTModel):
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
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
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
        )
        
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)
        
        self.audio_config = audio_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.seq_length = seq_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.length_normalized_loss = length_normalized_loss
        self.audio_encoder_activation_checkpoint = audio_encoder_activation_checkpoint
        self.feat_permute = feat_permute
        self.freeze_audio_encoder = audio_config.freeze_encoder
        self.freeze_audio_adaptor = audio_config.freeze_adaptor
        self.freeze_decoder = audio_config.freeze_decoder
        self.encoder_no_use_fused_attention = audio_config.encoder_no_use_fused_attention
        self.encoder_adaptor_sliding_window_size = audio_config.encoder_adaptor_sliding_window_size
        self.encoder_adaptor_sliding_window_left_context = audio_config.encoder_adaptor_sliding_window_left_context
        self.encoder_adaptor_bucketing_threshold = audio_config.encoder_adaptor_bucketing_threshold
        
        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # init speech tokenizer
        speech_token_extractor_conf = getattr(audio_config, 'speech_token_extractor_conf', None)
        speech_tokenizer_version = audio_config.tokenizer_version
        self.speech_tokenizer = SpeechTokenizer(
                speech_token_extractor_conf=speech_token_extractor_conf,
                version=speech_tokenizer_version,
                device="cuda"
            )
        
        # Audio encoder
        if self.pre_process:
            if self.audio_config.encoder_type == "none":
                logging.info(f'[MODEL INIT] audio encoder is None')
                self.audio_encoder = None
            else:
                if self.audio_config.encoder_type == "sensevoice":
                    self.audio_encoder = SenseVoiceEncoder(
                        config=audio_config,
                        pre_process=True,
                        post_process=True,
                    )
                elif self.audio_config.encoder_type == "sensevoicesmall":
                    self.audio_encoder = SensevoiceSmallEncoder(
                        config=audio_config,
                    )
                elif self.audio_config.encoder_type.startswith("qwen3"):
                    self.audio_encoder = Qwen3Encoder(
                        config=audio_config,
                        max_sequence_length=32768,
                    )
                else:
                    raise ValueError(f'[MODEL INIT] invalid audio encoder type: {self.audio_config.encoder_type}')
                if self.freeze_audio_encoder:
                    logging.info(f'[MODEL INIT] freeze audio encoder')
                    freeze_module(self.audio_encoder)
                if self.audio_config.encoder_lr_scale is not None and self.audio_config.encoder_lr_scale != 1.0:
                    logging.info(f'[MODEL INIT] set audio encoder lr scale to {self.audio_config.encoder_lr_scale}')
                    set_module_lr_scale(self.audio_encoder, self.audio_config.encoder_lr_scale)
        
        # Audio adaptor
        if self.pre_process:
            if self.audio_encoder is not None:
                self.audio_adaptor = AudioAdaptor(
                    config=audio_config,
                    pre_process=True,
                    post_process=True,
                )
                if self.freeze_audio_adaptor:
                    logging.info(f'[MODEL INIT] freeze audio adaptor')
                    freeze_module(self.audio_adaptor)
                if self.audio_config.adaptor_lr_scale is not None and self.audio_config.adaptor_lr_scale != 1.0:
                    set_module_lr_scale(self.audio_adaptor, self.audio_config.adaptor_lr_scale)
            else:
                logging.info(f'[MODEL INIT] audio encoder is None, so audio adaptor is None')
        
        # Freeze decoder
        if self.freeze_decoder:
            logging.info(f'[MODEL INIT] freeze decoder')
            freeze_module(self.decoder)
            if self.pre_process:
                freeze_module(self.embedding)
            if self.post_process:
                freeze_module(self.output_layer)
        if self.audio_config.decoder_lr_scale is not None and self.audio_config.decoder_lr_scale != 1.0:
            logging.info(f'[MODEL INIT] set decoder lr scale to {self.audio_config.decoder_lr_scale}')
            set_module_lr_scale(self.decoder, self.audio_config.decoder_lr_scale)
            if self.pre_process:
                set_module_lr_scale(self.embedding, self.audio_config.decoder_lr_scale)
            if self.post_process:
                set_module_lr_scale(self.output_layer, self.audio_config.decoder_lr_scale)
        
        # Language model
        # self.lm_model = GPTModel(
        #     config=config,
        #     transformer_layer_spec=transformer_layer_spec,
        #     vocab_size=vocab_size,
        #     max_sequence_length=max_sequence_length,
        #     pre_process=pre_process,
        #     post_process=post_process,
        #     fp16_lm_cross_entropy=fp16_lm_cross_entropy,
        #     parallel_output=parallel_output,
        #     share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        #     position_embedding_type=position_embedding_type,
        #     rotary_percent=rotary_percent,
        #     rotary_base=rotary_base,
        #     rope_scaling=rope_scaling,
        #     seq_len_interpolation_factor=seq_len_interpolation_factor,
        # )
        
        # Store configuration for reference
        self.input_size = audio_config.input_size
        self.downsample_rate = audio_config.downsample_rate
        self.llm_dtype = "bf16"  # Default dtype matching original

        self.init_extra_output_specs()

    def init_extra_output_specs(self):
        seq_length = self.seq_length // mpu.get_context_parallel_world_size()

        # extra output specs
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            self.extra_output_specs = [
                (["b", seq_length], torch.int64),  # labels_ids, only gets splitted by cp
                (["b", seq_length], torch.int32),  # loss_mask, only gets splitted by cp
            ]
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
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1 + len(self.extra_output_specs), f'input_tensor should only be length 1 + {len(self.extra_output_specs)} for FunBaseModel'

        self.decoder.set_input_tensor(input_tensor[0])

        self.labels_ids = None
        self.loss_mask = None
        self.packed_seq_params = None

        if len(self.extra_output_specs) > 0:
            self.labels_ids = input_tensor[1]
            self.loss_mask = input_tensor[2]

            if mpu.get_context_parallel_world_size() > 1:
                cu_seqlens_padded = input_tensor[3]
                max_seqlen = input_tensor[4]
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

    def get_passthrough_input_tensors(self, labels_ids, loss_mask, packed_seq_params) -> tuple[Tensor]:
        # passed from previous stage
        if self.labels_ids is not None:
            labels_ids = self.labels_ids
        if self.loss_mask is not None:
            loss_mask = self.loss_mask
        if self.packed_seq_params is not None:
            packed_seq_params = self.packed_seq_params
        return labels_ids, loss_mask, packed_seq_params


    def encode(self, speech: Tensor, speech_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Encode speech features - matching original implementation.
        
        Args:
            speech (Tensor): Input speech features
            speech_lengths (Tensor): Speech lengths
            
        Returns:
            tuple[Tensor, Tensor]: Encoded features and lengths
        """
        # always convert to bf16
        speech = speech.to(dtype=torch.bfloat16).contiguous()

        # print(f'[MODEL ENCODE] original speech shape: {speech.shape}, speech size: {speech.shape[0] * speech.shape[1]}, speech_lengths shape: {speech_lengths.shape}, max: {speech_lengths.max()}, min: {speech_lengths.min()}, sum: {speech_lengths.sum()}, util: {speech_lengths.sum() / (speech.shape[0] * speech.shape[1])}')

        old_fused_attn = get_nvte_fused_attn()
        if self.encoder_no_use_fused_attention:
            set_nvte_fused_attn("0")

        # if self.feat_permute:
        #     encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)
        # else:
        #     encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        # encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

        if self.encoder_no_use_fused_attention:
            set_nvte_fused_attn(old_fused_attn)

        return encoder_out, encoder_out_lens


    def forward_encoder_and_adaptor(self, speech: Tensor, speech_lengths: Tensor) -> tuple[Tensor, Tensor]:
        if self.encoder_adaptor_sliding_window_size > 0 and \
            self.encoder_adaptor_bucketing_threshold > 0:
            encoder_out, encoder_out_lens = process_with_sliding_window_and_bucketing(
                    speech,
                    speech_lengths,
                    self.encode,
                    self.audio_adaptor,
                    window_size=self.encoder_adaptor_sliding_window_size,
                    left_context=self.encoder_adaptor_sliding_window_left_context,
                    bin_threshold=self.encoder_adaptor_bucketing_threshold,
                    device="cuda"
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths) # speech: [b, t, d], speech_lengths: [b]; encoder_out: [b, t, d], encoder_out_lens: [b]
                # Audio adaptation
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens) # encoder_out: [b, t, d], encoder_out_lens: [b]
        return encoder_out, encoder_out_lens


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

        labels_ids, loss_mask, packed_seq_params = self.get_passthrough_input_tensors(labels_ids, loss_mask, packed_seq_params)

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

                            # tmp fix
                            if len(segment["tokens"]) < token_duration:
                                padding = torch.zeros(token_duration - len(segment["tokens"]), dtype=segment["tokens"].dtype, device=segment["tokens"].device)
                                speech_tokens = torch.cat([segment["tokens"], padding])
                                print (f'[MODEL FORWARD] segment tokens less than token_duration: {len(segment["tokens"])} vs {token_duration}')
                            else:
                                speech_tokens = segment["tokens"][-token_duration:]
                                print (f'[MODEL FORWARD] pick last token duration: {len(segment["tokens"])} vs {token_duration}')

                        slice_speech_tokens[-1].append(speech_tokens)

                if len(slice_speech_tokens[-1]) == 0:
                    # 纯文本，或者audio to text的形式
                    assert len(input_ids[i][input_ids[i] == -50]) == 0, f'[MODEL FORWARD] input_ids[i][input_ids[i] == -50] is not empty, {len(input_ids[i][input_ids[i] == -50])}'
                    slice_speech_tokens[-1] = torch.tensor([], dtype=torch.long)
                else:
                    slice_speech_tokens[-1] = torch.cat(slice_speech_tokens[-1])
                slice_speech_tokens[i] += speech_token_id_offset[0]

            for i in range(input_ids.size(0)):
                if speech_token_nums_per_sequence_chunk is not None:
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
                    input_ids[i][input_ids[i] == -50] = slice_speech_tokens[i][slice_speech_tokens_mask].long().to(input_ids.device)
                # # audio_placeholder
                # if len(input_ids[i][input_ids[i] == -50]) != len(slice_speech_tokens[i]):
                #     print (f'[MODEL FORWARD] input_ids[i][input_ids[i] == -50] length not match, {len(input_ids[i][input_ids[i] == -50])} vs {len(slice_speech_tokens[i])}, total input length: {len(input_ids[i])}')
                #     print (f'[MODEL FORWARD] item: {items[i]}')
                #     input_ids[i][input_ids[i] == -50] = 0
                else:
                    input_ids[i][input_ids[i] == -50] = slice_speech_tokens[i].long().to(input_ids.device)
            input_id_shift = torch.zeros_like(input_ids)
            input_id_shift[:, :-1] = input_ids[:, 1:]
            if labels_ids is not None:
                labels_ids = torch.where(labels_ids == -50, input_id_shift, labels_ids)
            if self.config.timers is not None:
                self.config.timers('forward-audio-tokenize-insert', log_level=2).stop()
        else:
            self.config.timers('forward-audio-tokenize-extract', log_level=2).start()
            self.config.timers('forward-audio-tokenize-extract', log_level=2).stop()
            self.config.timers('forward-audio-tokenize-insert', log_level=2).start()
            self.config.timers('forward-audio-tokenize-insert', log_level=2).stop()

        if self.pre_process:
            # Match original implementation exactly
            batch_size, token_num = input_ids.shape
            stats = {}
            input_ids[input_ids < 0] = 0
            inputs_embeds = self.embedding(input_ids=input_ids, position_ids=position_ids)
            
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
            inputs_embeds = inputs_embeds.permute(1, 0, 2)
            batch_size, _, dims = inputs_embeds.shape

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
            inputs_embeds = torch.where(speech_token_mask, speech_token_full, inputs_embeds)
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
        model_outputs = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds.to(dtype_map[self.llm_dtype]) if inputs_embeds is not None else None,
            labels=labels_ids,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        if self.config.timers is not None:
            self.config.timers('forward-gpt-model-forward', log_level=2).stop()

            # timer_keys = [
            #     'forward-audio-tokenize-extract',
            #     'forward-audio-tokenize-insert',
            #     'forward-audio-embedding-encode-real',
            #     'forward-audio-embedding-encode-fake',
            #     'forward-audio-embedding-insert',
            #     'forward-gpt-model-forward',
            # ]
            # self.config.timers.log(timer_keys)
        
        if loss_fake is not None:
            model_outputs += loss_fake * 0.0

        if not self.post_process:
            # pp non-last stage, pass extra_outputs to next stage
            extra_outputs = [labels_ids, loss_mask]
            if mpu.get_context_parallel_world_size() > 1:
                cu_seqlens_padded = pad_cu_seqlens(packed_seq_params["cu_seqlens_q"], labels_ids.shape[0], labels_ids.shape[1])
                extra_outputs.extend([cu_seqlens_padded, packed_seq_params["max_seqlen_q"]])
            extra_outputs = tuple(extra_outputs)
            assert len(extra_outputs) == len(self.extra_output_specs)
            return model_outputs, extra_outputs
        
        # pp last stage, only pass loss_mask to loss_func
        extra_outputs = (loss_mask,)
        assert len(extra_outputs) == 1
        return model_outputs, extra_outputs
        
        # if labels is not None:
        #     loss = model_outputs.loss
        # else:
        #     loss = model_outputs

        
        # # Calculate accuracy and stats - matching original implementation
        # if labels is not None:
        #     with torch.no_grad():
        #         preds = torch.argmax(model_outputs.logits, -1)
        #         acc_att = compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)
        #         stats["acc"] = acc_att

        #     stats["loss"] = torch.clone(loss.detach())
        #     stats["batch_size"] = batch_size

        #     stats["batch_size_x_tokens"] = token_num * batch_size
        #     stats["batch_size_real_tokens"] = attention_mask.sum().item()
        #     stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        #     if fbank_beg is not None:
        #         dialog_turns = (fbank_beg > 0).sum(-1)
        #         dialog_turns_max = torch.max(dialog_turns).int().item()
        #         dialog_turns_avg = dialog_turns.sum().item() / batch_size
        #         stats["dialog_turns_max"] = dialog_turns_max
        #         stats["dialog_turns_avg"] = dialog_turns_avg

        #     # Length normalized loss - matching original implementation
        #     if self.length_normalized_loss:
        #         batch_size = int((labels > 0 + 1).sum())
            
        #     # Force gatherable for DataParallel - matching original implementation
        #     loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        #     return loss, stats, weight
        # else:
        #     return loss
    

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