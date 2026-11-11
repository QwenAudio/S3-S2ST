# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Literal, Optional, List
import torch
from torch import Tensor
import logging
import traceback

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

from .audio_encoder import SenseVoiceEncoder
from .audio_adaptor import AudioAdaptor
from .speech_tokenizer import SpeechTokenizer

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


class MinMoS2TModel(LanguageModule):
    """MinMo Speech-to-Text model adapted for Megatron framework.
    
    This implementation exactly matches the original MinMo_S2T forward logic.
    
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
        max_sequence_length: int,
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
        super().__init__(config=config)
        
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)
        
        self.audio_config = audio_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.length_normalized_loss = length_normalized_loss
        self.audio_encoder_activation_checkpoint = audio_encoder_activation_checkpoint
        self.feat_permute = feat_permute
        
        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # init speech tokenizer
        speech_token_extractor_conf = getattr(audio_config, 'speech_token_extractor_conf', None)
        self.speech_tokenizer = SpeechTokenizer(
                speech_token_extractor_conf=speech_token_extractor_conf,
                device="cuda"
            )
        
        
        # Audio encoder
        if self.pre_process:
            self.audio_encoder = SenseVoiceEncoder(
                config=audio_config,
                pre_process=True,
                post_process=True,
            )
        
        # Audio adaptor
        if self.pre_process:
            self.audio_adaptor = AudioAdaptor(
                config=audio_config,
                pre_process=True,
                post_process=True,
            )
        
        # Language model
        self.lm_model = GPTModel(
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
        
        # Store configuration for reference
        self.input_size = audio_config.input_size
        self.downsample_rate = audio_config.downsample_rate
        self.llm_dtype = "bf16"  # Default dtype matching original
        
    def set_input_tensor(self, input_tensor) -> None:
        """Sets input tensor to the model.
        
        Args:
            input_tensor: Input tensor for the model
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for MinMoS2T'
        
        if self.pre_process:
            self.audio_hidden_state = input_tensor[0]
        else:
            self.lm_model.set_input_tensor(input_tensor[0])
    
    def encode(self, speech: Tensor, speech_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Encode speech features - matching original implementation.
        
        Args:
            speech (Tensor): Input speech features
            speech_lengths (Tensor): Speech lengths
            
        Returns:
            tuple[Tensor, Tensor]: Encoded features and lengths
        """
        if self.feat_permute:
            encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)
        else:
            encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        speech: Optional[Tensor] = None,
        speech_lengths: Optional[Tensor] = None,
        fbank_beg: Optional[Tensor] = None,
        fbank_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: dict = None,
        **kwargs,
    ) -> tuple[Tensor, dict, int]:
        """Forward function of the MinMo S2T model.
        
        This implementation exactly matches the original MinMo_S2T.forward method.
        
        Args:
            input_ids (Tensor): Input text token ids [batch, text_seq_len]
            position_ids (Tensor): Input text position ids [batch, text_seq_len]
            attention_mask (Tensor): Attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len]
            speech (Tensor, optional): Input audio features [batch, audio_seq_len, input_size]
            speech_lengths (Tensor, optional): Audio sequence lengths [batch]
            fbank_beg (Tensor, optional): Beginning positions for audio tokens [batch, num_turns]
            fbank_mask (Tensor, optional): Mask for audio tokens
            labels (Tensor, optional): Target text labels [batch, combined_seq_len]
            inference_params (InferenceParams, optional): Inference-time parameters
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters
            extra_block_kwargs (dict, optional): Extra block arguments
            **kwargs: Additional arguments including fake_token_len
            
        Returns:
            tuple[Tensor, dict, int]: Loss, stats, and weight
        """
        # Match original implementation exactly
        batch_size, token_num = input_ids.shape
        stats = {}
        input_ids[input_ids < 0] = 0
        
        # Get input embeddings from language model
        if self.pre_process:
            inputs_embeds = self.lm_model.embedding(input_ids=input_ids, position_ids=position_ids)
            
            # Process audio if provided - matching original logic exactly
            if speech is not None:
                if len(speech_lengths.size()) > 1:
                    speech_lengths = speech_lengths[:, 0]

                batch_size_speech, frames, _ = speech.shape

                # Audio encoding with activation checkpointing support
                if self.audio_encoder_activation_checkpoint:
                    from torch.utils.checkpoint import checkpoint
                    encoder_out, encoder_out_lens = checkpoint(
                        self.encode, speech, speech_lengths, use_reentrant=False
                    )
                else:
                    encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                # Audio adaptation
                encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

                batch_size, token_num, dims = inputs_embeds.shape
                fake_token_len = kwargs.get("fake_token_len")
                if fake_token_len is not None:
                    fake_token_len[fake_token_len < 0] = 0
                if fbank_beg is not None:
                    fbank_beg[fbank_beg < 0] = 0

                # Insert audio features into text embeddings - matching original logic exactly
                speech_idx = 0
                for batch_idx in range(batch_size):
                    if fbank_beg is not None:
                        for turn_id in range(fbank_beg.shape[1]):
                            fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                            if fbank_beg_idx > 0:
                                if fake_token_len is not None:
                                    speech_token_len = fake_token_len[batch_idx, turn_id]
                                else:
                                    speech_token_len = encoder_out_lens[speech_idx].item()
                                
                                speech_token = encoder_out[speech_idx, :speech_token_len, :]

                                try:
                                    inputs_embeds[
                                        batch_idx, fbank_beg_idx: fbank_beg_idx + speech_token_len, :
                                    ] = speech_token
                                except Exception as e:
                                    logging.error(f"{str(e)}, {traceback.format_exc()}")
                                    logging.info(
                                        f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, "
                                        f"fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, "
                                        f"encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, "
                                        f"fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                                    )
                                    # Fallback to using actual encoder output length
                                    speech_token_len = encoder_out_lens[speech_idx].item()
                                    speech_token = encoder_out[speech_idx, :speech_token_len, :]
                                    inputs_embeds[
                                        batch_idx, fbank_beg_idx: fbank_beg_idx + speech_token_len, :
                                    ] = speech_token

                                speech_idx += 1

                # Calculate stats - matching original implementation
                stats["batch_size_speech"] = batch_size_speech
                stats["batch_size_x_frames"] = frames * batch_size_speech
                stats["batch_size_real_frames"] = speech_lengths.sum().item()
                stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        else:
            inputs_embeds = None

        # Run language model with mixed precision - matching original implementation
        # with torch.cuda.amp.autocast(
        #     enabled=True if self.llm_dtype != "fp32" else False, 
        #     dtype=dtype_map[self.llm_dtype]
        # ):
        # if labels is not None:
        #     labels[labels == -1] = -100
        # if attention_mask is not None:
        #     attention_mask[attention_mask < 0] = 0
        if labels is not None:
            labels = torch.where(labels == -1, torch.tensor(-100, device=labels.device, dtype=labels.dtype), labels)
        if attention_mask is not None:
            attention_mask = torch.where(attention_mask < 0, torch.tensor(0, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask)
        
        model_outputs = self.lm_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds.to(dtype_map[self.llm_dtype]) if inputs_embeds is not None else None,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        
        return model_outputs
        
        if labels is not None:
            loss = model_outputs.loss
        else:
            loss = model_outputs

        
        # Calculate accuracy and stats - matching original implementation
        if labels is not None:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc_att = compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)
                stats["acc"] = acc_att

            stats["loss"] = torch.clone(loss.detach())
            stats["batch_size"] = batch_size

            stats["batch_size_x_tokens"] = token_num * batch_size
            stats["batch_size_real_tokens"] = attention_mask.sum().item()
            stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

            if fbank_beg is not None:
                dialog_turns = (fbank_beg > 0).sum(-1)
                dialog_turns_max = torch.max(dialog_turns).int().item()
                dialog_turns_avg = dialog_turns.sum().item() / batch_size
                stats["dialog_turns_max"] = dialog_turns_max
                stats["dialog_turns_avg"] = dialog_turns_avg

            # Length normalized loss - matching original implementation
            if self.length_normalized_loss:
                batch_size = int((labels > 0 + 1).sum())
            
            # Force gatherable for DataParallel - matching original implementation
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight
        else:
            return loss
    

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
        sharded_state_dict = MegatronModule.sharded_state_dict(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        
        return sharded_state_dict