# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Tuple, Dict, Any
from funasr.auto.auto_model import AutoModel


# Supported precision modes for speech token extractor
PRECISION_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    # "fp8": torch.float8_e4m3fn,  # Uncomment when fp8 support is needed
}


def _get_precision_dtype() -> Tuple[str, Optional[torch.dtype]]:
    """Get precision dtype from environment variable.
    
    Set SPEECH_TOKEN_EXTRACTOR_PRECISION to one of: fp32, fp16, bf16, fp8
    Default is fp32 if not set or invalid value.
    
    Returns:
        Tuple[str, Optional[torch.dtype]]: (precision_name, dtype)
            dtype is None if using default fp32 (no conversion needed)
    """
    precision = os.environ.get("SPEECH_TOKEN_EXTRACTOR_PRECISION", "fp32").lower()
    
    if precision not in PRECISION_DTYPE_MAP:
        logging.warning(
            f"Invalid SPEECH_TOKEN_EXTRACTOR_PRECISION='{precision}', "
            f"valid options: {list(PRECISION_DTYPE_MAP.keys())}. Using fp32."
        )
        precision = "fp32"
    
    # fp32 is default, no conversion needed
    if precision == "fp32":
        return precision, None
    
    return precision, PRECISION_DTYPE_MAP[precision]


def _should_compile() -> bool:
    """Check if torch.compile should be enabled.
    
    Set SPEECH_TOKEN_EXTRACTOR_COMPILE=1 to enable.
    """
    return os.environ.get("SPEECH_TOKEN_EXTRACTOR_COMPILE", "0").lower() in ("1", "true", "yes", "on")

DEFAULT_SPEECH_TOKEN_EXTRACTOR_CONF = {
    "enabled": True,
    "type": "automodel",
    "sliding_window_size": 4000,
    "sliding_window_left_context": 200,
    "sliding_window_bin_threshold": 1000,
    "downsample_factor": 4,
    "automodel_conf": {
        "model": "SenseVoiceL",
        "init_param": "/path/to/e0s1m.pt",
        "model_conf": {
            "lsm_weight": 0.1,
            "length_normalized_loss": True,
            "activation_checkpoint": True,
            "sos": "<|startoftranscript|>",
            "eos": "<|endoftext|>",
            "downsample_rate": 4,
            "use_padmask": True,
            "dims": {
                "n_mels": 128,
                "n_vocab": 60515,
                "n_audio_ctx": 1500,
                "n_audio_state": 1280,
                "n_audio_head": 20,
                "n_audio_layer": 32,
                "n_text_ctx": 448,
                "n_text_state": 1280,
                "n_text_head": 20,
                "n_text_layer": 32,
                "att_type": "default"
            }
        },
        "encoder": "SenseVoiceQuantizedEncoder",
        "encoder_conf": {
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
            "freeze_exclude_key": "attn",
            "quantize_layer_idx": 11,
            "normalized_quant_input": False,
            "quantizer_config": {
                "levels": [3, 3, 3, 3, 3, 3, 3, 3],
                "num_codebooks": 1,
                "name": "finite_scalar_quantizer"
            }
        },
        "tokenizer": "SenseVoiceTokenizer",
        "tokenizer_conf": {
            "is_multilingual": True,
            "num_languages": 8749,
            "vocab_path": "/path/to/multilingual.tiktoken"
        }
    }
}

DEFAULT_SPEECH_TOKEN_EXTRACTOR_CONF_12_5_Hz = {
    "enabled": True,
    "type": "automodel",
    "sliding_window_size": 4000,
    "sliding_window_left_context": 200,
    "sliding_window_bin_threshold": 1000,
    "downsample_factor": 8,
    "automodel_conf": {
        "model": "SenseVoiceL",
        "init_param": "/path/to/extract_model.pt",
        "model_conf": {
            "lsm_weight": 0.1,
            "length_normalized_loss": True,
            "activation_checkpoint": True,
            "sos": "<|startoftranscript|>",
            "eos": "<|endoftext|>",
            "downsample_rate": 4,
            "use_padmask": True,
            "dims": {
                "n_mels": 128,
                "n_vocab": 60515,
                "n_audio_ctx": 1500,
                "n_audio_state": 1280,
                "n_audio_head": 20,
                "n_audio_layer": 32,
                "n_text_ctx": 448,
                "n_text_state": 1280,
                "n_text_head": 20,
                "n_text_layer": 32,
                "att_type": "default"
            }
        },
        "encoder": "SenseVoiceQuantizedEncoder",
        "encoder_conf": {
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
            "freeze_exclude_key": "attn",
            "quantize_layer_idx": 11,
            "normalized_quant_input": False,
            "quantizer_config": {
                "levels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                "num_codebooks": 1,
                "downsample_rate": 2,
                "name": "finite_scalar_quantizer_down"
            }
        },
        "tokenizer": "SenseVoiceTokenizer",
        "tokenizer_conf": {
            "is_multilingual": True,
            "num_languages": 8749,
            "vocab_path": "/path/to/multilingual.tiktoken"
        }
    }
}

TOKENIZER_MAP = {
    "25hz": DEFAULT_SPEECH_TOKEN_EXTRACTOR_CONF,
    "12.5hz": DEFAULT_SPEECH_TOKEN_EXTRACTOR_CONF_12_5_Hz,
}

class SpeechTokenizer:
    """Speech token extractor for MinMo models.
    
    This class handles speech token extraction using FunASR AutoModel,
    supporting sliding window processing for long audio sequences.
    """
    
    # Default configuration based on MinMo_S2S_Interleave_1.7b_pretrain_test_shanghai_local_cache.yaml
    
    
    def __init__(
        self,
        speech_token_extractor_conf: Optional[Dict[str, Any]] = None,
        version: str = "25hz",
        device: str = "cuda"
    ):
        """Initialize speech tokenizer.
        
        Args:
            speech_token_extractor_conf (Optional[Dict[str, Any]]): Configuration for speech token extractor.
                If None, uses the default configuration.
            device (str): Device to run the model on
        """
        self.device = device
        self.version = version

        if self.version is None or str(self.version).lower() in ["none", "null"]:
            print("Not using speech tokenizer")
            return
        
        # Use default configuration if None is provided
        if speech_token_extractor_conf is None:
            print(f"Using default speech token extractor configuration of version {self.version}")
            default_conf = TOKENIZER_MAP[self.version]
            self.speech_token_extractor_conf = default_conf.copy()
        else:
            print("Using passed in speech_token_extractor_conf")
            self.speech_token_extractor_conf = speech_token_extractor_conf
        
        self.speech_token_extractor_type = self.speech_token_extractor_conf.get("type", "automodel")
        
        if self.speech_token_extractor_type == "automodel":
            self._init_automodel()
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")
    
    def _init_automodel(self):
        """Initialize AutoModel-based speech token extractor."""
        print(f'SpeechTokenExtraction use automodel')
        
        # Import and initialize AutoModel
        self.speech_token_extractor = AutoModel(
            **self.speech_token_extractor_conf.get("automodel_conf", {})
        )
        
        # Move model to device and set to eval mode
        self.speech_token_extractor.model = self.speech_token_extractor.model.to(self.device)
        self.speech_token_extractor.model.eval()
        
        # Precision mode (controlled by env var SPEECH_TOKEN_EXTRACTOR_PRECISION)
        self.precision_name, self.precision_dtype = _get_precision_dtype()
        if self.precision_dtype is not None:
            self.speech_token_extractor.model = self.speech_token_extractor.model.to(self.precision_dtype)
        print(f'SpeechTokenExtractor using {self.precision_name}')
        
        # torch.compile optimization (controlled by env var SPEECH_TOKEN_EXTRACTOR_COMPILE=1)
        if _should_compile():
            print('SpeechTokenExtractor enabling torch.compile on extract_token method')
            # Compile the extract_token method specifically since that's what we call
            self.speech_token_extractor.model.extract_token = torch.compile(
                self.speech_token_extractor.model.extract_token,
                dynamic=True,
            )
        else:
            print('SpeechTokenExtractor torch.compile disabled')

        # Configure sliding window parameters
        self.sliding_window_size = self.speech_token_extractor_conf.get("sliding_window_size", 4000)
        print(f'SpeechTokenExtractor sliding window size: {self.sliding_window_size}')
        
        self.sliding_window_left_context = self.speech_token_extractor_conf.get("sliding_window_left_context", 200)
        print(f'SpeechTokenExtractor sliding window left context: {self.sliding_window_left_context}')
        
        self.sliding_window_bin_threshold = self.speech_token_extractor_conf.get("sliding_window_bin_threshold", 1000)
        print(f'SpeechTokenExtractor sliding window bin threshold: {self.sliding_window_bin_threshold}')
        
        self.downsample_factor = self.speech_token_extractor_conf.get("downsample_factor", 4)
        print(f'SpeechTokenExtractor downsample factor: {self.downsample_factor}')
    
    def extract_speech_tokens(
        self, 
        segment_fbank: torch.Tensor, 
        segment_fbank_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract speech tokens from fbank features.
        
        Args:
            segment_fbank (torch.Tensor): Fbank features [batch, seq_len, feature_dim]
            segment_fbank_lengths (torch.Tensor): Fbank lengths [batch]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Speech tokens and lengths
        """
        if self.speech_token_extractor_type == "automodel":
            return self._extract_with_automodel(segment_fbank, segment_fbank_lengths)
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")
    
    def _extract_with_automodel(
        self, 
        segment_fbank: torch.Tensor, 
        segment_fbank_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract speech tokens using AutoModel with sliding window processing.
        
        Args:
            segment_fbank (torch.Tensor): Fbank features [batch, seq_len, feature_dim]
            segment_fbank_lengths (torch.Tensor): Fbank lengths [batch]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Speech tokens and lengths
        """
        if self.precision_dtype is not None:
            segment_fbank = segment_fbank.to(self.precision_dtype)
        with torch.no_grad():
            segment_speech_token, segment_speech_token_lengths = self.process_with_sliding_window(
                segment_fbank,
                segment_fbank_lengths,
                self.speech_token_extractor.model,
                window_size=self.sliding_window_size,
                left_context=self.sliding_window_left_context,
                bin_threshold=self.sliding_window_bin_threshold,
                downsample_factor=self.downsample_factor,
                device=self.device
            )
        return segment_speech_token, segment_speech_token_lengths
    
    @staticmethod
    def process_with_sliding_window(
        segment_fbank: torch.Tensor, 
        segment_fbank_lengths: torch.Tensor,
        model: nn.Module, 
        window_size: int = 4000, 
        left_context: int = 2000,
        bin_threshold: int = 256, 
        downsample_factor: int = 4, 
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio with sliding window for long sequences.
        
        Args:
            segment_fbank (torch.Tensor): Fbank features [batch, seq_len, feature_dim]
            segment_fbank_lengths (torch.Tensor): Fbank lengths [batch]
            model (nn.Module): Speech token extraction model
            window_size (int): Sliding window size
            left_context (int): Left context size for overlapping
            bin_threshold (int): Threshold for batching similar lengths
            downsample_factor (int): Downsampling factor for token length
            device (str): Device to run on
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed speech tokens and lengths
        """
        B, Lmax, D = segment_fbank.shape
        true_lengths = segment_fbank_lengths[:, 0]

        # How many tokens correspond to the overlapped part
        overlap_token_len = int(left_context / downsample_factor)

        # Step 1: create sliding window chunks
        chunks = []
        for idx in range(B):
            length = true_lengths[idx].item()
            feat = segment_fbank[idx, :length]  # Trim padded part

            start = 0
            chunk_idx_in_original = 0
            while start < length:
                end = min(start + window_size, length)
                if chunk_idx_in_original > 0:
                    context_start = start - left_context
                else:
                    context_start = start
                chunk = feat[context_start:end]  # (chunk_len, D)
                chunks.append({
                    "tensor": chunk,
                    "length": chunk.shape[0],
                    "orig_idx": idx,
                    "chunk_idx": chunk_idx_in_original
                })
                chunk_idx_in_original += 1
                if end == length:
                    break
                start += window_size

        # Step 2: sort by length to minimize padding
        chunks_sorted = sorted(chunks, key=lambda x: x["length"])

        # Step 3: bin chunks by length threshold
        bins = []
        current_bin = []
        current_bin_maxlen = 0
        for ch in chunks_sorted:
            if not current_bin:
                current_bin = [ch]
                current_bin_maxlen = ch["length"]
            else:
                if ch["length"] <= current_bin_maxlen + bin_threshold:
                    current_bin.append(ch)
                    current_bin_maxlen = max(current_bin_maxlen, ch["length"])
                else:
                    bins.append(current_bin)
                    current_bin = [ch]
                    current_bin_maxlen = ch["length"]
        if current_bin:
            bins.append(current_bin)

        # Step 4: run model per bin
        results = {}  # (orig_idx, chunk_idx) -> trimmed_output_tensor

        for bin_chunks in bins:
            maxlen = max(ch["length"] for ch in bin_chunks)
            batch_tensors = torch.stack([
                torch.cat([
                    ch["tensor"],
                    torch.zeros(maxlen - ch["length"], D, dtype=ch["tensor"].dtype, device=device)
                ], dim=0)
                for ch in bin_chunks
            ], dim=0)
            batch_lengths = torch.tensor([ch["length"] for ch in bin_chunks], device=device)

            # Pass through model
            batch_tensors = batch_tensors.permute(0, 2, 1)
            token_result, _ = model.extract_token(
                batch_tensors,
                data_lengths=batch_lengths,
                data_type="fbank",
                key=["any"],
                device=device
            )
            tokens = token_result[0]["token"]

            # Now assign each output explicitly
            for ch, out_tokens in zip(bin_chunks, tokens):
                max_token_len = math.ceil(ch["length"] / downsample_factor)
                if ch["chunk_idx"] == 0:
                    trimmed = out_tokens[:max_token_len]
                else:
                    drop = min(math.ceil(left_context / downsample_factor), max_token_len)
                    trimmed = out_tokens[drop:max_token_len]
                results[(ch["orig_idx"], ch["chunk_idx"])] = trimmed

        # Step 5: reassemble to original order
        final_outputs = []
        final_output_lengths = []
        for idx in range(B):
            chunk_idx = 0
            sample_tokens = []
            while (idx, chunk_idx) in results:
                sample_tokens.append(results[(idx, chunk_idx)])
                chunk_idx += 1
            if sample_tokens:
                cat_tokens = torch.cat(sample_tokens, dim=0)
                final_outputs.append(cat_tokens)
                final_output_lengths.append(cat_tokens.shape[0])
            else:
                final_outputs.append(torch.empty(0))  # no data
                final_output_lengths.append(0)

        # Optional sanity check: lengths match expected downsample length
        for i in range(B):
            expected_len = math.ceil(true_lengths[i].item() / downsample_factor)
            assert final_outputs[i].shape[0] == expected_len, \
                f"Mismatch for sample {i}: got {final_outputs[i].shape[0]}, expected {expected_len}"
        
        # Pad final_outputs to the same length
        final_outputs = torch.nn.utils.rnn.pad_sequence(final_outputs, batch_first=True, padding_value=0).to(device)
        final_output_lengths = torch.tensor(final_output_lengths, dtype=torch.int32)

        return final_outputs, final_output_lengths
