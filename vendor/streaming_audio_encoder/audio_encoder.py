# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import logging

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import AudioConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding as MegatronRotaryEmbedding
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

class Conv1d(nn.Module):
    """1D Convolution layer with proper initialization."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 config: Optional[TransformerConfig] = None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias)
        if config is not None:
            self.conv.weight.data = self.conv.weight.data.to(dtype=config.params_dtype)
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data.to(dtype=config.params_dtype)
        
        
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class LayerNorm(nn.LayerNorm):
    """Layer normalization with proper initialization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class Linear(nn.Linear):
    """Linear layer with proper initialization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)


def make_pad_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    """Create padding mask from lengths."""
    if max_len is None:
        max_len = lengths.max()
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(1).expand(batch_size, max_len)
    return seq_range_expand >= seq_length_expand


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# class RotaryEmbedding(nn.Module):
#     """Rotary position embedding."""
    
#     def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
        
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
        
#     def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
#         t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
#         freqs = torch.outer(t, self.inv_freq)
#         cos = torch.cos(freqs)
#         sin = torch.sin(freqs)
#         return cos, sin

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


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        
        qk = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            qk = qk.masked_fill(mask, min_value)
        
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)
        
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionSdpa(nn.Module):
    """Multi-head attention using scaled dot product attention."""
    
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        is_causal = kwargs.get("is_causal", False)
        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        
        if mask is not None:
            mask = mask.unsqueeze(1).to(torch.bool)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=scale
        )
        
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        
        attn_output = attn_output.transpose(1, 2).flatten(start_dim=2)
        return attn_output, None


class MultiHeadAttentionRoPE(nn.Module):
    """Multi-head attention with rotary position embedding."""
    
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
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        
        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        qk = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            qk = qk.masked_fill(mask, min_value)
        
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)
        
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionSdpaRoPE(nn.Module):
    """Multi-head attention with SDPA and rotary position embedding."""
    
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
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
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
        
        if mask is not None:
            mask = mask.unsqueeze(1).to(torch.bool)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=scale
        )
        
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        
        attn_output = attn_output.transpose(1, 2).flatten(start_dim=2)
        return attn_output, None


class MultiHeadAttentionFSMNRoPE(nn.Module):
    """Multi-head attention with FSMN and rotary position embedding."""
    
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
        
        # FSMN block
        self.fsmn_block = nn.Conv1d(
            linear_units, linear_units,
            kwargs.get("kernel_size", 31),
            stride=1, padding=0, groups=linear_units, bias=False
        )
        
        # Padding
        if kwargs.get("streaming", False):
            # A symmetric FSMN kernel would leak future frames even when
            # self-attention is block causal.  Use history-only memory in
            # streaming mode so the complete encoder respects chunk latency.
            left_padding = kwargs.get("kernel_size", 31) - 1
            right_padding = 0
        else:
            left_padding = (kwargs.get("kernel_size", 31) - 1) // 2
            left_padding = left_padding + kwargs.get("sanm_shfit", 0)
            right_padding = kwargs.get("kernel_size", 31) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))
        
    def fsmn(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply FSMN (Feed-forward Sequential Memory Network) to inputs."""
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
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        memory = self.fsmn(v, mask=mask)
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
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
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            qk = qk.masked_fill(mask, min_value)
        
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)
        
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionFSMNSdpaRoPE(nn.Module):
    """Multi-head attention with FSMN, SDPA and rotary position embedding."""
    
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
        
        # FSMN block
        self.fsmn_block = nn.Conv1d(
            linear_units, linear_units,
            kwargs.get("kernel_size", 31),
            stride=1, padding=0, groups=linear_units, bias=False
        )
        
        # Padding
        if kwargs.get("streaming", False):
            # See MultiHeadAttentionFSMNRoPE: streaming FSMN must not look at
            # frames from a future chunk.
            left_padding = kwargs.get("kernel_size", 31) - 1
            right_padding = 0
        else:
            left_padding = (kwargs.get("kernel_size", 31) - 1) // 2
            left_padding = left_padding + kwargs.get("sanm_shfit", 0)
            right_padding = kwargs.get("kernel_size", 31) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))
        
    def fsmn(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply FSMN (Feed-forward Sequential Memory Network) to inputs."""
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
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        memory = self.fsmn(v, mask=mask)
        # Keep ``mask`` as the per-frame padding mask used by FSMN.  Streaming
        # attention needs a separate [B, T, T] block-causal mask; feeding that
        # mask to FSMN would incorrectly reshape T*T entries into the sequence.
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
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
        
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = mask
        if attention_mask is not None:
            # SDPA boolean masks use True for entries that may be attended to.
            # Padding masks are [B, 1, T], streaming masks are [B, T, T].
            attention_mask = attention_mask.unsqueeze(1).to(torch.bool)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=is_causal, scale=scale
        )
        
        # Zero padded query positions.  Do not infer query validity by
        # transposing a [T, T] streaming mask: that would mix query/key axes.
        if mask is not None:
            valid_queries = mask.to(torch.bool).unsqueeze(-1)
            attn_output = attn_output.masked_fill(valid_queries.logical_not(), 0.0)
        
        attn_output = attn_output.transpose(1, 2).flatten(start_dim=2)
        return attn_output, None


# Attention type dictionary - matches the original implementation
att_type_dict = {
    "default": MultiHeadAttention,
    "sdpa": MultiHeadAttentionSdpa,
    "self_att": MultiHeadAttentionRoPE,
    "self_att_sdpa": MultiHeadAttentionSdpaRoPE,
    "self_att_fsmn": MultiHeadAttentionFSMNRoPE,
    "self_att_fsmn_sdpa": MultiHeadAttentionFSMNSdpaRoPE,
}


class EncoderLayerSANMLarge(nn.Module):
    """Encoder layer for SenseVoice audio encoder with multiple attention types."""
    
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        
        # Get attention type from kwargs, default to FSMN+SDPA+RoPE
        att_type = kwargs.get("att_type", "self_att_fsmn_sdpa")
        self.attn = att_type_dict[att_type](linear_units, attention_heads, **kwargs)
        self.attn_ln = LayerNorm(linear_units)
        
        # MLP layers
        n_mlp = linear_units * 4
        self.mlp = nn.Sequential(
            Linear(linear_units, n_mlp), 
            nn.GELU(), 
            Linear(n_mlp, linear_units)
        )
        self.mlp_ln = LayerNorm(linear_units)
        
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None, 
        **kwargs
    ) -> Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.attn_ln(x), mask=mask, **kwargs)[0]
        
        # MLP with residual connection
        x = x + self.mlp(self.mlp_ln(x))
        
        return x


class SenseVoiceEncoder(MegatronModule):
    """SenseVoice audio encoder adapted for Megatron framework.
    
    Args:
        config (AudioConfig): Audio configuration
        pre_process (bool): Whether to pre-process the input
        post_process (bool): Whether to post-process the output
    """
    
    def __init__(
        self,
        config: AudioConfig,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=config)
        
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        
        # Convolutional layers for downsampling
        if self.pre_process:
            self.conv1 = Conv1d(
                config.input_size, 
                config.linear_units, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                config=config
            )
            self.conv2 = Conv1d(
                config.linear_units, 
                config.linear_units, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                config=config
            )
        
        # Transformer blocks with configurable attention types
        self.blocks = nn.ModuleList([
            EncoderLayerSANMLarge(
                config.linear_units, 
                config.attention_heads,
                att_type=config.att_type,
                attention_dropout_rate=config.attention_dropout_rate,
                dropout_rate=config.dropout_rate,
                kernel_size=config.kernel_size,
                sanm_shfit=config.sanm_shfit,
                streaming=config.encoder_streaming,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
            )
            for _ in range(config.num_blocks)
            # for _ in range(1)
        ])
        
        self.ln_post = LayerNorm(config.linear_units)
        
        self.use_padmask = config.use_padmask
        self.downsample_rate = config.downsample_rate
        self.streaming = config.encoder_streaming
        self.streaming_chunk_ms = config.encoder_streaming_chunk_ms
        if self.streaming_chunk_ms <= 0:
            raise ValueError(
                f"audio encoder streaming chunk size must be positive, "
                f"got {self.streaming_chunk_ms} ms"
            )
        # Input fbank frames are 10 ms and the two convolutions each stride by
        # two, hence each encoder frame represents 40 ms.
        conv_downsample = self.conv1.conv.stride[0] * self.conv2.conv.stride[0]
        self.streaming_chunk_frames = max(
            1, round(self.streaming_chunk_ms / (10.0 * conv_downsample))
        )
        if self.streaming:
            logging.info(
                "Using streaming SenseVoice encoder attention: chunk=%s ms "
                "(%s encoder frames), layers=%s",
                self.streaming_chunk_ms,
                self.streaming_chunk_frames,
                len(self.blocks),
            )

    def _streaming_attention_mask(
        self, lengths: Tensor, sequence_length: int, device
    ) -> Tensor:
        """Return an SDPA allow-mask with block-causal visibility [B, T, T]."""
        positions = torch.arange(sequence_length, device=device)
        chunk_ids = torch.div(
            positions, self.streaming_chunk_frames, rounding_mode="floor"
        )
        # A query can see its complete current chunk and every older chunk.
        block_causal = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)
        valid = positions.unsqueeze(0) < lengths.to(device=device).unsqueeze(1)
        return block_causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
        
    def forward(
        self, 
        x: Tensor, 
        ilens: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of the audio encoder."""
        
        # 1. Pre-processing (Same as your original code)
        if self.pre_process:
            x = x.permute(0, 2, 1) # [b, t, d] -> [b, d, t]
            if self.streaming:
                # Match reference streaming encoder exactly: preserve
                # output stride/length while eliminating convolutional future
                # context with left-only padding.
                x = F.pad(x, (self.conv1.conv.kernel_size[0] - 1, 0))
                x = F.gelu(F.conv1d(
                    x, self.conv1.conv.weight, self.conv1.conv.bias,
                    stride=self.conv1.conv.stride, padding=0,
                    dilation=self.conv1.conv.dilation,
                    groups=self.conv1.conv.groups,
                ))
                x = F.pad(x, (self.conv2.conv.kernel_size[0] - 1, 0))
                x = F.gelu(F.conv1d(
                    x, self.conv2.conv.weight, self.conv2.conv.bias,
                    stride=self.conv2.conv.stride, padding=0,
                    dilation=self.conv2.conv.dilation,
                    groups=self.conv2.conv.groups,
                ))
            else:
                x = F.gelu(self.conv1(x))
                x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1) # [b, d, t] -> [b, t, d]
        
        n_frames = x.size(1)
        max_pos = n_frames
        
        # 2. Length calculation (Same as your original code)
        olens = None
        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1 + (ilens - self.conv1.conv.kernel_size[0] + 2 * self.conv1.conv.padding[0])
                    // self.conv1.conv.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1 + (olens - self.conv2.conv.kernel_size[0] + 2 * self.conv2.conv.padding[0])
                // self.conv2.conv.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        
        # 3. Mask creation (Same as your original code)
        # Streaming also needs frame validity even when --audio-use-padmask is
        # not set, both for padded keys and for the FSMN branch.
        if (self.use_padmask or self.streaming) and olens is not None:
            padding_mask = (~make_pad_mask(olens, max_len=max_pos))[:, None, :].to(torch.bool).to(x.device)
        else:
            padding_mask = None
        streaming_attention_mask = None
        if self.streaming:
            if olens is None:
                olens = torch.full(
                    (x.shape[0],), max_pos, dtype=torch.long, device=x.device
                )
                padding_mask = torch.ones(
                    (x.shape[0], 1, max_pos), dtype=torch.bool, device=x.device
                )
            streaming_attention_mask = self._streaming_attention_mask(
                olens, max_pos, x.device
            )
        
        # 4. Position IDs (Same as your original code)
        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        # ---------------------------------------------------------
        # 5. Apply transformer blocks with Recompute Logic
        # ---------------------------------------------------------
        for layer, block in enumerate(self.blocks):
            # Check if recompute is enabled in config
            # 'full' usually means checkpointing every layer
            if self.config.recompute_encoder:
                
                # Define a wrapper to map positional args from checkpoint 
                # to keyword args expected by block.forward
                def create_custom_forward(module):
                    def custom_forward(x_in, mask_in, pos_ids_in, attention_mask_in):
                        return module(
                            x_in,
                            mask=mask_in,
                            position_ids=pos_ids_in,
                            attention_mask=attention_mask_in,
                        )
                    return custom_forward

                # checkpoint(function, *args)
                # Note: args must be Tensors for autograd to track them, 
                # but mask/pos_ids don't require gradients, which is fine.
                x = checkpoint(
                    create_custom_forward(block),
                    x,
                    padding_mask,
                    position_ids,
                    streaming_attention_mask,
                    use_reentrant=False
                )
            else:
                # Standard forward pass
                x = block(
                    x,
                    mask=padding_mask,
                    position_ids=position_ids,
                    attention_mask=streaming_attention_mask,
                )
        # ---------------------------------------------------------
   
        x = self.ln_post(x)
        
        if ilens is None:
            return x
        else:
            return x, olens
    
    def output_size(self) -> int:
        """Return the output size of the encoder."""
        return self.config.linear_units


class SensevoiceSmallEncoder(MegatronModule):

    def __init__(self, config: AudioConfig):
        super().__init__(config=config)

        self.config = config

        from funasr.models.sense_voice.model_small import SenseVoiceEncoderSmall, SinusoidalPositionEncoder
        self.encoder = SenseVoiceEncoderSmall(
            input_size=560,
            output_size=512,
            attention_heads=4,
            linear_units=2048,
            num_blocks=50,
            tp_blocks=20,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            input_layer="pe",
            pos_enc_class=SinusoidalPositionEncoder,
            normalize_before=True,
            kernel_size=11,
            sanm_shfit=0,
            selfattention_layer_type="sanm",
        )

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        masks = (~make_pad_mask(ilens, max_len=xs_pad.shape[1]))[:, None, :].to(xs_pad.device)

        # xs_pad = xs_pad[:, :, :80]

        xs_pad *= self.encoder.output_size() ** 0.5

        xs_pad = self.encoder.embed(xs_pad)

        def checkpointed_forward(block, x, mask):
            if self.config.recompute_encoder:
                # Define a wrapper to map positional args from checkpoint 
                # to keyword args expected by block.forward
                def create_custom_forward(module):
                    def custom_forward(x_in, mask_in):
                        return module(x_in, mask=mask_in)[0]
                    return custom_forward

                # checkpoint(function, *args)
                # Note: args must be Tensors for autograd to track them, 
                # but mask/pos_ids don't require gradients, which is fine.
                x = checkpoint(
                    create_custom_forward(block),
                    x,
                    mask,
                    use_reentrant=False
                )
            else:
                # Standard forward pass
                x = block(x, mask=mask)[0]
            return x

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoder.encoders0):
            xs_pad = checkpointed_forward(encoder_layer, xs_pad, masks)
            # encoder_outs = checkpointed_forward(encoder_layer, xs_pad, masks)
            # xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoder.encoders):
            xs_pad = checkpointed_forward(encoder_layer, xs_pad, masks)
            # encoder_outs = checkpointed_forward(encoder_layer, xs_pad, masks)
            # xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.encoder.after_norm(xs_pad)

        # forward encoder2
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.encoder.tp_encoders):
            xs_pad = checkpointed_forward(encoder_layer, xs_pad, masks)
            # encoder_outs = checkpointed_forward(encoder_layer, xs_pad, masks)
            # xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.encoder.tp_norm(xs_pad)
        return xs_pad, olens

class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        # self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2, (1, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2, (1, 0)),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, ::2][:, :, ::2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv1dQwen(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def pack_sequence(x: torch.Tensor, lengths: torch.Tensor):
    '''
        x: [t, b, d]
        lengths: [b]
        output: [t', d], t' = sum(lengths)
    '''
    mask = (1 - make_pad_mask(lengths, x.shape[0]).to(torch.int32)).bool()
    return x.permute(1, 0, 2)[mask].unsqueeze(1)


def unpack_sequence(flat_x: torch.Tensor, lengths: torch.Tensor, total_length: int = None):
    """
    Reverses the pack_sequence operation.
    
    Args:
        flat_x: [sum(lengths), 1, d] - The flattened contiguous sequence.
        lengths: [b] - The number of valid tokens per batch.
        total_length: (Optional) The original max time dimension 't'. 
                      If None, defaults to lengths.max().
    
    Returns:
        output: [t, b, d] - The padded tensor with original layout.
    """
    flat_x = flat_x.squeeze(1)
    batch_size = lengths.size(0)
    dim = flat_x.size(1)
    
    # If original max length isn't provided, infer it from the longest sequence
    if total_length is None:
        total_length = int(lengths.max().item())

    # 1. Recreate the same mask used in packing [Batch, Time]
    # Note: We create this on the same device as flat_x
    mask = (torch.arange(total_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).to(flat_x.device)
    
    # 2. Initialize output tensor [Batch, Time, Dim] with zeros
    # We use Batch-First initially because flat_x is ordered by Batch
    output = torch.zeros(batch_size, total_length, dim, 
                         device=flat_x.device, dtype=flat_x.dtype)
    
    # 3. Fill the masked locations
    # Since flat_x was created via boolean indexing on a [B, T, D] tensor,
    # assigning it back using the same mask puts elements in the exact same order.
    output[mask] = flat_x
    
    # 4. Permute back to [Time, Batch, Dim] to match original input
    return output.permute(1, 0, 2)


def pack_rope(rotary_emb, lengths):
    '''
    rotary_emb: [max_seq_len, 1, 1, dim] (e.g., [96, 1, 1, 128])
    lengths: [batch_size] containing sequence lengths
    '''
    max_seq_len = rotary_emb.shape[0]
    batch_size = lengths.shape[0]
    device = lengths.device
    
    # 1. Create a grid of positions [batch_size, max_seq_len]
    #    Row 0: 0, 1, 2, ... T
    #    Row 1: 0, 1, 2, ... T
    position_ids = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 2. Create a boolean mask [batch_size, max_seq_len]
    #    True where position < length
    mask = position_ids < lengths.unsqueeze(1)
    
    # 3. Apply mask to flatten the position_ids
    #    This gives you [0, 1, ... L0-1, 0, 1, ... L1-1, ...]
    packed_indices = position_ids[mask]
    
    # 4. Gather the embeddings using the indices
    #    rotary_emb is indexed on the first dimension (sequence length)
    #    Output shape: [sum(lengths), 1, 1, 128]
    return rotary_emb[packed_indices]


class Qwen3Encoder(MegatronModule):

    def __init__(
        self,
        config: AudioConfig,
        max_sequence_length: int,
        position_embedding_type: str = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        self.position_embedding_type = position_embedding_type
        self.encoder_config = config.encoder_transformer_config
        self.encoder_layer_spec = config.encoder_transformer_layer_spec

        # self.embed = Conv2dSubsampling(config.input_size, self.encoder_config.hidden_size, 0.0)
        self.conv1 = Conv1dQwen(config.input_size, config.linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1dQwen(config.linear_units, self.encoder_config.hidden_size, kernel_size=3, stride=2, padding=1)

        if self.position_embedding_type == 'rope' and not self.encoder_config.multi_latent_attention:
            self.rotary_pos_emb = MegatronRotaryEmbedding(
                kv_channels=self.encoder_config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.encoder_config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                use_cpu_initialization=self.encoder_config.use_cpu_initialization,
                context_parallel_size=1
            )
        
        self.encoder = TransformerBlock(
            config=self.encoder_config,
            spec=self.encoder_layer_spec,
            pre_process=True,
            post_process=True,
        )

        self.use_padmask = config.use_padmask
        self.downsample_rate = config.downsample_rate
        self.use_packed_attention = config.encoder_use_packed_attention
        self.streaming = config.encoder_streaming
        self.streaming_chunk_ms = config.encoder_streaming_chunk_ms
        if self.streaming_chunk_ms <= 0:
            raise ValueError(
                f"audio encoder streaming chunk size must be positive, got {self.streaming_chunk_ms} ms"
            )
        # The input fbank has a 10 ms frame shift. Qwen3Encoder downsamples time
        # with both convolution strides before applying self-attention.
        conv_downsample = self.conv1.stride[0] * self.conv2.stride[0]
        self.streaming_chunk_frames = max(
            1, round(self.streaming_chunk_ms / (10.0 * conv_downsample))
        )
        if self.streaming and self.use_packed_attention:
            # THD packed metadata describes sequence boundaries only; it cannot
            # represent the block-causal mask required by streaming attention.
            logging.warning(
                "Audio encoder streaming attention disables packed attention"
            )
            self.use_packed_attention = False
        if self.streaming:
            logging.info(
                "Using streaming audio encoder attention: chunk=%s ms (%s encoder frames)",
                self.streaming_chunk_ms,
                self.streaming_chunk_frames,
            )
        if self.use_packed_attention:
            logging.info("Using packed attention for audio encoder")

    def _streaming_attention_mask(self, lengths: Tensor, sequence_length: int, device) -> Tensor:
        """Mask future chunks and padded keys; True entries are not attendable."""
        positions = torch.arange(sequence_length, device=device)
        chunk_ids = torch.div(positions, self.streaming_chunk_frames, rounding_mode='floor')
        # Query rows may attend all keys in their own chunk and in older chunks.
        future_chunks = chunk_ids.unsqueeze(0) > chunk_ids.unsqueeze(1)
        padded_keys = positions.unsqueeze(0) >= lengths.to(device=device).unsqueeze(1)
        return future_chunks.unsqueeze(0).unsqueeze(0) | padded_keys[:, None, None, :]
    
    def forward(self,
            x: Tensor,
            ilens: Optional[Tensor] = None,
            **kwargs
        ):
        '''
        x: [b, t, d]
        ilens: [b]

        output:
          x: [b, t, d]
          olens: 
        '''
        x = x.permute(0, 2, 1)  # [b, t, d] -> [b, d, t]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(2, 0, 1)  # [b, d, t] -> [t, b, d]

        # attention_mask = masks.squeeze(1).to(torch.int32)
        # attention_mask = masks.unsqueeze(2)
        attention_mask = None

        n_frames = x.shape[0]
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
                    (x.shape[1],), max_pos, dtype=torch.long, device=x.device
                )
            attention_mask = self._streaming_attention_mask(olens, max_pos, x.device)
        cu_seqlens = torch.zeros(olens.shape[0] + 1, dtype=torch.int32, device=x.device)
        cu_seqlens_padded = torch.arange(0, olens.shape[0] + 1, dtype=torch.int32, device=x.device) * x.shape[0]
        if self.use_padmask or self.use_packed_attention:
            cu_seqlens[1:] = torch.cumsum(olens, 0)
            max_seqlen = olens.max().to(torch.int32).cuda()
        else:
            cu_seqlens = torch.arange(0, x.shape[1] + 1, device='cuda', dtype=torch.int32) * max_pos
            max_seqlen = torch.Tensor([x.shape[0]]).to(torch.int32).cuda()

        if self.use_packed_attention:
            x = pack_sequence(x, olens)
            qkv_format = 'thd'
        else:
            qkv_format = 'bshd'

        rotary_pos_emb = None
        if self.position_embedding_type == 'rope' and not self.encoder_config.multi_latent_attention:
            if self.use_packed_attention:
                rotary_seq_len = x.shape[0]
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
                rotary_pos_emb = pack_rope(rotary_pos_emb, olens)   # cu_seqlens is not passed into apply_rotary_pos_emb() so we need to manually pack this
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    None, self.encoder, x, self.encoder_config
                )
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        x = x.contiguous()
        packed_seq_params = None if self.streaming else {
            'cu_seqlens_q': cu_seqlens,
            'cu_seqlens_kv': cu_seqlens,
            # 'cu_seqlens_q_padded': cu_seqlens_padded,
            # 'cu_seqlens_kv_padded': cu_seqlens_padded,
            'max_seqlen_q': max_seqlen,
            'max_seqlen_kv': max_seqlen,
            'qkv_format': qkv_format
        }
        x = self.encoder(
            hidden_states=x,
            attention_mask=attention_mask,
            inference_params=None,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        if self.use_packed_attention:
            x = unpack_sequence(x, olens, max_pos)

        return x.permute(1, 0, 2), olens  # [t, b, d] -> [b, t, d]
