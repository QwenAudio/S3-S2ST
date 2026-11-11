# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import AudioConfig


class LayerNorm(nn.LayerNorm):
    """Layer normalization module matching the original implementation."""
    
    def __init__(self, nout: int, dim: int = -1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization."""
        if self.dim == -1:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class Linear(nn.Linear):
    """Linear layer with proper initialization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention implementation matching the original."""
    
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0):
        super().__init__()
        assert n_feat % n_head == 0
        
        self.n_head = n_head
        self.n_feat = n_feat
        self.d_k = n_feat // n_head
        
        self.w_qs = Linear(n_feat, n_feat)
        self.w_ks = Linear(n_feat, n_feat)
        self.w_vs = Linear(n_feat, n_feat)
        self.linear = Linear(n_feat, n_feat)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        n_batch = query.size(0)
        q = self.w_qs(query).view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(key).view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(value).view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time2, 1)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        attn = self.dropout(attn)
        
        p_attn = torch.matmul(attn, v)
        p_attn = p_attn.transpose(1, 2).contiguous().view(n_batch, -1, self.n_feat)
        
        return self.linear(p_attn), attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed forward network matching the original."""
    
    def __init__(self, d_in: int, d_hid: int, dropout_rate: float = 0.0):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid)
        self.w_2 = Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """Encoder layer matching the original implementation from encoder.py.
    
    This implementation exactly matches the original EncoderLayer from funasr.
    """
    
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float = 0.0,
        normalize_before: bool = True,
        concat_after: bool = False,
        stochastic_depth_rate: float = 0.0,
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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, cache: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Compute encoded features.

        Args:
            x (Tensor): Input tensor (#batch, time, size).
            mask (Tensor): Mask tensor for the input (#batch, time).
            cache (Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            Tensor: Output tensor (#batch, time, size).
            Tensor: Mask tensor (#batch, time).
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
            x = residual + stoch_layer_coeff * self.dropout(self.self_attn(x_q, x, x, mask)[0])
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


def sequence_mask(lengths: Tensor, max_len: Optional[int] = None, flip: bool = False) -> Tensor:
    """Create sequence mask from lengths."""
    if max_len is None:
        max_len = lengths.max()
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(1).expand(batch_size, max_len)
    if flip:
        return seq_range_expand >= seq_length_expand
    else:
        return seq_range_expand < seq_length_expand


def block_causal_attention_mask(
    lengths: Tensor,
    max_len: int,
    chunk_size: int,
) -> Tensor:
    """Allow adaptor tokens to attend to their current and earlier chunks."""
    if chunk_size <= 0:
        raise ValueError(f"adaptor streaming chunk size must be positive: {chunk_size}")
    positions = torch.arange(max_len, device=lengths.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_ids = torch.div(positions, chunk_size, rounding_mode="floor")
    # Rows are queries and columns are keys.  Tokens inside the current closed
    # audio chunk may see one another, but never a later audio chunk.
    block_causal = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)
    return block_causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)


class AudioAdaptor(MegatronModule):
    """Audio adaptor to connect audio encoder with language model.
    
    This implementation matches the original Transformer adaptor from adaptor.py.
    
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
        
        self.k = config.adaptor_downsample_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.n_layer = config.adaptor_num_layers
        self.attention_heads = 8
        self.ffn_dim = config.ffn_dim
        self.attention_dropout_rate = 0.0
        self.dropout_rate = 0.0
        self.streaming_chunk_size = int(
            getattr(config, "adaptor_streaming_chunk_size", 0)
        )
        if self.streaming_chunk_size > 0:
            print(
                "[TRUE STREAMING ADAPTOR] block-causal chunk="
                f"{self.streaming_chunk_size} adaptor tokens",
                flush=True,
            )
        
        if self.pre_process:
            self.linear1 = Linear(self.encoder_dim * self.k, self.ffn_dim)
            self.relu = nn.ReLU()
            self.linear2 = Linear(self.ffn_dim, self.llm_dim)
        
        # Additional transformer layers if specified - matching original implementation
        self.blocks = None
        if self.n_layer > 0:
            self.blocks = nn.ModuleList([
                EncoderLayer(
                    self.llm_dim,
                    MultiHeadedAttention(
                        self.attention_heads,
                        self.llm_dim,
                        self.attention_dropout_rate,
                    ),
                    PositionwiseFeedForward(
                        self.llm_dim,
                        self.llm_dim // 4,
                        self.dropout_rate,
                    ),
                    self.dropout_rate,
                    normalize_before=True,
                    concat_after=False,
                    stochastic_depth_rate=0.0,
                )
                for _ in range(self.n_layer)
            ])
        
        # Create mask function
        self.make_pad_mask = lambda lengths, max_len=None: sequence_mask(lengths, max_len, flip=False)
    
    def forward(
        self, 
        x: Tensor, 
        ilens: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of the audio adaptor.
        
        This implementation exactly matches the original Transformer.forward method.
        
        Args:
            x (Tensor): Input audio features of shape [batch, seq_len, encoder_dim]
            ilens (Tensor, optional): Input lengths of shape [batch]
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: Adapted features and lengths
        """
        if self.pre_process:
            batch_size, seq_len, dim = x.size()
            
            # Downsample by factor k - matching original implementation
            chunk_num = (seq_len - 1) // self.k + 1
            pad_num = chunk_num * self.k - seq_len
            x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
            seq_len = x.size(1)
            
            # Reshape and apply linear layers
            x = x.contiguous()
            x = x.view(batch_size, chunk_num, dim * self.k)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            
            # Update lengths
            if ilens is not None:
                olens = (ilens - 1) // self.k + 1
            else:
                olens = None
        else:
            olens = ilens
        
        # Apply additional transformer blocks if specified - matching original implementation
        if self.blocks is not None and olens is not None:
            if self.streaming_chunk_size > 0:
                masks = block_causal_attention_mask(
                    olens, x.shape[1], self.streaming_chunk_size
                ).to(x.device)
            else:
                # Historical full-record behaviour: padding mask only.
                masks = self.make_pad_mask(
                    olens, max_len=x.shape[1]
                )[:, None, :].to(x.device)
            
            for layer, block in enumerate(self.blocks):
                x, masks = block(x, masks)
        
        return x, olens
