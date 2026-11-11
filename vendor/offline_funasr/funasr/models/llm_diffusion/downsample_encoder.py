# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Tuple, Union

import torch
import torch.utils.checkpoint as ckpt
from .embedding import (PositionalEncoding,
                        RelPositionalEncoding,
                        WhisperPositionalEncoding,
                        LearnablePositionalEncoding,
                        NoPositionalEncoding,
                        EspnetRelPositionalEncoding)
import torch.nn.functional as F
import torch.nn as nn

class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)

class LinearSubsampling2Dropout(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out_unified = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.Conv1d(idim, odim, kernel_size=2, stride=2, groups=1),
            Transpose(1, 2),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.dropout_rate = dropout_rate

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        time = x.size(1)
        x = self.out_unified(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x, pos_emb = self.pos_enc(x, offset)
        x_mask_out = x_mask[:, :, ::2]
        assert ((x_mask.sum(dim=-1) / 2) == x_mask_out.sum(dim=-1)).all()
        return x, pos_emb, x_mask_out

class LinearSubsampling4Dropout(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out_unified = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.Conv1d(idim, odim, kernel_size=4, stride=4, groups=1),
            Transpose(1, 2),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.dropout_rate = dropout_rate

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        time = x.size(1)
        x = self.out_unified(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x, pos_emb = self.pos_enc(x, offset)
        x_mask_out = x_mask[:, :, ::4]
        assert ((x_mask.sum(dim=-1) / 4) == x_mask_out.sum(dim=-1)).all()
        return x, pos_emb, x_mask_out


class LinearSubsampling8Dropout(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out_unified = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.Conv1d(idim, odim, kernel_size=8, stride=8, groups=1),
            Transpose(1, 2),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.dropout_rate = dropout_rate

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        """
        time = x.size(1)
        x = self.out_unified(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x, pos_emb = self.pos_enc(x, offset)
        x_mask_out = x_mask[:, :, ::8]
        assert ((x_mask.sum(dim=-1) / 8) == x_mask_out.sum(dim=-1)).all()
        return x, pos_emb, x_mask_out

class LinearSubsampling16Dropout(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out_unified = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.Conv1d(idim, odim, kernel_size=16, stride=16, groups=1),
            Transpose(1, 2),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.dropout_rate = dropout_rate

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        time = x.size(1)
        x = self.out_unified(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x, pos_emb = self.pos_enc(x, offset)
        x_mask_out = x_mask[:, :, ::16]
        assert ((x_mask.sum(dim=-1) / 16) == x_mask_out.sum(dim=-1)).all()
        return x, pos_emb, x_mask_out

class LinearSubsampling32Dropout(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out_unified = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.Conv1d(idim, odim, kernel_size=32, stride=32, groups=1),
            Transpose(1, 2),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.dropout_rate = dropout_rate

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        """
        time = x.size(1)
        x = self.out_unified(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x, pos_emb = self.pos_enc(x, offset)
        x_mask_out = x_mask[:, :, ::32]
        assert ((x_mask.sum(dim=-1) / 32) == x_mask_out.sum(dim=-1)).all()
        return x, pos_emb, x_mask_out


COSYVOICE_SUBSAMPLE_CLASSES = {
    "lineard2_dropout": LinearSubsampling2Dropout,
    "lineard4_dropout": LinearSubsampling4Dropout,
    "lineard8_dropout": LinearSubsampling8Dropout,
    "lineard16_dropout": LinearSubsampling16Dropout,
    "lineard32_dropout": LinearSubsampling32Dropout,
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

class LinearEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        global_cmvn: torch.nn.Module = None,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        return xs, masks

class TransformerDownsampleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2, hidden_dim=512, num_heads=4, num_layers=2, dropout=0.0):
        super(TransformerDownsampleBlock, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        #self.input_layer = nn.Linear(input_dim, hidden_dim*stride)
        self.input_layer = nn.Conv1d(input_dim*stride, hidden_dim*stride, kernel_size=1, groups=stride)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #self.positional_encoding = nn.Parameter(torch.randn(1, stride+1, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.stride = stride

    def forward(self, x, xs_lens):
        T = x.size(1)
        x_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        N_chunks = x.size(1) // self.stride
        x = x.reshape(x.size(0) * N_chunks, self.stride * x.size(2), 1)

        B,  _, _ = x.shape
        x = self.input_layer(x)
        x = x.reshape(x.size(0), self.stride, -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, L+1, D]
        # x = x + self.positional_encoding
        x = self.transformer_encoder(x)  # [B, L+1, D]
        output = x[:, 0, :]  # [B, D]

        output = output.reshape(B//N_chunks, N_chunks, -1)
        output = self.output_layer(output)
        x_mask_out = x_mask[:, :, ::self.stride]
        return output, x_mask_out
