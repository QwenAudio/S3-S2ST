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
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask

class LinearUpsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, input_size: int, output_size: int, stride: int = 2):
        super().__init__()
        self.channels = input_size
        self.out_channels = output_size
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.nn = nn.Linear(self.channels, self.out_channels)

    def output_size(self) -> int:
        return self.out_channels

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor, streaming) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs.transpose(1, 2).contiguous(), scale_factor=float(self.stride), mode="nearest")
        outputs = self.nn(outputs.transpose(1, 2).contiguous())
        T = outputs.size(1)
        masks = ~make_pad_mask(input_lengths * self.stride, T).unsqueeze(1)  # (B, 1, T)
        return outputs, masks

class LinearUpsample1D_V2(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, input_size: int, output_size: int, stride: int = 2):
        super().__init__()
        self.channels = input_size
        self.out_channels = output_size
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.nn = nn.Linear(self.channels, self.out_channels * stride)

    def output_size(self) -> int:
        return self.out_channels

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor, streaming=True) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.nn(inputs.contiguous())
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1] * self.stride, outputs.shape[2]//self.stride)
        T = outputs.size(1)
        if input_lengths is not None:
            masks = ~make_pad_mask(input_lengths * self.stride, T).unsqueeze(1)  # (B, 1, T)
        else:
            masks = None
        return outputs, masks

class TransformerUpsampleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2, hidden_dim=512, num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim*stride)
        #self.pos_emb = nn.Parameter(torch.randn(1, stride, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.stride = stride

    def forward(self, x, input_lengths):
        # z: [B, 1, D]
        B, L, D = x.shape
        x = self.input_layer(x)
        x = x.reshape(x.size(0)*x.size(1), self.stride, x.size(2)//self.stride)
        #x = x + self.pos_emb
        x = self.transformer(x)
        x = x.reshape(B, L*self.stride, -1)
        x = self.output_layer(x)
        T = x.size(1)
        if input_lengths is not None:
            masks = ~make_pad_mask(input_lengths * self.stride, T).unsqueeze(1)  # (B, 1, T)
        else:
            masks = None
        return x, masks
