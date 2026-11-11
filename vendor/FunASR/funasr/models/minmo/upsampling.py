#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.models.transformer.embedding import PositionalEncoding
import logging
from funasr.models.scama.utils import sequence_mask
from funasr.models.transformer.utils.nets_utils import sub_factor_to_params, pad_to_len
from typing import Optional, Tuple, Union
import math


class Conv1dUpsampling(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(
        self,
        idim,
        odim,
        kernel_size,
        stride,
        # pad,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(idim, odim, kernel_size, stride)
        # self.pad_fn = torch.nn.ConstantPad1d(pad, 0.0)
        self.stride = stride
        self.odim = odim

    def output_size(self) -> int:
        return self.odim

    def forward(self, x, x_len):
        """Subsample x."""
        x = x.transpose(1, 2)  # (b, d ,t)
        # x = self.pad_fn(x)
        x = F.relu(self.conv(x))
        # x = F.leaky_relu(self.conv(x), negative_slope=0.0)
        x = x.transpose(1, 2)  # (b, t ,d)

        if x_len is None:
            return x, None
        x_len = x_len * self.stride
        return x, x_len
