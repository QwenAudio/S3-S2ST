# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import numpy as np
import torch

from .common import yarn_get_mscale
from .dynamic_yarn_scaling_rope import DynamicYaRNScalingRotaryEmbedding
from .mrope import MRotaryEmbedding


class DynamicYaRNScalingMRotaryEmbedding(MRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[list[int]] = None,
        mrope_interleaved: bool = False,
        # YaRN parameters.
        *,
        scaling_factor: Optional[float] = None,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        dynamic_yarn: bool = False,
        semi_dynamic_yarn: bool = False,
        dynamic_scaling_factor: Optional[float] = None,
        truncate: bool = True,
    ) -> None:
        self.truncate = truncate
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
            scaling_factor=scaling_factor,
            extrapolation_factor=extrapolation_factor,
            attn_factor=attn_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )

        self.dynamic_yarn = dynamic_yarn
        self.semi_dynamic_yarn = semi_dynamic_yarn
        if dynamic_scaling_factor is not None:
            self.dynamic_scaling_factor = dynamic_scaling_factor
        else:
            self.dynamic_scaling_factor = scaling_factor

        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)

        # Initialize the cos_sin_cache for dynamic yarn.
        if self.semi_dynamic_yarn or self.dynamic_yarn:
            scaled_cache = self._compute_cos_sin_cache_yarn(
                self.dynamic_scaling_factor,
                float(yarn_get_mscale(self.dynamic_scaling_factor) * attn_factor),
            )
            scaled_cache = scaled_cache.to(torch.get_default_dtype())

            # merge
            merged_cache = torch.cat((self.cos_sin_cache, scaled_cache), dim=0)
            delattr(self, "cos_sin_cache")
            self.register_buffer("cos_sin_cache", merged_cache, persistent=False)

    def _compute_inv_freq_yarn(
        self,
        scaling_factor: float,
    ) -> torch.Tensor:
        return DynamicYaRNScalingRotaryEmbedding._compute_inv_freq_yarn(
            self, scaling_factor
        )

    def _compute_cos_sin_cache_yarn(
        self,
        scaling_factor: Optional[float] = None,
        mscale: Optional[float] = None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        return DynamicYaRNScalingRotaryEmbedding._compute_cos_sin_cache_yarn(
            self, scaling_factor, mscale, seq_len
        )

    def update_cos_sin_cache(
        self,
        positions: torch.Tensor,
        subquery_start_loc: list[int] = None,
        orig_seq_lens: Union[list[int], np.ndarray, torch.Tensor] = None,
    ):
        return DynamicYaRNScalingRotaryEmbedding.update_cos_sin_cache(
            self, positions, subquery_start_loc, orig_seq_lens
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return MRotaryEmbedding.forward(self, positions, query, key)
