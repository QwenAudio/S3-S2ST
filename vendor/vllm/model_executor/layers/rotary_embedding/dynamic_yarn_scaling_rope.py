# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import numpy as np
import torch

from .base import RotaryEmbedding
from .common import yarn_find_correction_range, yarn_get_mscale, yarn_linear_ramp_mask


class DynamicYaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        dynamic_yarn: bool = False,
        semi_dynamic_yarn: bool = False,
        dynamic_scaling_factor: Optional[float] = None,
        truncate: bool = True,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.dynamic_yarn = dynamic_yarn
        self.semi_dynamic_yarn = semi_dynamic_yarn
        if dynamic_scaling_factor is not None:
            self.dynamic_scaling_factor = dynamic_scaling_factor
        else:
            self.dynamic_scaling_factor = scaling_factor
        self.truncate = truncate

        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
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
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache_yarn(
        self,
        scaling_factor: Optional[float] = None,
        mscale: Optional[float] = None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        if scaling_factor is None:
            scaling_factor = self.scaling_factor
        if mscale is None:
            mscale = self.mscale
        if seq_len is None:
            seq_len = self.max_position_embeddings
        if seq_len > self.max_position_embeddings:
            scaling_factor = seq_len / self.max_position_embeddings
            mscale = yarn_get_mscale(scaling_factor) * self.attn_factor
        inv_freq = self._compute_inv_freq_yarn(scaling_factor)
        t = torch.arange(
            max(seq_len, self.max_position_embeddings * scaling_factor),
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def update_cos_sin_cache(
        self,
        positions: torch.Tensor,
        subquery_start_loc: list[int] = None,
        orig_seq_lens: Union[list[int], np.ndarray, torch.Tensor] = None,
    ):
        """Inplace update the cos_sin_cache at the required positions
        where the position < max_position_embeddings.

        It works without undefined behavior, as

        - for prefill, we assumes that different long sequence's prefill
          won't be batched;
        - for decoding, if two sequences's decoding position is same, they
          must have the same seq length, and requires the same cos_sin_cache;
        - splitfuse (chunked-prefill) is not considered yet.
        """
        if subquery_start_loc is None:
            return

        # Update the recomputed cos_sin_cache to [0:total_query_len]
        # and revise the positions to the new positions

        offset = self.max_position_embeddings
        for i in range(len(subquery_start_loc) - 1):
            start = subquery_start_loc[i]
            end = subquery_start_loc[i + 1]
            if start == end:
                continue
            seq_len = orig_seq_lens[i]
            if seq_len <= self.max_position_embeddings:
                continue
            query_len = end - start

            if self.dynamic_yarn:
                # update the cos_sin_cache at the required positions
                scaling_factor = seq_len / self.max_position_embeddings
                mscale = yarn_get_mscale(scaling_factor) * self.attn_factor
                inv_freq = self._compute_inv_freq_yarn(scaling_factor).to(
                    positions.device
                )
                freqs = torch.einsum("i,j -> ij", positions[start:end], inv_freq)
                cos = freqs.cos() * mscale
                sin = freqs.sin() * mscale

                # revise positions
                positions[start:end] -= positions[start].item() - offset

                # update the cos_sin_cache
                self.cos_sin_cache[offset : offset + query_len] = torch.cat(
                    (cos, sin), dim=-1
                )
                offset += query_len
            else:
                # revise positions
                positions[start:end] += self.max_position_embeddings
