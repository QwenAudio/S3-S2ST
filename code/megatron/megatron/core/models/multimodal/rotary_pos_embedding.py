import torch
from torch import Tensor, nn
from typing import List, Optional
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding
)
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank


class Qwen2VLRotaryEmbedding(RotaryEmbedding):
    """Chunked M-RoPE used by Qwen2-VL / Qwen2.5-VL."""

    def forward(self, position_ids: torch.Tensor, mrope_section: List[int] = [16, 24, 24]) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1) # shape (3, bs, dim, 1)
        seq_expanded = seq[:, :, None, :].float()  # shape (3, bs, 1, seq_length)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3) # shape (3, bs, seq_length, dim)
        
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1) # shape (3, bs, seq_length, 2 * dim)
        else:
            bs = freqs.shape[1]
            emb = torch.stack((freqs.view(3, bs, -1, 1), freqs.view(3, bs, -1, 1)), dim=-1).view(
                3, bs, freqs.shape[0], -1
            )
        
        # generate freqs with mrope_section
        # shape (bs, seq_length, 2 * dim)
        mrope_section = mrope_section * 2
        emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)
        
        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 1)
        return emb


class Qwen3_5MoeRotaryEmbedding(RotaryEmbedding):
    """Interleaved M-RoPE used by Qwen3.5-MoE.

    Unlike Qwen2VLRotaryEmbedding which uses a chunked layout [TTT...HHH...WWW...],
    this uses an interleaved layout [THWTHWTHW...TT] controlled by mrope_section.
    Also supports partial_rotary_factor through the parent class rotary_percent.
    """

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: Optional[List[int]] = None,
    ) -> Tensor:
        """Forward pass of interleaved multimodal RoPE embedding.

        Args:
            position_ids: shape [3, batch_size, seq_len] for 3D positions (T, H, W),
                          or [batch_size, seq_len] for 1D positions.
            mrope_section: section sizes for interleaving, e.g. [11, 11, 10].
                          If None, defaults to [11, 11, 10].

        Returns:
            Tensor: Embeddings after applying interleaved M-RoPE.
                    Shape: [seq_len, batch_size, 1, 2*dim]
        """
        if mrope_section is None:
            mrope_section = [11, 11, 10]

        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # If 2D position_ids [bs, seq_len], expand to 3D [3, bs, seq_len]
        if seq.ndim == 2:
            seq = seq[None, ...].expand(3, seq.shape[0], -1)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # inv_freq shape: (dim // 2,)
        # expanded: (3, bs, dim//2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, seq.shape[1], -1, 1
        )
        # seq_expanded: (3, bs, 1, seq_len)
        seq_expanded = seq[:, :, None, :].float()

        # freqs: (3, bs, seq_len, dim//2)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)

        # Apply interleaved M-RoPE: reorganize from [TTT...HHH...WWW] to [THWTHW...TT]
        freqs = self._apply_interleaved_mrope(freqs, mrope_section)

        # Double for cos/sin: (bs, seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_len, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()

        if parallel_state.get_context_parallel_world_size() > 1:
            emb = get_pos_emb_on_this_cp_rank(emb, 1)

        return emb

    @staticmethod
    def _apply_interleaved_mrope(
        freqs: torch.Tensor,
        mrope_section: List[int],
    ) -> torch.Tensor:
        """Apply interleaved M-RoPE layout.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.

        Args:
            freqs: (3, bs, seq_len, dim//2) - frequencies for T, H, W dimensions
            mrope_section: (3,) - section sizes, e.g. [11, 11, 10]

        Returns:
            freqs_t: (bs, seq_len, dim//2) - interleaved frequencies
        """
        # Start with T dimension as the base
        freqs_t = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t
