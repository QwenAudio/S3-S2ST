import torch
import os
from typing import Optional, Union, List
import math


def build_cu_seqlens_for_packed_batch(
    cu_seqlens_per_row: Union[List[torch.Tensor], torch.Tensor],
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """
    Build a single cu_seqlens for TE thd format when batch_size > 1 and each row
    is packed (multiple segments per row).

    When input_ids are [batch_size, seq_len] with each row packed to 8K, the
    Attention layer with qkv_format='thd' flattens QKV to (batch_size * seq_len, H, D)
    in order: row0's seq_len, then row1's, ... So segment boundaries must be
    global indices into this flattened sequence.

    Padding at the end of each row:
        If each sequence has padding at the end (e.g. real length 6000, padded to 8192),
        treat the padding as the last segment: for that row pass
        [0, seg1_end, seg2_end, ..., real_end, seq_len]. The last segment
        [real_end, seq_len] is then the padding; real tokens (in earlier segments)
        will not attend to it (different segment), and padding positions only
        attend within their own segment (their output is typically ignored).
        Example: row with real end at 6000 -> [0, 100, 250, 6000, 8192].

    Args:
        cu_seqlens_per_row: Either a list of 1D tensors, one per batch row, each
            shape [num_segments_in_row + 1] with values in [0, seq_len], e.g.
            for row 0: [0, 100, 250, 8192]; for row 1: [0, 200, 8192].
            With end padding: [0, 100, 250, 6000, 8192] (last segment = padding).
            Or a 2D tensor [batch_size, max_num_segments+1] with padding (e.g. -1)
            for shorter rows; will be interpreted row by row.
        seq_len: Length of each row (e.g. 8192). Each row's last boundary must equal this.
        device: Output device. Inferred from cu_seqlens_per_row if None.
        dtype: Output dtype; TE expects int32.

    Returns:
        cu_seqlens: 1D tensor [total_num_segments + 1], suitable for
            packed_seq_params["cu_seqlens_q"] / cu_seqlens_kv with qkv_format="thd".

    max_seqlen for packed_seq_params:
        Use max_seqlen_from_cu_seqlens(cu_seqlens) to get the maximum segment length
        for packed_seq_params["max_seqlen_q"] and ["max_seqlen_kv"] (self-attn: same).
    """
    if isinstance(cu_seqlens_per_row, torch.Tensor):
        assert cu_seqlens_per_row.dim() == 2
        batch_size = cu_seqlens_per_row.size(0)
        device = device or cu_seqlens_per_row.device
        parts = []
        for i in range(batch_size):
            row = cu_seqlens_per_row[i]
            valid = (row >= 0) & (row <= seq_len)
            if valid.any():
                row = row[valid].to(device)
                row = row.unique(sorted=True)
                if row[0].item() != 0:
                    row = torch.cat([torch.tensor([0], device=device, dtype=row.dtype), row])
                if row[-1].item() != seq_len:
                    row = torch.cat([row, torch.tensor([seq_len], device=device, dtype=row.dtype)])
            else:
                row = torch.tensor([0, seq_len], device=device, dtype=torch.int64)
            offset = i * seq_len
            parts.append((offset + row).to(dtype))
        cu_seqlens = torch.cat(parts)
    else:
        batch_size = len(cu_seqlens_per_row)
        device = device or cu_seqlens_per_row[0].device
        parts = []
        for i in range(batch_size):
            row = cu_seqlens_per_row[i].to(device)
            if row.dim() == 0:
                row = row.unsqueeze(0)
            assert row.dim() == 1 and row.size(0) >= 2 and row[0].item() == 0 and row[-1].item() == seq_len, (
                f"Row {i}: expected [0, ..., seq_len], got shape {row.shape} values {row.tolist()}"
            )
            offset = i * seq_len
            parts.append((offset + row).to(dtype))
        cu_seqlens = torch.cat(parts)
    return cu_seqlens


def max_seqlen_from_cu_seqlens(cu_seqlens: torch.Tensor) -> int:
    """
    Maximum segment length from cumulative sequence lengths.
    Use for packed_seq_params["max_seqlen_q"] / ["max_seqlen_kv"] with TE thd format.

    Each segment has length cu_seqlens[i+1] - cu_seqlens[i]; this returns the max over i.
    """
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        return 0
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(lengths.max().item())


def build_packed_segment_causal_attention_mask(
    cu_seqlens: torch.Tensor,
    seq_len: int,
    device: torch.device,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Build attention mask for packed SFT sequences: causal within each segment,
    no attention across segments. Used with TransformerEngine DotProductAttention
    when attn_mask_type is "arbitrary".

    Semantics:
        - Position q can attend to position k iff: q and k are in the same
          segment AND k <= q (causal within segment).
        - Mask[q, k] = True means (q, k) is masked out (blocked); False means allowed.

    Args:
        cu_seqlens: Cumulative sequence lengths, shape [num_segments + 1], e.g.
            [0, 100, 250, 8192] for 3 segments of length 100, 150, 7942.
            Segment s is the index range [cu_seqlens[s], cu_seqlens[s+1]).
        seq_len: Total sequence length (e.g. 8192). Must equal cu_seqlens[-1].
        device: Device for the output mask.
        batch_size: Output will be [batch_size, 1, seq_len, seq_len]. Use > 1
            when each batch row shares the same segment boundaries.
        dtype: Output dtype; use torch.bool for TE.

    Returns:
        mask: Shape [batch_size, 1, seq_len, seq_len], dtype bool.
            True = masked out (block attention), False = allow attention.
    """
    assert cu_seqlens.dim() == 1 and cu_seqlens.size(0) >= 2
    assert seq_len == cu_seqlens[-1].item(), (seq_len, cu_seqlens[-1].item())
    cu_seqlens = cu_seqlens.to(device=device)

    # [seq_len]: segment_id[i] = segment index for position i
    segment_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    for s in range(cu_seqlens.size(0) - 1):
        start = cu_seqlens[s].item()
        end = cu_seqlens[s + 1].item()
        segment_id[start:end] = s

    # [seq_len, 1] vs [1, seq_len] -> [seq_len, seq_len]
    seg_q = segment_id.unsqueeze(1)
    seg_k = segment_id.unsqueeze(0)
    different_segment = seg_q != seg_k

    # Causal: block when k > q
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_block = k_idx > q_idx

    # Block attention where: different segment OR causal (k > q)
    mask = different_segment | causal_block
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    if batch_size > 1:
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
    return mask.to(dtype=dtype)


def pad_cu_seqlens(cu_seqlens, batch_size, seq_length):
    cu_seqlens_padded = torch.ones(batch_size * seq_length, dtype=cu_seqlens.dtype, device=cu_seqlens.device) * -1
    cu_seqlens_padded[:cu_seqlens.numel()] = cu_seqlens
    cu_seqlens_padded = cu_seqlens_padded.view(batch_size, seq_length)
    return cu_seqlens_padded


def unpad_cu_seqlens(cu_seqlens_padded):
    cu_seqlens = cu_seqlens_padded.view(-1)
    return cu_seqlens[cu_seqlens >= 0].contiguous()


def get_nvte_fused_attn(default=None):
    return os.environ.get("NVTE_FUSED_ATTN", default)


def set_nvte_fused_attn(value: Optional[str] = None):
    '''
    The encoder input is variable length, fused attn runs tuning for each shape, so is very slow. Bucketing will speed up.
    NVTE_FUSED_ATTN = 1 will fallback to flash attn.
    '''
    if value is not None:
        os.environ["NVTE_FUSED_ATTN"] = value
    else:
        os.environ.pop("NVTE_FUSED_ATTN", None)


def freeze_module(module: torch.nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def set_module_lr_scale(module: torch.nn.Module, scale: float):
    for param in module.parameters():
        param.lr_scale = scale


def process_with_sliding_window_and_bucketing(
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    encode_fn,
    audio_adaptor_fn,
    window_size: int = 10000000,
    left_context: int = 0,
    bin_threshold: int = 200,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sliding window + bucketing for continuous feature encoder+adaptor pipeline.

    Args:
        speech (Tensor): [B, T_in, D_in]
        speech_lengths (Tensor): [B] or [B,1]
        encode_fn: function(speech_chunk, lengths_chunk) -> (enc_out, enc_out_lens)
        audio_adaptor_fn: function(enc_out, enc_out_lens) -> (adapted_out, adapted_lens)
        window_size (int): max window length in raw input frames for each chunk
        left_context (int): overlap (raw frames) for subsequent chunks
        bin_threshold (int): length difference threshold for binning
        device (str): CUDA/CPU

    Returns:
        (final_outputs, final_output_lengths):
            final_outputs: [B, T_total_out, D_out], padded
            final_output_lengths: [B], int tensor
    """
    if len(speech_lengths.size()) > 1:
        speech_lengths = speech_lengths[:, 0]

    B, Lmax, Din = speech.shape
    true_lengths = speech_lengths

    # Step 1: create sliding window chunks from raw input
    chunks = []
    for idx in range(B):
        length = int(true_lengths[idx].item())
        feat = speech[idx, :length]  # [len, Din]

        start = 0
        chunk_idx_in_original = 0
        while start < length:
            end = min(start + window_size, length)
            if chunk_idx_in_original > 0:
                context_start = max(0, start - left_context)
            else:
                context_start = start
            chunk_tensor = feat[context_start:end]  # [chunk_len, Din]
            chunks.append({
                "tensor": chunk_tensor,
                "length": chunk_tensor.shape[0],
                "orig_idx": idx,
                "chunk_idx": chunk_idx_in_original,
                "raw_left_context": start - context_start
            })
            chunk_idx_in_original += 1
            if end >= length:
                break
            start += window_size

    # Step 2: sort by length
    chunks_sorted = sorted(chunks, key=lambda x: x["length"])

    # Step 3: bin chunks to reduce padding
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

    # Step 4: process each bin
    results = {}  # (orig_idx, chunk_idx) -> Tensor
    for bin_chunks in bins:
        maxlen = max(ch["length"] for ch in bin_chunks)
        batch_tensors = torch.stack([
            torch.cat([
                ch["tensor"],
                torch.zeros(maxlen - ch["length"], Din, dtype=ch["tensor"].dtype, device=device)
            ], dim=0)
            for ch in bin_chunks
        ], dim=0)  # [B_bin, maxlen, Din]
        batch_lengths = torch.tensor([ch["length"] for ch in bin_chunks], device=device)

        # Pass through encoder
        enc_out, enc_out_lens = encode_fn(batch_tensors, batch_lengths)
        # Pass through adaptor
        adapted_out, adapted_out_lens = audio_adaptor_fn(enc_out, enc_out_lens)

        # Infer downsample ratio from raw frames to adapted frames
        ds_factor = batch_lengths.float() / adapted_out_lens.float()

        for ch, out_tensor, out_len, ds in zip(
            bin_chunks, adapted_out, adapted_out_lens, ds_factor
        ):
            raw_left_ctx = ch["raw_left_context"]
            # Convert raw_left_ctx to output frame units
            drop = math.floor(raw_left_ctx / ds.item()) if ch["chunk_idx"] > 0 else 0
            trimmed = out_tensor[drop:int(out_len.item())]
            results[(ch["orig_idx"], ch["chunk_idx"])] = trimmed

    # Step 5: reassemble
    final_outputs = []
    final_output_lengths = []
    for idx in range(B):
        chunk_idx = 0
        sample_parts = []
        while (idx, chunk_idx) in results:
            sample_parts.append(results[(idx, chunk_idx)])
            chunk_idx += 1
        if sample_parts:
            cat_tokens = torch.cat(sample_parts, dim=0)
            final_outputs.append(cat_tokens)
            final_output_lengths.append(cat_tokens.shape[0])
        else:
            final_outputs.append(torch.empty(0, device=device))
            final_output_lengths.append(0)

    # Pad output to batch tensor
    final_outputs = torch.nn.utils.rnn.pad_sequence(final_outputs, batch_first=True, padding_value=0)
    final_output_lengths = torch.tensor(final_output_lengths, dtype=torch.int32)

    return final_outputs, final_output_lengths