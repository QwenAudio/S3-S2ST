import os
import unicodedata
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import AutoConfig, AutoTokenizer

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)


def contains_cj(text):
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        if 'CJK' in name or 'KATAKANA' in name or 'HIRAGANA' in name:
            return 1
    return 0

def contains_latin(text):
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        if 'LATIN' in name:
            return 1
    return 0

def contains_only_special(text):
    for char in text:
        if char.isspace():
            continue
        try:
            unicodedata.name(char)
        except ValueError:
            continue
        if unicodedata.category(char).startswith('L'):
            return 0
    return 1

def get_lang_masks(vocab_size, tokenizer):
    cj_labels, latin_labels, special_labels, lowres_labels = [], [], [], []
    for tok in range(vocab_size):
        tok_text = tokenizer.decode(tok)
        cj_label = contains_cj(tok_text)
        latin_label = contains_latin(tok_text)
        special_label = contains_only_special(tok_text)
        cj_labels.append(cj_label)
        latin_labels.append(latin_label)
        special_labels.append(special_label)
        lowres_labels.append((cj_label == 0) and 
                             (latin_label == 0) and (special_label == 0))
    current_device = torch.cuda.current_device()
    lang_masks = torch.tensor([cj_labels, latin_labels, special_labels, 
                               lowres_labels]).to(current_device)
    return lang_masks

def get_control_toks(vocab_size, tokenizer):
    special_toks = []
    for vs in tokenizer.special_tokens_map.values():
        if type(vs) is str:
            assert(len(tokenizer.encode(vs, add_special_tokens=False)) == 1)
            special_toks.append(tokenizer.encode(vs, add_special_tokens=False)[0])
        else:
            for v in vs:
                assert(len(tokenizer.encode(v, add_special_tokens=False)) == 1)
                special_toks.append(tokenizer.encode(v, add_special_tokens=False)[0])
    control_toks_mask = torch.zeros(vocab_size)
    for tok in range(vocab_size):
        tok_text = tokenizer.decode([tok])
        if len(tok_text) >= 2 and tok_text[0] == '<' and tok_text[-1] == '>':
            if tok not in special_toks:
                special_toks.append(tok)
    logger.info_once(f"CodeSwitchGate special tokens: {special_toks}")
    for tok in special_toks:
        control_toks_mask[tok] = 1
    control_toks_mask = torch.tensor(control_toks_mask).bool()
    return control_toks_mask


class InterventionSampler(nn.Module):
    def __init__(self):
        super().__init__()

        assert envs.VLLM_ENABLE_CODE_SWITCH_GATE, (
            "InterventionSampler should only be initialized when "
            "VLLM_ENABLE_CODE_SWITCH_GATE is set to True."
        )

        # Maximum number of output tokens to check during intervention operations
        self.max_output_len = envs.VLLM_CODE_SWITCH_GATE_MAX_TOKENS_TO_CHECK
        logger.info_once(
            f"max output length to check in code_switch gate: {self.max_output_len}"
        )

        self.top_p1 = 0.999
        self.top_k1 = 5

        self.top_p2 = 0.95
        self.top_k2 = 20

        tok_path = envs.VLLM_CODE_SWITCH_TOK_PATH
        logger.info_once(f"tokenizer path: {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        config = AutoConfig.from_pretrained(tok_path)

        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            vocab_size = config.text_config.vocab_size

        lang_masks = get_lang_masks(vocab_size, tokenizer)
        logger.info_once(
            f"initial lang_masks: {lang_masks[0].sum()}, "
            f"{lang_masks[1].sum()}, {lang_masks[2].sum()}, "
            f"{lang_masks[3].sum()}"
        )
        control_toks_mask = get_control_toks(vocab_size, tokenizer)
        lang_masks[:, control_toks_mask] = 0
        lang_masks[2, control_toks_mask] = 1
        logger.info_once(
            f"initial lang_masks: {lang_masks[0].sum()}, "
            f"{lang_masks[1].sum()}, {lang_masks[2].sum()}, "
            f"{lang_masks[3].sum()}"
        )
        assert((lang_masks.int().sum(dim=0) == 1).all())

        tok_to_lang = lang_masks.int().argmax(dim=0)
        self.tok_to_lang = tok_to_lang
        # tokens come from sampling_tensor are padded with vocab_size, set it to symbol
        self.tok_to_lang = torch.cat(
            (
                self.tok_to_lang,
                torch.tensor([2], device=self.tok_to_lang.device)
            ),
            dim=0
        )
        self.lang_masks = lang_masks.to(torch.bool)

    def forward(
        self,
        logits: torch.Tensor,
        logits_before_apply_temperature: torch.Tensor,
        batch_lang_logits: torch.Tensor,
        output_tokens_t: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        num_tokens = logits.shape[0]
        
        if sampling_metadata.top_p is not None:
            top_p = expand_batch_to_tokens(
                sampling_metadata.top_p,
                spec_decode_metadata.cu_num_draft_tokens,
                num_tokens,
            ) if spec_decode_metadata is not None else sampling_metadata.top_p

            top_p1 = torch.full(top_p.size(), self.top_p1, device=logits.device)
            top_p2 = torch.full(top_p.size(), self.top_p2, device=logits.device)
        else:
            top_p = None
            top_p1 = None
            top_p2 = None

        if sampling_metadata.top_k is not None:
            top_k = expand_batch_to_tokens(
                sampling_metadata.top_k,
                spec_decode_metadata.cu_num_draft_tokens,
                num_tokens,
            ) if spec_decode_metadata is not None else sampling_metadata.top_k
            top_k1 = torch.full(top_k.size(), self.top_k1, device=logits.device)
            top_k2 = torch.full(top_k.size(), self.top_k2, device=logits.device)
        else:
            top_k = None
            top_k1 = None
            top_k2 = None

        if envs.VLLM_CODE_SWITCH_GATE_USE_RAW_LOGITS:
            assert logits_before_apply_temperature is not None
            [logits1] = _apply_top_k_top_p_fused(logits, ((top_p, top_k),))
            [logits2, logits3] = _apply_top_k_top_p_fused(
                logits_before_apply_temperature, 
                ((top_p1, top_k1), (top_p2, top_k2)),
            )
        else:
            [logits1, logits2, logits3] = _apply_top_k_top_p_fused(
                logits,
                ((top_p, top_k), (top_p1, top_k1), (top_p2, top_k2)),
            )

        if output_tokens_t is None:
            output_tokens_t = make_tensor_with_pad(
                sampling_metadata.output_token_ids,
                logits.shape[-1],
                device='cpu',
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(logits.device, non_blocking=True)

        if output_tokens_t.shape[1] > self.max_output_len:
            output_tokens_t = output_tokens_t[:, :self.max_output_len]

        if spec_decode_metadata is not None:
            batch_size = output_tokens_t.shape[0]
            assert (
                batch_size == 0 or
                spec_decode_metadata.cu_num_draft_tokens.shape[0] == batch_size 
            )

            output_tokens_t = expand_batch_to_tokens_2d_triton(
                output_tokens_t,
                spec_decode_metadata.cu_num_draft_tokens,
                num_tokens,
            )

        logits = self.apply_lang_mask(
            logits1, logits2, logits3, batch_lang_logits, 
            output_tokens_t, full_logits=logits, use_full_logits=True,
        )

        return logits

    def apply_lang_mask(
        self,
        logits_topk_topp: torch.Tensor,
        logits_k5_p999: torch.Tensor,
        logits_k20_p95: torch.Tensor,
        batch_lang_logits: torch.Tensor,
        output_tokens_tensor: torch.Tensor,
        full_logits: Optional[torch.Tensor] = None,
        use_full_logits: bool = False,
    ) -> Optional[SamplerOutput]:

        assert logits_topk_topp.shape[0] == batch_lang_logits.shape[0]
        assert logits_topk_topp.shape == logits_k5_p999.shape
        assert logits_topk_topp.shape == logits_k20_p95.shape

        original_batch_logits = logits_topk_topp
        # Check if the output tensor is empty or batch sizes mismatch
        if (
            output_tokens_tensor.shape[1] == 0
            or output_tokens_tensor.shape[0] != logits_topk_topp.shape[0]
        ):
            return original_batch_logits

        # 1. update batch logits
        combine_batch_logits = update_batch_logits(
            logits_topk_topp, 
            logits_k5_p999,
            logits_k20_p95
        )
  
        vocab_size = logits_topk_topp.shape[-1]
        has_output_tokens = (output_tokens_tensor != vocab_size).any(dim=-1)
        # it left-padding, so can't take from the last
        last_token_ids_tensor = output_tokens_tensor
        tok_langs_tensor = self.tok_to_lang[last_token_ids_tensor]

        seq_len = tok_langs_tensor.shape[1]
        mask = (tok_langs_tensor != 2)
        has_non_value = mask.any(dim=1)

        rev_mask = mask.flip(dims=[1])
        offsets = rev_mask.float().argmax(dim=1)

        last_pos = seq_len - 1 - offsets

        last_pos = torch.where(has_non_value, last_pos, 
                               torch.full_like(last_pos, seq_len - 1))
        last_pos = torch.clamp(last_pos, 0, seq_len - 1)

        batch_idx = torch.arange(tok_langs_tensor.size(0), 
                                 device=tok_langs_tensor.device)
        last_tok_langs = tok_langs_tensor[batch_idx, last_pos]

        # 2. Perform batched calculations
        # Check which language categories are present in the filtered logits
        logits_lang_labels_sampling = batch_varied_top_k_p_in_logits_opt(
            logits_topk_topp,
            self.lang_masks
        )

        logits_lang_labels_topk = (
            batch_varied_top_k_p_in_logits_opt(logits_k5_p999, self.lang_masks)
            | batch_varied_top_k_p_in_logits_opt(logits_k20_p95, self.lang_masks)
        )

        pred_lang_labels = compute_pred_lang_labels(batch_lang_logits)

        original_pred_lang_labels = pred_lang_labels.clone()
        pred_lang_labels.scatter_(dim=1, index=last_tok_langs.unsqueeze(-1), value=True)
        # no intervention if last all symbol
        mask = (last_tok_langs == 2)
        pred_lang_labels[mask, :] = True

        # Determine which languages to mask out
        langs_to_mask = ~pred_lang_labels
        langs_to_mask[:, 2:4] = False  # Never mask 'special' or 'low-res' categories

        # 3. Determine which sequences need intervention
        apply_intervention = compute_intervention(
            logits_lang_labels_sampling,
            langs_to_mask,
            logits_lang_labels_topk,
            original_pred_lang_labels,
            has_output_tokens,
        )

        # 4. apply intervention
        output_logits = compute_mask_and_output_logits(
            langs_to_mask, 
            self.lang_masks, 
            apply_intervention, 
            combine_batch_logits, 
            original_batch_logits,
        )

        return output_logits

# NOTE: The kernel is copy from v1.sample.rejection_sampler to avoid circular import
MAX_SPEC_LEN = 128
def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x

# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)

@triton.jit
def cumsum_block_kernel(
    X_ptr, Y_ptr, BLOCK_SUMS,
    stride_xb, stride_xv,
    stride_yb, stride_yv,
    V, num_blocks, BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch index
    pid_v = tl.program_id(1)  # block index within row

    offs = pid_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < V

    # load a chunk
    x_ptrs = X_ptr + pid_b * stride_xb + offs * stride_xv
    vals = tl.load(x_ptrs, mask=mask, other=0.0)

    # cumsum inside one thread block
    cumsum_in_block = tl.cumsum(vals, axis=0)

    # store results
    y_ptrs = Y_ptr + pid_b * stride_yb + offs * stride_yv
    tl.store(y_ptrs, cumsum_in_block, mask=mask)

    # store last value
    last_val = tl.sum(vals)
    tl.store(BLOCK_SUMS + pid_b * num_blocks + pid_v, last_val)

@triton.jit
def add_offsets_kernel(
    Y_ptr, BLOCK_OFFSETS,
    stride_yb, stride_yn,
    N, BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)

    if pid_blk == 0:
        return

    offset_val = tl.load(
        BLOCK_OFFSETS + pid_b * (tl.cdiv(N, BLOCK_SIZE)) + pid_blk - 1
    )

    offs = pid_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    y_ptrs = Y_ptr + pid_b * stride_yb + offs * stride_yn
    vals = tl.load(y_ptrs, mask=mask, other=0.0)
    vals += offset_val
    tl.store(y_ptrs, vals, mask=mask)

def triton_cumsum(x: torch.Tensor, block_size=1024, num_warps=4):
    B, V = x.shape
    output = torch.empty_like(x)
    num_blocks = (V + block_size - 1) // block_size
    block_sums = torch.empty((B, num_blocks), device=x.device, dtype=x.dtype)

    cumsum_block_kernel[(B, num_blocks)](
        x, output, block_sums,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        V, num_blocks,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )

    block_sums = torch.cumsum(block_sums, dim=-1)

    add_offsets_kernel[(B, num_blocks)](
        output, block_sums,
        output.stride(0), output.stride(1),
        V,
        BLOCK_SIZE=block_size,
        num_warps=4
    )

    return output

@triton.jit
def kernel_max_2d(x_ptr, partial_max_ptr,
                  M, N,
                  BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * N
    offs = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)

    tl.store(partial_max_ptr + pid_m * tl.num_programs(1) + pid_n, max_val)


@triton.jit
def kernel_sum_2d(x_ptr, partial_sum_ptr, global_max_ptr,
                  M, N,
                  BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * N

    offs = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    gmax = tl.load(global_max_ptr + pid_m)
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float('inf'))
    val = tl.exp(x - gmax)
    sum_val = tl.sum(val, axis=0)

    tl.store(partial_sum_ptr + pid_m * tl.num_programs(1) + pid_n, sum_val)


@triton.jit
def kernel_softmax_2d(x_ptr, y_ptr, global_max_ptr, global_sum_ptr,
                      M, N,
                      BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * N

    offs = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    gmax = tl.load(global_max_ptr + pid_m)
    gsum = tl.load(global_sum_ptr + pid_m)

    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float('inf'))
    y = tl.exp(x - gmax) / gsum

    tl.store(y_ptr + row_start + offs, y, mask=mask)


def triton_softmax_2d(x: torch.Tensor, dtype, block_size=1024):
    assert x.dim() == 2

    x.to(dtype)
    M, N = x.shape

    grid = lambda meta: (M, (N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    num_blocks = grid({'BLOCK_SIZE': block_size})[1]
    partial_max = torch.empty((M * num_blocks,), device=x.device, dtype=torch.float32)

    # Step 1. local max
    kernel_max_2d[grid](x, partial_max, M, N, BLOCK_SIZE=block_size)

    # global max of each row
    global_max = partial_max.view(M, num_blocks).max(dim=1).values

    # Step 2. local sum
    partial_sum = torch.empty_like(partial_max)
    kernel_sum_2d[grid](x, partial_sum, global_max, M, N, BLOCK_SIZE=block_size)

    # global sum of each row
    global_sum = partial_sum.view(M, num_blocks).sum(dim=1)

    # Step 3. Perform the final Softmax
    y = torch.empty_like(x)
    kernel_softmax_2d[grid](x, y, global_max, global_sum, M, N, BLOCK_SIZE=block_size)
    return y


def expand_batch_to_tokens_2d_triton(
    x: torch.Tensor,  # [batch_size, feat_dim]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
) -> torch.Tensor:
    """Expand 2D tensor [batch_size, feat_dim] to [num_tokens, feat_dim]
    based on number of tokens per batch in cu_num_tokens.
    """
    batch_size, feat_dim = x.shape
    # assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty((num_tokens, feat_dim))

    padded_block_feat = max(32, (1 << feat_dim.bit_length()))

    expand_kernel_2d[(batch_size, )](
        expanded_x, x, cu_num_tokens,
        feat_dim=feat_dim,
        BLOCK_FEAT=padded_block_feat,
        num_warps=4,
    )
    return expanded_x


@triton.jit
def expand_kernel_2d(
    output_ptr,        # [num_tokens, feat_dim]
    input_ptr,         # [batch_size, feat_dim]
    cu_num_tokens_ptr, # [batch_size]
    feat_dim,
    BLOCK_FEAT: tl.constexpr, # Integer power of 2, must be >= feat_dim
):
    pid_b = tl.program_id(0)  # batch index

    start_idx = tl.where(pid_b == 0, 0, tl.load(cu_num_tokens_ptr + pid_b - 1))
    end_idx   = tl.load(cu_num_tokens_ptr + pid_b)
    num_tokens = end_idx - start_idx

    # Use a fixed BLOCK_FEAT to ensure the compile-time constants meet the requirements.
    feat_offsets = tl.arange(0, BLOCK_FEAT)
    mask_feat = feat_offsets < feat_dim
    src_vec = tl.load(input_ptr + pid_b * feat_dim + feat_offsets,
                      mask=mask_feat,
                      other=0.0)

    for t in range(num_tokens):
        base_out = (start_idx + t) * feat_dim
        tl.store(output_ptr + base_out + feat_offsets, src_vec, mask=mask_feat)


@torch.compile
def apply_top_p_k_atom(
    logits_sort: torch.Tensor,
    logits_idx: torch.Tensor,
    probs_sum: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
):
    logits_sort = logits_sort.clone()
    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = torch.empty_like(logits_sort).scatter_(dim=-1,
                                                    index=logits_idx,
                                                    src=logits_sort)
    
    return logits


def _apply_top_k_top_p_fused(
    logits: torch.Tensor, # [N, V], float32
    p_k_lst: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],  # [[N, ], [N, ]]
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    probs_sort = logits_sort.softmax(dim=-1) # [N, V], float32
    # probs_sort = triton_softmax_2d(logits_sort, dtype=torch.float)

    if logits.shape[0] == 1:
        probs_sum = probs_sort.cumsum(dim=-1)  # [N, V], float32
    else:
        probs_sum = triton_cumsum(probs_sort)

    logits_lst = []
    for (p, k) in p_k_lst:
        logits_after_apply = apply_top_p_k_atom(logits_sort, logits_idx, 
                                                probs_sum, p, k)
        logits_lst.append(logits_after_apply)

    return logits_lst

@torch.compile
def update_batch_logits(
    logits_topk_topp: torch.Tensor,  # [N, V]
    logits_k5_p999: torch.Tensor,    # [N, V]
    logits_k20_p95: torch.Tensor,    # [N, V]
):
    combine_batch_logits = logits_topk_topp.clone()
    mask_k5_p999 = ~torch.isneginf(logits_k5_p999)
    mask_k20_p95 = ~torch.isneginf(logits_k20_p95)

    combine_batch_logits = torch.where(
        mask_k5_p999,
        logits_k5_p999,
        combine_batch_logits,
    )

    combine_batch_logits = torch.where(
        mask_k20_p95,
        logits_k20_p95,
        combine_batch_logits,
    )

    return combine_batch_logits

@torch.compile
def get_last_pos(
    tok_langs_tensor: torch.Tensor, 
):
    seq_len = tok_langs_tensor.shape[1]
    mask = (tok_langs_tensor != 2)

    rev_mask = mask.flip(dims=[1])
    offsets = rev_mask.float().argmax(dim=1)
    last_pos = seq_len - 1 - offsets

    return last_pos


## this expects logits_batch to be processed such that non topk topp tokens are -inf
@torch.compile
def batch_varied_top_k_p_in_logits_opt(
    logits_batch: torch.Tensor,      # [N, V]
    lang_masks:   torch.Tensor,      # [C, V] bool
) -> torch.Tensor:
    # A value of True here indicates a non -inf logits, 
    # i.e., logits that could be selected.
    keep_mask_exp = ~torch.isneginf(logits_batch).unsqueeze(1) # [N, 1, V]

    # Offline lang_masks mapping each vocabulary token to its language category.
    lang_masks_exp = lang_masks.unsqueeze(0) # [1, C, V]

    # Languages that can potentially be selected from the input.
    # i.e, when N=1, [T, F, T, F] means the possible outputs are Chinese and symbols.
    any_lang = (keep_mask_exp & lang_masks_exp).any(dim=2) # [N, C]
    return any_lang

@torch.compile
def compute_pred_lang_labels(
    batch_lang_logits: torch.Tensor,  # [N, C]
) -> torch.Tensor:
    
    batch_lang_logits_sigmoid = batch_lang_logits.sigmoid()

    pred_lang_labels = batch_lang_logits_sigmoid >= 0.5

    max_idx = batch_lang_logits_sigmoid.argmax(dim=-1, keepdim=True)
    best_is_true = torch.zeros_like(pred_lang_labels).scatter_(-1, max_idx, 1).bool()
    
    pred_lang_labels = pred_lang_labels | best_is_true

    return pred_lang_labels


@torch.compile
def compute_intervention(
    logits_lang_labels_sampling: torch.Tensor,  # [N, C]
    langs_to_mask: torch.Tensor,                # [C, V]
    logits_lang_labels_topk: torch.Tensor,      # [N, ]
    original_pred_lang_labels: torch.Tensor,    # [N, V]
    has_output_tokens: torch.Tensor,            # [N, V]
) -> torch.Tensor:
    
    intervention_has_effect = (logits_lang_labels_sampling & langs_to_mask).any(dim=1)
    target_in_topk = (logits_lang_labels_topk & original_pred_lang_labels).any(dim=1)
    apply_intervention = intervention_has_effect & target_in_topk & has_output_tokens

    return apply_intervention

@torch.compile
def compute_mask_and_output_logits(
    langs_to_mask: torch.Tensor,         # [N, C]
    lang_masks: torch.Tensor,            # [C, V]
    apply_intervention: torch.Tensor,    # [N, ]
    combine_batch_logits: torch.Tensor,  # [N, V]
    original_batch_logits: torch.Tensor, # [N, V]
) -> torch.Tensor:
    vocab_mask = (langs_to_mask[:, :, None] & lang_masks[None, :, :]).any(dim=1)
    vocab_mask = vocab_mask & apply_intervention.unsqueeze(1)
    combine_batch_logits = combine_batch_logits.masked_fill(vocab_mask, -torch.inf)

    # if no intervention, use original logits
    # otherwise use more tokens to ensure lowres is included
    result = torch.where(
        apply_intervention[:, None],
        combine_batch_logits,
        original_batch_logits,
    )

    return result