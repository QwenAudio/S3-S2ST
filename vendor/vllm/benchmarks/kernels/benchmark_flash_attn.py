#!/usr/bin/env python3
"""
Bench flash_attn_varlen_func with paged KV-cache and multiple "case" workloads.

Features fixed/added in this patch:
- Restores a robust `manual_bench` fallback (CUDA events) if Triton's `do_bench` is unavailable.
- Uses `bench()` wrapper to choose Triton's `do_bench` when available, otherwise falls back to `manual_bench`.
- Ensures `block_table` entries reference the allocated number of KV pages (no out-of-range indices).
- Calls `flash_attn_varlen_func` using keyword arguments and supports paged KV via `block_table`.
- Computes and prints time, estimated FLOPs, TFLOP/s, estimated memory bytes, and GB/s for each workload.

Usage:
  python flash_attn_varlen_bench.py --B 16 --K 1 --min-seq-len 1024 --max-seq-len 10240

Requirements:
  - PyTorch with CUDA
  - flash-attn installed (flash_attn_varlen_func)
  - Triton (optional, speeds up benchmarking harness)

"""

import argparse
import random
from typing import List

import tqdm
import torch
import torch.nn.functional as F

# Try to import Triton's do_bench; fall back if unavailable
try:
    import triton
    from triton.testing import do_bench as triton_do_bench
except Exception:
    triton = None
    triton_do_bench = None


# Try to import flash_attn_varlen_func from common locations
def _try_import_varlen():
    last_err = None
    for path in [
        "vllm.attention.utils.fa_utils",
        "vllm_flash_attn.flash_attn_interface",
        "vllm_flash_attn.flash_attn_triton",
        "vllm_flash_attn",
    ]:
        try:
            mod = __import__(path, fromlist=["flash_attn_varlen_func"])
            fn = getattr(mod, "flash_attn_varlen_func")
            print(f'Loaded flash_attn_varlen_func from "{path}"')
            return fn
        except Exception as e:
            print(f'Failed to load flash_attn_varlen_func from "{path}": "{e}"')
            last_err = e
    raise ImportError(
        "Could not import flash_attn_varlen_func from flash-attn. Last error: %s"
        % (last_err,)
    )


# ------------------ Benchmark helpers ------------------


def manual_bench(fn, warmup: int = 25, iters: int = 100) -> float:
    """Benchmark using CUDA events and return median time in milliseconds."""
    # warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def bench(fn, warmup: int = 25, iters: int = 100) -> float:
    """Use Triton's do_bench if available, otherwise fallback to manual_bench.

    Returns: mean time in milliseconds.
    """
    if triton_do_bench is not None:
        # triton.testing.do_bench accepts a python callable and returns ms
        # Keep parameters consistent with manual_bench semantics
        try:
            ms = triton_do_bench(fn, rep=iters, warmup=warmup, return_mode="mean")
            # ensure GPU work completed
            torch.cuda.synchronize()
            return float(ms)
        except Exception:
            # If triton fails for any reason, fallback to manual
            return manual_bench(fn, warmup, iters)
    else:
        return manual_bench(fn, warmup, iters)


# ------------------ Workload & KV paging helpers ------------------


def gen_case2_q_lens(B: int, K: int, seed: int = 0) -> List[int]:
    """Generate B positive integers in [1, 2*K] that sum to B*K."""
    rng = random.Random(seed + 123)
    max_query_len = min(2 * K + 1, 8)
    q_lens = [rng.randint(1, max_query_len + 1) for _ in range(B)]
    total = sum(q_lens)
    scale = (B * K) / total
    q_lens = [max(1, min(max_query_len, int(round(x * scale)))) for x in q_lens]
    diff = B * K - sum(q_lens)
    while diff > 0:
        for i in range(B):
            if diff == 0:
                break
            if q_lens[i] < max_query_len:
                q_lens[i] += 1
                diff -= 1
    while diff < 0:
        for i in range(B):
            if diff == 0:
                break
            if q_lens[i] > 1:
                q_lens[i] -= 1
                diff += 1
    assert all(1 <= q <= max_query_len for q in q_lens), (
        f"Out of bound q_lens: {q_lens}"
    )
    assert sum(q_lens) == B * K
    return q_lens


# ------------------ Estimators ------------------


def estimate_attention_flops(
    q_lens: List[int], k_lens: List[int], H_q: int, dh: int
) -> int:
    total = 0
    for nq, nk in zip(q_lens, k_lens):
        total += 4 * H_q * nq * (nq + nk) * dh
    return total


def estimate_bandwidth_bytes(
    q_lens: List[int],
    k_lens: List[int],
    H_q: int,
    H_kv: int,
    dh: int,
    dtype: torch.dtype,
) -> int:
    elem = (
        torch.finfo(dtype).bits // 8
        if dtype.is_floating_point
        else torch.iinfo(dtype).bits // 8
    )
    read_q = sum(q_lens) * H_q * dh * elem
    read_kv = sum(k_lens) * H_kv * dh * elem * 2
    write_o = sum(q_lens) * H_q * dh * elem
    return int(read_q + read_kv + write_o)


# ------------------ Benchmark runner ------------------


def run_baseline_and_cases(
    B: int,
    K: int,
    min_seq_len: int,
    max_seq_len: int,
    head_dim: int,
    H_q: int,
    H_kv: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    num_case: int,
    causal: bool,
    fa_version: int,
):
    flash_attn_varlen_func = _try_import_varlen()

    if max_seq_len is None or min_seq_len == max_seq_len:
        max_seq_len = min_seq_len + 1

    # random KV context lengths per sequence
    seq_lens = [random.randint(min_seq_len, max_seq_len) for _ in range(B)]
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.cumsum(seqused_k, dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_ctx = max(seq_lens) + 2 * K + 1
    # number of pages needed to cover the longest sequence
    num_pages = B * ((max_ctx + block_size - 1) // block_size)

    # allocate paged KV cache: [num_pages, block_size, H_kv, head_dim]
    k_cache = torch.randn(
        num_pages, block_size, H_kv, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)

    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, -1)
    results = []

    # Cases: multiple random distributions summing to B*K
    for i in tqdm.tqdm(range(num_case)):
        q_lens_case = gen_case2_q_lens(B, K, seed=i)
        cu_q_case = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(q_lens_case), dim=0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        total_q_case = sum(q_lens_case)
        seqused_k_with_q = seqused_k + torch.tensor(
            q_lens_case, dtype=torch.int32, device=device
        )
        q_case = torch.randn(total_q_case, H_q, head_dim, device=device, dtype=dtype)
        print(f"Case {i} q_lens: {q_lens_case}, sum: {total_q_case}")

        def case_op():
            return flash_attn_varlen_func(
                q=q_case,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_q_case,
                seqused_k=seqused_k_with_q,
                block_table=block_table,
                max_seqlen_q=int(max(q_lens_case)),
                max_seqlen_k=max_ctx,
                dropout_p=0.0,
                softmax_scale=None,
                causal=causal,
                fa_version=fa_version,
            )

        ms_case = bench(case_op, warmup=25, iters=iters)
        flops_case = estimate_attention_flops(q_lens_case, seq_lens, H_q, head_dim)
        bytes_case = estimate_bandwidth_bytes(
            q_lens_case, seq_lens, H_q, H_kv, head_dim, dtype
        )
        tflops_case = (
            (flops_case / 1e12) / (ms_case / 1e3) if ms_case > 0 else float("nan")
        )
        gbs_case = (bytes_case / 1e9) / (ms_case / 1e3) if ms_case > 0 else float("nan")

        results.append(
            {
                "name": f"case_{i}",
                "time_ms": ms_case,
                "FLOPs": flops_case,
                "TFLOP/s": tflops_case,
                "Bytes": bytes_case,
                "GB/s": gbs_case,
            }
        )
        print(
            f"Case {i}: time={ms_case:.3f} ms | TFLOP/s={tflops_case:.2f} | GB/s={gbs_case:.2f}"
        )

    # Baseline: equal query length K per sequence
    q_lens_baseline = [K] * B
    cu_q_baseline = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens_baseline), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    total_q_baseline = sum(q_lens_baseline)
    seqused_k_with_q = seqused_k + torch.tensor(
        q_lens_baseline, dtype=torch.int32, device=device
    )
    q_baseline = torch.randn(
        total_q_baseline, H_q, head_dim, device=device, dtype=dtype
    )
    print(f"Baseline q_lens: {q_lens_baseline}, sum: {total_q_baseline}")

    # Function wrapper for baseline
    def baseline_op():
        return flash_attn_varlen_func(
            q=q_baseline,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_q_baseline,
            seqused_k=seqused_k_with_q,
            block_table=block_table,
            max_seqlen_q=K,
            max_seqlen_k=max_ctx,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
            fa_version=fa_version,
        )

    ms_baseline = bench(baseline_op, warmup=25, iters=iters)

    # Estimate baseline FLOPs/BW
    flops_baseline = estimate_attention_flops(q_lens_baseline, seq_lens, H_q, head_dim)
    bytes_baseline = estimate_bandwidth_bytes(
        q_lens_baseline, seq_lens, H_q, H_kv, head_dim, dtype
    )
    tflops_baseline = (
        (flops_baseline / 1e12) / (ms_baseline / 1e3)
        if ms_baseline > 0
        else float("nan")
    )
    gbs_baseline = (
        (bytes_baseline / 1e9) / (ms_baseline / 1e3)
        if ms_baseline > 0
        else float("nan")
    )

    results.append(
        {
            "name": "baseline",
            "time_ms": ms_baseline,
            "FLOPs": flops_baseline,
            "TFLOP/s": tflops_baseline,
            "Bytes": bytes_baseline,
            "GB/s": gbs_baseline,
        }
    )
    print(
        f"Baseline: time={ms_baseline:.3f} ms | TFLOP/s={tflops_baseline:.2f} | GB/s={gbs_baseline:.2f}"
    )

    return results


# ------------------ CLI ------------------


def main():
    parser = argparse.ArgumentParser(
        description="FlashAttention varlen paged-KV benchmark (baseline vs. case)"
    )
    parser.add_argument("--B", type=int, default=16, help="Query length")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--min-seq-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num-case", type=int, default=0)
    parser.add_argument("--head-dim", type=int, default=128, help="Dimension of head")
    parser.add_argument("--hq", type=int, default=16, help="Num of query heads")
    parser.add_argument("--hkv", type=int, default=1, help="Num of key/value heads")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--fa-version", type=int, default=3)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(args)

    results = run_baseline_and_cases(
        args.B,
        args.K,
        args.min_seq_len,
        args.max_seq_len,
        args.head_dim,
        args.hq,
        args.hkv,
        args.block_size,
        dtype,
        device,
        args.iters,
        args.num_case,
        args.causal,
        args.fa_version,
    )

    print("\n=== Summary ===")
    for r in results:
        print(
            f"{r['name']}: time={r['time_ms']:.3f} ms | TFLOP/s={r['TFLOP/s']:.2f} | GB/s={r['GB/s']:.2f} | FLOPs={r['FLOPs']:,} | Bytes={r['Bytes']:,}"
        )


if __name__ == "__main__":
    main()
