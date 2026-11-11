import argparse
import contextlib
import json
import multiprocessing
import os
import queue
import sys
import time
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig


def benchmark_config(
    backend: str,
    m: int,
    k: int,
    n: int,
    use_bias: bool,
    dtype: torch.dtype,
    num_iters: int = 10,
) -> tuple[float, float]:
    A = torch.rand(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.rand(n, k, dtype=torch.bfloat16, device="cuda")
    if use_bias:  # noqa: SIM108
        bias = torch.rand(n, dtype=torch.bfloat16, device="cuda")
    else:
        bias = None
    if dtype != torch.bfloat16:
        if backend == "torch" or backend == "cutlass":
            from vllm import _custom_ops as ops

            A, A_scale = ops.scaled_fp8_quant(A)
            B, B_scale = ops.scaled_fp8_quant(B)
        elif backend == "deepgemm":
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                per_block_cast_to_fp8,
                per_token_cast_to_fp8,
            )

            A, A_scale = per_token_cast_to_fp8(A)
            B, B_scale = per_block_cast_to_fp8(B)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def run():
        if dtype == torch.bfloat16:
            torch.mm(A, B)
        elif backend == "torch":
            torch._scaled_mm(
                A,
                B.t(),
                out_dtype=torch.bfloat16,
                scale_a=A_scale,
                scale_b=B_scale.t(),
                bias=bias,
            )
        elif backend == "cutlass":
            from vllm import _custom_ops as ops

            ops.cutlass_scaled_mm(
                A,
                B.t(),
                out_dtype=torch.bfloat16,
                scale_a=A_scale,
                scale_b=B_scale.t(),
                bias=bias,
            )
        elif backend == "deepgemm":
            from vllm.utils.deep_gemm import fp8_gemm_nt

            C = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            fp8_gemm_nt((A, A_scale), (B, B_scale), C)
            if use_bias:
                C.add_(bias)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        run()
    end_event.record()
    torch.cuda.synchronize()
    kernel_time = start_event.elapsed_time(end_event) * 1000 / num_iters  # us
    tflops = 2 * m * n * k / kernel_time / 1e6  # TFLOPS
    return kernel_time, tflops


class BenchmarkWorker:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @contextlib.contextmanager
    def lock_device(self, device_queue: queue.Queue = None):
        if device_queue is not None:
            device = device_queue.get()
            torch.cuda.set_device(device)
            yield
            device_queue.put(device)
        else:
            yield

    def do_remote(
        self,
        args: tuple[queue.Queue, tuple[torch.dtype, str, int, int, int, bool]],
    ):
        device_queue, args = args
        with self.lock_device(device_queue):
            return self.do(args)

    def do(
        self,
        args: tuple[str, bool, torch.dtype, int, int, int],
    ) -> tuple[tuple[torch.dtype, str, int, int, int], float, float]:
        backend, tune, dtype, op, m, k, n, use_bias = args
        kernel_time, tflops = benchmark_config(
            backend, m, k, n, use_bias, dtype, num_iters=10
        )
        if tune and hasattr(torch.cuda, "tunable"):
            torch.cuda.tunable.write_file()
        return (backend, dtype, op, m, k, n, use_bias), kernel_time, tflops


def expand_batch_sizes(
    start_batch_size: int,
    max_batch_size: int,
    sparse_batch_size: bool,
) -> list[int]:
    if sparse_batch_size:
        candidates = (
            list(range(64))
            + list(range(64, min(128, max_batch_size + 4), 4))
            + list(range(128, min(256, max_batch_size + 8), 8))
            + list(range(256, min(512, max_batch_size + 16), 16))
            + list(range(512, min(1024, max_batch_size + 32), 32))
            + list(range(1024, min(2048, max_batch_size + 64), 64))
            + list(range(2048, min(4096, max_batch_size + 128), 128))
            + list(range(4096, min(8192, max_batch_size + 256), 256))
            + list(range(8192, min(16384, max_batch_size + 512), 512))
        )
        candidates = [bs for bs in candidates if bs <= max_batch_size]
        return sorted(set(candidates))
    else:
        return list(range(start_batch_size, max_batch_size))


def generate_gemm_configs(
    config: PretrainedConfig,
    batch_size: int,
    hidden_size: int,
    padded_hidden_size: Optional[int],
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
    tensor_parallel_size: int,
) -> list[tuple[str, int, int, int, bool]]:
    qkv_bias = getattr(config, "qkv_bias", True)
    gemms = []
    # qkv proj: b x h * h x ((nh + 2 * nkvh) / tp * hd)
    if padded_hidden_size is None:
        head_dim = hidden_size // num_attention_heads
    else:
        head_dim = padded_hidden_size // num_attention_heads
    num_key_value_heads = (
        max(1, num_key_value_heads // tensor_parallel_size) * tensor_parallel_size
    )
    num_local_heads = (
        num_attention_heads + 2 * num_key_value_heads
    ) // tensor_parallel_size
    gemms.append(
        ("qkv_proj", batch_size, hidden_size, num_local_heads * head_dim, qkv_bias)
    )
    # o proj: b x (nh / tp * hd) * (h / tp) x h
    gemms.append(
        (
            "o_proj",
            batch_size,
            num_attention_heads // tensor_parallel_size * head_dim,
            hidden_size,
            False,
        )
    )
    # gate_up_proj: b x h * h * (2 * il / tp)
    if intermediate_size != 0:
        gemms.append(
            (
                "gate_up_proj",
                batch_size,
                hidden_size,
                2 * intermediate_size // tensor_parallel_size,
                False,
            )
        )
        # down_proj: b x (il / tp) * (il / tp) x h
        gemms.append(
            (
                "down_proj",
                batch_size,
                intermediate_size // tensor_parallel_size,
                hidden_size,
                False,
            )
        )
    return gemms


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model)
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    tensor_parallel_size = args.tp_size
    hidden_size = config.hidden_size
    padded_hidden_size = getattr(config, "padded_hidden_size", None)
    if hasattr(config, "shared_expert_intermediate_size"):
        intermediate_size = config.shared_expert_intermediate_size
    else:
        intermediate_size = config.intermediate_size

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp8":
        if "ROCM_HOME" in os.environ or "ROCM_PATH" in os.environ:
            dtype = torch.float8_e4m3fnuz
        else:
            dtype = torch.float8_e4m3fn

    if args.backend == "deepgemm":
        assert len(args.block_shape) == 2

    if args.batch_size is None:
        batch_sizes = expand_batch_sizes(
            args.start_batch_size, args.max_batch_size, args.sparse_batch_size
        )
    else:
        batch_sizes = [args.batch_size]

    gemms = []
    for batch_size in batch_sizes:
        gemms.extend(
            generate_gemm_configs(
                config,
                batch_size,
                hidden_size,
                padded_hidden_size,
                num_attention_heads,
                num_key_value_heads,
                intermediate_size,
                tensor_parallel_size,
            )
        )

    if args.tune:
        os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
        os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
        os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "0"
        os.environ["PYTORCH_TUNABLEOP_NUMERICAL_CHECK"] = "1"
    else:
        os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
        os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
        os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "0"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = args.tunableop_results

    num_gpus = torch.cuda.device_count()
    worker = BenchmarkWorker()
    if args.spawn:
        m = multiprocessing.Manager()
        devices = m.Queue()
        for i in range(num_gpus):
            devices.put(i)
        print(f"Spawn {num_gpus} processes for benchmarking")
    else:
        device = 0
        torch.cuda.set_device(device)

    def _distribute(inputs: list[Any]) -> list[Any]:
        messages = []
        with tqdm(inputs, desc="Benchmarking GEMM") as t:
            for i, input_args in enumerate(t):
                t.set_postfix_str(f"{input_args}")
                config, kernel_time, tflops = worker.do(
                    (
                        args.backend,
                        args.tune,
                        dtype,
                    )
                    + input_args
                )
                t.update(1)
                message = f"Benchmarking GEMM: {config}: {kernel_time:.2f} us, {tflops:.4f} TFLOPS"
                t.set_description(message)
                messages.append(message)
        if not args.tune:
            for message in messages:
                print(message, file=sys.stderr, flush=True)

    def _distribute_mp(inputs: list[Any]) -> list[Any]:
        with multiprocessing.Pool(num_gpus) as pool:
            async_results = []
            for input_args in inputs:
                async_results.append(
                    (
                        input_args,
                        pool.apply_async(
                            worker.do_remote,
                            [
                                (
                                    devices,
                                    (
                                        args.backend,
                                        args.tune,
                                        dtype,
                                    )
                                    + input_args,
                                )
                            ],
                        ),
                    )
                )

            messages = []
            with tqdm(async_results, desc="Benchmarking GEMM") as t:
                for i, (input_args, async_result) in enumerate(t):
                    config, kernel_time, tflops = async_result.get()
                    t.update(1)
                    message = f"Benchmarking GEMM: {config}: {kernel_time:.2f} us, {tflops:.4f} TFLOPS"
                    t.set_description(message)
                    messages.append(message)
            if not args.tune:
                for message in messages:
                    print(message, file=sys.stderr, flush=True)

    print(f"Start benchmarking over {len(gemms)} configurations...")
    start = time.perf_counter()
    if args.dry_run:
        print(
            "\n".join(
                [f"{(gemm[0], gemm[3], gemm[1], gemm[2], gemm[4])}" for gemm in gemms]
            )
        )
    elif args.spawn:
        _distribute_mp(gemms)
    else:
        _distribute(gemms)
    end = time.perf_counter()
    print(f"Benchmarking took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "-tp", type=int, default=1)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp8"], default="fp8")
    parser.add_argument(
        "--backend", type=str, choices=["torch", "cutlass", "deepgemm"], default="torch"
    )
    parser.add_argument("--block-shape", type=json.loads, default=None)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--start-batch-size", type=int, required=False, default=16)
    parser.add_argument("--max-batch-size", type=int, required=False, default=8192)
    parser.add_argument("--sparse-batch-size", action="store_true")
    parser.add_argument(
        "--tunableop-results", type=str, required=False, default="tunableop_results.csv"
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(args)
