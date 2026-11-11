# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_PORT: int | None = None
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    VLLM_USE_TRITON_FLASH_ATTN: bool = True
    VLLM_USE_BLADNN_FLASH_ATTN: bool = False
    VLLM_FLASH_ATTN_USE_UPSTREAM: Optional[bool] = None
    VLLM_FLASH_ATTN_PREFILL_VERSION: Optional[int] = None
    VLLM_VL_USE_UPSTREAM_FA: bool = True
    VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE: int = 256
    VLLM_V1_USE_PREFILL_DECODE_ATTENTION: bool = False
    VLLM_FLASH_ATTN_VERSION: int | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 1800
    VLLM_API_KEY: str | None = None
    VLLM_DEBUG_LOG_API_SERVER_RESPONSE: bool = False
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_ENDPOINT_URL: str | None = None
    VLLM_MODEL_REDIRECT_PATH: str | None = None
    VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")
    VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")
    VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"
    VLLM_NO_USAGE_STATS: bool = True
    VLLM_DISABLE_FLASHINFER_PREFILL: bool = False
    VLLM_DO_NOT_TRACK: bool = True
    VLLM_USAGE_SOURCE: str = ""
    VLLM_CONFIGURE_LOGGING: bool = True
    VLLM_LOGGING_LEVEL: str = "INFO"
    VLLM_LOGGING_PREFIX: str = ""
    VLLM_LOGGING_STREAM: str = "ext://sys.stdout"
    VLLM_LOGGING_CONFIG_PATH: str | None = None
    VLLM_LOGGING_COLOR: str = "auto"
    NO_COLOR: bool = False
    VLLM_LOG_STATS_INTERVAL: float = 5.0
    VLLM_TRACE_FUNCTION: int = 0
    VLLM_TRACE_FUNCTION_OUT_PATH: Optional[str] = None
    VLLM_ATTENTION_BACKEND: str | None = None
    VLLM_ATTENTION_DISABLE_PD_SEPARATION: bool = "PPU_HOME" not in os.environ
    VLLM_USE_FLASHINFER_SAMPLER: bool = False
    VLLM_USE_FLASHINFER_RMSNORM: bool = False
    VLLM_MLA_FP8_ATTENTION: bool = False
    VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD: int = -1
    VLLM_DSA_USE_TILELANG_THRESHOLD: int = -1
    VLLM_DSA_MIN_HEADS_FOR_BF16_PREFILL: int = 8
    VLLM_DSA_USE_TILELANG_INDEXER: bool = False
    VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD: int = -1
    VLLM_DSA_USE_CONTEXT_PARALLEL_ALLTOALL: bool = False
    VLLM_PP_LAYER_PARTITION: str | None = None
    VLLM_CPU_KVCACHE_SPACE: int | None = 0
    VLLM_CPU_OMP_THREADS_BIND: str = ""
    VLLM_CPU_NUM_OF_RESERVED_CPU: int | None = None
    VLLM_CPU_SGL_KERNEL: bool = False
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, "xla_cache")
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    VLLM_FUSED_MOE_CHUNK_SIZE: int = 16 * 1024
    VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING: bool = True
    VLLM_MOE_USE_BLADNN: bool = False
    VLLM_MOE_USE_ACEXT: bool = False
    VLLM_MOE_USE_DEEPEP: bool = False
    VLLM_MOE_USE_PERMUTE: bool = False
    VLLM_INT8_DENSE_USE_DEEPGEMM: bool = False
    VLLM_MOE_USE_DEEPEP_NORMAL: bool = True
    VLLM_MOE_RANDOM_GATING: bool = False
    VLLM_MOE_BALANCED_GATING: bool = False
    VLLM_MOE_TOKEN_DROP: bool = False
    VLLM_MOE_TOKEN_DROP_STATS: int = 0
    VLLM_MOE_EXPERT_REMAPPING: Optional[str] = None
    VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK: Optional[int] = None
    VLLM_CONFIGURE_DEEP_GEMM_NUM_SMS: bool = False
    VLLM_FP8_USE_BLADNN: bool = False
    VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX: str = ""
    VLLM_BLADNN_GEMM_CHUNK_SIZE: int = -1
    VLLM_TENSOR_PARALLEL_VOCAB: bool = True
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl", "shm"] = "auto"
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    VLLM_USE_RAY_WRAPPED_PP_COMM: bool = True
    VLLM_XLA_USE_SPMD: bool = False
    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, "assets")
    VLLM_ASSETS_CACHE_MODEL_CLEAN: bool = False
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10
    VLLM_MEDIA_URL_ALLOW_REDIRECTS: bool = True
    VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    VLLM_VIDEO_LOADER_BACKEND: str = "opencv"
    VLLM_MEDIA_CONNECTOR: str = "http"
    VLLM_TARGET_DEVICE: str = "cuda"
    VLLM_QUANTIZATION_CONFIG: Optional[str] = None
    VLLM_QUANTIZATION_LAYER_WISE: bool = False
    VLLM_QUANTIZATION_MULTIMODAL: bool = False
    VLLM_FORCE_DETOKENIZE: Optional[bool] = None
    VLLM_ENABLE_CPU_GROUP: bool = True
    VLLM_ENABLE_PYTHON_GC_INTERVAL: int = 0
    VLLM_DUAL_CHUNK_ATTENTION_CONFIG: Optional[str] = None
    VLLM_DUAL_CHUNK_ATTENTION_FUSED_ROPE: Optional[bool] = None
    VLLM_FP8_SPARSE_ATTENTION: Optional[bool] = None
    VLLM_MAIN_CUDA_VERSION: str = "12.9"
    VLLM_FLOAT32_MATMUL_PRECISION: Literal["ieee", "tf32"] = "ieee"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    VLLM_VL_BLOCKSPARSE_ATTENTION: Optional[bool] = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX: bool = False
    VLLM_DOCKER_BUILD_CONTEXT: bool = False
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    VLLM_PRINT_TRITON_CACHE_COMPILATIONS: bool = False
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None
    CMAKE_VLLM_ENABLE_PPU: bool = "PPU_HOME" in os.environ
    CMAKE_VLLM_ENABLE_ATTENTION_KERNEL: bool = False
    CMAKE_VLLM_ENABLE_CUTLASS_KERNEL: bool = not CMAKE_VLLM_ENABLE_PPU
    CMAKE_VLLM_ENABLE_GPTQ_KERNEL: bool = False
    CMAKE_VLLM_ENABLE_MARLIN_KERNEL: bool = False
    VERBOSE: bool = False
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = True
    VLLM_TEST_FORCE_FP8_MARLIN: bool = False
    VLLM_TEST_DEVICE_COUNT: int = -1
    VLLM_TEST_NUM_NODES: int = -1
    VLLM_RPC_TIMEOUT: int = 10000  # ms
    VLLM_GENERATE_TIMEOUT: int = 0  # seconds, 0 means no timeout
    VLLM_RESPONSE_TIMEOUT: int = 0  # seconds, 0 means no timeout
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    VLLM_PLUGINS: list[str] | None = None
    VLLM_LORA_RESOLVER_CACHE_DIR: str | None = None
    # Deprecated env variables for profiling, kept for backward compatibility
    # See also vllm/config/profiler.py and `--profiler-config` argument
    VLLM_TORCH_CUDA_PROFILE: str | None = None
    VLLM_TORCH_PROFILER_DIR: str | None = None
    VLLM_TORCH_PROFILER_RECORD_SHAPES: str | None = None
    VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY: str | None = None
    VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM: str | None = None
    VLLM_TORCH_PROFILER_WITH_STACK: str | None = None
    VLLM_TORCH_PROFILER_WITH_FLOPS: str | None = None
    VLLM_TORCH_PROFILER_USE_GZIP: str | None = None
    VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL: str | None = None
    VLLM_PROFILER_DELAY_ITERS: str | None = None
    VLLM_PROFILER_MAX_ITERS: str | None = None
    # End of deprecated env variables for profiling
    VLLM_USE_AOT_COMPILE: bool = False
    VLLM_USE_BYTECODE_HOOK: bool = True
    VLLM_FORCE_AOT_LOAD: bool = False
    VLLM_USE_TRITON_AWQ: bool = False
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    VLLM_SKIP_P2P_CHECK: bool = False
    VLLM_DISABLED_KERNELS: list[str] = []
    VLLM_DISABLE_PYNCCL: bool = False
    VLLM_ROCM_USE_AITER: bool = False
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
    VLLM_ROCM_USE_AITER_LINEAR: bool = True
    VLLM_ROCM_USE_AITER_MOE: bool = True
    VLLM_ROCM_USE_AITER_ASMMOE: bool = True
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True
    VLLM_ROCM_USE_AITER_MLA: bool = True
    VLLM_ROCM_USE_AITER_MHA: bool = True
    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: bool = False
    VLLM_ROCM_USE_AITER_TRITON_ROPE: bool = False
    VLLM_ROCM_USE_AITER_FP8BMM: bool = True
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False
    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False
    VLLM_ROCM_USE_AITER_TRITON_GEMM: bool = True
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True
    VLLM_ROCM_FP8_PADDING: bool = True
    VLLM_ROCM_MOE_PADDING: bool = True
    VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
    VLLM_LOG_REQ_KV_LENS: bool = True
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    VLLM_SERVER_DEV_MODE: bool = False
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    VLLM_MLA_DISABLE: bool = False
    VLLM_MLA_KEEP_BLOCKWISE_QUANT: bool = False
    VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY: bool = False
    VLLM_TEST_SKIP_ATTENTION: bool = False
    VLLM_TEST_SKIP_MLP: bool = False
    VLLM_TEST_SKIP_DEEPEP_PREFILL: bool = False
    VLLM_TEST_SKIP_MLP_ROUTED_EXPERTS: bool = False
    VLLM_TEST_SKIP_MLP_SHARED_EXPERTS: bool = False
    VLLM_TEST_SKIP_MLP_GEMM: bool = False
    VLLM_TEST_SKIP_CUDA_GRAPH: bool = False
    VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH: int = 32
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0
    VLLM_RAY_BUNDLE_INDICES: str = ""
    VLLM_CUDART_SO_PATH: str | None = None
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: int = -1
    VLLM_DP_SIZE: int = 1
    VLLM_USE_STANDALONE_COMPILE: bool = True
    VLLM_DP_MASTER_IP: str = ""
    VLLM_DP_MASTER_PORT: int = 0
    VLLM_DP_SPMD_MODE: bool = False
    VLLM_SERVE_WITH_TORCHRUN: bool = False
    VLLM_MOE_DP_CHUNK_SIZE: int = 256
    VLLM_GDN_USE_BLADNN: bool = False
    VLLM_ENABLE_MOE_DP_CHUNK: bool = True
    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    VLLM_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"] = "strict"
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None
    VLLM_FLATTEN_TP_ACROSS_DP: bool = True
    VLLM_CUDA_VISIBLE_DEVICES: Optional[list[int]] = None
    VLLM_DISTRIBUTED_USE_RANKS: Optional[list[int]] = None
    VLLM_VL_FASTV_K: int = 0
    VLLM_VL_FASTV_R: float = 0.5
    VLLM_VL_FASTV_THRESHOLD: float = -1
    VLLM_VL_PROCESSOR_DO_RESIZE: bool = False
    VLLM_VL_FAST_PROCESSOR: bool = True
    VLLM_VL_VISION_NUM_GRID_PER_SIDE: int = -1
    VLLM_MXFP4_USE_MARLIN: bool | None = None
    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool = False
    VLLM_DEEPEPLL_NVFP4_FP8_DISPATCH: bool = False
    VLLM_DEEPEPHT_NVFP4_DISPATCH: bool = False
    VLLM_DEEPEPHT_NVFP4_FP8_DISPATCH: bool = False
    VLLM_V1_USE_OUTLINES_CACHE: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_TPU_MOST_MODEL_LEN: int | None = None
    VLLM_TPU_USING_PATHWAYS: bool = False
    VLLM_USE_DEEP_GEMM: Optional[bool] = None
    VLLM_MOE_USE_DEEP_GEMM: bool = True
    VLLM_USE_DEEP_GEMM_E8M0: bool = True
    VLLM_USE_LEGACY_PER_TOKEN_GROUP_QUANT: bool = False
    VLLM_DEEP_GEMM_WARMUP: Literal[
        "skip",
        "full",
        "relax",
    ] = "skip"
    VLLM_SKIP_DEEP_GEMM_WARMUP: Optional[bool] = None
    VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool = True
    VLLM_USE_FLASHINFER_MOE_FP16: bool = False
    VLLM_USE_FLASHINFER_MOE_FP8: bool = False
    VLLM_USE_FLASHINFER_MOE_FP4: bool = False
    VLLM_FLASHINFER_MOE_BACKEND: Literal["throughput", "latency", "masked_gemm"] = (
        "latency"
    )
    VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024
    VLLM_XGRAMMAR_CACHE_MB: int = 0
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_MOONCAKE_BOOTSTRAP_PORT: int = 8998
    VLLM_ALL2ALL_BACKEND: Literal[
        "naive",
        "pplx",
        "deepep_high_throughput",
        "deepep_low_latency",
        "allgather_reducescatter",
        "flashinfer_all2allv",
    ] = "allgather_reducescatter"
    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_SLEEP_WHEN_IDLE: bool = False
    VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS: float = 30
    VLLM_PD_TRANSFER_TIMEOUT_SECONDS: float = 30
    VLLM_PD_CONNMANAGER_CAP: int = 128
    VLLM_EPD_TRANSFER_TIMEOUT_SECONDS: int = 10
    VLLM_EPD_ENABLE_GLOBAL_ENCODER_CACHE: bool = False
    VLLM_PD_NUM_LOGITS_SLOTS: int = 256
    VLLM_KVS_IO_TIMEOUT_SECONDS: float = 30
    VLLM_KVS_ON_MIN_LENGTH: int = 1024
    VLLM_USE_FILE_PATH_MMD_CACHE: bool = False
    VLLM_OMNI_PREPROCESS_ON_GPU: bool = False
    VLLM_PREPROCESS_MMD_CACHE_SIZE: int = 100
    VLLM_OMNI_T2T_USE_ZMQ: bool = True
    VLLM_OMNI_T2C_USE_ZMQ: bool = True
    VLLM_OMNI_SKIP_TEXT_ENCODE: bool = False
    VLLM_VLO_V2V_USE_ZMQ: bool = True
    VLLM_KVCACHE_BLOCK_FILE_PATH: str = "/tmp/vllm_kvcache_blocks"
    VLLM_USE_XXHASH: bool = False
    VLLM_TEST_PERSIST_BLOCK_RATIO: float = 1.0
    VLLM_TEST_USE_DIO: bool = False
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    VLLM_KV_CACHE_LAYOUT: Literal["NHD", "HND"] | None = None
    VLLM_COMPUTE_NANS_IN_LOGITS: bool = False
    VLLM_USE_NVFP4_CT_EMULATIONS: bool = False
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: Literal[
        "FP", "INT8", "INT6", "INT4", "NONE"
    ] = "NONE"
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 480
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: int = 480
    VLLM_USE_CUDNN_PREFILL: bool = False
    VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL: bool = False
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False
    VLLM_LOOPBACK_IP: str = ""
    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = False
    VLLM_ENABLE_RESPONSES_API_STORE: bool = False
    VLLM_USE_TRTLLM_ATTENTION: str | None = None
    VLLM_NVFP4_GEMM_BACKEND: str | None = None
    VLLM_USE_FLASHINFER_CUTLASS_FP4_GEMM: bool = True
    VLLM_USE_BLADNN_FP4_GEMM: bool = False
    VLLM_USE_BLADNN_MOE_FP4: bool = False
    VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4: bool = False
    VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4: bool = False
    VLLM_USE_FP8_ATTN_PROJ_FOR_NVFP4: bool = False
    VLLM_NVFP4_GLOBAL_SCALE_WITH_NAN_TO_NUM: bool = False
    VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION: bool = False
    VLLM_HAS_FLASHINFER_CUBIN: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: bool = False
    VLLM_ROCM_FP8_MFMA_PAGE_ATTN: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: bool = False
    VLLM_ALLREDUCE_USE_SYMM_MEM: bool = True
    VLLM_TUNED_CONFIG_FOLDER: str | None = None
    VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str] = set()
    VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT: bool = False
    VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool = False
    VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool = False
    VLLM_CUSTOM_SCOPES_FOR_PROFILING: bool = False
    VLLM_NVTX_SCOPES_FOR_PROFILING: bool = False
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True
    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "VLLM_OBJECT_STORAGE_SHM_BUFFER"
    VLLM_DEEPEP_BUFFER_SIZE_MB: int = 1024
    VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool = False
    VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL: bool = False
    VLLM_DBO_COMM_SMS: int = 20
    VLLM_PATTERN_MATCH_DEBUG: str | None = None
    VLLM_DEBUG_DUMP_PATH: str | None = None
    VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = True
    VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = True
    VLLM_USE_NCCL_SYMM_MEM: bool = False
    VLLM_NCCL_INCLUDE_PATH: str | None = None
    VLLM_USE_FBGEMM: bool = False
    VLLM_GC_DEBUG: str = ""
    VLLM_USE_QUICK_LOADER: Optional[bool] = False
    VLLM_USE_SWAP_ENGINE: bool = False
    VLLM_ENABLE_LLUMNIX_V0: bool = False
    VLLM_ENABLE_LLUMNIX: bool = False
    LLUMNIX_DETAILED_MIG_STATUS: bool = False
    LLUMNIX_MAX_REQ_MIG_IN: int = 1
    LLUMNIX_MAX_TOKEN_MIG_IN: int = 10000
    LLUMNIX_MAX_BLOCK_RATIO_MIG_IN: float = 0.3
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False
    VLLM_FUSE_QKNORM_AND_ROPE: bool = False
    VLLM_FUSE_QKNORM_ROPE_AND_KVCACHE_WRITE: bool = False
    VLLM_ENABLE_GUARD_MODEL: bool = False
    VLLM_ENABLE_DASHLOG: bool = True
    VLLM_KVT_MAX_DELAY_MS: int = 0
    VLLM_TPEP_REDUCE_SCATTER: bool = False
    VLLM_ENABLE_ELASTIC: bool = False
    VLLM_ELASTIC_COLL_BUFFER_BYTES: int = 1024 * 1024
    VLLM_ELASTIC_MASK_NODE: bool = True
    VLLM_USE_MIGRATION_BACKEND_RETURN_TOKEN: bool = False
    VLLM_DEBUG_WORKSPACE: bool = False
    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool = False
    VLLM_ENABLE_CODE_SWITCH_GATE: bool = False
    VLLM_CODE_SWITCH_TOK_PATH: str = ""
    VLLM_CODE_SWITCH_INTERMEDIATE_DIM: int = 0
    VLLM_CODE_SWITCH_GATE_USE_RAW_LOGITS: bool = False
    VLLM_ATTN_INPUT_USE_RS: bool = False
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int = 256
    VLLM_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"] = "binary"
    VLLM_USE_V2_MODEL_RUNNER: bool = False
    VLLM_DISABLE_ASYNC_SCHED_TIMING: bool = False
    VLLM_DISABLE_ASYNC_SCHED_DP_SYNC: bool = True
    VLLM_ENABLE_PHASED_DURATION_LOGGING: bool = False
    VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS: bool = False
    VLLM_OMNI_THINKER_AVENCODER_CACHE: bool = False
    VLLM_OMNI_THINKER_AVENCODER_CACHE_SIZE: int = 100
    VLLM_OMNI_XVECTOR_CACHE_SIZE: int = 100
    VLLM_OMNI_ENABLE_TURN_BASE: bool = False
    VLLM_ENFORCE_GENERATE_CONFIG: bool = False
    VLLM_OMNI_CODE2WAV_CHUNK_SIZE: int = 4
    VLLM_OMNI_CODE2WAV_MAX_BATCH_SIZE: int = 16
    VLLM_OMNI_CODE2WAV_ENABLE_DYNAMIC_CHUNK: bool = False
    VLLM_OMNI_CODE2WAV_DYNAMIC_CHUNK_SIZE_LIST: list[int] = [4, 6, 8]
    VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE: bool = False
    VLLM_OMNI_REALTIME_VIDEO_FPS = 1
    VLLM_OMNI_ENABLE_TALKER: bool = False
    VLLM_COSYVOICE_EXPORT_LLM_HIDDEN_FOR_FM: bool = False
    VLLM_ENABLE_RUND_APPLICATION_TEMPLATE: bool = False
    VLLM_KV_TRANS_PROTOCOL: str = "rdma"
    VLLM_ENCODER_CACHE_LIMIT: int = 0
    VLLM_ENABLE_BYPASS_TASK: bool = True
    VLLM_KVS_USE_REQUEST_HASH: bool = False
    VLLM_ENABLE_SAMPLER_DP: bool = False
    VLLM_ENABLE_MODEL_RUNNER_WARMUP: bool = False
    VLLM_QSA_USE_FP8_INDEXER: bool = True
    VLLM_QSA_PREFILL_USE_TL_INDEXER: bool = False
    VLLM_QSA_DECODE_USE_TL_INDEXER: bool = False
    VLLM_QSA_PREFILL_ATTN_BACKEND: str = 'triton'
    VLLM_QSA_PREFILL_USE_CP: bool = False  # both indexter and attn use q-dim cp
    VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP: bool = False
    VLLM_QSA_DECODE_ATTN_BACKEND: str = 'pai-fa3'
    VLLM_QWEN_TTS_GENERATE_MODE: str = "default"


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def disable_compile_cache() -> bool:
    return bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0")))


def use_aot_compile() -> bool:
    from vllm.model_executor.layers.batch_invariant import (
        vllm_is_batch_invariant,
    )
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    default_value = (
        "1"
        if is_torch_equal_or_newer("2.10.0.dev") and not disable_compile_cache()
        else "0"
    )

    return (
        not vllm_is_batch_invariant()
        and os.environ.get("VLLM_USE_AOT_COMPILE", default_value) == "1"
    )


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """
    Create a lambda that validates environment variable
    containing comma-separated values against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default list of values if not set
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables
        dict that returns list of strings
    """

    def _get_validated_env_list() -> list[str]:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Split comma-separated values and strip whitespace
        values = [v.strip() for v in value.split(",") if v.strip()]

        if not values:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        # Validate each value
        for val in values:
            if not case_sensitive:
                check_value = val.lower()
                check_choices = [choice.lower() for choice in actual_choices]
            else:
                check_value = val
                check_choices = actual_choices

            if check_value not in check_choices:
                raise ValueError(
                    f"Invalid value '{val}' in {env_name}. "
                    f"Valid options: {actual_choices}."
                )

        return values

    return _get_validated_env_list


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """
    Creates a lambda which that validates environment variable
    containing comma-separated values against allowed choices which
    returns choices as a set.
    """

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
    """
    if "VLLM_PORT" not in os.environ:
        return None

    port = os.getenv("VLLM_PORT", "0")

    try:
        return int(port)
    except ValueError as err:
        from urllib.parse import urlparse

        parsed = urlparse(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/stable/serving/env_vars.html"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


# The start-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]

logger = logging.getLogger(__name__)

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of vLLM, supporting [cuda (by default),
    # rocm, cpu]
    "VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),
    # Main CUDA version of vLLM. This follows PyTorch but can be overridden.
    "VLLM_MAIN_CUDA_VERSION": lambda: os.getenv("VLLM_MAIN_CUDA_VERSION", "").lower()
    or "12.9",
    # Controls PyTorch float32 matmul precision mode within vLLM workers.
    # Accepted values:
    #   - "ieee" (default): force full IEEE FP32 matmul precision.
    #   - "tf32": enable TensorFloat32-based fast matmul.
    "VLLM_FLOAT32_MATMUL_PRECISION": env_with_choices(
        "VLLM_FLOAT32_MATMUL_PRECISION",
        "ieee",
        ["ieee", "tf32"],
        case_sensitive=False,
    ),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # If set, vllm will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED": lambda: os.environ.get("VLLM_USE_PRECOMPILED", "")
    .strip()
    .lower()
    in ("1", "true")
    or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),
    # If set, skip adding +precompiled suffix to version string
    "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": lambda: bool(
        int(os.environ.get("VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX", "0"))
    ),
    # Used to mark that setup.py is running in a Docker build context,
    # in order to force the use of precompiled binaries.
    "VLLM_DOCKER_BUILD_CONTEXT": lambda: os.environ.get("VLLM_DOCKER_BUILD_CONTEXT", "")
    .strip()
    .lower()
    in ("1", "true"),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": env_with_choices(
        "CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]
    ),
    # Enable PPU kernel
    "CMAKE_VLLM_ENABLE_PPU": lambda: bool(
        int(os.getenv("CMAKE_VLLM_ENABLE_PPU", None) or "PPU_HOME" in os.environ)
    ),
    # Enable Attention kernel
    "CMAKE_VLLM_ENABLE_ATTENTION_KERNEL": lambda: bool(
        int(os.getenv("CMAKE_VLLM_ENABLE_ATTENTION_KERNEL", "0"))
    ),
    # Enable Cutlass kernel
    "CMAKE_VLLM_ENABLE_CUTLASS_KERNEL": lambda: bool(
        int(
            os.getenv("CMAKE_VLLM_ENABLE_CUTLASS_KERNEL", None)
            or "PPU_HOME" not in os.environ
        )
    ),
    # Enable GPTQ kernel
    "CMAKE_VLLM_ENABLE_GPTQ_KERNEL": lambda: bool(
        int(os.getenv("CMAKE_VLLM_ENABLE_GPTQ_KERNEL", "0"))
    ),
    # Enable Marlin kernel
    "CMAKE_VLLM_ENABLE_MARLIN_KERNEL": lambda: bool(
        int(os.getenv("CMAKE_VLLM_ENABLE_MARLIN_KERNEL", "0"))
    ),
    # If set, vllm will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for vLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    "VLLM_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for vLLM cache files
    # Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set
    "VLLM_CACHE_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "vllm"),
        )
    ),
    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),
    # used in distributed environment to manually set the communication port
    # Note: if VLLM_PORT is set, and some code asks for multiple ports, the
    # VLLM_PORT will be used as the first port, and the rest will be generated
    # by incrementing the VLLM_PORT value.
    "VLLM_PORT": get_vllm_port,
    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    "VLLM_RPC_BASE_PATH": lambda: os.getenv(
        "VLLM_RPC_BASE_PATH", tempfile.gettempdir()
    ),
    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE": lambda: os.environ.get(
        "VLLM_USE_MODELSCOPE", "False"
    ).lower()
    == "true",
    # Interval in seconds to log a warning message when the ring buffer is full
    "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")
    ),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH": lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),
    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # flag to control if vllm should use triton flash attention
    "VLLM_USE_TRITON_FLASH_ATTN": lambda: (
        os.environ.get("VLLM_USE_TRITON_FLASH_ATTN", "False").lower() in ("true", "1")
    ),
    # flag to control if vllm should use bladnn flash attention
    "VLLM_USE_BLADNN_FLASH_ATTN": lambda: (
        os.environ.get("VLLM_USE_BLADNN_FLASH_ATTN", "False").lower() in ("true", "1")
    ),
    # flag to control if vllm should use upstream flash attention in the flash-attention
    # backend.
    "VLLM_FLASH_ATTN_USE_UPSTREAM": lambda: (
        None
        if "VLLM_FLASH_ATTN_USE_UPSTREAM" not in os.environ
        else os.environ.get("VLLM_FLASH_ATTN_USE_UPSTREAM", "False").lower()
        in ("true", "1")
    ),
    # flag to control if we enforce FA2 during prefill with the flash-attention backend.
    "VLLM_FLASH_ATTN_PREFILL_VERSION": lambda: maybe_convert_int(
        os.environ.get("VLLM_FLASH_ATTN_PREFILL_VERSION", None)
    ),
    # flag to control if vllm should use upstream flash attention for VL models.
    "VLLM_VL_USE_UPSTREAM_FA": lambda: (
        os.environ.get("VLLM_VL_USE_UPSTREAM_FA", "True").lower() in ("true", "1")
    ),
    # flag to control the chunk size (in MB) for sleeping memory allocations under ROCm
    "VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE": lambda: int(
        os.environ.get("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE", "256")
    ),
    # Use separate prefill and decode kernels for V1 attention instead of
    # the unified triton kernel.
    "VLLM_V1_USE_PREFILL_DECODE_ATTENTION": lambda: (
        os.getenv("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "False").lower()
        in ("true", "1")
    ),
    # Force vllm to use a specific flash-attention version (2 or 3), only valid
    # when using the flash-attention backend.
    "VLLM_FLASH_ATTN_VERSION": lambda: maybe_convert_int(
        os.environ.get("VLLM_FLASH_ATTN_VERSION", None)
    ),
    # Internal flag to enable Dynamo fullgraph capture
    "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE": lambda: bool(
        os.environ.get("VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"
    ),
    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.9 this is
    # enabled by default.
    "VLLM_USE_STANDALONE_COMPILE": lambda: os.environ.get(
        "VLLM_USE_STANDALONE_COMPILE", "1"
    )
    == "1",
    # Debug pattern matching inside custom passes.
    # Should be set to the fx.Node name (e.g. 'getitem_34' or 'scaled_mm_3').
    "VLLM_PATTERN_MATCH_DEBUG": lambda: os.environ.get(
        "VLLM_PATTERN_MATCH_DEBUG", None
    ),
    # Dump fx graphs to the given directory.
    # It will override CompilationConfig.debug_dump_path if set.
    "VLLM_DEBUG_DUMP_PATH": lambda: os.environ.get("VLLM_DEBUG_DUMP_PATH", None),
    # Feature flag to enable/disable AOT compilation. This will ensure
    # compilation is done in warmup phase and the compilation will be
    # reused in subsequent calls.
    "VLLM_USE_AOT_COMPILE": use_aot_compile,
    # Feature flag to enable/disable bytecode in
    # TorchCompileWithNoGuardsWrapper.
    "VLLM_USE_BYTECODE_HOOK": lambda: bool(
        int(os.environ.get("VLLM_USE_BYTECODE_HOOK", "1"))
    ),
    # Force vllm to always load AOT compiled models from disk. Failure
    # to load will result in a hard error when this is enabled.
    # Will be ignored when VLLM_USE_AOT_COMPILE is disabled.
    "VLLM_FORCE_AOT_LOAD": lambda: os.environ.get("VLLM_FORCE_AOT_LOAD", "0") == "1",
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # timeout for each iteration in the engine
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": lambda: int(
        os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "1800")
    ),
    # API key for vLLM API server
    "VLLM_API_KEY": lambda: os.environ.get("VLLM_API_KEY", None),
    # Whether to log responses from API Server for debugging
    "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": lambda: os.environ.get(
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "False"
    ).lower()
    == "true",
    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID": lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY": lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL": lambda: os.environ.get("S3_ENDPOINT_URL", None),
    # Usage stats collection
    "VLLM_USAGE_STATS_SERVER": lambda: os.environ.get(
        "VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"
    ),
    "VLLM_NO_USAGE_STATS": lambda: os.environ.get("VLLM_NO_USAGE_STATS", "1") == "1",
    "VLLM_DISABLE_FLASHINFER_PREFILL": lambda: os.environ.get(
        "VLLM_DISABLE_FLASHINFER_PREFILL", "0"
    )
    == "1",
    "VLLM_DO_NOT_TRACK": lambda: (
        os.environ.get("VLLM_DO_NOT_TRACK", None)
        or os.environ.get("DO_NOT_TRACK", None)
        or "1"
    )
    == "1",
    "VLLM_USAGE_SOURCE": lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),
    # Logging configuration
    # If set to 0, vllm will not configure logging
    # If set to 1, vllm will configure logging using the default configuration
    #    or the configuration file specified by VLLM_LOGGING_CONFIG_PATH
    "VLLM_CONFIGURE_LOGGING": lambda: bool(
        int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))
    ),
    "VLLM_LOGGING_CONFIG_PATH": lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),
    # this is used for configuring the default logging level
    "VLLM_LOGGING_LEVEL": lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),
    # this is used for configuring the default logging stream
    "VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout"),
    # if set, VLLM_LOGGING_PREFIX will be prepended to all log messages
    "VLLM_LOGGING_PREFIX": lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),
    # Controls colored logging output. Options: "auto" (default, colors when terminal),
    # "1" (always use colors), "0" (never use colors)
    "VLLM_LOGGING_COLOR": lambda: os.getenv("VLLM_LOGGING_COLOR", "auto"),
    # Standard unix flag for disabling ANSI color codes
    "NO_COLOR": lambda: os.getenv("NO_COLOR", "0") != "0",
    # If set, vllm will log stats at this interval in seconds
    # If not set, vllm will log stats every 10 seconds.
    "VLLM_LOG_STATS_INTERVAL": lambda: val
    if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0
    else 5.0,
    # Trace function calls
    # If set to 1, vllm will trace function calls
    # Useful for debugging
    "VLLM_TRACE_FUNCTION": lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),
    # If set, vllm will write the trace function calls log to this file
    # else it will be written to the default system tmp directory
    "VLLM_TRACE_FUNCTION_OUT_PATH": lambda: os.getenv(
        "VLLM_TRACE_FUNCTION_OUT_PATH", None
    ),
    # Backend for attention computation
    # Example options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "FLASHINFER": use flashinfer
    # - "FLASHMLA": use FlashMLA
    # - "FLASH_ATTN_MLA": use FlashAttention for MLA
    # - "FLASHINFER_MLA": use FlashInfer for MLA
    # - "CUTLASS_MLA": use CUTLASS for MLA
    # All possible options loaded dynamically from AttentionBackendEnum
    "VLLM_ATTENTION_BACKEND": env_with_choices(
        "VLLM_ATTENTION_BACKEND",
        None,
        lambda: list(
            __import__(
                "vllm.attention.backends.registry", fromlist=["AttentionBackendEnum"]
            ).AttentionBackendEnum.__members__.keys()
        ),
    ),
    # If set, vllm will use flashinfer sampler
    "VLLM_USE_FLASHINFER_SAMPLER": lambda: bool(
        int(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0"))
    ),
    # Enable better performance for PPU with v1 engine: separate prefill
    # and decode requests in a batch.
    "VLLM_ATTENTION_DISABLE_PD_SEPARATION": lambda: os.getenv(
        "VLLM_ATTENTION_DISABLE_PD_SEPARATION", "PPU_HOME" not in os.environ
    ),
    # If set, vllm will use flashinfer rmsnorm
    "VLLM_USE_FLASHINFER_RMSNORM": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_RMSNORM", "0"))
    ),
    # if set, vllm will use fp8 attention for mla prefill
    "VLLM_MLA_FP8_ATTENTION": lambda: bool(
        int(os.getenv("VLLM_MLA_FP8_ATTENTION", "0"))
    ),
    # if set, vllm will use dense mla attention for sparse_mla models (dpsk v32).
    "VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD": lambda: int(
        os.getenv("VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD", "-1")
    ),
    # The query length threshold that vllm will use dense tilelang implementation for
    # the prefill of sparse_mla models (dpsk v32) to avoid padding, -1 means always
    # disable, 0 means always enable.
    "VLLM_DSA_USE_TILELANG_THRESHOLD": lambda: int(
        os.getenv("VLLM_DSA_USE_TILELANG_THRESHOLD", "-1")
    ),
    # Threshold for use bf16 prefill kernel for sparse mla models when fp8 kvcache is
    # in use. Default is 8 in vLLM, but we found in some cases using bf16 prefill
    # kernel yields better performance even with less heads (with TP8).
    "VLLM_DSA_MIN_HEADS_FOR_BF16_PREFILL": lambda: int(
        os.getenv("VLLM_DSA_MIN_HEADS_FOR_BF16_PREFILL", "8")
    ),
    # if set, vllm will use tilelang's indexer during prefill for sparse mla.
    "VLLM_DSA_USE_TILELANG_INDEXER": lambda: bool(
        int(os.getenv("VLLM_DSA_USE_TILELANG_INDEXER", "0"))
    ),
    # The query length threshold that vllm will use context parallel for sparse
    # flashmla's prefill.
    "VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD": lambda: int(
        os.getenv("VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD", "-1")
    ),
    # if set, vllm will use all_to_all for context parallel in sparse flashmla's
    # prefill.
    "VLLM_DSA_USE_CONTEXT_PARALLEL_ALLTOALL": lambda: bool(
        int(os.getenv("VLLM_DSA_USE_CONTEXT_PARALLEL_ALLTOALL", "0"))
    ),
    # if set, vllm will use fa3 do fp8 attention prefill with bf16 kvcache
    "VLLM_FLASH_ATTN_FP8_ATTENTION": lambda: bool(
        int(os.getenv("VLLM_FLASH_ATTN_FP8_ATTENTION", "0"))
    ),
    # if set, will use reduce-scatter for attn input's instead of ar for tp
    "VLLM_ATTN_INPUT_USE_RS": lambda: bool(
        int(os.getenv("VLLM_ATTN_INPUT_USE_RS", "0"))
    ),
    # Pipeline stage partition strategy
    "VLLM_PP_LAYER_PARTITION": lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),
    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "VLLM_CPU_KVCACHE_SPACE": lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0"))
    if "VLLM_CPU_KVCACHE_SPACE" in os.environ
    else None,
    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "VLLM_CPU_OMP_THREADS_BIND": lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "auto"),
    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "VLLM_CPU_NUM_OF_RESERVED_CPU": lambda: int(
        os.getenv("VLLM_CPU_NUM_OF_RESERVED_CPU", "0")
    )
    if "VLLM_CPU_NUM_OF_RESERVED_CPU" in os.environ
    else None,
    # (CPU backend only) whether to use SGL kernels, optimized for small batch.
    "VLLM_CPU_SGL_KERNEL": lambda: bool(int(os.getenv("VLLM_CPU_SGL_KERNEL", "0"))),
    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": env_with_choices(
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto", ["auto", "nccl", "shm"]
    ),
    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph.
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM": lambda: bool(
        int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
    ),
    # If the env var is set, it uses a Ray Communicator wrapping
    # vLLM's pipeline parallelism communicator to interact with Ray's
    # Compiled Graph. Otherwise, it uses Ray's NCCL communicator.
    "VLLM_USE_RAY_WRAPPED_PP_COMM": lambda: bool(
        int(os.getenv("VLLM_USE_RAY_WRAPPED_PP_COMM", "1"))
    ),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "VLLM_WORKER_MULTIPROC_METHOD": env_with_choices(
        "VLLM_WORKER_MULTIPROC_METHOD", "spawn", ["spawn", "fork"]
    ),
    # Path to the cache for storing downloaded assets
    "VLLM_ASSETS_CACHE": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "vllm", "assets"),
        )
    ),
    # If the env var is set, we will clean model file in
    # this path $VLLM_ASSETS_CACHE/model_streamer/$model_name
    "VLLM_ASSETS_CACHE_MODEL_CLEAN": lambda: bool(
        int(os.getenv("VLLM_ASSETS_CACHE_MODEL_CLEAN", "0"))
    ),
    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "VLLM_IMAGE_FETCH_TIMEOUT": lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),
    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "VLLM_VIDEO_FETCH_TIMEOUT": lambda: int(
        os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")
    ),
    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "VLLM_AUDIO_FETCH_TIMEOUT": lambda: int(
        os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")
    ),
    # Whether to allow HTTP redirects when fetching from media URLs.
    # Default to True
    "VLLM_MEDIA_URL_ALLOW_REDIRECTS": lambda: bool(
        int(os.getenv("VLLM_MEDIA_URL_ALLOW_REDIRECTS", "1"))
    ),
    # Max number of workers for the thread pool handling
    # media bytes loading. Set to 1 to disable parallel processing.
    # Default is 8
    "VLLM_MEDIA_LOADING_THREAD_COUNT": lambda: int(
        os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")
    ),
    # Maximum filesize in MB for a single audio file when processing
    # speech-to-text requests. Files larger than this will be rejected.
    # Default is 25 MB
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": lambda: int(
        os.getenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "25")
    ),
    # Backend for Video IO
    # - "opencv": Default backend that uses OpenCV stream buffered backend.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_VIDEO_LOADER_BACKEND": lambda: os.getenv(
        "VLLM_VIDEO_LOADER_BACKEND", "opencv"
    ),
    # Media connector implementation.
    # - "http": Default connector that supports fetching media via HTTP.
    #
    # Custom implementations can be registered
    # via `@MEDIA_CONNECTOR_REGISTRY.register("my_custom_media_connector")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "VLLM_MEDIA_CONNECTOR": lambda: os.getenv("VLLM_MEDIA_CONNECTOR", "http"),
    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "VLLM_XLA_CACHE_PATH": lambda: os.path.expanduser(
        os.getenv(
            "VLLM_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
        )
    ),
    # If set, assert on XLA recompilation after each execution step.
    "VLLM_XLA_CHECK_RECOMPILATION": lambda: bool(
        int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION", "0"))
    ),
    # Enable SPMD mode for TPU backend.
    "VLLM_XLA_USE_SPMD": lambda: bool(int(os.getenv("VLLM_XLA_USE_SPMD", "0"))),
    # vLLM MoE chunk size for the fused_moe triton kernel.
    "VLLM_FUSED_MOE_CHUNK_SIZE": lambda: int(
        os.getenv(
            "VLLM_FUSED_MOE_CHUNK_SIZE", "8192" if "PPU_HOME" in os.environ else "16384"
        )
    ),
    # Control whether to use fused MoE activation chunking. Current chunking
    # logic is incompatible with torch.compile and causes IMA. See issue
    # https://github.com/vllm-project/vllm/issues/19631.
    "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING": lambda: bool(
        int(os.getenv("VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING", "1"))
    ),
    # If set, use bladnn fused_moe kernel for MoE layers.
    "VLLM_MOE_USE_BLADNN": lambda: bool(int(os.getenv("VLLM_MOE_USE_BLADNN", "0"))),
    # If set, debug bladnn fused_moe kernel for MoE layers.
    "VLLM_MOE_DEBUG_BLADNN": lambda: bool(int(os.getenv("VLLM_MOE_DEBUG_BLADNN", "0"))),
    # If set, use acext fused_moe kernel for MoE layers.
    "VLLM_MOE_USE_ACEXT": lambda: bool(int(os.getenv("VLLM_MOE_USE_ACEXT", "0"))),
    # Enable DeepEP interface for prefill/decoding process during EP.
    "VLLM_MOE_USE_DEEPEP": lambda: bool(int(os.getenv("VLLM_MOE_USE_DEEPEP", "0"))),
    # Enable Permute/Unpermute instead of deepep for prefill/decoding process during EP.
    "VLLM_MOE_USE_PERMUTE": lambda: bool(int(os.getenv("VLLM_MOE_USE_PERMUTE", "0"))),
    # Enable Dense Linear for PPU INT8 
    "VLLM_INT8_DENSE_USE_DEEPGEMM": lambda: bool(
        int(os.getenv("VLLM_INT8_DENSE_USE_DEEPGEMM", "0"))
    ),
    # Enable DeepEP interface for prefill process during EP.
    "VLLM_MOE_USE_DEEPEP_NORMAL": lambda: bool(
        int(os.getenv("VLLM_MOE_USE_DEEPEP_NORMAL", "1"))
    ),
    "VLLM_CONFIGURE_DEEP_GEMM_NUM_SMS": lambda: bool(
        int(os.getenv("VLLM_CONFIGURE_DEEP_GEMM_NUM_SMS", "0"))
    ),
    # If set, use random gating for MoE routing, for benchmark purpose.
    "VLLM_MOE_RANDOM_GATING": lambda: (
        os.getenv("VLLM_MOE_RANDOM_GATING", "False").lower() in ("true", "1")
    ),
    # If set, use balanced gating for MoE routing, for benchmark purpose.
    "VLLM_MOE_BALANCED_GATING": lambda: (
        os.getenv("VLLM_MOE_BALANCED_GATING", "False").lower() in ("true", "1")
    ),
    # If set, MoE layer will apply token dropping policy when decoding,
    # will improve performance.
    "VLLM_MOE_TOKEN_DROP": lambda: (
        os.getenv("VLLM_MOE_TOKEN_DROP", "False").lower() in ("true", "1")
    ),
    # If set, will collect token drop stats for MoE layers.
    "VLLM_MOE_TOKEN_DROP_STATS": lambda: int(
        os.getenv("VLLM_MOE_TOKEN_DROP_STATS", "0")
    ),
    # If set, enable expert remapping by loading this configuration.
    "VLLM_MOE_EXPERT_REMAPPING": lambda: os.getenv("VLLM_MOE_EXPERT_REMAPPING", None),
    # If set, use this oracle as the value for num max dispatched tokens per
    # rank for deepep.
    "VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK": lambda: None
    if "VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK" not in os.environ
    else int(os.environ["VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK"]),
    # If set, use bladnn kernel for FP8 dense gemm operations.
    "VLLM_FP8_USE_BLADNN": lambda: bool(int(os.getenv("VLLM_FP8_USE_BLADNN", "0"))),
    "VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX": lambda: os.getenv(
        "VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX", ""
    ),
    # If set, chunking inputs to chunks when using bladnn fp8/int8 kernels.
    "VLLM_BLADNN_GEMM_CHUNK_SIZE": lambda: int(
        os.getenv("VLLM_BLADNN_GEMM_CHUNK_SIZE", "-1")
    ),
    # If set, enable tensor parallel for vocab, otherwise replicated layer
    # will be used to reduce the communication overhead.
    "VLLM_TENSOR_PARALLEL_VOCAB": lambda: bool(
        int(os.getenv("VLLM_TENSOR_PARALLEL_VOCAB", "1"))
    ),
    # If set, vllm will skip the deprecation warnings.
    "VLLM_NO_DEPRECATION_WARNING": lambda: bool(
        int(os.getenv("VLLM_NO_DEPRECATION_WARNING", "0"))
    ),
    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(
        int(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))
    ),
    # If set, vllm will print Triton cache updates
    "VLLM_PRINT_TRITON_CACHE_COMPILATIONS": lambda: bool(
        int(os.getenv("VLLM_PRINT_TRITON_CACHE_COMPILATIONS", "0"))
    ),
    # If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": lambda: (
        os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1").strip().lower()
        in ("1", "true")
    ),
    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "VLLM_TEST_FORCE_FP8_MARLIN": lambda: (
        os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower()
        in ("1", "true")
    ),
    # If set, using the given value as num of gpus per node for testing.
    "VLLM_TEST_DEVICE_COUNT": lambda: int(os.getenv("VLLM_TEST_DEVICE_COUNT", "-1")),
    # If set, using the given value as num of nodes for testing.
    "VLLM_TEST_NUM_NODES": lambda: int(os.getenv("VLLM_TEST_NUM_NODES", "-1")),
    # If set, force using given load format for loading models.
    "VLLM_TEST_FORCE_LOAD_FORMAT": lambda: os.getenv(
        "VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"
    ),
    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "VLLM_RPC_TIMEOUT": lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),
    # Total timeout in seconds for request generation, 0 means no timeout
    "VLLM_GENERATE_TIMEOUT": lambda: int(os.getenv("VLLM_GENERATE_TIMEOUT", "0")),
    # Timeout in seconds between responses, 0 means no timeout
    # If no response is received within this time since the last response,
    # the request will timeout
    "VLLM_RESPONSE_TIMEOUT": lambda: int(os.getenv("VLLM_RESPONSE_TIMEOUT", "0")),
    # Timeout in seconds for keeping HTTP connections alive in API server
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(
        os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")
    ),
    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "VLLM_PLUGINS": lambda: None
    if "VLLM_PLUGINS" not in os.environ
    else os.environ["VLLM_PLUGINS"].split(","),
    # a local directory to look in for unrecognized LoRA adapters.
    # only works if plugins are enabled and
    # VLLM_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    "VLLM_LORA_RESOLVER_CACHE_DIR": lambda: os.getenv(
        "VLLM_LORA_RESOLVER_CACHE_DIR", None
    ),
    # Enables torch CUDA profiling if set to 1.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_CUDA_PROFILE": lambda: os.getenv("VLLM_TORCH_CUDA_PROFILE"),
    # Enables torch profiler if set.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_DIR": lambda: os.getenv("VLLM_TORCH_PROFILER_DIR"),
    # Enable torch profiler to record shapes if set to 1.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_RECORD_SHAPES": lambda: (
        os.getenv("VLLM_TORCH_PROFILER_RECORD_SHAPES")
    ),
    # Enable torch profiler to profile memory if set to 1.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": lambda: (
        os.getenv("VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY")
    ),
    # Enable torch profiler to profile stack if set to 1.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_WITH_STACK": lambda: (
        os.getenv("VLLM_TORCH_PROFILER_WITH_STACK")
    ),
    # Enable torch profiler to profile flops if set to 1.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_WITH_FLOPS": lambda: (
        os.getenv("VLLM_TORCH_PROFILER_WITH_FLOPS")
    ),
    # Disable torch profiling of the AsyncLLMEngine process if set to 1.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM": lambda: (
        os.getenv("VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM")
    ),
    # Delay number of iterations before starting profiling when using
    # the torch/torch CUDA profiler. If set to 0, will start profiling immediately.
    # Deprecated, see profiler_config.
    "VLLM_PROFILER_DELAY_ITERS": lambda: (os.getenv("VLLM_PROFILER_DELAY_ITERS")),
    # Maximum number of iterations to profile when using the torch/torch CUDA profiler.
    # If set to 0, will not limit the number of iterations.
    "VLLM_PROFILER_MAX_ITERS": lambda: os.getenv("VLLM_PROFILER_MAX_ITERS"),
    # Control whether torch profiler gzip-compresses profiling files.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_USE_GZIP": lambda: os.getenv("VLLM_TORCH_PROFILER_USE_GZIP"),
    # Control whether torch profiler dumps the self_cuda_time_total table.
    # Set to 0 to disable dumping the table.
    # Deprecated, see profiler_config.
    "VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL": lambda: (
        os.getenv("VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL")
    ),
    # If set, vLLM will use Triton implementations of AWQ.
    "VLLM_USE_TRITON_AWQ": lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),
    # If set, allow loading or unloading lora adapters in runtime,
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": lambda: (
        os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower()
        in ("1", "true")
    ),
    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set VLLM_SKIP_P2P_CHECK=0
    # so that vLLM can verify if p2p is actually working.
    # See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "VLLM_SKIP_P2P_CHECK": lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",
    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "VLLM_DISABLED_KERNELS": lambda: []
    if "VLLM_DISABLED_KERNELS" not in os.environ
    else os.environ["VLLM_DISABLED_KERNELS"].split(","),
    # Swaps the all reduce backend that we use to coordinate the DP padding
    # information from NCCL to gloo.
    # Disable pynccl (using torch.distributed instead)
    "VLLM_DISABLE_PYNCCL": lambda: (
        os.getenv("VLLM_DISABLE_PYNCCL", "False").lower() in ("true", "1")
    ),
    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "VLLM_ROCM_USE_AITER": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter paged attention.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_PAGED_ATTN": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_PAGED_ATTN", "True").lower() in ("true", "1")
    ),
    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    "VLLM_ROCM_USE_AITER_LINEAR": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_LINEAR", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter moe ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MOE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter asm moe ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_ASMMOE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_ASMMOE", "True").lower() in ("true", "1")
    ),
    # use aiter rms norm op if aiter ops are enabled.
    "VLLM_ROCM_USE_AITER_RMSNORM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_RMSNORM", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter mla ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MLA": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter mha ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MHA": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter fp4 gemm asm.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FP4_ASM_GEMM", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter rope.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_TRITON_ROPE": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter triton fp8 bmm kernel
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_FP8BMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FP8BMM", "True").lower() in ("true", "1")
    ),
    # Use AITER triton unified attention for V1 attention
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()
        in ("true", "1")
    ),
    # Whether to use aiter fusion shared experts ops.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower()
        in ("true", "1")
    ),
    # Whether to use aiter triton kernels for gemm ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_TRITON_GEMM": lambda: (
        os.getenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", "True").lower() in ("true", "1")
    ),
    # use rocm skinny gemms
    "VLLM_ROCM_USE_SKINNY_GEMM": lambda: (
        os.getenv("VLLM_ROCM_USE_SKINNY_GEMM", "True").lower() in ("true", "1")
    ),
    # Pad the fp8 weights to 256 bytes for ROCm
    "VLLM_ROCM_FP8_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),
    # Pad the weights for the moe kernel
    "VLLM_ROCM_MOE_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_MOE_PADDING", "1"))),
    # custom paged attention kernel for MI3* cards
    "VLLM_ROCM_CUSTOM_PAGED_ATTN": lambda: (
        os.getenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4 or NONE
    # Recommended for large models to get allreduce
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": env_with_choices(
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
        "NONE",
        ["FP", "INT8", "INT6", "INT4", "NONE"],
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": lambda: (
        os.getenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower()
        in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    ),
    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT": lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT": lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT": lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),
    # If set, enable multiprocessing in LLM for the V1 code path.
    "VLLM_ENABLE_V1_MULTIPROCESSING": lambda: bool(
        int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))
    ),
    # If set, enable logging batch size in the VLLM engine.
    "VLLM_LOG_BATCHSIZE_INTERVAL": lambda: float(
        os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")
    ),
    # If set, enable log each requests kv cache lens in a batch
    "VLLM_LOG_REQ_KV_LENS": lambda: bool(int(os.getenv("VLLM_LOG_REQ_KV_LENS", "1"))),
    # If set, disable compile cache for torch compilation.
    "VLLM_DISABLE_COMPILE_CACHE": disable_compile_cache,
    # Quantization config from the environment variables.
    "VLLM_QUANTIZATION_CONFIG": lambda: os.getenv("VLLM_QUANTIZATION_CONFIG", None),
    # Quantization config from the environment variables.
    "VLLM_QUANT_SKIP_FC2": lambda: os.getenv("VLLM_QUANT_SKIP_FC2", None),
    # If set, quantization will be done layer-wise, which is helpful to reduce
    # the peek memory usage, but may has problem for checkpoints where tensors
    # are unexpectedly ordered.
    "VLLM_QUANTIZATION_LAYER_WISE": lambda: bool(
        int(os.getenv("VLLM_QUANTIZATION_LAYER_WISE", "0"))
    ),
    # If set, quantization (e.g., fp8, minmax) will be applied to multi-modal
    # layer as well (e.g., qwen2_vl, qwen2_audio).
    "VLLM_QUANTIZATION_MULTIMODAL": lambda: bool(
        int(os.getenv("VLLM_QUANTIZATION_MULTIMODAL", "0"))
    ),
    # Force detokenize
    "VLLM_FORCE_DETOKENIZE": lambda: (
        None
        if "VLLM_FORCE_DETOKENIZE" not in os.environ
        else (os.getenv("VLLM_FORCE_DETOKENIZE", "False").lower() in ("true", "1"))
    ),
    # Enable CPU group
    "VLLM_ENABLE_CPU_GROUP": lambda: (
        os.getenv("VLLM_ENABLE_CPU_GROUP", "True").lower() in ("true", "1")
    ),
    # Enable python GC for every speficied round to avoid impact of
    # full GC on TPOT.
    #
    # It is recommanded to set as 1200 (20 tps for 1 minute).
    #
    # See also: https://docs.python.org/3/library/gc.html
    "VLLM_ENABLE_PYTHON_GC_INTERVAL": lambda: int(
        os.getenv("VLLM_ENABLE_PYTHON_GC_INTERVAL", "0")
    ),
    # DCA configurations for sparse attention for long context.
    "VLLM_DUAL_CHUNK_ATTENTION_CONFIG": lambda: os.getenv(
        "VLLM_DUAL_CHUNK_ATTENTION_CONFIG", None
    ),
    # DCA: use optimized fused rope.
    "VLLM_DUAL_CHUNK_ATTENTION_FUSED_ROPE": lambda: (
        os.getenv("VLLM_DUAL_CHUNK_ATTENTION_FUSED_ROPE", "False").lower()
        in ("true", "1")
    ),
    # If set, sparse attention will use fp8 to compute
    "VLLM_FP8_SPARSE_ATTENTION": lambda: (
        os.getenv("VLLM_FP8_SPARSE_ATTENTION", "False").lower() in ("true", "1")
    ),
    # If set, vllm will enable guard model support
    "VLLM_ENABLE_GUARD_MODEL": lambda: (
        os.getenv("VLLM_ENABLE_GUARD_MODEL", "False").lower() in ("true", "1")
    ),
    # If set, sparse attention will use block sparse attention
    "VLLM_VL_BLOCKSPARSE_ATTENTION": lambda: (
        os.getenv("VLLM_VL_BLOCKSPARSE_ATTENTION", "False").lower() in ("true", "1")
    ),
    # If set, vllm will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "VLLM_SERVER_DEV_MODE": lambda: bool(int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))),
    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": lambda: int(
        os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")
    ),
    # If set, vLLM will disable the MLA attention optimizations.
    "VLLM_MLA_DISABLE": lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),
    # If set, vLLM will keep the blockwise quantization for MLA attention
    # even the model is quantized with fp4/fp5.
    "VLLM_MLA_KEEP_BLOCKWISE_QUANT": lambda: bool(
        int(os.getenv("VLLM_MLA_KEEP_BLOCKWISE_QUANT", "0"))
    ),
    # If set, vLLM will only quantize the routed experts.
    "VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY": lambda: bool(
        int(os.getenv("VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY", "0"))
    ),
    # If set, vLLM will skip the attention layer for benchmarking purposes.
    "VLLM_TEST_SKIP_ATTENTION": lambda: os.environ.get(
        "VLLM_TEST_SKIP_ATTENTION", "False"
    ).lower()
    in ("1", "true"),
    # If set, vLLM will skip the mlp layer for benchmarking purposes.
    "VLLM_TEST_SKIP_MLP": lambda: os.environ.get("VLLM_TEST_SKIP_MLP", "False").lower()
    in ("1", "true"),
    "VLLM_TEST_SKIP_DEEPEP_PREFILL": lambda: os.environ.get(
        "VLLM_TEST_SKIP_DEEPEP_PREFILL", "False"
    ).lower()
    in ("1", "true"),
    # If set, vLLM will skip the routed experts in mlp layer for benchmarking purposes.
    "VLLM_TEST_SKIP_MLP_ROUTED_EXPERTS": lambda: os.environ.get(
        "VLLM_TEST_SKIP_MLP_ROUTED_EXPERTS", "False"
    ).lower()
    in ("1", "true"),
    # If set, vLLM will skip the shared experts in mlp layer for benchmarking purposes.
    "VLLM_TEST_SKIP_MLP_SHARED_EXPERTS": lambda: os.environ.get(
        "VLLM_TEST_SKIP_MLP_SHARED_EXPERTS", "False"
    ).lower()
    in ("1", "true"),
    # If set, vLLM will skip the mlp gemm for benchmarking purposes.
    "VLLM_TEST_SKIP_MLP_GEMM": lambda: os.environ.get(
        "VLLM_TEST_SKIP_MLP_GEMM", "False"
    ).lower()
    in ("1", "true"),
    # If set, vLLM will skip the cuda graph (but keep padding) for debugging purposes.
    "VLLM_TEST_SKIP_CUDA_GRAPH": lambda: os.environ.get(
        "VLLM_TEST_SKIP_CUDA_GRAPH", "False"
    ).lower()
    in ("1", "true"),
    # If set, vLLM will pick up the provided Flash Attention MLA
    # max number splits for cuda graph decode
    "VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH": lambda: int(
        os.getenv("VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH", "32")
    ),
    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as vLLM.
    "VLLM_RAY_PER_WORKER_GPUS": lambda: float(
        os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")
    ),
    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "VLLM_RAY_BUNDLE_INDICES": lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),
    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable VLLM_CUDART_SO_PATH.
    "VLLM_CUDART_SO_PATH": lambda: os.getenv("VLLM_CUDART_SO_PATH", None),
    # Rank of the process in the data parallel setting
    "VLLM_DP_RANK": lambda: int(os.getenv("VLLM_DP_RANK", "0")),
    # Rank of the process in the data parallel setting.
    # Defaults to VLLM_DP_RANK when not set.
    "VLLM_DP_RANK_LOCAL": lambda: int(
        os.getenv("VLLM_DP_RANK_LOCAL", sys.modules[__name__].VLLM_DP_RANK)
    ),
    # World size of the data parallel setting
    "VLLM_DP_SIZE": lambda: int(os.getenv("VLLM_DP_SIZE", "1")),
    # IP address of the master node in the data parallel setting
    "VLLM_DP_MASTER_IP": lambda: os.getenv(
        "VLLM_DP_MASTER_IP", os.getenv("MASTER_ADDR", "127.0.0.1")
    ),
    # Port of the master node in the data parallel setting
    "VLLM_DP_MASTER_PORT": lambda: int(
        os.getenv("VLLM_DP_MASTER_PORT", os.getenv("MASTER_PORT", "0"))
    ),
    # If set, vllm will use the SPMD (one core engine per LLM) mode for DP.
    "VLLM_DP_SPMD_MODE": lambda: os.getenv("VLLM_DP_SPMD_MODE", "False").lower()
    in ("1", "true"),
    # If set, vllm will revise the port and vineyard socket argument DP ranks
    # are launched with torchrun.
    "VLLM_SERVE_WITH_TORCHRUN": lambda: os.getenv(
        "VLLM_SERVE_WITH_TORCHRUN", "False"
    ).lower()
    in ("1", "true"),
    # If set (the default behaviour), when EP is not enabled, the TP size/rank
    # will be flatten across dp ranks, which, is in use when we want to run
    # attention DP + MoE TP.
    #
    # In some cases, we want to run DP with two or more standalone TP instances,
    # e.g., deploying two TP-4 P instances inside a 8-GPUs pod on spectrum.
    "VLLM_FLATTEN_TP_ACROSS_DP": lambda: os.environ.get(
        "VLLM_FLATTEN_TP_ACROSS_DP", "True"
    ).lower()
    in ("1", "true"),
    # In the context of executing MoE models with Data-Parallel, Expert-Parallel
    # and Batched All-to-All dispatch/combine kernels, VLLM_MOE_DP_CHUNK_SIZE
    # dictates the quantum of tokens that can be dispatched from a DP
    # rank. All DP ranks process the activations in VLLM_MOE_DP_CHUNK_SIZE
    # units.
    "VLLM_MOE_DP_CHUNK_SIZE": lambda: int(os.getenv("VLLM_MOE_DP_CHUNK_SIZE", "256")),
    # If set, use bladnn gdn_recur kernel for GDN layers.
    "VLLM_ENABLE_MOE_DP_CHUNK": lambda: bool(
        int(os.getenv("VLLM_ENABLE_MOE_DP_CHUNK", "1"))
    ),
    # If set, use bladnn gdn kernel for GDN layers.
    "VLLM_GDN_USE_BLADNN": lambda: bool(int(os.getenv("VLLM_GDN_USE_BLADNN", "0"))),
    # Randomize inputs during dummy runs when using Data Parallel
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": lambda: os.environ.get(
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", "0"
    )
    == "1",
    # Strategy to pack the data parallel ranks for Ray.
    # Available options:
    # - "fill":
    #   for DP master node, allocate exactly data-parallel-size-local DP ranks,
    #   for non-master nodes, allocate as many DP ranks as can fit;
    # - "strict":
    #   allocate exactly data-parallel-size-local DP ranks to each picked node;
    # - "span":
    #   Should be used only when a single DP rank requires multiple nodes.
    #   allocate one DP rank over as many nodes as required for set world_size;
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "VLLM_RAY_DP_PACK_STRATEGY": lambda: os.getenv(
        "VLLM_RAY_DP_PACK_STRATEGY", "strict"
    ),
    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "VLLM_CI_USE_S3": lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",
    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "VLLM_MODEL_REDIRECT_PATH": lambda: os.environ.get(
        "VLLM_MODEL_REDIRECT_PATH", None
    ),
    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "VLLM_MARLIN_USE_ATOMIC_ADD": lambda: os.environ.get(
        "VLLM_MARLIN_USE_ATOMIC_ADD", "0"
    )
    == "1",
    # Used to dispatch visible devices inside vLLM rather via the environment
    # variable CUDA_VISIBLE_DEVICES, as it is known to cause issues when
    # launching with DeepEP on H20 with 4 NICs environments.
    "VLLM_CUDA_VISIBLE_DEVICES": lambda: []
    if "VLLM_CUDA_VISIBLE_DEVICES" not in os.environ
    else [
        int(x)
        for x in (
            os.environ["VLLM_CUDA_VISIBLE_DEVICES"].split(",")
            if os.environ["VLLM_CUDA_VISIBLE_DEVICES"]
            else []
        )
    ],
    # Specify ranks to use when launching vLLM within a pre-setup
    # torch.distributed environment.
    "VLLM_DISTRIBUTED_USE_RANKS": lambda: []
    if "VLLM_DISTRIBUTED_USE_RANKS" not in os.environ
    else [
        int(x)
        for x in (
            os.environ["VLLM_DISTRIBUTED_USE_RANKS"].split(",")
            if os.environ["VLLM_DISTRIBUTED_USE_RANKS"]
            else []
        )
    ],
    # FastV parameters, used to control the layer from which pruning begins.
    # If it is set to 0, it means pruning is disabled.
    "VLLM_VL_FASTV_K": lambda: int(os.getenv("VLLM_VL_FASTV_K", "0")),
    # FastV parameters, used to control the pruning ratio.
    "VLLM_VL_FASTV_R": lambda: float(os.getenv("VLLM_VL_FASTV_R", "0.0")),
    # FastV parameters, used to control the threshold for pruning. If the overall
    # attention recall ratio is greater than this parameter,
    # no pruning will be performed.
    # A value of -1 indicates that pruning is always enabled.
    "VLLM_VL_FASTV_THRESHOLD": lambda: float(
        os.getenv("VLLM_VL_FASTV_THRESHOLD", "-1")
    ),
    # VL: do resize inside model execution. Default to false as some
    # preprocessors in the pipeline may already resize the input.
    "VLLM_VL_PROCESSOR_DO_RESIZE": lambda: (
        os.getenv("VLLM_VL_PROCESSOR_DO_RESIZE", "False").lower() in ("true", "1")
    ),
    # VL: use fast processor in transformers.
    "VLLM_VL_FAST_PROCESSOR": lambda: (
        os.getenv("VLLM_VL_FAST_PROCESSOR", "True").lower() in ("true", "1")
    ),
    # VL: use fixed vision grid num in fast processor.
    # It's for first qwen3-vl-30b-a3b model
    "VLLM_VL_VISION_NUM_GRID_PER_SIDE": lambda: (
        int(os.getenv("VLLM_VL_VISION_NUM_GRID_PER_SIDE", "-1"))
    ),
    # Whether to use marlin kernel in mxfp4 quantization method
    "VLLM_MXFP4_USE_MARLIN": lambda: maybe_convert_bool(
        os.environ.get("VLLM_MXFP4_USE_MARLIN", None)
    ),
    # The activation dtype for marlin kernel
    "VLLM_MARLIN_INPUT_DTYPE": env_with_choices(
        "VLLM_MARLIN_INPUT_DTYPE", None, ["int8", "fp8"]
    ),
    # Whether to use DeepEPLL kernels for NVFP4 quantization and dispatch method
    # only supported on Blackwell GPUs and with
    # https://github.com/deepseek-ai/DeepEP/pull/341
    "VLLM_DEEPEPLL_NVFP4_DISPATCH": lambda: bool(
        int(os.getenv("VLLM_DEEPEPLL_NVFP4_DISPATCH", "0"))
    ),
    # Whether to use DeepEPLL kernels with fp8 dispatch for fp4 quantization.
    # TODO: not recommanded as the current dequant_fp8 kernel is not optimized.
    "VLLM_DEEPEPLL_NVFP4_FP8_DISPATCH": lambda: bool(
        int(os.getenv("VLLM_DEEPEPLL_NVFP4_FP8_DISPATCH", "0"))
    ),
    # Whether to use DeepEPHT kernels for NVFP4 quantization and dispatch method
    # only supported on Blackwell GPUs and with
    # https://github.com/deepseek-ai/DeepEP/pull/341
    "VLLM_DEEPEPHT_NVFP4_DISPATCH": lambda: bool(
        int(os.getenv("VLLM_DEEPEPHT_NVFP4_DISPATCH", "0"))
    ),
    # Whether to use DeepEPHT kernels with fp8 dispatch for fp4 quantization.
    # TODO: not recommanded as the current dequant_fp8 kernel is not optimized.
    "VLLM_DEEPEPHT_NVFP4_FP8_DISPATCH": lambda: bool(
        int(os.getenv("VLLM_DEEPEPHT_NVFP4_FP8_DISPATCH", "0"))
    ),
    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "VLLM_V1_USE_OUTLINES_CACHE": lambda: os.environ.get(
        "VLLM_V1_USE_OUTLINES_CACHE", "0"
    )
    == "1",
    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "VLLM_TPU_BUCKET_PADDING_GAP": lambda: int(
        os.environ["VLLM_TPU_BUCKET_PADDING_GAP"]
    )
    if "VLLM_TPU_BUCKET_PADDING_GAP" in os.environ
    else 0,
    "VLLM_TPU_MOST_MODEL_LEN": lambda: maybe_convert_int(
        os.environ.get("VLLM_TPU_MOST_MODEL_LEN", None)
    ),
    # Whether using Pathways
    "VLLM_TPU_USING_PATHWAYS": lambda: bool(
        "proxy" in os.getenv("JAX_PLATFORMS", "").lower()
    ),
    # Allow use of DeepGemm kernels for fused moe ops.
    "VLLM_USE_DEEP_GEMM": lambda: (
        None
        if (
            "VLLM_USE_DEEP_GEMM" not in os.environ
            and "VLLM_FP8_USE_DEEPGEMM" not in os.environ
            and "VLLM_MOE_USE_DEEPGEMM" not in os.environ
        )
        else (
            os.environ.get("VLLM_USE_DEEP_GEMM", "0").strip().lower() in ("1", "true")
            or os.environ.get("VLLM_FP8_USE_DEEPGEMM", "0").strip().lower()
            in ("1", "true")
            or os.environ.get("VLLM_MOE_USE_DEEPGEMM", "0").strip().lower()
            in ("1", "true")
        )
    ),
    # Allow use of DeepGemm specifically for MoE fused ops (overrides only MoE).
    "VLLM_MOE_USE_DEEP_GEMM": lambda: bool(
        int(os.getenv("VLLM_MOE_USE_DEEP_GEMM", "1"))
    ),
    # Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs.
    "VLLM_USE_DEEP_GEMM_E8M0": lambda: bool(
        int(os.getenv("VLLM_USE_DEEP_GEMM_E8M0", "1"))
    ),
    # Whether to use the legacy per-token group quant kernel when deepgemm is used.
    # The legacy kernel has optimization for scaling factor layout, but has known
    # issue under cudagraph with current develop branch.
    "VLLM_USE_LEGACY_PER_TOKEN_GROUP_QUANT": lambda: bool(
        int(os.getenv("VLLM_USE_LEGACY_PER_TOKEN_GROUP_QUANT", "0"))
    ),
    # DeepGemm JITs the kernels on-demand. The warmup attempts to make DeepGemm
    # JIT all the required kernels before model execution so there is no
    # JIT'ing in the hot-path. However, this warmup increases the engine
    # startup time by a couple of minutes.
    # Available options:
    #  - "skip"  : Skip warmup.
    #  - "full"  : Warmup deepgemm by running all possible gemm shapes the
    #   engine could encounter.
    #  - "relax" : Select gemm shapes to run based on some heuristics. The
    #   heuristic aims to have the same effect as running all possible gemm
    #   shapes, but provides no guarantees.
    "VLLM_DEEP_GEMM_WARMUP": env_with_choices(
        "VLLM_DEEP_GEMM_WARMUP",
        "skip",
        [
            "skip",
            "full",
            "relax",
        ],
    ),
    # For backwards compatibility, allow skipping deep gemm warmup via this variable.
    "VLLM_SKIP_DEEP_GEMM_WARMUP": lambda: (
        None
        if "VLLM_SKIP_DEEP_GEMM_WARMUP" not in os.environ
        else os.getenv("VLLM_SKIP_DEEP_GEMM_WARMUP", "False").lower() in ("true", "1")
    ),
    # Whether to use fused grouped_topk used for MoE expert selection.
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK": lambda: bool(
        int(os.getenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"))
    ),
    # Allow use of FlashInfer MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP16": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP16", "0"))
    ),
    # Allow use of FlashInfer MoE kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP8": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP8", "0"))
    ),
    # Allow use of FlashInfer CUTLASS kernels for fused moe ops.
    "VLLM_USE_FLASHINFER_MOE_FP4": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_FP4", "0"))
    ),
    # If set to 1, use the FlashInfer
    # MXFP8 (activation) x MXFP4 (weight) MoE backend.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "0"))
    ),
    # If set to 1, use the FlashInfer CUTLASS backend for
    # MXFP8 (activation) x MXFP4 (weight) MoE.
    # This is separate from the TRTLLMGEN path controlled by
    # VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS", "0"))
    ),
    # If set to 1, use the FlashInfer
    # BF16 (activation) x MXFP4 (weight) MoE backend.
    "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "0"))
    ),
    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),
    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(
        os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256")
    ),
    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "VLLM_ALLOW_INSECURE_SERIALIZATION": lambda: bool(
        int(os.getenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0"))
    ),
    # IP address used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_HOST": lambda: os.getenv(
        "VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"
    ),
    # Port used for NIXL handshake between remote agents.
    "VLLM_NIXL_SIDE_CHANNEL_PORT": lambda: int(
        os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")
    ),
    # Port used for Mooncake handshake between remote agents.
    "VLLM_MOONCAKE_BOOTSTRAP_PORT": lambda: int(
        os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "8998")
    ),
    # all2all backend for vllm's expert parallel communication
    # Available options:
    # - "naive": naive all2all implementation using broadcasts
    # - "allgather_reducescatter": all2all implementation based on allgather and
    #  reducescatter
    # - "pplx": use pplx kernels
    # - "deepep_high_throughput", use deepep high-throughput kernels
    # - "deepep_low_latency", use deepep low-latency kernels
    # - "flashinfer_all2allv", use flashinfer alltoallv kernels for mnnvl
    "VLLM_ALL2ALL_BACKEND": env_with_choices(
        "VLLM_ALL2ALL_BACKEND",
        "allgather_reducescatter",
        [
            "naive",
            "pplx",
            "deepep_high_throughput",
            "deepep_low_latency",
            "allgather_reducescatter",
            "flashinfer_all2allv",
        ],
    ),
    # Flashinfer MoE backend for vLLM's fused Mixture-of-Experts support.
    # Both require compute capability 10.0 or above.
    # Available options:
    # - "throughput":  [default]
    #     Uses CUTLASS kernels optimized for high-throughput batch inference.
    # - "latency":
    #     Uses TensorRT-LLM kernels optimized for low-latency inference.
    "VLLM_FLASHINFER_MOE_BACKEND": env_with_choices(
        "VLLM_FLASHINFER_MOE_BACKEND",
        "latency",
        ["throughput", "latency", "masked_gemm"],
    ),
    # Control the workspace buffer size for the FlashInfer backend.
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": lambda: int(
        os.getenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(394 * 1024 * 1024))
    ),
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE CUTLASS Kernel. This value is used to create a buffer for
    # the blockscale tensor of activations NVFP4 Quantization.
    # This is used to prevent the kernel from running out of memory.
    "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE": lambda: int(
        os.getenv("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")
    ),
    # Specifies the thresholds of the communicated tensor sizes under which
    # vllm should use flashinfer fused allreduce. The variable should be a
    # JSON with the following format:
    #     { <world size>: <max size in mb> }
    # Unspecified world sizes will fall back to
    #     { 2: 64, 4: 1, <everything else>: 0.5 }
    "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB": lambda: json.loads(
        os.getenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB", "{}")
    ),
    # MoE routing strategy selector.
    # See `RoutingSimulator.get_available_strategies()` # for available
    # strategies.
    # Custom routing strategies can be registered by
    # RoutingSimulator.register_strategy()
    # Note: custom strategies may not produce correct model outputs
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": lambda: os.environ.get(
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""
    ).lower(),
    # Regex timeout for use by the vLLM tool parsing plugins.
    "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")
    ),
    # Reduce CPU usage when vLLM is idle. Enabling this will incur small
    # latency penalty when a request eventually comes.
    "VLLM_SLEEP_WHEN_IDLE": lambda: bool(int(os.getenv("VLLM_SLEEP_WHEN_IDLE", "0"))),
    # Control the PD transfer timeouts.
    "VLLM_PD_TRANSFER_TIMEOUT_SECONDS": lambda: float(
        os.getenv("VLLM_PD_TRANSFER_TIMEOUT_SECONDS", "30")
    ),
    "VLLM_PD_CONNMANAGER_CAP": lambda: int(os.getenv("VLLM_PD_CONNMANAGER_CAP", "128")),
    "VLLM_EPD_TRANSFER_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_EPD_TRANSFER_TIMEOUT_SECONDS", "10")
    ),
    "VLLM_EPD_ENABLE_GLOBAL_ENCODER_CACHE": lambda: (
        os.getenv(
            "VLLM_EPD_ENABLE_GLOBAL_ENCODER_CACHE", "False"
        ).lower() in ("true", "1")
    ),
    "VLLM_PD_NUM_LOGITS_SLOTS": lambda: int(
        os.getenv("VLLM_PD_NUM_LOGITS_SLOTS", "256")
    ),
    "VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS": lambda: float(
        os.getenv("VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS", "30")
    ),
    # Control vineyard kv store io timeouts.
    "VLLM_KVS_IO_TIMEOUT_SECONDS": lambda: float(
        os.getenv("VLLM_KVS_IO_TIMEOUT_SECONDS", "30")
    ),
    # Control whether vineyard kv store is enabled for this request.
    "VLLM_KVS_ON_MIN_LENGTH": lambda: int(os.getenv("VLLM_KVS_ON_MIN_LENGTH", "1024")),
    # Use filesystem cache for multimodal data in OmniLLMEngine.
    "VLLM_USE_FILE_PATH_MMD_CACHE": lambda: (
        os.getenv("VLLM_USE_FILE_PATH_MMD_CACHE", "False").lower() in ("true", "1")
    ),
    "VLLM_PREPROCESS_MMD_CACHE_SIZE": lambda: int(
        os.getenv("VLLM_PREPROCESS_MMD_CACHE_SIZE", "100")
    ),
    "VLLM_OMNI_PREPROCESS_ON_GPU": lambda: (
        os.getenv("VLLM_OMNI_PREPROCESS_ON_GPU", "False").lower() in ("true", "1")
    ),
    # Use ZMQ for OmniLLMEngine T2T and T2C communication.
    "VLLM_OMNI_T2T_USE_ZMQ": lambda: (
        os.getenv("VLLM_OMNI_T2T_USE_ZMQ", "True").lower() in ("true", "1")
    ),
    "VLLM_OMNI_T2C_USE_ZMQ": lambda: (
        os.getenv("VLLM_OMNI_T2C_USE_ZMQ", "True").lower() in ("true", "1")
    ),
    "VLLM_OMNI_SKIP_TEXT_ENCODE": lambda: (
        os.getenv("VLLM_OMNI_SKIP_TEXT_ENCODE", "False").lower() in ("true", "1")
    ),
    # Control the path used for persisted kvcache block files.
    "VLLM_KVCACHE_BLOCK_FILE_PATH": lambda: os.getenv(
        "VLLM_KVCACHE_BLOCK_FILE_PATH", "/tmp/vllm_kvcache_blocks"
    ),
    # Control whether to use xxhash for Prefix Caching
    "VLLM_USE_XXHASH": lambda: bool(int(os.getenv("VLLM_USE_XXHASH", "0"))),
    # For test persisting kv cache blocks partial match
    "VLLM_TEST_PERSIST_BLOCK_RATIO": lambda: float(
        os.getenv("VLLM_TEST_PERSIST_BLOCK_RATIO", "1.0")
    ),
    # For test loading kv cache blocks using directio
    "VLLM_TEST_USE_DIO": lambda: bool(int(os.getenv("VLLM_TEST_USE_DIO", "0"))),
    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "VLLM_MQ_MAX_CHUNK_BYTES_MB": lambda: int(
        os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")
    ),
    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
    ),
    # KV Cache layout used throughout vllm.
    # Some common values are:
    # - NHD
    # - HND
    # Where N=num_blocks, H=num_heads and D=head_size. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    "VLLM_KV_CACHE_LAYOUT": env_with_choices(
        "VLLM_KV_CACHE_LAYOUT", None, ["NHD", "HND"]
    ),
    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "VLLM_COMPUTE_NANS_IN_LOGITS": lambda: bool(
        int(os.getenv("VLLM_COMPUTE_NANS_IN_LOGITS", "0"))
    ),
    # Controls whether or not emulations are used for NVFP4
    # generations on machines < 100 for compressed-tensors
    # models
    "VLLM_USE_NVFP4_CT_EMULATIONS": lambda: bool(
        int(os.getenv("VLLM_USE_NVFP4_CT_EMULATIONS", "0"))
    ),
    # Time (in seconds) after which the KV cache on the producer side is
    # automatically cleared if no READ notification is received from the
    # consumer. This is only applicable when using NixlConnector in a
    # disaggregated decode-prefill setup.
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": lambda: int(
        os.getenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "480")
    ),
    # If set, use quick model loader to load model
    "VLLM_USE_QUICK_LOADER": lambda: bool(int(os.getenv("VLLM_USE_QUICK_LOADER", "0"))),
    # If set, use the vllm swap engine
    "VLLM_USE_SWAP_ENGINE": lambda: bool(int(os.getenv("VLLM_USE_SWAP_ENGINE", "0"))),
    # If set, enable Llumnix V0
    "VLLM_ENABLE_LLUMNIX_V0": lambda: bool(
        int(os.getenv("VLLM_ENABLE_LLUMNIX_V0", "0"))
    ),
    # If set, enable Llumnix V1
    "VLLM_ENABLE_LLUMNIX": lambda: bool(int(os.getenv("VLLM_ENABLE_LLUMNIX", "0"))),
    # If set, collect detailed migration status for migration
    "LLUMNIX_DETAILED_MIG_STATUS": lambda: bool(
        int(os.getenv("LLUMNIX_DETAILED_MIG_STATUS", "0"))
    ),
    # Set to limit the max number of requests being migrated simultaneously
    "LLUMNIX_MAX_REQ_MIG_IN": lambda: int(os.getenv("LLUMNIX_MAX_REQ_MIG_IN", "1")),
    # Set to limit the max tokens of requests being migrated simultaneously
    # only works when LLUMNIX_DETAILED_MIG_STATUS=True
    "LLUMNIX_MAX_TOKEN_MIG_IN": lambda: int(
        os.getenv("LLUMNIX_MAX_TOKEN_MIG_IN", "10000")
    ),
    # Set to limit the max block ratio of requests being migrated simultaneously
    # only works when LLUMNIX_DETAILED_MIG_STATUS=True
    "LLUMNIX_MAX_BLOCK_RATIO_MIG_IN": lambda: float(
        os.getenv("LLUMNIX_MAX_BLOCK_RATIO_MIG_IN", "0.3")
    ),
    # Timeout (in seconds) for MooncakeConnector in PD disaggregated setup.
    "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": lambda: int(
        os.getenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "480")
    ),
    # Controls whether or not to use cudnn prefill
    "VLLM_USE_CUDNN_PREFILL": lambda: bool(
        int(os.getenv("VLLM_USE_CUDNN_PREFILL", "0"))
    ),
    # Controls whether to use TRT-LLM ragged DeepSeek prefill
    "VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL": lambda: bool(
        int(os.getenv("VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL", "0"))
    ),
    # If set to 1/True, use the TRTLLM attention backend in flashinfer.
    # If set to 0/False, use the default attention backend in flashinfer.
    # If not set, auto-detect the attention backend in flashinfer.
    "VLLM_USE_TRTLLM_ATTENTION": lambda: (
        None
        if "VLLM_USE_TRTLLM_ATTENTION" not in os.environ
        else os.environ["VLLM_USE_TRTLLM_ATTENTION"].lower() in ("1", "true")
    ),
    # If set to 1, when we use fp8 kv, we do not quantize Q to fp8
    "VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION": lambda: bool(
        int(os.getenv("VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION", "0"))
    ),
    # If set, it means we pre-downloaded cubin files and flashinfer will
    # read the cubin files directly.
    "VLLM_HAS_FLASHINFER_CUBIN": lambda: bool(
        int(os.getenv("VLLM_HAS_FLASHINFER_CUBIN", "0"))
    ),
    # Supported options:
    # - "flashinfer-cudnn": use flashinfer cudnn GEMM backend
    # - "flashinfer-trtllm": use flashinfer trtllm GEMM backend
    # - "flashinfer-cutlass": use flashinfer cutlass GEMM backend
    # - <none>: automatically pick an available backend
    "VLLM_NVFP4_GEMM_BACKEND": env_with_choices(
        "VLLM_NVFP4_GEMM_BACKEND",
        None,
        ["flashinfer-cudnn", "flashinfer-trtllm", "flashinfer-cutlass", "cutlass"],
    ),
    "VLLM_USE_FLASHINFER_CUTLASS_FP4_GEMM": lambda: bool(
        int(os.getenv("VLLM_USE_FLASHINFER_CUTLASS_FP4_GEMM", "1"))
    ),
    "VLLM_USE_BLADNN_FP4_GEMM": lambda: bool(
        int(os.getenv("VLLM_USE_BLADNN_FP4_GEMM", "0"))
    ),
    "VLLM_USE_BLADNN_MOE_FP4": lambda: bool(
        int(os.getenv("VLLM_USE_BLADNN_MOE_FP4", "0"))
    ),
    "VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4": lambda: bool(
        int(os.getenv("VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4", "0"))
    ),
    "VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4": lambda: bool(
        int(os.getenv("VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4", "0"))
    ),
    "VLLM_USE_FP8_ATTN_PROJ_FOR_NVFP4": lambda: bool(
        int(os.getenv("VLLM_USE_FP8_ATTN_PROJ_FOR_NVFP4", "0"))
    ),
    "VLLM_NVFP4_GLOBAL_SCALE_WITH_NAN_TO_NUM": lambda: bool(
        int(os.getenv("VLLM_NVFP4_GLOBAL_SCALE_WITH_NAN_TO_NUM", "0"))
    ),
    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "VLLM_ENABLE_CUDAGRAPH_GC": lambda: bool(
        int(os.getenv("VLLM_ENABLE_CUDAGRAPH_GC", "0"))
    ),
    # If set, use the fuse QKNorm and RoPE kernel
    "VLLM_FUSE_QKNORM_AND_ROPE": lambda: bool(
        int(os.getenv("VLLM_FUSE_QKNORM_AND_ROPE", "0"))
    ),
    # If set, use the fuse QKNorm, RoPE and KV cache write kernel
    "VLLM_FUSE_QKNORM_ROPE_AND_KVCACHE_WRITE": lambda: bool(
        int(os.getenv("VLLM_FUSE_QKNORM_ROPE_AND_KVCACHE_WRITE", "0"))
    ),
    # If set, enable dashlog integration for info/error logs.
    "VLLM_ENABLE_DASHLOG": lambda: os.environ.get("VLLM_ENABLE_DASHLOG", "True").lower()
    in ("1", "true"),
    # Maximum delay in milliseconds for KVT retry mechanism.
    # If this value is greater than 17, additional delays will be added
    # at 100ms intervals beyond the default delay list.
    "VLLM_KVT_MAX_DELAY_MS": lambda: int(os.getenv("VLLM_KVT_MAX_DELAY_MS", "0")),
    # Use reduce_scatter instead of all_reuduce when tp_size > 1 and ep_size > 1.
    "VLLM_TPEP_REDUCE_SCATTER": lambda: bool(
        int(os.getenv("VLLM_TPEP_REDUCE_SCATTER", "0"))
    ),
    # If set, enable elastic EP failover (shrink)
    "VLLM_ENABLE_ELASTIC": lambda: bool(int(os.getenv("VLLM_ENABLE_ELASTIC", "0"))),
    # Collective communication (AllGather) buffer size in DeepEP
    # when elastic is enabled, DP communication should be replaced with DeepEP
    "VLLM_ELASTIC_COLL_BUFFER_BYTES": lambda: int(
        os.getenv("VLLM_ELASTIC_COLL_BUFFER_BYTES", "1048576")
    ),
    # Mask a rank or a node under rank failure.
    "VLLM_ELASTIC_MASK_NODE": lambda: bool(
        int(os.getenv("VLLM_ELASTIC_MASK_NODE", "1"))
    ),
    # Use migration backend to return token. Generally, this is used for test.
    "VLLM_USE_MIGRATION_BACKEND_RETURN_TOKEN": lambda: bool(
        int(os.getenv("VLLM_USE_MIGRATION_BACKEND_RETURN_TOKEN", "0"))
    ),
    # Disable padding to CUDA graph capture batch sizes.
    # TODO(wentao): https://github.com/vllm-project/vllm/issues/23378
    # After the issue is fixed, we can remove this flag.
    "VLLM_DISABLE_PAD_FOR_CUDAGRAPH": lambda: bool(
        int(os.getenv("VLLM_DISABLE_PAD_FOR_CUDAGRAPH", "0"))
    ),
    # Used to force set up loopback IP
    "VLLM_LOOPBACK_IP": lambda: os.getenv("VLLM_LOOPBACK_IP", ""),
    # Used to set the process name prefix for vLLM processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "VLLM".
    "VLLM_PROCESS_NAME_PREFIX": lambda: os.getenv("VLLM_PROCESS_NAME_PREFIX", "VLLM"),
    # Allow chunked local attention with hybrid kv cache manager.
    # Currently using the Hybrid KV cache manager with chunked local attention
    # in the Llama4 models (the only models currently using chunked local attn)
    # causes a latency regression. For this reason, we disable it by default.
    # This flag is used to allow users to enable it if they want to (to save on
    # kv-cache memory usage and enable longer contexts)
    # TODO(lucas): Remove this flag once latency regression is resolved.
    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(
        int(os.getenv("VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "0"))
    ),
    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, vLLM's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the vLLM server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "VLLM_ENABLE_RESPONSES_API_STORE": lambda: bool(
        int(os.getenv("VLLM_ENABLE_RESPONSES_API_STORE", "0"))
    ),
    # If set, use the fp8 mfma in rocm paged attention.
    "VLLM_ROCM_FP8_MFMA_PAGE_ATTN": lambda: bool(
        int(os.getenv("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0"))
    ),
    # Whether to use pytorch symmetric memory for allreduce
    "VLLM_ALLREDUCE_USE_SYMM_MEM": lambda: bool(
        int(os.getenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1"))
    ),
    # Experimental: use this to enable MCP tool calling for non harmony models
    "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": lambda: bool(
        int(os.getenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0"))
    ),
    # Allows vllm to find tuned config under customized folder
    "VLLM_TUNED_CONFIG_FOLDER": lambda: os.getenv("VLLM_TUNED_CONFIG_FOLDER", None),
    # Valid values are container,code_interpreter,web_search_preview
    # ex VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=container,code_interpreter
    # If the server_label of your mcp tool is not in this list it will
    # be completely ignored.
    "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": env_set_with_choices(
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
        default=[],
        choices=["container", "code_interpreter", "web_search_preview"],
    ),
    # Allows harmony instructions to be injected on system messages
    "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": lambda: bool(
        int(os.getenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0"))
    ),
    # Enable automatic retry when tool call JSON parsing fails
    # If enabled, returns an error message to the model to retry
    # If disabled (default), raises an exception and fails the request
    "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY": lambda: bool(
        int(os.getenv("VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "0"))
    ),
    # Add optional custom scopes for profiling, disable to avoid overheads
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING": lambda: bool(
        int(os.getenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "0"))
    ),
    # Add optional nvtx scopes for profiling, disable to avoid overheads
    "VLLM_NVTX_SCOPES_FOR_PROFILING": lambda: bool(
        int(os.getenv("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))
    ),
    # Represent block hashes in KV cache events as 64-bit integers instead of
    # raw bytes. Defaults to True for backward compatibility.
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": lambda: bool(
        int(os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))
    ),
    # Name of the shared memory buffer used for object storage.
    # Only effective when mm_config.mm_processor_cache_type == "shm".
    "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": lambda: os.getenv(
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", "VLLM_OBJECT_STORAGE_SHM_BUFFER"
    ),
    # The size in MB of the buffers (NVL and RDMA) used by DeepEP
    "VLLM_DEEPEP_BUFFER_SIZE_MB": lambda: int(
        os.getenv("VLLM_DEEPEP_BUFFER_SIZE_MB", "1024")
    ),
    # Force DeepEP to use intranode kernel for inter-node communication in
    # high throughput mode. This is useful archive higher prefill throuhgput
    # on system supports multi-node nvlink (e.g GB200).
    "VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE", "0"))
    ),
    # Allow DeepEP to use MNNVL (multi-node nvlink) for internode_ll kernel,
    # turn this for better latency on GB200 like system
    "VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL": lambda: bool(
        int(os.getenv("VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL", "0"))
    ),
    # The number of SMs to allocate for communication kernels when running DBO
    # the rest of the SMs on the device will be allocated to compute
    "VLLM_DBO_COMM_SMS": lambda: int(os.getenv("VLLM_DBO_COMM_SMS", "20")),
    # Enable max_autotune & coordinate_descent_tuning in inductor_config
    # to compile static shapes passed from compile_sizes in compilation_config
    # If set to 1, enable max_autotune; By default, this is enabled (1)
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": lambda: bool(
        int(os.getenv("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"))
    ),
    # If set to 1, enable coordinate_descent_tuning;
    # By default, this is enabled (1)
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": lambda: bool(
        int(os.getenv("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"))
    ),
    # Flag to enable NCCL symmetric memory allocation and registration
    "VLLM_USE_NCCL_SYMM_MEM": lambda: bool(
        int(os.getenv("VLLM_USE_NCCL_SYMM_MEM", "0"))
    ),
    # NCCL header path
    "VLLM_NCCL_INCLUDE_PATH": lambda: os.environ.get("VLLM_NCCL_INCLUDE_PATH", None),
    # Flag to enable FBGemm kernels on model execution
    "VLLM_USE_FBGEMM": lambda: bool(int(os.getenv("VLLM_USE_FBGEMM", "0"))),
    # GC debug config
    # - VLLM_GC_DEBUG=0: disable GC debugger
    # - VLLM_GC_DEBUG=1: enable GC debugger with gc.collect elpased times
    # - VLLM_GC_DEBUG='{"top_objects":5}': enable GC debugger with
    #                                      top 5 collected objects
    "VLLM_GC_DEBUG": lambda: os.getenv("VLLM_GC_DEBUG", ""),
    # Debug workspace allocations.
    # logging of workspace resize operations.
    "VLLM_DEBUG_WORKSPACE": lambda: bool(int(os.getenv("VLLM_DEBUG_WORKSPACE", "0"))),
    # Disables parallel execution of shared_experts via separate cuda stream
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(
        int(os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0"))
    ),
    # controls whether to open code switch gate
    "VLLM_ENABLE_CODE_SWITCH_GATE": lambda: (
        os.getenv("VLLM_ENABLE_CODE_SWITCH_GATE", "False").lower() in ("true", "1")
    ),
    # the path of tokenizer_config, only takes effect when
    # VLLM_ENABLE_CODE_SWITCH_GATE=1
    "VLLM_CODE_SWITCH_TOK_PATH": lambda: os.environ.get(
        "VLLM_CODE_SWITCH_TOK_PATH", ""
    ),
    # Used to avoid CUDA out-of-memory issues caused by overly long single outputs.
    "VLLM_CODE_SWITCH_GATE_MAX_TOKENS_TO_CHECK": lambda: int(
        os.getenv("VLLM_CODE_SWITCH_GATE_MAX_TOKENS_TO_CHECK", 4096)
    ),
    "VLLM_CODE_SWITCH_INTERMEDIATE_DIM": lambda: int(
        os.getenv("VLLM_CODE_SWITCH_INTERMEDIATE_DIM", 0)
    ),
    "VLLM_CODE_SWITCH_GATE_USE_RAW_LOGITS": lambda: bool(
        os.getenv("VLLM_CODE_SWITCH_GATE_USE_RAW_LOGITS", "False").lower()
        in ("true", "1")
    ),
    # Limits when we run shared_experts in a separate stream.
    # We found out that for large batch sizes, the separate stream
    # execution is not beneficial (most likely because of the input clone)
    # TODO(alexm-redhat): Tune to be more dynamic based on GPU type
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": lambda: int(
        int(os.getenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256))
    ),
    # Format for saving torch.compile cache artifacts
    # - "binary": saves as binary file
    #     Safe for multiple vllm serve processes accessing the same torch compile cache.
    # - "unpacked": saves as directory structure (for inspection/debugging)
    #     NOT multiprocess safe - race conditions may occur with multiple processes.
    #     Allows viewing and setting breakpoints in Inductor's code output files.
    "VLLM_COMPILE_CACHE_SAVE_FORMAT": env_with_choices(
        "VLLM_COMPILE_CACHE_SAVE_FORMAT", "binary", ["binary", "unpacked"]
    ),
    # Flag to enable v2 model runner.
    "VLLM_USE_V2_MODEL_RUNNER": lambda: bool(
        int(os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0"))
    ),
    # Disable step duration timing for asynchronous scheduling
    "VLLM_DISABLE_ASYNC_SCHED_TIMING": lambda: bool(
        int(os.getenv("VLLM_DISABLE_ASYNC_SCHED_TIMING", "0"))
    ),
    # Disable async metadata synchronization across DP ranks when asynchronous
    # scheduling is on.
    "VLLM_DISABLE_ASYNC_SCHED_DP_SYNC": lambda: bool(
        int(os.getenv("VLLM_DISABLE_ASYNC_SCHED_DP_SYNC", "1"))
    ),
    "VLLM_ENABLE_PHASED_DURATION_LOGGING": lambda: bool(
        int(os.getenv("VLLM_ENABLE_PHASED_DURATION_LOGGING", "0"))
    ),
    "VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS": lambda: (
        os.getenv("VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS", "False").lower()
        in ("true", "1")
    ),
    "VLLM_OMNI_THINKER_AVENCODER_CACHE": lambda: (
        os.getenv("VLLM_OMNI_THINKER_AVENCODER_CACHE", "False").lower() in ("true", "1")
    ),
    "VLLM_OMNI_THINKER_AVENCODER_CACHE_SIZE": lambda: int(
        os.getenv("VLLM_OMNI_THINKER_AVENCODER_CACHE_SIZE", "100")
    ),
    "VLLM_OMNI_ENABLE_TURN_BASE": lambda: (
        os.getenv("VLLM_OMNI_ENABLE_TURN_BASE", "False").lower() in ("true", "1")
    ),
    # Cache size for zero-shot xvector features to reduce loading overhead
    "VLLM_OMNI_XVECTOR_CACHE_SIZE": lambda: int(
        os.getenv("VLLM_OMNI_XVECTOR_CACHE_SIZE", "100")
    ),
    # If set, sampling parameters are forced to use the generate config.
    "VLLM_ENFORCE_GENERATE_CONFIG": lambda: (
        os.getenv("VLLM_ENFORCE_GENERATE_CONFIG", "False").lower() in ("true", "1")
    ),
    "VLLM_OMNI_CODE2WAV_CHUNK_SIZE": lambda: int(
        os.getenv("VLLM_OMNI_CODE2WAV_CHUNK_SIZE", "4")
    ),
    "VLLM_OMNI_CODE2WAV_MAX_BATCH_SIZE": lambda: int(
        os.getenv("VLLM_OMNI_CODE2WAV_MAX_BATCH_SIZE", "16")
    ),
    "VLLM_OMNI_CODE2WAV_ENABLE_DYNAMIC_CHUNK": lambda: (
        os.getenv("VLLM_OMNI_CODE2WAV_ENABLE_DYNAMIC_CHUNK", "False").lower()
        in ("true", "1")
    ),
    "VLLM_OMNI_CODE2WAV_DYNAMIC_CHUNK_SIZE_LIST": lambda: [
        int(x) for x in
        os.getenv("VLLM_OMNI_CODE2WAV_DYNAMIC_CHUNK_SIZE_LIST", "4,6,8").split(",")
    ],
    "VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE": lambda: (
        os.getenv("VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE", "False").lower() in ("true", "1")
    ),
    "VLLM_OMNI_REALTIME_VIDEO_FPS": lambda: int(
        os.getenv("VLLM_OMNI_REALTIME_VIDEO_FPS", "1")
    ),
    "VLLM_ENABLE_RUND_APPLICATION_TEMPLATE": lambda: bool(
        int(os.getenv("VLLM_ENABLE_RUND_APPLICATION_TEMPLATE", "0"))
    ),
    # KV transfer protocol: rdma or tcp
    "VLLM_KV_TRANS_PROTOCOL": lambda: os.getenv("VLLM_KV_TRANS_PROTOCOL", "rdma"),
    "VLLM_ENCODER_CACHE_LIMIT": lambda: int(os.getenv("VLLM_ENCODER_CACHE_LIMIT", "0")),
    "VLLM_ENABLE_BYPASS_TASK": lambda: (
        os.getenv("VLLM_ENABLE_BYPASS_TASK", "True").lower() in ("true", "1")
    ),
    "VLLM_KVS_USE_REQUEST_HASH": lambda: bool(
        int(os.getenv("VLLM_KVS_USE_REQUEST_HASH", "0"))
    ),
    "VLLM_ENABLE_SAMPLER_DP": lambda: (
        os.getenv("VLLM_ENABLE_SAMPLER_DP", "False").lower() in ("true", "1")
    ),
    # If set, rerun the model runner's profile run with attention after kvcache been
    # initialized, to warmup the possible JIT complation in attn implementation.
    "VLLM_ENABLE_MODEL_RUNNER_WARMUP": lambda: (
        os.getenv("VLLM_ENABLE_MODEL_RUNNER_WARMUP", "False").lower() in ("true", "1")
    ),
    "VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING": lambda: (
        os.getenv("VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING", "False").lower()
        in ("true", "1")
    ),
    "VLLM_OMNI_ENABLE_TALKER": lambda: (
        os.getenv("VLLM_OMNI_ENABLE_TALKER", "False").lower() in
        ("true", "1")
    ),
    # Export the CosyVoice codec-LLM sampled hidden states for unified FM
    # decoding. Registered here so it is propagated to vLLM/Ray workers.
    "VLLM_COSYVOICE_EXPORT_LLM_HIDDEN_FOR_FM": lambda: (
        os.getenv("VLLM_COSYVOICE_EXPORT_LLM_HIDDEN_FOR_FM", "False").lower()
        in ("true", "1")
    ),
    "VLLM_QWEN_TTS_GENERATE_MODE": lambda: os.getenv(
        "VLLM_QWEN_TTS_GENERATE_MODE", "default"
    ).lower(),
    # qsa envs
    "VLLM_QSA_USE_FP8_INDEXER": lambda: (
        os.getenv("VLLM_QSA_USE_FP8_INDEXER", "True").lower() in
        ("true", "1")
    ),
    "VLLM_QSA_PREFILL_USE_TL_INDEXER": lambda: (
        os.getenv("VLLM_QSA_PREFILL_USE_TL_INDEXER", "False").lower() in
        ("true", "1")
    ),
    "VLLM_QSA_DECODE_USE_TL_INDEXER": lambda: (
        os.getenv("VLLM_QSA_DECODE_USE_TL_INDEXER", "False").lower() in
        ("true", "1")
    ),
    "VLLM_QSA_PREFILL_ATTN_BACKEND": env_with_choices(
        "VLLM_QSA_PREFILL_ATTN_BACKEND", "triton", ["triton", "pai-fa3","tilelang","torch"] # noqa: E501
    ),
    "VLLM_QSA_PREFILL_USE_CP": lambda: (
        os.getenv("VLLM_QSA_PREFILL_USE_CP", "False").lower() in
        ("true", "1")
    ),
    "VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP": lambda: (
        os.getenv("VLLM_QSA_PREFILL_ONLY_INDEXER_USE_CP", "False").lower() in
        ("true", "1")
    ),
    "VLLM_QSA_DECODE_ATTN_BACKEND": env_with_choices(
        "VLLM_QSA_DECODE_ATTN_BACKEND", "pai-fa3", ["fa2", "pai-fa3","fa3"]
    ),
    
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """
    Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Allow __setattr__ and __delattr__ to be defined, see also: PEP-0726.
class ModifiableModule(ModuleType):
    def __setattr__(self, name, value):
        if name in environment_variables:
            if isinstance(value, bool):
                os.environ[name] = "1" if value else "0"
            else:
                os.environ[name] = str(value)
            environment_variables[name] = lambda: value

            if hasattr(__getattr__, "cache_clear"):
                __getattr__.cache_clear()
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __delattr__(self, name):
        raise AttributeError(
            "attribute {name!r} cannot be deleted from module {__name__!r}"
        )


sys.modules[__name__].__class__ = ModifiableModule


def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache"""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """
    Enables caching of environment variables. This is useful for performance
    reasons, as it avoids the need to re-evaluate environment variables on
    every call.

    NOTE: Currently, it's invoked after service initialization to reduce
    runtime overhead. This also means that environment variables should NOT
    be updated after the service is initialized.
    """
    if _is_envs_cache_enabled():
        # Avoid wrapping functools.cache multiple times
        return
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """
    Resets the environment variables cache. It could be used to isolate environments
    between unit tests.
    """
    global __getattr__
    # If __getattr__ is wrapped by functions.cache, unwrap the caching layer.
    if _is_envs_cache_enabled():
        __getattr__ = __getattr__.__wrapped__


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compile_factors() -> dict[str, object]:
    """Return env vars used for torch.compile cache keys.

    Start with every known vLLM env var; drop entries in `ignored_factors`;
    hash everything else. This keeps the cache key aligned across workers."""

    ignored_factors: set[str] = {
        "MAX_JOBS",
        "VLLM_RPC_BASE_PATH",
        "VLLM_USE_MODELSCOPE",
        "VLLM_RINGBUFFER_WARNING_INTERVAL",
        "VLLM_DEBUG_DUMP_PATH",
        "VLLM_PORT",
        "VLLM_CACHE_ROOT",
        "LD_LIBRARY_PATH",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
        "VLLM_CI_USE_S3",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_HOST_IP",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "VLLM_USAGE_STATS_SERVER",
        "VLLM_NO_USAGE_STATS",
        "VLLM_DO_NOT_TRACK",
        "VLLM_LOGGING_LEVEL",
        "VLLM_LOGGING_PREFIX",
        "VLLM_LOGGING_STREAM",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_COLOR",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
        "VLLM_TUNED_CONFIG_FOLDER",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        "VLLM_SLEEP_WHEN_IDLE",
        "VLLM_IMAGE_FETCH_TIMEOUT",
        "VLLM_VIDEO_FETCH_TIMEOUT",
        "VLLM_AUDIO_FETCH_TIMEOUT",
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
        "VLLM_MEDIA_LOADING_THREAD_COUNT",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
        "VLLM_VIDEO_LOADER_BACKEND",
        "VLLM_MEDIA_CONNECTOR",
        "VLLM_ASSETS_CACHE",
        "VLLM_ASSETS_CACHE_MODEL_CLEAN",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_CPU_OMP_THREADS_BIND",
        "VLLM_CPU_NUM_OF_RESERVED_CPU",
        "VLLM_CPU_MOE_PREPACK",
        "VLLM_CPU_SGL_KERNEL",
        "VLLM_TEST_FORCE_LOAD_FORMAT",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
        "VLLM_ENCODER_CACHE_LIMIT",
    }

    from vllm.config.utils import normalize_value

    factors: dict[str, object] = {}
    for factor, getter in environment_variables.items():
        if factor in ignored_factors:
            continue

        try:
            raw = getter()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping environment variable %s while hashing compile factors: %s",
                factor,
                exc,
            )
            continue

        factors[factor] = normalize_value(raw)

    ray_noset_env_vars = [
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",
    ]

    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))

    return factors
