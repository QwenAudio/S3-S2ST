# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import queue
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from enum import Enum
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import better_exceptions
import msgspec
import zmq

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import CurrentSpan, init_tracer, trace_execution
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.math_utils import round_up
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ExternalData,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.dpcoord import Participant
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.hybrid_connector import send_bypass_task
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.version import __version__ as VLLM_VERSION

if envs.VLLM_ENABLE_LLUMNIX:
    from llumnix.instance_info import InstanceStatus
    from llumnix.vllm_llumlet_proxy import (
        StepPhase,
        init_instance_status_with_scheduler,
    )
    from llumnix.vllm_llumlet_proxy import VLLMLlumletProxy as LlumletProxy
else:
    InstanceStatus = dict[str, str]

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


def _add_output(output: dict[int, EngineCoreOutputs],
                input: dict[int, list[EngineCoreOutput]]):
    for client_index, kvconn_output in input.items():
        if client_index in output:
            output[client_index].outputs.extend(kvconn_output)
        else:
            output[client_index] = EngineCoreOutputs(
                outputs=kvconn_output
            )
    return


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        driver_tensor_queue_union: Any | None = None,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s, vllm path: %s",
                VLLM_VERSION,
                vllm_config,
                __file__,
            )

            for key in dir(envs):
                if key and key.startswith("VLLM"):
                    logger.info("%s=%s", key, getattr(envs, key))
            envvars: list[tuple[str, str]] = []
            for key, value in sorted(os.environ.items()):
                if key and any(key.startswith(k) for k in [
                    "CUDA", "NCCL", "ACCL", "TORCH"
                ]):
                    envvars.append((key, value))
            for key, value in sorted(envvars):
                logger.info("%s=%s", key, value)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config, driver_tensor_queue_union)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.attn_backends: list[type[AttentionBackend]] = self.collective_rpc(
            "get_attn_backends_type"
        )[0]

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        # Effective drafter max model length
        effective_drafter_max_model_len = vllm_config.model_config.max_model_len
        speculative_config = vllm_config.speculative_config
        if (
            speculative_config
            and speculative_config.draft_model_config is not None
            and speculative_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = (
                speculative_config.draft_model_config.max_model_len
            )
        self.effective_drafter_max_model_len = effective_drafter_max_model_len
        logger.info(
            "Effective draft max model len is: "
            f"{self.effective_drafter_max_model_len}"
        )
        assert (
            self.effective_drafter_max_model_len
            >= vllm_config.model_config.max_model_len
        ), (
            "Draft model max length must be no smaller than the target model "
            f"max length. Got draft: {self.effective_drafter_max_model_len}, "
            f"target: {vllm_config.model_config.max_model_len}."
        )

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            # NOTE: when Eagle is enabled, use the next token for hash 
            # calculation only in prefill node of PD disagg.
            use_eagle_token = (
                vllm_config.speculative_config is not None
                and vllm_config.speculative_config.use_eagle()
                and vllm_config.kv_transfer_config is not None
                and vllm_config.kv_transfer_config.is_kv_producer
            )
            if use_eagle_token:
                logger.info_once(
                    "Using the next token for the block hash calculation "
                    "in P nodes when Eagle is enabled."
                )

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn, use_eagle_token,
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling
        # NOTE: If distributed_executor_backend is not "mp", such as "uni", EngineCore
        # cpu time can not overlap with Worker's cpu time, so don't pre-calculate dp
        # metadata for best performance.
        self.synchronize_dp_metadata = (
            (not envs.VLLM_DISABLE_ASYNC_SCHED_DP_SYNC)
            and (vllm_config.parallel_config.distributed_executor_backend == "mp")
            # For hybrid models, it is hard to know if the sequence is in the
            # first step at the decode node when P/D disaggregation is on.
            and (not vllm_config.model_config.is_hybrid)
        )
        self.cudagraph_dispatcher: CudagraphDispatcher | None = None
        if self.async_scheduling:
            self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
        else:
            self.cudagraph_dispatcher = None

        self.use_cuda_graph = (
            vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and not vllm_config.model_config.enforce_eager
        )
        if vllm_config.compilation_config.cudagraph_capture_sizes:
            self.cudagraph_batch_sizes = sorted(
                vllm_config.compilation_config.cudagraph_capture_sizes
            )
        else:
            self.cudagraph_batch_sizes = []
        self.num_spec_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        self.uniform_decode_query_len = (
            1
            if not vllm_config.speculative_config
            else 1 + vllm_config.speculative_config.num_speculative_tokens
        )

        if envs.VLLM_ENABLE_LLUMNIX:
            self.instance_status: dict[str, str] = init_instance_status_with_scheduler(
                self.scheduler
            )
            self.llumlet_proxy: LlumletProxy = None
        self._external_data_buffer: dict[str, queue.Queue] = defaultdict(queue.Queue)

        # Init vLLM tracer using OTEL API
        self.tracer = init_tracer(__name__)

        # Clean mask buffer before actual run, since warmup
        # may cause some ranks to be masked unexpectedly.
        self.masked_ranks: set[int] = set()
        if envs.VLLM_ENABLE_ELASTIC:
            self.model_executor.collective_rpc("clean_deepep_mask_buffer")

        self.aborts_queue = queue.Queue[list[str]]()

        if self.use_cuda_graph and self.cudagraph_dispatcher is not None:
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                vllm_config.compilation_config.cudagraph_mode,
                self.uniform_decode_query_len)
        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                if (
                    envs.VLLM_USE_SWAP_ENGINE
                    and vllm_config.cache_config.num_gpu_blocks_override is not None
                ):
                    available_gpu_memory = [0 for _ in range(len(kv_cache_specs))]
                else:
                    available_gpu_memory = (
                        self.model_executor.determine_available_memory()
                    )
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        kvconn = self.scheduler.get_kv_connector()
        if kvconn:
            eaten = kvconn.on_add_req(request)
            if eaten:
                return

        self.scheduler.add_request(request)

    def update_metric(
        self,
        step_duration: float | None,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        expert_imbalance_ratio: float | None,
        intra_expert_imbalance_ratio: float | None,
    ):
        if step_duration is None:
            return

        num_prefill_groups = len(
            [
                req_id
                for req_id, num_new_tokens
                in scheduler_output.num_scheduled_tokens.items()
                if num_new_tokens > 1 + self.num_spec_tokens
            ]
        )
        if num_prefill_groups > 0:
            for output in engine_core_outputs.values():
                output.prefill_step_duration = step_duration
                output.decode_step_duration = None
                # TODO(yancey.yx): imbalance ratio on prefill step
                output.expert_imbalance_ratio = None
                output.intra_expert_imbalance_ratio = None
        else:
            for output in engine_core_outputs.values():
                output.prefill_step_duration = None
                output.decode_step_duration = step_duration
                output.expert_imbalance_ratio = expert_imbalance_ratio
                output.intra_expert_imbalance_ratio = intra_expert_imbalance_ratio

    def get_instance_status(self) -> InstanceStatus:
        return self.instance_status

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        kvconn = self.scheduler.get_kv_connector()
        if kvconn:
            for reqid in request_ids:
                kvconn.on_abort_req(reqid, reason="core")
            return

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    def update_request_with_data(
        self, request_id: str, external_data: dict[str, Any]
    ) -> None:
        """
        Update an existing request with external data (tensors, token ids, etc.)
        """
        self._external_data_buffer[request_id].put(external_data)

    def get_external_data_for_request(self, request_id: str) -> ExternalData | None:
        """Retrieve external data for a specific request."""
        try:
            return self._external_data_buffer[request_id].get_nowait()
        except queue.Empty:
            return None

    def clear_external_data_for_request(self, request_id: str) -> None:
        """
        Clear external data for a specific request after they've been used.
        """
        if request_id in self._external_data_buffer:
            del self._external_data_buffer[request_id]

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except SystemExit:
            raise
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def _log_err_callback(self, scheduler_output: SchedulerOutput):
        """Log error details of a future that's not expected to return a result."""

        def callback(f, sched_output=scheduler_output):
            with self.log_error_detail(sched_output):
                result = f.result()
                assert result is None

        return callback

    def _wait_model_output_future(
        self,
        model_output_future: \
            Future[ModelRunnerOutput | None] | Future[ModelRunnerOutput]) -> tuple[
            ModelRunnerOutput | None, 
            dict[int, list[EngineCoreOutput]]]:
        return model_output_future.result(), {}

    @trace_execution("engine_core")
    def step(self) -> tuple[dict[int, EngineCoreOutputs] | set[int], bool, None]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        If elastic is enabled and timeout is detected, 
        the first element in the tuple should be a set to indicate 
        the failed ranks (set[int]).
        """
        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.update_instance_status(
                self.instance_status, step_phase=StepPhase.STEP_BEGIN
            )

        span = CurrentSpan()
        kvconn = self.scheduler.get_kv_connector()
        kvconn_outputs: dict[int, list[EngineCoreOutput]] | None = None
        if kvconn:
            kvconn_outputs = kvconn.step()
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            if envs.VLLM_ENABLE_LLUMNIX:
                self.llumlet_proxy.update_instance_status(
                    self.instance_status, step_phase=StepPhase.BEFORE_RETURN
                )
            engine_core_outputs: dict[int, EngineCoreOutputs] = dict()
            _add_output(engine_core_outputs, kvconn_outputs or {})
            return engine_core_outputs, False, None

        span.start_timer("schedule")
        scheduler_output = self.scheduler.schedule()
        span.end_timer()

        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.update_instance_status(
                self.instance_status,
                step_phase=StepPhase.AFTER_SCHEDULE,
                scheduler_output=scheduler_output,
            )

        _dppart: Participant | None = getattr(self, "_dppart", None)
        if _dppart is not None and scheduler_output.total_num_scheduled_tokens > 0:
            _dppart.new_step()

        future: Future[ModelRunnerOutput | None] = \
            self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with self.log_error_detail(scheduler_output):
            span.start_timer("execute_model")
            model_output, kvconn_outputs2 = self._wait_model_output_future(future)
            span.end_timer()
            if model_output is None:
                span.start_timer("sample_tokens")
                model_output = self.model_executor.sample_tokens(grammar_output)
                span.end_timer()

        if envs.VLLM_ENABLE_ELASTIC and model_output.ranks_to_mask is not None:
            return (  # type: ignore
                model_output.ranks_to_mask,
                scheduler_output.total_num_scheduled_tokens > 0,
                scheduler_output
            )

        # Clear external tensors after request is finished.
        for request_id in scheduler_output.finished_req_ids:
            self.clear_external_data_for_request(request_id)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )  # type: ignore

        for req_id in list(self.scheduler.requests_requiring_external_data):  # type: ignore
            external_data = self.get_external_data_for_request(req_id)
            if external_data is not None:
                self.scheduler.update_external_data(req_id, external_data)
                # When all extra data has been consumed, use padding data.
                if external_data.is_padding:
                    self.update_request_with_data(req_id, external_data)

        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.update_instance_status(
                self.instance_status,
                step_phase=StepPhase.AFTER_UPDATE,
                scheduler_output=scheduler_output,
                model_output=model_output,
            )

        if model_output.step_duration:
            self.update_metric(
                model_output.step_duration,
                scheduler_output,
                model_output,
                engine_core_outputs,
                model_output.expert_imbalance_ratio,
                model_output.intra_expert_imbalance_ratio,
            )
        _add_output(engine_core_outputs, kvconn_outputs or {})
        _add_output(engine_core_outputs, kvconn_outputs2 or {})

        return (
            engine_core_outputs,
            scheduler_output.total_num_scheduled_tokens > 0,
            None,
        )

    def _pad_for_sequence_parallelism_and_cudagraph(
        self,
        num_scheduled_tokens: int,
    ) -> int:
        if num_scheduled_tokens == 0:
            return -1

        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.vllm_config.compilation_config.pass_config.enable_sp and tp_size > 1:
            num_scheduled_tokens = round_up(num_scheduled_tokens, tp_size)

        if (
            self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and hasattr(self, "cudagraph_batch_sizes")
            and self.cudagraph_batch_sizes
            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
        ):
            # Use CUDA graphs.
            # Add padding to the batch size.
            num_scheduled_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens
            )

        return num_scheduled_tokens

    def _make_scheduled_dp_metadata(
        self,
        scheduler_output: SchedulerOutput,
    ):
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_scheduled_reqs = len(scheduler_output.num_scheduled_tokens)
        num_tokens_padded = self._pad_for_sequence_parallelism_and_cudagraph(
            num_scheduled_tokens
        )

        # Disable DP padding when running eager
        allow_dp_padding = (
            self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        )

        is_prompt_batch_local = False
        is_prefill_only_local = False
        input_fits_in_drafter = True
        max_num_scheduled_tokens = 0
        if num_scheduled_tokens > 0:
            for req_id, tokens in scheduler_output.num_scheduled_tokens.items():
                max_num_scheduled_tokens = max(max_num_scheduled_tokens, tokens)
                req_scheduled_spec_decode_token = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
                )
                is_decode = (
                    tokens == self.uniform_decode_query_len
                    or tokens == len(req_scheduled_spec_decode_token) + 1
                )
                input_fits_in_drafter = input_fits_in_drafter and (
                    scheduler_output.scheduled_seq_lens.get(req_id, 0)
                    + tokens + self.num_spec_tokens
                    <= self.effective_drafter_max_model_len
                )
                is_prompt_batch_local = is_prompt_batch_local or (not is_decode)
                is_prefill_only_local = is_prefill_only_local and (not is_decode)

        uniform_decode = (
            not is_prompt_batch_local
            and max_num_scheduled_tokens == self.uniform_decode_query_len
            and num_scheduled_tokens == num_scheduled_reqs * max_num_scheduled_tokens
        )

        dispatch_cudagraph = (
            lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(  # type: ignore
                num_tokens=num_tokens,
                has_lora=False,
                uniform_decode=uniform_decode,
                disable_full=disable_full,
            )
            if not self.vllm_config.model_config.enforce_eager
            else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
        )
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, (
                self.vllm_config.model_config.is_encoder_decoder
                and len(scheduler_output.scheduled_encoder_inputs) > 0
            )
        )

        # NOTE: async scheduling does not support microbatching for pre-compute
        # ScheduledDPMetaData, as the request's order of SchedulerOutput may diff
        # from InputBatch's in GPUModelRunner. Its support will cause large code diff
        # from upstream and hard to maintain. So we disable it here.
        scheduler_output.scheduled_dp_metadata = coordinate_batch_across_dp(
            num_tokens_unpadded=num_scheduled_tokens,
            num_reqs=num_scheduled_reqs,
            allow_microbatching=False,
            allow_dp_padding=allow_dp_padding,
            cudagraph_mode=cudagraph_mode.value,
            parallel_config=self.vllm_config.parallel_config,
            num_tokens_padded=num_tokens_padded,
            is_prompt_batch_local=is_prompt_batch_local,
            is_prefill_only_local=is_prefill_only_local,
            uniform_decode=uniform_decode,
            input_fits_in_drafter=input_fits_in_drafter,
            group=getattr(self, "dp_group", None),
        )
        logger.debug("%r", scheduler_output.scheduled_dp_metadata)

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool, SchedulerOutput | None]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.update_instance_status(
                self.instance_status, step_phase=StepPhase.STEP_BEGIN
            )

        kvconn = self.scheduler.get_kv_connector()
        kvconn_outputs: dict[int, list[EngineCoreOutput]] | None = None
        if kvconn:
            kvconn_outputs = kvconn.step()

        engine_core_outputs = None
        scheduler_output = None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()

            if envs.VLLM_ENABLE_LLUMNIX:
                self.llumlet_proxy.update_instance_status(
                    self.instance_status,
                    step_phase=StepPhase.AFTER_SCHEDULE,
                    scheduler_output=scheduler_output,
                )

            _dppart: Participant | None = getattr(self, "_dppart", None)
            if _dppart is not None and scheduler_output.total_num_scheduled_tokens > 0:
                _dppart.new_step()

            # NOTE: trigger dp sync when "model_executed" is True, otherwise
            # the sync will be triggered from other ranks in "_process_engine_step".
            if (_dppart is not None
                and self.async_scheduling
                and self.synchronize_dp_metadata
                and scheduler_output.total_num_scheduled_tokens > 0
            ):
                self._make_scheduled_dp_metadata(scheduler_output)

            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    outputs: dict[int, EngineCoreOutputs] = dict()
                    _add_output(outputs, kvconn_outputs or {})
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return outputs, True, None

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            if envs.VLLM_ENABLE_LLUMNIX:
                self.llumlet_proxy.update_instance_status(
                    self.instance_status, step_phase=StepPhase.BEFORE_RETURN
                )
            outputs = dict()
            _add_output(outputs, kvconn_outputs or {})
            return outputs, False, None

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output, kvconn_outputs2 = self._wait_model_output_future(future)
            assert model_output is not None

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            sche_spec_decode_tokens = None
            if self.use_spec_decode:
                sche_spec_decode_tokens = (
                    self.scheduler.update_scheduled_draft_token_ids(
                        deferred_scheduler_output, model_output
                    )
                )
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output,
                scheduled_spec_decode_tokens=sche_spec_decode_tokens,
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.update_instance_status(
                self.instance_status,
                step_phase=StepPhase.AFTER_UPDATE,
                scheduler_output=scheduler_output,
                model_output=model_output,
            )

        # metric
        if model_output.step_duration:
            self.update_metric(
                model_output.step_duration,
                scheduler_output,
                model_output,
                engine_core_outputs,
                model_output.expert_imbalance_ratio,
                model_output.intra_expert_imbalance_ratio,
            )

        _add_output(engine_core_outputs, kvconn_outputs or {})
        _add_output(engine_core_outputs, kvconn_outputs2 or {})

        # For async scheduling.
        scheduled_batch = (
            scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens > 0
        )
        dummy_scheduler_output = None if scheduled_batch else scheduler_output

        return engine_core_outputs, model_executed, dummy_scheduler_output

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                if isinstance(ids, str):
                    # Should be a list here, but also handle string just in case.
                    ids = (ids,)
                request_ids.extend(ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()
        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def initialize_kv_transfer(self):
        self.model_executor.initialize_kv_transfer()
        self.scheduler.create_kv_connector()

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self, **kwargs) -> ModelRunnerOutput:
        return self.model_executor.execute_dummy_batch(**kwargs)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def save_quick_loader_model(
        self,
        path: str,
        parallel_num: int,
        parallel_type: str = "tp",
        quantization_type: str | None = None,
        dynamic_minmax_config: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_quick_loader_model(
            path=path,
            parallel_num=parallel_num,
            parallel_type=parallel_type,
            quantization_type=quantization_type,
            dynamic_minmax_config=dynamic_minmax_config,
            max_size=max_size,
        )

    def save_quick_loader_model_for_drafter(
        self,
        path: str,
        parallel_num: int,
        parallel_type: str = "tp",
        quantization_type: str | None = None,
        dynamic_minmax_config: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_quick_loader_model_for_drafter(
            path=path,
            parallel_num=parallel_num,
            parallel_type=parallel_type,
            quantization_type=quantization_type,
            dynamic_minmax_config=dynamic_minmax_config,
            max_size=max_size,
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def record_triton_cache_inode(self) -> None:
        """
        Record Triton cache inode mappings for conflict detection.
        """

        from vllm.triton_utils.custom_cache_manager import CustomCacheManager
        CustomCacheManager.record_triton_cache_inode()
    
        self.collective_rpc("record_triton_cache_inode")

    def save_tensorized_model(
        self,
        tensorizer_config,
    ) -> None:
        self.model_executor.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    def migrate_out(self, dst_engine_host: str, dst_engine_port: str,
                    migration_param: dict) -> bool:
        return self.llumlet_proxy.migrate_out(migration_param, dst_engine_host,
                                              dst_engine_port)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


def _core_init(core, vllm_config: VllmConfig):
    if vllm_config.kv_transfer_config is None:
        return
    from vllm.v1.hybrid_connector.engine_proxy import core_init

    core_init(core, vllm_config)
    return


def _use_bypass(core: EngineCore) -> bool:
    if not envs.VLLM_ENABLE_BYPASS_TASK:
        return False
    kvconn = core.scheduler.get_kv_connector()
    if kvconn is None:
        return False
    from vllm.v1.hybrid_connector import HybridConnector
    if not isinstance(kvconn, HybridConnector):
        return False
    from vllm.v1.hybrid_connector.engine_proxy import is_statemodel
    return not is_statemodel()


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    RANK_MASKED = b'RANK_MASKED'

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
        driver_tensor_queue_union: Any | None = None,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )

            self._init_data_parallel(vllm_config)
            _core_init(self, vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                driver_tensor_queue_union=driver_tensor_queue_union,
            )
            
            if envs.VLLM_ENABLE_LLUMNIX:
                self.llumlet_proxy = LlumletProxy(vllm_config, self.scheduler,
                                                  engine_index)
                addresses = self.llumlet_proxy.add_llumlet_address(addresses)

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

        self._enable_bypass = _use_bypass(self)
        self.task_executor: ThreadPoolExecutor | None = None
        if self._enable_bypass:
            logger.info("enable bypass")
            self.task_executor = ThreadPoolExecutor(max_workers=1)

        # In batch mode, start a background heartbeat thread to maintain upstream
        # connections by sending heartbeats every heartbeat_interval for requests
        # in scheduler_proxy and the waiting queue.
        if self.vllm_config.scheduler_config.policy == "batch":
            threading.Thread(
                target=self._heartbeat_loop,
                args=(self.vllm_config.scheduler_config.heartbeat_interval,),
                daemon=True,
            ).start()

    @property
    def engines_running(self) -> bool:
        return False

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        if "AQUILA_RPC_PROTOCOL" not in os.environ:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.error(
                    "EngineCore failed to start. Exception:\n%s",
                    "".join(better_exceptions.format_exception(*sys.exc_info())),
                )
            else:
                logger.error(
                    "EngineCore encountered a fatal error. Exception:\n%s",
                    "".join(better_exceptions.format_exception(*sys.exc_info())),
                )
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> tuple[bool, SchedulerOutput | None]:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed, scheduler_outputd = self.step_fn()

        if envs.VLLM_ENABLE_ELASTIC and isinstance(outputs, set):
            self.handle_rank_timeout(outputs, scheduler_outputd)
            outputs = {}
        else:
            outputs = cast(dict[int, Any], outputs)

        if outputs:
            from vllm.v1.hybrid_connector.migration import backend as migration

            migration.on_outputs(outputs)
        # If enable Llumnix, transfer outputs to APIServer using Llumnix's
        # output_forwarder.
        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.forward_outputs(outputs)
        else:
            # Put EngineCoreOutputs into the output queue.
            for output in outputs.items() if outputs else ():
                self.output_queue.put_nowait(output)

        # Post-step hook.
        self.post_step(model_executed)

        return model_executed, scheduler_outputd

    def handle_rank_timeout(
        self,
        ranks_to_mask: set[int],
        scheduler_output: SchedulerOutput | None
    ):
        self.masked_ranks.update(ranks_to_mask)
        msg = EngineCoreProc.RANK_MASKED + bytes(f'{list(ranks_to_mask)}',
                                                 encoding='utf-8')
        logger.info("EngineCore sends msg %s to core clients", msg.decode())
        self.output_queue.put_nowait(msg)
        if scheduler_output is not None:
            self.scheduler.revert_request_outputs(scheduler_output)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""
        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.EXTERNAL_DATA:
            self.update_request_with_data(request.request_id, request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            if call_id is None:
                # Not interested in the result
                return

            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # If enable Llumnix, shut down llumlet_proxy
        if envs.VLLM_ENABLE_LLUMNIX:
            self.llumlet_proxy.shutdown()
        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()
        external_data_decoder = MsgpackDecoder(ExternalData)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    elif request_type == EngineCoreRequestType.EXTERNAL_DATA:
                        request = external_data_decoder.decode(data_frames)
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                if envs.VLLM_ENABLE_ELASTIC \
                    and isinstance(output, bytes) \
                    and output.startswith(EngineCoreProc.RANK_MASKED):
                    for socket in sockets:
                        socket.send(output)
                    continue
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _heartbeat_loop(self, heartbeat_interval: float = 0.15):
        """Heartbeat loop for batch scheduler. Sends periodic heartbeats to
        clients during batch scheduling for requests in scheduler_proxy and
        the waiting queue to maintain connections liveness."""

        assert self.vllm_config.scheduler_config.policy == "batch"

        logger.info(
            "Starting heartbeat thread with interval %.3f seconds", heartbeat_interval
        )

        while True:
            try:
                outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
                for (
                    req_id,
                    request,
                ) in self.scheduler.scheduler_proxy.unprocessed_requests.items():  # type: ignore
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=[],
                        )
                    )
                for request in self.scheduler.waiting_queue:  # type: ignore
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                        )
                    )

                # Send heartbeat packets to the client via ZMQ socket.
                if outputs:
                    for client_index, outs in outputs.items():
                        self.output_queue.put_nowait(
                            (client_index, EngineCoreOutputs(outputs=outs))
                        )
                time.sleep(heartbeat_interval)
            except Exception as e:
                logger.error("Heartbeat loop failed: %s", str(e))

    def _wait_model_output_future(
        self,
        model_output_future: \
            Future[ModelRunnerOutput | None] | Future[ModelRunnerOutput]) -> tuple[
            ModelRunnerOutput | None, 
            dict[int, list[EngineCoreOutput]]]:
        # For the GDN state model, tokens share state and do not support transfer
        # during computation.
        kvconn_outputs: dict[int, list[EngineCoreOutput]] = dict()
        if not self._enable_bypass:
            return model_output_future.result(), kvconn_outputs

        def wait_model_output():
            model_output_or_ex = None
            try:
                model_output_or_ex = model_output_future.result()
            except Exception as e:
                model_output_or_ex = e # type: ignore
            self.input_queue.put_nowait(
                (EngineCoreRequestType.EXECUTE_MODEL_RESULT, model_output_or_ex)
            )
            return

        assert self.task_executor is not None
        self.task_executor.submit(wait_model_output)

        temp_items: list[tuple[EngineCoreRequestType, Any]] = []
        def _should_process_now(item: tuple[EngineCoreRequestType, Any]) -> bool:
            reqtype, req = item
            if reqtype == EngineCoreRequestType.ADD:
                return True
            if reqtype == EngineCoreRequestType.ABORT:
                return True
            if reqtype == EngineCoreRequestType.UTILITY:
                _, _, method_name, _ = req
                if method_name == '_start_step':
                    return True
                if method_name == 'migrate_out':
                    return True
            temp_items.append(item)
            return False

        def _requeue_deferred_items():
            for item in reversed(temp_items):
                self.input_queue.queue.appendleft(item)
            return

        kvconn = self.scheduler.get_kv_connector()
        assert kvconn is not None
        while True:
            if not kvconn.has_requests():
                item = self.input_queue.get()  # blocking op
                if item[0] == EngineCoreRequestType.EXECUTE_MODEL_RESULT:
                    _requeue_deferred_items()
                    if isinstance(item[1], Exception):
                        # kvconn_outputs will be discard
                        raise item[1]
                    else:
                        return item[1], kvconn_outputs
                if _should_process_now(item):
                    self._handle_client_request(*item)

            # process input queue
            while not self.input_queue.empty():
                item = self.input_queue.get_nowait()
                if item[0] == EngineCoreRequestType.EXECUTE_MODEL_RESULT:
                    _requeue_deferred_items()
                    if isinstance(item[1], Exception):
                        raise item[1]
                    else:
                        return item[1], kvconn_outputs
                if _should_process_now(item):
                    self._handle_client_request(*item)

            if envs.VLLM_ENABLE_LLUMNIX:
                self.llumlet_proxy.update_instance_status(
                    self.instance_status, step_phase=StepPhase.BYPASS_STEP_BEGIN
                )

            # process kvconn step
            step_outputs = kvconn.step()

            if envs.VLLM_ENABLE_LLUMNIX:
                self.llumlet_proxy.update_instance_status(
                    self.instance_status, step_phase=StepPhase.BYPASS_STEP_END
                )

            if step_outputs:
                for cliidx, outputs in step_outputs.items():
                    if cliidx in kvconn_outputs:
                        kvconn_outputs[cliidx].extend(outputs)
                    else:
                        kvconn_outputs[cliidx] = outputs

            # process kvconn tasks
            schedout: SchedulerOutput = SchedulerOutput.make_empty()
            meta = kvconn.build_connector_meta(schedout)
            send_bypass_task(meta)
        return  # dead code


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        driver_tensor_queue_union: Any | None = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            dp_rank,
            driver_tensor_queue_union=driver_tensor_queue_union,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self._dppart = Participant(vllm_config, self)
        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )

            if not vllm_config.parallel_config.data_parallel_external_lb:
                vllm_config.kv_transfer_config.engine_available_port += (
                    local_dp_rank
                )

            logger.info("adjust kvt cfg: %s", vllm_config.kv_transfer_config)
            logger.info(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                num_running_reqs=counts[0],
                num_waiting_reqs=counts[1],
                step_counter=self.step_counter,
                current_wave=self.current_wave,
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def _process_engine_step(self):
        forward_executed, scheduler_output = super()._process_engine_step()
        self._maybe_publish_request_counts()
        if forward_executed:
            return

        need_to_remove_finished = (
            scheduler_output is not None and len(scheduler_output.finished_req_ids) > 0
        )

        if (
            self.engines_running and not self.scheduler.has_unfinished_requests()
        ) or need_to_remove_finished:
            self._dppart.new_step()
            scheduled_dp_metadata = None
            if self.async_scheduling and self.synchronize_dp_metadata:
                # Disable DP padding when running eager
                allow_dp_padding = (
                    self.vllm_config.compilation_config.cudagraph_mode
                    != CUDAGraphMode.NONE
                )
                num_tokens_padded = self._pad_for_sequence_parallelism_and_cudagraph(
                    self.uniform_decode_query_len)
                dispatch_cudagraph = (
                    lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(  # type: ignore
                        num_tokens=num_tokens,
                        has_lora=False,
                        uniform_decode=True,
                        disable_full=disable_full,
                    )
                    if not self.vllm_config.model_config.enforce_eager
                    else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
                )
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded, (
                        self.vllm_config.model_config.is_encoder_decoder
                        and scheduler_output is not None
                        and len(scheduler_output.scheduled_encoder_inputs) > 0
                    )
                )
                scheduled_dp_metadata = coordinate_batch_across_dp(
                    num_tokens_unpadded=self.uniform_decode_query_len,
                    num_reqs=0,
                    allow_microbatching=False,
                    allow_dp_padding=allow_dp_padding,
                    parallel_config=self.vllm_config.parallel_config,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                    is_prompt_batch_local=False,
                    is_prefill_only_local=False,
                    uniform_decode=True,
                    input_fits_in_drafter=True,  # Dummy batch always fits in drafter
                    group=self.dp_group,
                )
            model_output = self.execute_dummy_batch(
                scheduler_output=scheduler_output,
                scheduled_dp_metadata=scheduled_dp_metadata,
            )

            if envs.VLLM_ENABLE_ELASTIC and model_output.ranks_to_mask is not None:
                self.handle_rank_timeout(
                    model_output.ranks_to_mask,
                    scheduler_output
                )
                return

            if scheduler_output is not None:
                self.scheduler.update_from_output(scheduler_output, model_output)
        return

    @property
    def engines_running(self) -> bool:
        return self._dppart.engines_running()

    def _start_step(self, step: int):
        return self._dppart.core_start_step(step)

    def start_step_threadsafe(self, step: int):
        # client_idx, call_id, method_name, args
        req = (-1, None, "_start_step", (step,))
        self.input_queue.put_nowait((EngineCoreRequestType.UTILITY, req))
        return

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.error(
                "EngineCore encountered a fatal error. Exception:\n%s",
                "".join(better_exceptions.format_exception(*sys.exc_info())),
            )
            raise
        finally:
            self.shutdown()
