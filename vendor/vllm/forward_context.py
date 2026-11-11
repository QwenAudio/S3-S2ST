# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import better_exceptions
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import (
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ubatch_utils import UBatchSlices, is_second_ubatch_empty
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class DropStats:

    def __init__(self, interval: int = 500):
        self.interval = interval
        self.drop_totals: dict[int, torch.Tensor] = {}
        self.recv_totals: dict[int, torch.Tensor] = {}
        self.counters: dict[int, torch.Tensor] = {}
        logger.info("Init DropStats, interval: %d", interval)

    def update(self, drop_tensor: torch.Tensor, recv_tensor: torch.Tensor,
               layer_index: int):
        """
        Update the stats and print the stats when the interval is reached.
        """
        if layer_index not in self.drop_totals:
            self.drop_totals[layer_index] = torch.zeros_like(
                drop_tensor, device=drop_tensor.device, dtype=torch.int64)
            self.recv_totals[layer_index] = torch.zeros_like(
                recv_tensor, device=recv_tensor.device, dtype=torch.int64)
            self.counters[layer_index] = torch.zeros(1,
                                                     device=drop_tensor.device,
                                                     dtype=torch.int64)

        self.drop_totals[layer_index].add_(drop_tensor)
        self.recv_totals[layer_index].add_(recv_tensor)
        self.counters[layer_index].add_(1)

    def print_stats(self, layer_index: int, rank: int, ep_size: int, group):
        """
        Check and print the stats when the interval is reached.
        """
        if torch.cuda.is_current_stream_capturing():
            if layer_index in self.drop_totals:
                self.drop_totals[layer_index].zero_()
                self.recv_totals[layer_index].zero_()
                self.counters[layer_index].zero_()
            return

        if (layer_index not in self.counters
                or self.drop_totals.get(layer_index) is None):
            return

        counter_val = self.counters[layer_index].item()
        if counter_val < self.interval:
            return

        drop_total = self.drop_totals[layer_index]
        recv_total = self.recv_totals[layer_index]

        drop_list = [torch.zeros_like(drop_total) for _ in range(ep_size)]
        recv_list = [torch.zeros_like(recv_total) for _ in range(ep_size)]

        dist.all_gather(drop_list, drop_total, group=group)
        dist.all_gather(recv_list, recv_total, group=group)

        drop_total.zero_()
        recv_total.zero_()
        self.counters[layer_index].zero_()

        if rank == 0:
            drop_list_t = torch.stack(drop_list)
            recv_list_t = torch.stack(recv_list)

            total_drop = drop_list_t.sum().item()
            total_recv = recv_list_t.sum().item()
            global_ratio = 100.0 * total_drop / max(total_recv, 1e-8)

            # Compute average expert drop ratio
            # drop_list_t: (ep_size, num_experts)
            expert_ratios = torch.where(
                recv_list_t > 0, 100.0 * drop_list_t / recv_list_t,
                torch.zeros_like(drop_list_t, dtype=torch.float32))
            valid_experts = (recv_list_t > 0).sum().item()
            avg_ratio = (expert_ratios.sum().item() /
                         valid_experts if valid_experts > 0 else 0.0)

            if layer_index == -1:
                logger.info(
                    '[All Layers] Stats (last %d steps): Global Drop Ratio: '
                    '%.2f%% (%d/%d), Average Expert Drop Ratio: %.2f%%',
                    counter_val, global_ratio, total_drop, total_recv,
                    avg_ratio)
            else:
                logger.info(
                    '[Layer %d] Stats (last %d steps): Global Drop Ratio: '
                    '%.2f%% (%d/%d), Average Expert Drop Ratio: %.2f%%',
                    layer_index, counter_val, global_ratio, total_drop,
                    total_recv, avg_ratio)


_DROP_STATS: DropStats | None = None


def get_drop_ctx() -> DropStats:
    global _DROP_STATS
    if _DROP_STATS is None:
        _DROP_STATS = DropStats(interval=envs.VLLM_MOE_TOKEN_DROP_STATS)
    return _DROP_STATS


track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


def _get_num_input_tokens(
    attn_metadata: Any,
    num_tokens: int,
) -> int:
    if attn_metadata is not None:
        # attn_metadata: Union[list[AttnMetadataDict], AttnMetadataDict]
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0]
            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))

        if hasattr(attn_metadata, "num_prefill_tokens"):
            # for v0 attention backends
            num_input_tokens = (
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
            )
        elif hasattr(attn_metadata, "num_input_tokens"):
            # for v1 attention backends
            num_input_tokens = attn_metadata.num_input_tokens
        else:
            # for v1 attention backends
            num_input_tokens = num_tokens
    else:
        num_input_tokens = num_tokens
    return num_input_tokens


@dataclass
class EPMetadata:
    is_prompt_batch: bool = False
    num_fused_moe_chunks: int = 1

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        is_prompt_batch: bool = False,
        num_tokens: int = 0,
    ) -> "EPMetadata":
        num_fused_moe_chunks = (
            num_tokens + envs.VLLM_FUSED_MOE_CHUNK_SIZE - 1
        ) // envs.VLLM_FUSED_MOE_CHUNK_SIZE

        return EPMetadata(
            is_prompt_batch=is_prompt_batch,
            num_fused_moe_chunks=num_fused_moe_chunks,
        )


class BatchDescriptor(NamedTuple):
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False
    """
    Whether this batch has active LoRA adapters.
    """

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor":
        """
        Return a relaxed version of current batch descriptor that is still compatible
        with PIECEWISE cudagraphs (or mixed prefill-decode FA cudagraphs).
        """
        return BatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    max_num_tokens: int,
    chunk_idx: int,
) -> list[int]:
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)
    sp_size = len(sp_tokens)

    local_size = [-1] * sp_size
    for i in range(sp_size):
        # Take into account sharding if MoE activation is sequence parallel.
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert (
            parallel_config.data_parallel_size > 1
            or parallel_config.all2all_backend == "deepep_low_latency"
        )
        dp_rank = parallel_config.data_parallel_rank

        assert (
            num_tokens_across_dp_cpu[dp_rank] == num_tokens
        ), f"{num_tokens_across_dp_cpu[dp_rank]=} {num_tokens=}"
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return DPMetadata(max_tokens_across_dp_cpu, num_tokens_across_dp_cpu)

    @contextmanager
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ):
        """
        Context manager to compute and temporarily set the per-rank local token
        sizes for a specific chunk during chunked forward execution.

        This is necessary to ensure each DP (data parallel) rank processes its
        designated portion of tokens in lockstep with others, even when the
        token counts are uneven or some ranks have completed their input early.

        For chunked execution, we break up the total tokens on each rank into
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given
        `chunk_idx`, this context manager sets `self.local_sizes` to the number
        of tokens to process in that chunk on each rank.

        `self.local_sizes` is only valid inside the context.

        Args:
            sequence_parallel_size: When Attn is TP and MoE layers are EP,
                                    we use SP between the layers to avoid
                                    redundant ops. We need this value to
                                    compute the chunked sizes.
            max_chunk_size_per_rank: The max number of tokens each rank is
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        self.local_sizes = _compute_chunked_local_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
            max_chunk_size_per_rank,
            chunk_idx,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int):
        """
        Context manager for setting self.local_sizes. Same as self.chunked_sizes
        but without any chunking.
        """
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.
    # In this case the input to the MoEs will be distributed w.r.t both
    # DP and TP rank.
    # When sp_size==1, this is just the cummulative num tokens across DP.
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each 
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass

    # set dynamically for each forward pass
    ep_metadata: EPMetadata | None = None
    dp_metadata: DPMetadata | None = None

    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: UBatchSlices | None = None

    num_pad_tokens: torch.Tensor | None = None

    num_tokens: int | None = None

    attn_input_can_use_rs: bool = False

    def __post_init__(self):
        assert (
            self.cudagraph_runtime_mode.valid_runtime_modes()
        ), f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"


_forward_context: ForwardContext | None = None


def has_forward_context() -> bool:
    """Check if forward context is currently set.

    Returns:
        True if forward context is set, False otherwise.
    """
    global _forward_context
    return _forward_context is not None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    ep_metadata: EPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    num_pad_tokens: torch.Tensor | None = None,
    num_tokens: int | None = None,
    attn_input_can_use_rs: bool = False,
):
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        ep_metadata=ep_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        num_pad_tokens=num_pad_tokens,
        num_tokens=num_tokens,
        attn_input_can_use_rs=attn_input_can_use_rs,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def maybe_override_execution_stream(
    execution_stream: torch.cuda.Stream | None = None,
):
    """Execute model forward within the given stream, required by fluid-eplb."""
    if execution_stream is None:
        yield
    else:
        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(execution_stream):
            if current_stream != execution_stream:
                execution_stream.wait_stream(current_stream)
            yield
            if current_stream != execution_stream:
                current_stream.wait_stream(execution_stream)


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    is_prompt_batch: bool = False,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    num_pad_tokens: torch.Tensor | None = None,
    skip_eplb: bool = False,
    execution_stream: torch.cuda.Stream | None = None,
    common_attn_metadata: Any | None = None,  # for logging
    input_batch: Any | None = None,  # for logging
    scheduler_output: Any | None = None,  # for logging
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    if vllm_config.parallel_config.data_parallel_size > 1 and (
        attn_metadata is not None or num_tokens is not None
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if num_tokens_across_dp is None:
            assert ubatch_slices is None
            assert num_tokens is not None
            # Note that when async scheduling is on, the sychronize will be called
            # with the CPU group, as it will be overlapped with the execution of
            # target model on GPU, it won't be a performance concern.
            scheduled_dp_metadata = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                num_reqs=num_tokens,  # use num_tokens.
                allow_microbatching=False,
                # already padded across dp before the draft step
                allow_dp_padding=False,
                cudagraph_mode=cudagraph_runtime_mode.value,
                parallel_config=vllm_config.parallel_config,
                is_prompt_batch_local=is_prompt_batch,
                is_prefill_only_local=False,
            )
            num_tokens_across_dp = scheduled_dp_metadata.num_tokens_across_dp
            assert num_tokens_across_dp is not None

        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config,
            _get_num_input_tokens(
                attn_metadata, num_tokens or 0
            ),
            num_tokens_across_dp,
        )
    elif vllm_config.parallel_config.all2all_backend == "deepep_low_latency" and (
        attn_metadata is not None or num_tokens is not None
    ):
        max_tokens_across_dp = _get_num_input_tokens(
            attn_metadata, num_tokens or 0
        )
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config,
            max_tokens_across_dp,
            torch.tensor([max_tokens_across_dp], dtype=torch.int32),
        )

    ep_metadata: EPMetadata | None = None
    if (
        vllm_config.parallel_config.enable_expert_parallel
        and vllm_config.parallel_config.expert_parallel_size > 1
    ):
        if dp_metadata is None:
            max_tokens_across_dp = _get_num_input_tokens(
                attn_metadata, num_tokens or 0
            )
        else:
            max_tokens_across_dp = dp_metadata.max_tokens_across_dp_cpu.item()
        ep_metadata = EPMetadata.make(
            vllm_config.parallel_config,
            vllm_config.scheduler_config,
            is_prompt_batch,
            max_tokens_across_dp,
        )

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    attn_input_can_use_rs = is_prompt_batch and envs.VLLM_ATTN_INPUT_USE_RS
    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine,
        dp_metadata,
        ep_metadata,
        cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        num_pad_tokens=num_pad_tokens,
        num_tokens=num_tokens,
        attn_input_can_use_rs=attn_input_can_use_rs,
    )

    try:
        with (
            override_forward_context(forward_context),
            (maybe_override_execution_stream(execution_stream)
             if execution_stream else contextlib.nullcontext())
        ):
            if envs.VLLM_MOE_TOKEN_DROP and (envs.VLLM_MOE_TOKEN_DROP_STATS > 0):
                for layer_index in list(get_drop_ctx().counters.keys()):
                    get_drop_ctx().print_stats(layer_index, get_expert_parallel_rank(),
                                               get_expert_parallel_world_size(),
                                               get_ep_group().device_group)
            yield
    except Exception:
        logger.error(
            "Dumping input data for V1 LLM engine (v%s) with config: %s. "
            "Exception:\n%s",
            VLLM_VERSION,
            get_current_vllm_config(),
            "".join(better_exceptions.format_exception(*sys.exc_info())),
        )
        try:
            logger.error(
                "Dumping forward context due to exception: "
                f"dp_metadata={forward_context.dp_metadata}, "
                f"ep_metadata={forward_context.ep_metadata}, "
                f"cudagraph_runtime_mode={forward_context.cudagraph_runtime_mode}, "
                f"batch_descriptor={forward_context.batch_descriptor}, "
                f"ubatch_slices={forward_context.ubatch_slices}, "
                f"num_pad_tokens={forward_context.num_pad_tokens}, "
                f"num_tokens={forward_context.num_tokens}, "
                f"attn_input_can_use_rs={forward_context.attn_input_can_use_rs}, "
                f"input_batch={input_batch}, "
                f"scheduler_output={scheduler_output}, "
                f"{common_attn_metadata=}."
            )
        except Exception:
            logger.exception("Error preparing object to dump")
        raise
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform

            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )

        # Cleanup the deep ep buffer after each prefill forward pass
        if ep_metadata:
            # NOTE: when normal kernel is not used, don't clean buffer
            if (ep_metadata.is_prompt_batch
                and vllm_config.model_config.is_mixture_of_experts
                and envs.VLLM_MOE_USE_DEEPEP
                and envs.VLLM_MOE_USE_DEEPEP_NORMAL
            ):
                from vllm.model_executor.layers.fused_moe.expert_parallel import (
                    cleanup_deep_ep_buffer_after_prefill,
                )

                cleanup_deep_ep_buffer_after_prefill(vllm_config=vllm_config)


@dataclass
class ExpertParallelContext:
    n_nodes: int
    n_gpus_per_node: int
    n_experts: int

    origin_expert_stats: torch.Tensor  # [layer_id, expert_id] -> stat
    remapped_expert_stats: torch.Tensor  # [layer_id, expert_id]

    stats_enabled: bool = False
    stats_logging_internal: int = 0

    def __post_init__(self):
        self.reset_expert_stats()

    def reset_expert_stats(self):
        if self.origin_expert_stats is None or self.remapped_expert_stats is None:
            return
        self.origin_expert_stats.zero_()
        self.remapped_expert_stats.zero_()


_EXPERT_PARALLEL_CONTEXT: ExpertParallelContext | None = None


def get_expert_parallel_context(vllm_config=None) -> ExpertParallelContext:
    global _EXPERT_PARALLEL_CONTEXT

    if vllm_config is None and _EXPERT_PARALLEL_CONTEXT is None:
        vllm_config: VllmConfig = get_current_vllm_config()

    if _EXPERT_PARALLEL_CONTEXT is None:
        from vllm.model_executor.layers.fused_moe.layer import load_expert_allocation

        n_gpus_per_node = vllm_config.parallel_config.tensor_parallel_size
        n_nodes = vllm_config.parallel_config.world_size_across_dp // n_gpus_per_node

        if vllm_config.model_config.is_omni3_model():
            hf_config = vllm_config.model_config.hf_config.thinker_config.text_config
        else:
            hf_config = vllm_config.model_config.hf_config

        if hasattr(hf_config, 'n_routed_experts'):
            num_experts = hf_config.n_routed_experts
        elif hasattr(hf_config, "num_experts"):
            num_experts = hf_config.num_experts
        else:
            return None  # Not a MoE model
        ep_groups = load_expert_allocation(
            num_experts,
            getattr(
                vllm_config.parallel_config.eplb_config, "num_redundant_experts", 0
            ),
            layer_idx=-1,
        )
        global_num_experts = sum(len(group) for group in ep_groups)

        if hasattr(hf_config, "num_hidden_layers"):
            num_layers = hf_config.num_hidden_layers
            if (
                vllm_config.speculative_config is not None
                and vllm_config.speculative_config.method in ["deepseek_mtp", "eagle3"]
            ):
                num_layers += vllm_config.speculative_config.num_speculative_tokens
        else:
            raise ValueError("num_layers cannot be resolved from the model config.")

        stats_enabled = vllm_config.scheduler_config.enable_moe_expert_stats

        if stats_enabled:
            origin_expert_stats = torch.zeros(
                (num_layers, num_experts),
                dtype=torch.int64,
                device=vllm_config.device_config.device,
            )
            remapped_expert_stats = torch.zeros(
                (num_layers, global_num_experts),
                dtype=torch.int64,
                device=vllm_config.device_config.device,
            )
        else:
            origin_expert_stats = None
            remapped_expert_stats = None

        _EXPERT_PARALLEL_CONTEXT = ExpertParallelContext(
            n_nodes=n_nodes,
            n_gpus_per_node=n_gpus_per_node,
            n_experts=global_num_experts,
            origin_expert_stats=origin_expert_stats,
            remapped_expert_stats=remapped_expert_stats,
            stats_enabled=stats_enabled,
            stats_logging_internal=vllm_config.scheduler_config.moe_expert_stats_logging_interval,
        )

    return _EXPERT_PARALLEL_CONTEXT
