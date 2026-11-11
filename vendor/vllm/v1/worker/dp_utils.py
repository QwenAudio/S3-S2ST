# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy as np
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_ppu
from vllm.v1.core.sched.output import ScheduledDPMetaData
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_second_ubatch_empty,
)

logger = init_logger(__name__)


# We found that on PPU, if the device group not been called before
# initializing the deepep buffers, the gather() call inside the
# Buffer.__init__() will hang.
#
# As a workaround, we ensure the device group is called before deepep
# initialization. It must happen as we will have a profile run.
_get_device_and_group_call_once = is_ppu()


def _get_device_and_group(
    parallel_config: ParallelConfig,
    group: dist.ProcessGroup | None = None,
) -> tuple[str, dist.ProcessGroup]:
    if group is not None:
        assert parallel_config.disable_nccl_for_dp_synchronization
        device = "cpu"
        return device, group

    # Use the actual device assigned to the DP group, not just the device type
    device = get_dp_group().device
    group = get_dp_group().device_group

    # Transferring this tensor from GPU to CPU will introduce a GPU sync
    # point that could adversely affect performance of vllm with asynch
    # scheduling. This environment variable exists to quickly disable
    # this optimization if we run into this case.

    global _get_device_and_group_call_once

    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once(
            "Using CPU all reduce to synchronize DP padding between ranks."
        )
    if (not _get_device_and_group_call_once
        and parallel_config.disable_nccl_for_dp_synchronization
    ):
        device = "cpu"
        group = get_dp_group().cpu_group
    _get_device_and_group_call_once = False
    return device, group


def run_deepep_allgather(
    should_ubatch: bool,
    should_dp_pad: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    num_reqs: int,
    is_prompt_batch_local: bool,
    is_prefill_only_local: bool,
    input_fits_in_drafter: bool,
    parallel_config: ParallelConfig,
    device: str | None = None,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.expert_parallel import (
        get_deepep_rank,
        try_deepep_allgather,
    )

    dp_size = parallel_config.data_parallel_size
    ep_size = parallel_config.expert_parallel_size
    ep_rank = get_deepep_rank()
    tensor = torch.zeros(ep_size, 9, device=device, dtype=torch.int32)
    tensor[ep_rank] = torch.tensor(
        [
            orig_num_tokens_per_ubatch,
            padded_num_tokens_per_ubatch,
            1 if should_ubatch else 0,
            1 if should_dp_pad else 0,
            cudagraph_mode,
            num_reqs,
            1 if is_prompt_batch_local else 0,
            1 if is_prefill_only_local else 0,
            1 if input_fits_in_drafter else 0,
        ],
        dtype=torch.int32,
        device=device
    )
    torch.cuda.synchronize()
    try_deepep_allgather(tensor, use_default_stream=True)
    # torch.cuda.synchronize()
    ep_stride_per_dp_rank = ep_size // dp_size  # type: ignore[operator]
    ep_indices_per_dp_rank = torch.arange(
        0, ep_size, ep_stride_per_dp_rank, device=device, dtype=torch.long)
    gathered_tensor = tensor[ep_indices_per_dp_rank].transpose(0, 1)
    return gathered_tensor


def _run_ar(
    should_ubatch: bool,
    should_dp_pad: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    num_reqs: int,
    is_prompt_batch_local: bool,
    is_prefill_only_local: bool,
    input_fits_in_drafter: bool,
    parallel_config: ParallelConfig,
    device: str | None = None,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    tensor = torch.zeros(9, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0
    tensor[3][dp_rank] = 1 if should_dp_pad else 0
    tensor[4][dp_rank] = cudagraph_mode
    tensor[5][dp_rank] = num_reqs
    tensor[6][dp_rank] = 1 if is_prompt_batch_local else 0
    tensor[7][dp_rank] = 1 if is_prefill_only_local else 0
    tensor[8][dp_rank] = 1 if input_fits_in_drafter else 0
    dist.all_reduce(tensor, group=group)
    return tensor


def _post_process_ubatch(tensor: torch.Tensor) -> bool:
    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    # First determine if we are going to be ubatching.
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    if not should_ubatch:
        return False
    # If the DP ranks are planning to ubatch, make sure that
    # there are no "empty" second ubatches
    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_second_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens):
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        # If DP padding is enabled, ensure that each rank is processing the same number
        # of tokens
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[4, :].min().item())


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    num_reqs: int,
    should_attempt_ubatching: bool,
    should_attempt_dp_padding: bool,
    cudagraph_mode: int,
    is_prompt_batch_local: bool,
    is_prefill_only_local: bool,
    input_fits_in_drafter: bool,
    parallel_config: ParallelConfig,
    group: dist.ProcessGroup | None = None,
) -> tuple[
    bool,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.
    When running microbatched or if should_attempt_dp_padding is True, all
    ranks will be padded out so that the run with the same number of tokens

    3. Synchronizes cudagraph_mode across ranks by taking the minimum.

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including any DP padding.
        num_tokens_per_rank: A tensor containing the total number of
        tokens per-microbatch for each DP rank without any DP padding.
        num_reqs_per_rank: A tensor containing the number of requests
        per-microbatch for each DP rank.
        is_prompt_batch_per_rank: A tensor indicating if each DP rank
        has a prompt batch or not.
        is_prefill_only_per_rank: A tensor indicating if each DP rank
        has a prefill-only batch or not.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    from vllm.model_executor.layers.fused_moe.expert_parallel import (
        is_deepep_buffer_initialized,
    )

    assert num_tokens_padded >= num_tokens_unpadded, (
        f"Padded number of tokens ({num_tokens_padded}) must be "
        f">= unpadded number of tokens ({num_tokens_unpadded})."
    )

    device, group = _get_device_and_group(parallel_config, group)

    # Coordinate between the DP ranks via an All Reduce
    # to determine the total number of tokens that each rank
    # will run and if we are using ubatching or not.
    kwargs = dict(
        should_ubatch=should_attempt_ubatching,
        should_dp_pad=should_attempt_dp_padding,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        num_reqs=num_reqs,
        is_prompt_batch_local=is_prompt_batch_local,
        is_prefill_only_local=is_prefill_only_local,
        input_fits_in_drafter=input_fits_in_drafter,
        parallel_config=parallel_config,
        device=device,
        group=group,
    )
    if envs.VLLM_ENABLE_ELASTIC and is_deepep_buffer_initialized():
        tensor = run_deepep_allgather(**kwargs)
    else:
        tensor = _run_ar(**kwargs)

    should_dp_pad = bool(torch.all(tensor[3] == 1).item())

    # DP ranks should all have the same value for should_attempt_dp_padding.
    assert should_attempt_dp_padding == should_dp_pad, (
        "DP padding setting mismatch between ranks: "
        f"{should_attempt_dp_padding} != {should_dp_pad}, "
        f"should_dp_pad across ranks are: {tensor[3]}."
    )

    # Check conditions for microbatching
    should_ubatch = _post_process_ubatch(tensor)

    if should_ubatch and not should_dp_pad:
        logger.debug_once(
            "Microbatching has been triggered and requires DP padding. "
            "Enabling DP padding even though it has been explicitly "
            "disabled.",
            scope="global",
        )
        should_dp_pad = True

    # Pad all DP ranks up to the maximum token count across ranks if
    # should_dp_pad is True
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )

    num_tokens_per_rank = tensor[0, :].cpu()
    num_reqs_per_rank = tensor[5, :].cpu()
    is_prompt_batch_per_rank = tensor[6, :].cpu()
    is_prefill_only_per_rank = tensor[7, :].cpu()
    input_fits_in_drafter_per_rank = tensor[8, :].cpu()

    # Synchronize cudagraph_mode across ranks (take min)
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    return (
        should_ubatch,
        num_tokens_after_padding,
        num_tokens_per_rank,
        num_reqs_per_rank,
        synced_cudagraph_mode,
        is_prompt_batch_per_rank,
        is_prefill_only_per_rank,
        input_fits_in_drafter_per_rank,
    )


@record_function_or_nullcontext("coordinate_batch_across_dp")
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    num_reqs: int,
    allow_microbatching: bool,
    allow_dp_padding: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    cudagraph_mode: int = 0,
    is_prompt_batch_local: bool = False,
    is_prefill_only_local: bool = False,
    uniform_decode: bool | None = None,
    num_scheduled_tokens_per_request: np.ndarray | None = None,
    input_fits_in_drafter: bool = False,
    scheduled_dp_metadata: ScheduledDPMetaData | None = None,
    group: dist.ProcessGroup | None = None,
) -> ScheduledDPMetaData:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding
        allow_microbatching: If microbatching should be attempted
        allow_dp_padding: If all DP ranks should be padded up to the same value
        parallel_config: The parallel config
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch
            only contains single token decodes
        num_scheduled_tokens_per_request: Only used if allow_microbatching is True. The
            number of tokens per request.
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL)

    Returns: SchedulerDPMetadata, includes:
        - num_pad_tokens
        - num_tokens_across_dp
        - num_reqs_per_rank
        - num_tokens_per_rank
        - synced_cudagraph_mode
        - is_prompt_batch
        - all_prefill
        - all_decode
        - should_ubatch
    """

    if scheduled_dp_metadata is not None:
        # Already provided, by asynchronous scheduling.
        return scheduled_dp_metadata

    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return ScheduledDPMetaData(
            num_pad_tokens=0,
            num_tokens_across_dp=None,
            num_tokens_per_rank=None,
            num_reqs_per_rank=None,
            is_prompt_batch=is_prompt_batch_local,
            all_prefill=is_prompt_batch_local and is_prefill_only_local,
            all_decode=not is_prefill_only_local,
            should_ubatch=False,
            input_fits_in_drafter=input_fits_in_drafter,
            cudagraph_mode=cudagraph_mode,
        )

    # If the caller has explicitly enabled microbatching.
    should_attempt_ubatching = False
    if allow_microbatching:
        # Check preconditions for microbatching
        assert uniform_decode is not None
        should_attempt_ubatching = check_ubatch_thresholds(
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded

    (
        should_ubatch,
        num_tokens_after_padding,
        num_tokens_per_rank,
        num_reqs_per_rank,
        synced_cudagraph_mode,
        is_prompt_batch_per_rank,
        is_prefill_only_per_rank,
        input_fits_in_drafter_per_rank,
    ) = _synchronize_dp_ranks(
        num_tokens_unpadded=num_tokens_unpadded,
        num_tokens_padded=num_tokens_padded,  # type: ignore[arg-type]
        num_reqs=num_reqs,
        should_attempt_ubatching=should_attempt_ubatching,
        should_attempt_dp_padding=allow_dp_padding,
        cudagraph_mode=cudagraph_mode,
        is_prompt_batch_local=is_prompt_batch_local,
        is_prefill_only_local=is_prefill_only_local,
        input_fits_in_drafter=input_fits_in_drafter,
        parallel_config=parallel_config,
        group=group,
    )

    is_prompt_batch = torch.sum(is_prompt_batch_per_rank).item() > 0
    all_prefill = (
        torch.sum(is_prompt_batch_per_rank).item()
        == parallel_config.data_parallel_size
        and torch.sum(is_prefill_only_per_rank).item()
        == parallel_config.data_parallel_size
    )
    all_decode = torch.sum(is_prompt_batch_per_rank).item() == 0
    input_fits_in_drafter = (
        torch.sum(input_fits_in_drafter_per_rank).item()
        == parallel_config.data_parallel_size
    )

    return ScheduledDPMetaData(
        num_pad_tokens=int(num_tokens_after_padding[0].item()) - num_tokens_unpadded,
        num_tokens_across_dp=num_tokens_after_padding,
        num_reqs_per_rank=num_reqs_per_rank,
        num_tokens_per_rank=num_tokens_per_rank,
        is_prompt_batch=is_prompt_batch,
        all_prefill=all_prefill,
        all_decode=all_decode,
        should_ubatch=should_ubatch,
        input_fits_in_drafter=input_fits_in_drafter,
        cudagraph_mode=synced_cudagraph_mode,
    )
