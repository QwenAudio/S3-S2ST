# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import torch

from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def check_stop(
    request: Request,
    max_model_len: int,
    vocab_size: int = 0,
) -> bool:
    assert not request.pooling_params

    if vocab_size and request.output_token_ids[-1] >= vocab_size:
        logger.warning(
            "Request %s aborted due to invalid token id, request: %r",
            request.request_id,
            request,
        )
        request.status = RequestStatus.FINISHED_ABORTED
        request.stop_reason = vocab_size  # type: ignore[assignment]
        return True

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id: int | list[int] = request.output_token_ids[-1]
    extra_args = sampling_params.extra_args or {}
    repeated_token_stop_len = int(extra_args.get("stop_on_repeated_token_len") or 0)
    target_repeated_token_stop_len = int(
        extra_args.get("stop_on_target_repeated_token_len") or 0
    )
    target_repeated_token_ids = {
        int(token_id)
        for token_id in (extra_args.get("stop_on_repeated_token_ids") or [])
    }
    repeated_ngram_stop_len = int(extra_args.get("stop_on_repeated_ngram_len") or 0)
    repeated_ngram_min = int(extra_args.get("stop_on_repeated_ngram_min") or 2)
    repeated_ngram_max = int(extra_args.get("stop_on_repeated_ngram_max") or 8)
    if (
        target_repeated_token_stop_len > 0
        and isinstance(last_token_id, int)
        and last_token_id in target_repeated_token_ids
        and request.num_output_tokens >= target_repeated_token_stop_len
    ):
        repeated = 1
        output_token_ids = request.output_token_ids
        for token_id in reversed(output_token_ids[:-1]):
            if token_id != last_token_id:
                break
            repeated += 1
            if repeated >= target_repeated_token_stop_len:
                request.status = RequestStatus.FINISHED_STOPPED
                request.stop_reason = (
                    f"target_repeated_token:{last_token_id}:{repeated}"
                )
                return True

    if (
        repeated_token_stop_len > 0
        and request.num_output_tokens >= repeated_token_stop_len
    ):
        repeated = 1
        output_token_ids = request.output_token_ids
        for token_id in reversed(output_token_ids[:-1]):
            if token_id != last_token_id:
                break
            repeated += 1
            if repeated >= repeated_token_stop_len:
                request.status = RequestStatus.FINISHED_STOPPED
                request.stop_reason = (
                    f"repeated_token:{last_token_id}:{repeated}"
                )
                return True

    if repeated_ngram_stop_len > 0 and request.num_output_tokens >= repeated_ngram_stop_len:
        output_token_ids = request.output_token_ids
        min_ngram = max(1, repeated_ngram_min)
        max_ngram = max(min_ngram, repeated_ngram_max)
        max_ngram = min(max_ngram, repeated_ngram_stop_len // 2)
        for ngram_size in range(min_ngram, max_ngram + 1):
            tail = output_token_ids[-ngram_size:]
            repeated_tokens = ngram_size
            pos = len(output_token_ids) - ngram_size
            while pos - ngram_size >= 0:
                pos -= ngram_size
                if output_token_ids[pos:pos + ngram_size] != tail:
                    break
                repeated_tokens += ngram_size
                if repeated_tokens >= repeated_ngram_stop_len:
                    request.status = RequestStatus.FINISHED_STOPPED
                    request.stop_reason = (
                        f"repeated_ngram:{ngram_size}:{repeated_tokens}"
                    )
                    return True

    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    for stop_token_id in sampling_params.stop_token_ids or ():
        if isinstance(stop_token_id, int):
            last_token_id = request.output_token_ids[-1]
        else:
            last_token_id = list(request.output_token_ids[-len(stop_token_id) :])
        if last_token_id == stop_token_id:
            request.status = RequestStatus.FINISHED_STOPPED
            if isinstance(stop_token_id, int):
                request.stop_reason = last_token_id  # type: ignore[assignment]
            else:
                request.stop_reason = last_token_id[-1]  # type: ignore[assignment,index]
            return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
