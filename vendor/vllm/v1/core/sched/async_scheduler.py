# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_async = True

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            if (
                request.num_computed_tokens
                == request.num_tokens
                + request.num_output_placeholders
                + cur_num_spec_tokens
            ):
                # The request will generate a new token plus num_spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add placeholders for the new tokens in spec_token_ids.
                # We will update the actual spec token ids in the worker process.
                request.spec_token_ids = [-1] * self.num_spec_tokens

        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens
        )

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )
        # Replace the pre-allocated placeholder token with the actual token.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped

    def update_scheduled_draft_token_ids(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[str, list[int]] | None:
        ''' Update scheduled draft token ids for structured output requests.
        This func return a new dict instead of modifying in-place, which allows
        ori scheduled_spec_decode_tokens trigger update_from_output properly.
        '''
        # If requests have no draft_token_ids (e.g., input_fits_in_drafter=False),
        # GPUModelRunner has already removed its scheduled spec decode tokens.
        # The scheduler still expects them to exist, so we don't need pass them
        # into updated_scheduled_draft_token_ids
        if model_runner_output.draft_token_ids is None:
            return {}

        max_req_index = len(model_runner_output.draft_token_ids)
        updated_scheduled_draft_token_ids: dict[str, list[int]] = {}
        req_ids = list(scheduler_output.scheduled_spec_decode_tokens.keys())
        for req_id in req_ids:
            need_struct = (
                req := self.requests.get(req_id)
            ) and req.use_structured_output
            if not need_struct:
                continue

            ori_spec_tokens = scheduler_output.scheduled_spec_decode_tokens[req_id]
            ori_spec_decode_len = len(ori_spec_tokens)
            req_index = model_runner_output.req_id_to_index.get(req_id)
            # If a request isn't in the pre-step batch (e.g., skipped due to budget
            # limits), GPUModelRunner will have no draft_token_ids for it. Directly
            # use padding -1.
            if req_index is None or ori_spec_decode_len == 0:
                updated_scheduled_draft_token_ids[req_id] = [-1] * ori_spec_decode_len
                continue

            # If req_index exceeds the shape of draft_token_ids for unknown reason,
            # we defensively to reject all draft tokens.
            if req_index >= max_req_index:
                updated_scheduled_draft_token_ids[req_id] = [-1] * ori_spec_decode_len
                logger.warning(
                    f"Request {req_id} exceeds the shape of draft_token_ids for unknown"
                    "race condition, rejecting all draft tokens."
                )
                continue

            draft_tokens = model_runner_output.draft_token_ids[req_index].copy()

            # 1. Truncate first:
            #    Only tokens that the scheduler will actually execute should be kept.
            #    Extra tokens from the drafter should NOT be passed into.
            if len(draft_tokens) > ori_spec_decode_len:
                draft_tokens = draft_tokens[:ori_spec_decode_len]

            # 2. Validate tokens (grammar-based filtering).
            request = self.requests.get(req_id)
            assert request is not None
            if self.structured_output_manager.should_advance(request):
                struct_output_req = request.structured_output_request
                assert struct_output_req is not None
                assert struct_output_req.grammar is not None
                draft_tokens = struct_output_req.grammar.validate_tokens(draft_tokens)

            # 3. Pad with -1 if shorter:
            #    Padding avoids NaNs in later sampling and prevents the grammar FSM
            #    from stepping into invalid states. During sampling, -1 will be mapped
            #    to vocab_size, allowing fast rejection.
            cur_len = len(draft_tokens)
            if cur_len < ori_spec_decode_len:
                draft_tokens.extend([-1] * (ori_spec_decode_len - cur_len))

            updated_scheduled_draft_token_ids[req_id] = draft_tokens

        return updated_scheduled_draft_token_ids
