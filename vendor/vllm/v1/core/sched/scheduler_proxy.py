# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Optional, Tuple

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class SchedulerProxyInterface(ABC):
    """Base class: secondary scheduler interface for preprocessing and
    prioritizing requests before they enter the main scheduler,
    designed for batched inference scenarios."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a new request."""
        raise NotImplementedError

    @abstractmethod
    def get_requests_num(self) -> int:
        """Return the number of currently unprocessed requests."""
        raise NotImplementedError

    @abstractmethod
    def has_requests(self) -> bool:
        """Check whether there are any requests remaining."""
        raise NotImplementedError

    @abstractmethod
    def pop_request(self) -> Optional[Request]:
        """Pop a request (may return None if no request exists)."""
        raise NotImplementedError

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the proxy scheduler."""
        raise NotImplementedError


class BatchSchedulerProxy(SchedulerProxyInterface):
    """
    A secondary scheduler that prioritizes and organizes incoming requests
    before they are submitted to the primary execution engine. Designed for
    batched inference scenarios, this class implements a hybrid scheduling
    strategy combining lexicographical ordering and deadline-based prioritization.

    This scheduler ensures:
    - Improved Prefix cache hit rate: Requests with similar prompt prefixes
      are scheduled consecutively, maximizing cache hit rates and reducing
      memory overhead.
    - Timely processing guarantee: Ensures the requests are not "starved" by
      using a deadline-based queue to prioritize their execution.

    Internally, it maintains three heaps:
    - [_prefix_min_heap]: min-heap for lexicographically increasing prompt token IDs.
    - [_prefix_max_heap]: max-heap (via negated token IDs) for decreasing order.
    - [_deadline_heap]: min-heap based on request deadlines for timeout protection.

    The dual heap mechanism allows alternating between ascending and descending
    lexicographical order, improving cache efficiency.
    """

    def __init__(self, timeout: float = 2700.0):
        """
        Initialize the BatchSchedulerProxy.

        :param timeout: Forced scheduling deadline in seconds.
            All requests must be scheduled before this time, or they will be
            prioritized and forcibly popped to ensure timely processing.
        """
        self._prefix_max_heap: list[tuple[tuple[int, ...], str, Request]] = []
        self._prefix_min_heap: list[tuple[tuple[int, ...], str, Request]] = []
        self._deadline_heap: list[tuple[float, str, Request]] = []
        self._unprocessed_requests: dict[str, Request] = {}

        self._last_popped_token_ids: Optional[tuple[int, ...]] = None
        self._pop_from_min_heap: bool = True
        self.timeout = timeout

        logger.info(
            "BatchSchedulerProxy initialized with timeout=%.1f seconds", self.timeout
        )

    def add_request(self, request: Request) -> None:
        """Add a new request into the proxy."""
        assert request.prompt_token_ids is not None
        token_ids = tuple(request.prompt_token_ids)
        if (
            self._last_popped_token_ids is None
            or token_ids >= self._last_popped_token_ids
        ):
            heapq.heappush(
                self._prefix_min_heap, (token_ids, request.request_id, request)
            )
        else:
            # Max heap via negated token IDs
            heapq.heappush(
                self._prefix_max_heap,
                (tuple(-x for x in token_ids), request.request_id, request),
            )

        heapq.heappush(
            self._deadline_heap,
            (request.arrival_time + self.timeout, request.request_id, request),
        )
        self._unprocessed_requests[request.request_id] = request

    def get_requests_num(self) -> int:
        """Get the number of currently unprocessed requests."""
        return len(self._unprocessed_requests)

    def has_requests(self) -> bool:
        """Check if there are unprocessed requests."""
        return self.get_requests_num() > 0

    def pop_request(self) -> Optional[Request]:
        """
        Pop the next request to be processed.
        Prioritize requests that are about to time out; otherwise, process them in
        lexicographical order.
        """
        if not self.has_requests():
            return None

        while self._deadline_heap:
            deadline, req_id, request = self._deadline_heap[0]
            if deadline > time.time():
                break

            heapq.heappop(self._deadline_heap)
            if req_id not in self._unprocessed_requests:
                continue
            del self._unprocessed_requests[req_id]
            return request

        if self._pop_from_min_heap:
            while self._prefix_min_heap:
                _, req_id, request = heapq.heappop(self._prefix_min_heap)
                if req_id not in self._unprocessed_requests:
                    continue

                del self._unprocessed_requests[req_id]
                assert request.prompt_token_ids is not None
                self._last_popped_token_ids = tuple(request.prompt_token_ids)
                return request
            self._pop_from_min_heap = False

        if not self._pop_from_min_heap:
            while self._prefix_max_heap:
                _, req_id, request = heapq.heappop(self._prefix_max_heap)
                if req_id not in self._unprocessed_requests:
                    continue

                del self._unprocessed_requests[req_id]
                assert request.prompt_token_ids is not None
                self._last_popped_token_ids = tuple(request.prompt_token_ids)
                return request
            self._pop_from_min_heap = True

        return None

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """
        Remove multiple specific requests from the proxy scheduler.

        This only removes the requests from _unprocessed_requests.
        The heaps are not modified, and stale entries will be skipped during
        pop_request() (lazy removal).
        """
        for req in requests:
            req_id = req.request_id
            if req_id in self._unprocessed_requests:
                del self._unprocessed_requests[req_id]

    @property
    def unprocessed_requests(self) -> dict[str, Request]:
        """
        Provide read-only access to the current dictionary of unprocessed requests.
        Returns a copy of the internal dictionary to prevent external modification
        of internal state.
        """
        return dict(self._unprocessed_requests)
