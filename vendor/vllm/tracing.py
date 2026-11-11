# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextvars
import functools
import inspect
import json
import os
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import nvtx

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate"]

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: str | None = None
try:
    from opentelemetry import propagate, trace
    from opentelemetry.context.context import Context
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.trace import (
        ReadableSpan,
        Span,
        SpanProcessor,
        TracerProvider,
    )
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.sdk.trace.id_generator import IdGenerator
    from opentelemetry.trace import (
        SpanKind,
        Tracer,
        set_span_in_context,
        set_tracer_provider,
    )
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback

    otel_import_error_traceback = traceback.format_exc()
    logger.info(otel_import_error_traceback)

_is_stria_available = False
try:
    from stria.tracing import NanoTracer, extract_parent_context, init_nano_backend

    _is_stria_available = True
except ImportError:
    logger.info("stria is not available.")


def get_env(param: str, value):
    value = type(value)(os.getenv(param.lower(), value))
    return type(value)(os.getenv(param, value))


def is_otel_available() -> bool:
    return _is_otel_imported


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")


# ==============================
# Tracing Levels
# ==============================
class TracingLevel(Enum):
    """
    The level of tracing.
    """

    NOOTEL = 0
    ENGINE = 1
    WORKER = 2



_vllm_global_tracing_level = 0
_otel_log_dir = None
if _is_stria_available:
    config_str = os.environ.get("OTEL_CONFIG")
    try:
        otel_config = json.loads(config_str)
        if not isinstance(otel_config, dict):
            raise TypeError("OTEL_CONFIG must be a JSON dictionary.")

        # Set OTel tracing level
        otel_tracing_level = str(otel_config.get("tracing_level", "WORKER"))
        if otel_tracing_level not in list(TracingLevel.__members__.keys()):
            otel_tracing_level = "NOOTEL"
        _vllm_global_tracing_level = TracingLevel[otel_tracing_level].value

        # Set OTel log dump directory
        _otel_log_dir = otel_config.get("dump_dir", "/path/to/otel_log")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse OTEL_CONFIG JSON. Disabling OTEL. Error: {e}")


logger.info(
    "vLLM tracing level: %s, dump to: %s",
    TracingLevel(_vllm_global_tracing_level),
    _otel_log_dir,
)

# ===================================
# Trace provider & Tracer
# ===================================
_nano_tracer = None
_is_tracer_provider_initialized = False
def init_tracer_provider():
    global _is_tracer_provider_initialized, _nano_tracer
    if _is_tracer_provider_initialized:
        return
    _is_tracer_provider_initialized = True

    if _is_stria_available:
        logger.info("Initializing Nano-OTel High Performance Backend...")
        try:
            init_nano_backend(
                log_dir=_otel_log_dir,
                max_bytes=5*1024*1024,
                backup_count=50,
                batch_size=256,
                flush_interval_ms=1000
            )
            _nano_tracer = NanoTracer("vllm")
        except Exception as e:
            logger.error(f"Failed to init NanoTracer: {e}. Tracing Disabled.")
    else:
        logger.error("stria not found. Tracing Disabled.")
            

def init_tracer(
    instrumenting_module_name: Optional[str] = None, 
    end_point: Optional[str] = None) -> Tracer:
    if _vllm_global_tracing_level > TracingLevel.NOOTEL.value:
        init_tracer_provider()
        return _nano_tracer
    else:
        return None


# ==============================
# Per-request tracing
# ==============================
class TraceWrapper:
    def __init__(self, trace_headers: Optional[Union[Dict[str, str], str]] = None):
        if _vllm_global_tracing_level > TracingLevel.NOOTEL.value:
            self.tracer = init_tracer(__name__)
        else:
            self.tracer = None
            
        if isinstance(trace_headers, str):
            self.trace_headers = {"traceparent": trace_headers}
        else:
            self.trace_headers = trace_headers
        
        self.active_spans: Dict[str, any] = {}

    # =====================
    # Helper for Otel span
    # =====================
    def _start_span(
        self,
        span_name: str,
        parent_span: Union[str, Tuple[int, int], None] = None
    ):
        if self.tracer is None:
            return
        
        # Retrieve context from active_spans if parent_span_name is provided
        parent_context = None
        if parent_span is not None:
            if isinstance(parent_span, str):
                if parent_span in self.active_spans:
                    # Prioritize context derived from the active span name
                    parent_context = self.active_spans[parent_span].context()
            else:
                parent_context = parent_span
        
        span = self.tracer.start_span(span_name, parent_context)
        self.active_spans[span_name] = span

    def _end_span(self, span_name: str, extra_attrs: Optional[Mapping] = None):
        if span_name in self.active_spans:
            span = self.active_spans.pop(span_name)
            if extra_attrs is not None:
                span.set_attributes(extra_attrs)
            span.end()
    
    
    # =========================
    # Helpers for Engine Core
    # ========================
    # Extract traceparent from span
    def traceparent(self, span_name: str):
        if self.tracer is None:
            return None
        
        if span_name in self.active_spans:
            span = self.active_spans[span_name]
            return span.traceparent()
        else:
            return None

    # Full-life span of vLLM Request
    def on_req_start(self):
        if self.tracer is None:
            return

        self._start_span(
            span_name="llm_request"
        )

    def on_req_finish(self, request_id):
        if self.tracer is None:
            return
        
        self._end_span(
            span_name="llm_request", 
            extra_attrs={"request_id": request_id}
        )

    # Span of KV cache load/store
    def on_req_kv_start(self, op: str):
        if self.tracer is None or op not in ["load", "save"]:
            return

        self._start_span(
            span_name=f"kv_{op}",
            parent_span="llm_request"
        )

    def on_req_kv_end(self, op: str):
        if self.tracer is None or op not in ["load", "save"]:
            return

        self._end_span(span_name=f"kv_{op}")


# ========================================
# Per-batch tracing
# ========================================
_CURRENT_BSMGR: contextvars.ContextVar["BatchedSpanManager"] = contextvars.ContextVar(
    "_CURRENT_BSMGR"
)


class CurrentSpan:
    def __init__(self):
        self.bs_mgr = _CURRENT_BSMGR.get(None)

    def start_timer(self, name):
        if self.bs_mgr is not None:
            self.bs_mgr.start_timer(name)

    def end_timer(self):
        if self.bs_mgr is not None:
            self.bs_mgr.end_timer()

    def set_attr(self, key, value):
        if self.bs_mgr is not None:
            self.bs_mgr.set_attr(key, value)

    def collect_batch_metadata(self, reqs_iter, num_scheduled_tokens):
        if self.bs_mgr is not None:
            return self.bs_mgr.collect_batch_metadata(
                reqs_iter, num_scheduled_tokens)
        
    def collect_from_model_output(self, scheduler_output, model_runner_output):
        if self.bs_mgr is not None:
            return self.bs_mgr.collect_from_model_output(
                scheduler_output, model_runner_output)


class BatchedSpanManager(AbstractContextManager):
    def __init__(self, tracer, span_name: str, scheduler_output=None):
        self.batch_span = None
        self.span_attrs = dict()
        self.timer_stack: List[Tuple[int, str]] = []
        self.scheduler_output = scheduler_output
        self.trace_id_hex = None

        if (
            tracer is not None
            and _vllm_global_tracing_level > TracingLevel.ENGINE.value
        ):
            start = time.time_ns()

            # Get span context
            if scheduler_output is None:
                parent_context = None
            else:
                parent_context = extract_parent_context(scheduler_output.trace_carrier)
                
            # Create a batch span
            self.batch_span = tracer.start_span(span_name, parent_context)

            self.span_attrs["create_span"] = int((time.time_ns() - start) * 1e-3)
            self._ctx_token = _CURRENT_BSMGR.set(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.batch_span is not None:
            # Forcefully close un-ended timers in the stack
            now = time.time_ns()
            while self.timer_stack:
                start_ns, name = self.timer_stack.pop()
                duration_us = int((now - start_ns) * 1e-3)
                self.span_attrs[name] = f"{start_ns},{duration_us}"
            self.batch_span.set_attributes(self.span_attrs)
            self.batch_span.end()
            _CURRENT_BSMGR.reset(self._ctx_token)

    def end(self):
        self.__exit__(None, None, None)

    def start_timer(self, name):
        if self.batch_span is not None:
            self.timer_stack.append((time.time_ns(), name))

    def end_timer(self):
        if self.batch_span is not None:
            if not self.timer_stack:
                logger.warning(
                    "Mismatched timer calls: end_timer() called with empty stack."
                )
                return

            # Pop the most recent timer (timestamp, name) from stack.
            start_ns, name = self.timer_stack.pop()
            duration_us = int((time.time_ns() - start_ns) * 1e-3)
            self.span_attrs[name] = f"{start_ns},{duration_us}"

    def set_attr(self, key, value):
        self.span_attrs.update({key: value})

    def collect_batch_metadata(self, reqs_iter, num_scheduled_tokens):
        """
        Helper method to collect tracing metadata from a stream of requests.
        """
        if self.batch_span is None:
            return

        start_ns = time.time_ns()
        num_scheduled_tok = []
        num_computed_tok = []
        req_state = []

        # Iterate through the chain once
        for req in reqs_iter:
            num_scheduled_tok.append(str(num_scheduled_tokens[req.request_id]))
            num_computed_tok.append(str(req.num_computed_tokens))
            if req.prefill_done:
                req_state.append("1")
            else:
                req_state.append("0")

        # Set span attributes
        self.span_attrs.update(
            {
                "num_scheduled_tok": ",".join(num_scheduled_tok),
                "num_computed_tok": ",".join(num_computed_tok),
                "req_state": ",".join(req_state),
                "collect_meta": int((time.time_ns() - start_ns) * 1e-3),
            }
        )

        # Get span context for Core to Worker tracing
        return self.batch_span.traceparent()

    def collect_from_model_output(self, scheduler_output, model_runner_output):
        if self.batch_span is None or model_runner_output is None:
            return
        
        start_ns = time.time_ns()
        draft_tok_nums = []
        accept_tok_nums = []
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        
        for req_id, _ in num_scheduled_tokens.items():
            scheduled_spec_token_ids = scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                req_index = model_runner_output.req_id_to_index.get(req_id)
                generated_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )
                num_accepted = (
                    len(generated_token_ids) - 1 if generated_token_ids else 0
                )
                draft_tok_nums.append(str(num_draft_tokens))
                accept_tok_nums.append(str(num_accepted))
            else:
                draft_tok_nums.append("0")
                accept_tok_nums.append("0")
        
        self.span_attrs.update(
            {
                "draft_tok_nums": ",".join(draft_tok_nums),
                "accept_tok_nums": ",".join(accept_tok_nums),
                "collect_output": int((time.time_ns() - start_ns) * 1e-3),
            }
        )

def trace_execution(span_name: str):
    def decorator(func):
        # -----------------------------------------------------------
        # [Pre-computation] Executed once at decoration time with zero runtime overhead.
        # Locate the index of the scheduler_output parameter
        # self is handled separately in the wrapper
        # Example: for (self, a, sched_out), indices are self:0, a:1, sched_out:2.
        # In *args, sched_out corresponds to index 1.
        # -----------------------------------------------------------
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        target_arg_name = "scheduler_output"
        arg_pos_index = -1

        if target_arg_name in params:
            full_index = params.index(target_arg_name)
            arg_pos_index = full_index - 1

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            tracer = getattr(self, "tracer", None)
            if tracer is None:
                return func(self, *args, **kwargs)

            sched_out = None

            if target_arg_name in kwargs:
                # Check kwargs first (O(1))
                sched_out = kwargs[target_arg_name]
            elif arg_pos_index >= 0 and len(args) > arg_pos_index:
                # Check args using pre-computed index (O(1)).
                sched_out = args[arg_pos_index]

            with BatchedSpanManager(tracer, span_name, sched_out):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator