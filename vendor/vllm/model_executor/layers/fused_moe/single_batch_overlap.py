import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config


@dataclass
class CombineOverlapArgs:
    overlap: bool = False
    stream: Optional[torch.cuda.Stream] = None
    recv_stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    num_sms: int = 8
    num_next_clean_int: int = -1
    signal: Optional[torch.Tensor] = None
    reduce_hidden: int = 4096
    device_sms: int = -1


_combine_overlap = None


def get_reduce_hidden() -> int:
    return _combine_overlap.reduce_hidden if _combine_overlap is not None else 0


def get_comb_overlap_args() -> CombineOverlapArgs:
    assert _combine_overlap is not None
    return _combine_overlap


def init_and_get_overlap(
    local_num_experts: int,
    global_num_experts: int,
    hidden_size: int,
    device: torch.device,
) -> bool:
    global _combine_overlap

    vllm_config = get_current_vllm_config()
    overlap = False if vllm_config.combine_overlap_config is None \
        else vllm_config.combine_overlap_config.overlap
    ceil_div = lambda x, y: (x + y - 1) // y

    if overlap and _combine_overlap is not None:
        # combine overlap does not support dynamic hidden_size yet
        assert _combine_overlap.num_next_clean_int == global_num_experts * ceil_div(
            hidden_size, _combine_overlap.reduce_hidden) * 4

    if _combine_overlap is None and overlap:
        reduce_hidden = vllm_config.combine_overlap_config.reduce_hidden
        assert reduce_hidden > 0
        # signal_pad_size is big enough to cover dpsk and qwen cases
        signal_pad_size = ceil_div(
            local_num_experts * ceil_div(hidden_size, reduce_hidden) * 4 + 16,
            128) * 128
        _combine_overlap = CombineOverlapArgs(
            overlap=True,
            stream=torch.cuda.Stream(),
            recv_stream=torch.cuda.Stream(),
            wait_event=torch.cuda.Event(),
            num_sms=vllm_config.combine_overlap_config.combine_sm_num,
            num_next_clean_int=global_num_experts *
            ceil_div(hidden_size, reduce_hidden) * 4,
            signal=torch.zeros(signal_pad_size,
                               dtype=torch.int32,
                               device=device),
            reduce_hidden=reduce_hidden,
        )

        import deep_ep
        sig = inspect.signature(deep_ep.Buffer.low_latency_dispatch)
        assert 'num_next_clean_int' in sig.parameters, "DeepEP does not support overlap"
        sig = inspect.signature(deep_ep.Buffer.low_latency_combine)
        assert 'num_sms' in sig.parameters, "DeepEP does not support overlap"
        assert 'reduce_hidden' in sig.parameters, "DeepEP does not support overlap"
        assert 'moe_signal_pad' in sig.parameters, "DeepEP does not support overlap"
        assert 'num_next_clean_int' in sig.parameters, "DeepEP does not support overlap"
        assert 'recv_stream' in sig.parameters, "DeepEP does not support overlap"
        assert 'recv_count' in sig.parameters, "DeepEP does not support overlap"
        import deep_gemm
        _combine_overlap.device_sms = deep_gemm.get_num_sms()

    return overlap


def combine_overlap_sync(overlap=False):
    if not overlap:
        return
    import deep_gemm
    original_num_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(original_num_sms - _combine_overlap.num_sms)
    _combine_overlap.signal.zero_()
    _combine_overlap.wait_event.record(torch.cuda.current_stream())
    _combine_overlap.stream.wait_event(_combine_overlap.wait_event)


@contextmanager
def combine_overlap_consumer(overlap=False):
    if not overlap:
        yield
        return
    _combine_overlap.recv_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_combine_overlap.stream):
        yield


def wait_combine_finish(combine_recv_hook):
    combine_recv_hook()
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(_combine_overlap.stream)
    current_stream.wait_stream(_combine_overlap.recv_stream)
    import deep_gemm
    updated_num_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(updated_num_sms + _combine_overlap.num_sms)
