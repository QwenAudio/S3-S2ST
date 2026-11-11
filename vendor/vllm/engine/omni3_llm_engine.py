#!/usr/bin/env python3

import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import glob
import itertools
import multiprocessing
import os
import pickle
import queue
import signal
import subprocess
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import psutil
import torch
import zmq
import zmq.asyncio
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerResizeMLP,
)
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

import vllm.envs as envs
from vllm.config import LoadConfig, ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.envs import (
    VLLM_ENFORCE_GENERATE_CONFIG,
    VLLM_OMNI_CODE2WAV_CHUNK_SIZE,
    VLLM_OMNI_CODE2WAV_MAX_BATCH_SIZE,
    VLLM_OMNI_PREPROCESS_ON_GPU,
    VLLM_OMNI_T2C_USE_ZMQ,
    VLLM_RPC_TIMEOUT,
)
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.qwen3_omni_codec_decoder import (
    load_qwen3_omni_dac_from_config,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.config import get_config, try_get_generation_config
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils.cache import LRUCache
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.engine import ExternalData
from vllm.v1.engine.async_llm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine
from vllm.v1.engine.output_processor import RequestOutputCollector

POLLING_TIMEOUT_MS = 10000
VLLM_RPC_SUCCESS_STR = "SUCCESS"

IPC_INPUT_EXT = "_input_socket"
IPC_OUTPUT_EXT = "_output_socket"
IPC_HEALTH_EXT = "_health_socket"
IPC_DATA_EXT = "_data_socket"
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR),)

logger = init_logger(__name__)

@dataclass
class RPCStartupResponse:
    tracing_enabled: bool

class RPCStartupRequest(Enum):
    IS_SERVER_READY = 1

@contextlib.contextmanager
def envvars(
    key: Union[str, dict[str, str]],
    value: Union[str, None] = None,
) -> Generator[os._Environ, Any, Any]:
    if isinstance(key, str):
        if value is None:
            items = dict()
        else:
            items: dict[str, str] = {key: value}
    else:
        items: dict[str, str] = key
    original_items = dict()
    for k, v in items.items():
        original_items[k] = os.environ.get(k, None)
        os.environ[k] = v

    yield os.environ

    for k, v in original_items.items():
        if v is not None:
            os.environ[k] = v
        else:
            del os.environ[k]


class SynchronizedGenerator(Generator[RequestOutput, None, None]):

    def __init__(
        self,
        # generator: AsyncGenerator[RequestOutput, None],
        collector: RequestOutputCollector | AsyncGenerator[RequestOutput, None],
        loop: asyncio.AbstractEventLoop,
    ):
        # self._generator = generator
        self._collector = collector
        self._loop = loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if isinstance(self._collector, AsyncGenerator):
                return asyncio.run_coroutine_threadsafe(
                    self._collector.__anext__(),
                    self._loop,
                ).result()
            elif isinstance(self._collector, RequestOutputCollector):
                # vllm v1 use RequestOutputCollector
                out = asyncio.run_coroutine_threadsafe(
                    self._collector.get(),
                    self._loop,
                ).result()
                # vllm v1 use finished to stop iteration
                if out.finished:
                    raise StopIteration
                return out
            else:
                raise NotImplementedError(
                    "Unknown collector type: {}".format(type(self._collector))
                )
        except StopAsyncIteration as e:
            raise StopIteration from e
        except concurrent.futures._base.CancelledError as e:
            raise StopIteration from e

    def send(self, value):
        return self.__next__()

    def throw(self, type, value=None, traceback=None):
        raise StopIteration


# create a new class for code2wav engine request
class Code2WavChunk:

    def __init__(
        self,
        request_id: str,
        valid_chunk: int,
        code_list: List[torch.Tensor],
        total_codec_num: int,
        flushed: bool,
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.valid_chunk = valid_chunk
        self.code_list = code_list
        self.total_codec_num = total_codec_num
        self.flushed = flushed
        self.finished = finished

    def __str__(self) -> str:
        return (
            f"request_id: {self.request_id}, "
            f"code_list: {len(self.code_list)}, "
            f"total_codec_num: {self.total_codec_num}, "
            f"valid_chunk: {self.valid_chunk}, "
            f"flushed: {self.flushed}, "
            f"finished: {self.finished}"
        )

    def __repr__(self) -> str:
        return (
            f"request_id: {self.request_id}, "
            f"code_list: {len(self.code_list)}, "
            f"total_codec_num: {self.total_codec_num}, "
            f"valid_chunk: {self.valid_chunk}, "
            f"flushed: {self.flushed}, "
            f"finished: {self.finished}"
        )


class ChunkScheduler:

    def __init__(self, max_batch_size=1):
        logger.info("[ChunkScheduler]: init with max_batch_size: %s", max_batch_size)
        self.chunks: list[Code2WavChunk] = []
        self.max_batch_size = max_batch_size
        self.lock = threading.Lock()

    def add_chunk(self, chunk: Code2WavChunk):
        with self.lock:
            self.chunks.append(chunk)

    def schedule(self) -> list[Code2WavChunk]:
        """Select the list of chunks to be executed based on priority and
        dependency relationships.
        """
        with self.lock:
            if len(self.chunks) <= 0:
                return []
            logger.debug(
                "[ChunkScheduler] schedule: queue=%s rids=%s",
                len(self.chunks),
                [chunk.request_id for chunk in self.chunks],
            )
            selected_chunks = []
            selected_requests = []
            for chunk in self.chunks:
                # select one chunk for each request
                if chunk.request_id not in selected_requests:
                    selected_chunks.append(chunk)
                    selected_requests.append(chunk.request_id)
                    if len(selected_chunks) >= self.max_batch_size:
                        break
            for chunk in selected_chunks:
                # remove the chunk selected from scheduler
                self.chunks.remove(chunk)
            return selected_chunks

    def clear_chunk(self, request_id):
        with self.lock:
            new_list = []
            for chunk in self.chunks:
                if chunk.request_id != request_id:
                    new_list.append(chunk)
            self.chunks = new_list


class _MQOmniCode2WavEngine:

    def __init__(
        self,
        ipc_path: str,
        model_path: str,
        enable_torch_compile: bool,
        N_code_size: int = 16,
        codec_chunk_size: int = 1,
        codec_left_context: int = 25,
        codec_right_context: int = 0,
        codec_upsample_rate: int = 1920,
        device: Union[int, str] = "cuda",
        talker_to_code2wav_queue=None,
    ) -> None:
        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = torch.device(device)

        logger.info(
            "Code2WavEngine starting up on device %s, with model %s",
            self.device,
            model_path,
        )
        config_path = os.path.join(model_path, "config.yaml")
        checkpoint_path = os.path.join(model_path, "model_weights.pt")

        logger.info("loading qwen3 codec 2 wav model from %s", model_path)
        self.code2wav = load_qwen3_omni_dac_from_config(
            config_path, checkpoint_path, device
        )
        logger.info("qwen3 codec 2 wav model loaded")

        # DAC model related parameters
        self.N_code_size = N_code_size
        self.codec_chunk_size = codec_chunk_size
        self.codec_left_context = codec_left_context
        self.codec_right_context = codec_right_context
        self.codec_upsample_rate = codec_upsample_rate
        # warm up and padding config
        self.max_batch_size = VLLM_OMNI_CODE2WAV_MAX_BATCH_SIZE
        self.padding_interval = (
            self.codec_left_context + self.codec_right_context + self.codec_chunk_size
        )
        self.padding_interval = (self.padding_interval // 8 + 1) * 8

        self.code2wav_progress_error: LRUCache = LRUCache(capacity=1024)

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # ZMQ High Water Mark for high concurrency scenarios
        # Increase buffer to prevent blocking under load
        ZMQ_HWM = 1000

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.setsockopt(zmq.constants.RCVHWM, ZMQ_HWM)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        # use queue instead of zmq
        self.talker_to_code2wav_queue = talker_to_code2wav_queue

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.setsockopt(zmq.constants.SNDHWM, ZMQ_HWM)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # Send heartbeats back to client.
        self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
        self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Error state.
        self._errored_with: Optional[BaseException] = None

        self.chunk_scheduler = ChunkScheduler(self.max_batch_size)
        # warm up code2wav
        self._warmup_mode()
        self.error_lock = threading.Lock()
        self.engine_infer_loop_thread = None

    def _warmup_mode(self):
        logger.info("start warming up code2wav for 1 to %d batch", self.max_batch_size)
        max_chunk_size = (
            self.codec_left_context + self.codec_chunk_size + self.codec_right_context
        )
        for b in range(1, self.max_batch_size + 1):
            for i in range(1, max_chunk_size // self.padding_interval + 1 + 1):
                start_time = time.time() * 1000
                fake_input = torch.zeros(
                    (b, i * self.padding_interval, self.N_code_size),
                    dtype=torch.int64,
                    device=self.device,
                )
                with torch.no_grad():
                    self.code2wav.decode(fake_input).to(torch.float32)
                end_time = time.time() * 1000
                logger.info(
                    "warm up code2wav batch %d, length %d, cost: %f ms",
                    b,
                    i * self.padding_interval,
                    end_time - start_time,
                )
        logger.info("code2wav warm up done")

    @staticmethod
    def parse_key(fname: str, key: str) -> str:
        if fname == key:
            return "default"
        return fname.split("_")[0].lower()

    # @property
    # def dead_error(self) -> BaseException:
    #    if self._errored_with is not None:
    #        return ENGINE_DEAD_ERROR(self._errored_with)
    #    else:
    #        return ENGINE_DEAD_ERROR()

    def start(self):
        try:
            try:
                logger.debug("Starting Startup Loop.")
                self.run_startup_loop()
                logger.debug("Starting Engine Loop.")
                self.engine_infer_loop_thread = Thread(target=self.engine_infer_loop)
                self.engine_infer_loop_thread.start()
                self.engine_recv_loop()
                # self.run_engine_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug("Shutting down MQOmniCode2WavEngine.")
        finally:
            logger.debug("MQOmniCode2WavEngine is shut down.")
            self._set_errored(BaseException("Code2wav Finally Shutdown"))
            if self.engine_infer_loop_thread is not None:
                self.engine_infer_loop_thread.join()
            self.cleanup()

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.code2wav
        del self.chunk_scheduler

    @contextmanager
    def make_data_socket(self) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Startup loop for sending data from Engine -> Client."""

        with self.make_data_socket() as socket:
            response: Union[RPCStartupResponse, BaseException]
            try:
                identity, message = socket.recv_multipart(copy=False)
                request: RPCStartupRequest = pickle.loads(message.buffer)

                # Handle the query from the Client.
                if request == RPCStartupRequest.IS_SERVER_READY:
                    response = RPCStartupResponse(False)
            except Exception as e:
                response = e

            socket.send_multipart((identity, pickle.dumps(response)), copy=False)

    def engine_recv_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            # Get new inputs, may contains multiple requests with more than one chunk
            # Schedule arrived requests
            if self._get_errored() is not None:
                break
            try:
                if VLLM_OMNI_T2C_USE_ZMQ:
                    # Poll until there is work to do.
                    while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                        # When there's no work, check on engine health and send
                        # health status back to client
                        self._health_check()
                    while self.input_socket.poll(timeout=0) != 0:
                        frame: Frame = self.input_socket.recv(copy=False)
                        (
                            request_id,
                            valid_chunk,
                            codec_list,
                            total_codec_num,
                            flushed,
                            finished,
                        ) = pickle.loads(frame.buffer)
                        if flushed and finished:
                            # receive abort, clear chunks in list
                            logger.info(
                                "[%s] C2W_RECV_ABORT", request_id
                            )
                            self.chunk_scheduler.clear_chunk(request_id)
                            continue
                        logger.debug(
                            "[%s] C2W_RECV_ZMQ tokens=%d chunk_size=%d",
                            request_id,
                            total_codec_num,
                            len(codec_list),
                        )
                        new_chunk = Code2WavChunk(
                            request_id,
                            valid_chunk,
                            codec_list,
                            total_codec_num,
                            flushed,
                            finished,
                        )
                        self.chunk_scheduler.add_chunk(new_chunk)
                else:
                    # use zmq to do health check instead
                    new_frame = None
                    while new_frame is None:
                        # wait for new request
                        with suppress(queue.Empty):
                            new_frame = self.talker_to_code2wav_queue.get(
                                timeout=POLLING_TIMEOUT_MS / 1000
                            )
                        if new_frame is None:
                            self._health_check()

                    while new_frame is not None:
                        # poll all request in queue to chunk scheduler
                        (
                            request_id,
                            valid_chunk,
                            codec_list,
                            total_codec_num,
                            flushed,
                            finished,
                        ) = new_frame
                        if flushed and finished:
                            # receive abort, clear chunks in list
                            logger.info(
                                "[%s] C2W_RECV_ABORT", request_id
                            )
                            self.chunk_scheduler.clear_chunk(request_id)
                            try:
                                new_frame = self.talker_to_code2wav_queue.get_nowait()
                            except queue.Empty:
                                break
                            continue
                        new_chunk = Code2WavChunk(
                            request_id,
                            valid_chunk,
                            codec_list,
                            total_codec_num,
                            flushed,
                            finished,
                        )
                        self.chunk_scheduler.add_chunk(new_chunk)
                        logger.debug(
                            "[%s] C2W_RECV_QUEUE tokens=%d chunk_size=%d",
                            request_id,
                            total_codec_num,
                            len(codec_list),
                        )
                        try:
                            new_frame = self.talker_to_code2wav_queue.get_nowait()
                        except queue.Empty:
                            break
            except Exception as e:
                self._set_errored(e)
                self._send_unhealthy(e)
                raise e
            continue

    def engine_infer_loop(self):
        """Core busy loop of the LLMEngine."""
        logger.info("Code2wav infer loop start")
        while True:
            # Get new inputs, may contains multiple requests with more than one chunk
            # Schedule arrived requests
            if self._get_errored() is not None:
                break
            chunks_list = self.chunk_scheduler.schedule()
            try:
                if len(chunks_list) == 0:
                    time.sleep(0.02)
                else:
                    # Execute selected requests
                    request_outputs = self.step_batch(chunks_list)
                    # Send outputs back to client
                    self.send_outputs_batch(request_outputs)
            except Exception as e:
                self._set_errored(e)
                self._send_unhealthy(e)
                raise e
            continue

    def step_batch(
        self, chunks_list: list[Code2WavChunk]
    ) -> list[tuple[str, np.ndarray, bool, int]]:
        request_outputs = []
        # filter illegal chunks
        max_padding_to = max([len(chunk.code_list) for chunk in chunks_list])
        max_chunk_size = (
            self.codec_left_context + self.codec_chunk_size + self.codec_right_context
        )
        max_padding_to = min(max_padding_to, max_chunk_size)
        # padding to n*padding interval
        if max_padding_to % self.padding_interval != 0:
            max_padding_to += (
                self.padding_interval - max_padding_to % self.padding_interval
            )
        batched_codec = torch.zeros(
            (len(chunks_list), max_padding_to, self.N_code_size),
            dtype=chunks_list[0].code_list[0].dtype,
            device=self.device,
        )
        for idx, chunk in enumerate(chunks_list):
            if len(chunk.code_list) > max_padding_to:
                chunk.code_list = chunk.code_list[:max_padding_to]
            for cidx, code in enumerate(chunk.code_list):
                batched_codec[idx, cidx, :] = code

        start_time = time.time() * 1000
        with torch.no_grad():
            audio_outputs = self.code2wav.decode(batched_codec).to(torch.float32)
        end_time = time.time() * 1000
        cost_ms = end_time - start_time
        
        logger.info(
            "C2W_BATCH batch=%d len=%d cost=%.0fms",
            len(chunks_list),
            max_padding_to,
            cost_ms,
        )
        # WARNING: Slow batch inference
        if cost_ms > 500:
            logger.warning(
                "C2W_BATCH_SLOW batch=%d cost=%.0fms (>500ms)",
                len(chunks_list),
                cost_ms,
            )
        for idx, chunk in enumerate(chunks_list):
            audio_start_idx = (
                len(chunk.code_list) - self.codec_right_context - chunk.valid_chunk
            ) * self.codec_upsample_rate
            audio_end_idx = (
                len(chunk.code_list) - self.codec_right_context
            ) * self.codec_upsample_rate
            if chunk.finished:
                # finish with audio=None signal to client
                audio = None
                request_outputs.append(
                    (chunk.request_id, audio, chunk.finished, chunk.total_codec_num)
                )
            else:
                if chunk.flushed:
                    if self.codec_right_context > 0:
                        # flush right context chunk
                        audio_start_idx = (
                            len(chunk.code_list) - self.codec_right_context
                        ) * self.codec_upsample_rate
                        audio_end_idx = len(chunk.code_list) * self.codec_upsample_rate
                        audio = audio_outputs[idx][0][audio_start_idx:audio_end_idx]
                        request_outputs.append(
                            (
                                chunk.request_id,
                                audio.detach().cpu().numpy(),
                                chunk.finished,
                                chunk.total_codec_num,
                            )
                        )
                    else:
                        # no right context chunk and flush, return empty audio
                        audio = np.empty((0), dtype=np.float32)
                        request_outputs.append(
                            (
                                chunk.request_id,
                                audio,
                                chunk.finished,
                                chunk.total_codec_num,
                            )
                        )
                else:
                    if len(chunk.code_list) > max_padding_to:
                        # already pruned code list, should not be here
                        audio_start_idx = (
                            -(chunk.valid_chunk + self.codec_right_context)
                            * self.codec_upsample_rate
                        )
                        audio_end_idx = (
                            -self.codec_right_context * self.codec_upsample_rate
                        )
                    # regular case, return audio chunk
                    audio = audio_outputs[idx][0][audio_start_idx:audio_end_idx]
                    request_outputs.append(
                        (
                            chunk.request_id,
                            audio.detach().cpu().numpy(),
                            chunk.finished,
                            chunk.total_codec_num,
                        )
                    )
        return request_outputs

    def send_outputs_batch(self, request_outputs: list[tuple[str, np.ndarray, bool]]):
        # Pack all batch results into a SINGLE ZMQ message to avoid
        # per-message event loop poll overhead on the client side.
        # With N separate messages, the client pays ~16ms event loop
        # contention per poll; batching reduces N polls to 1.
        batch_items = []
        for request_output in request_outputs:
            if request_output is None:
                continue
            elif isinstance(request_output, Tuple):
                request_id, audio, finished, output_tokens = (
                    request_output
                )
                batch_items.append(
                    (request_id, audio, output_tokens)
                )
                if finished and audio is not None:
                    batch_items.append(
                        (request_id, None, output_tokens)
                    )
            else:
                logger.warning(
                    "Invalid request output: %r", request_output
                )
        if not batch_items:
            return
        send_start = time.time()
        output_bytes = pickle.dumps(batch_items)
        self.output_socket.send_multipart(
            (output_bytes,), copy=False
        )
        send_duration_ms = (time.time() - send_start) * 1000
        logger.debug(
            "C2W_ZMQ_SEND_BATCH count=%d time=%.0fms "
            "size=%.1fKB",
            len(batch_items),
            send_duration_ms,
            len(output_bytes) / 1024,
        )
        if send_duration_ms > 500:
            logger.warning(
                "C2W_ZMQ_SEND_BATCH_SLOW count=%d "
                "time=%.0fms (>500ms)",
                len(batch_items),
                send_duration_ms,
            )
        return

    def _health_check(self):
        # Send unhealthy if engine has already errored
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)
        try:
            self._send_healthy()
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)

    def _send_outputs(self, outputs: Any):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            send_start = time.time()
            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes,), copy=False)
            
            send_duration_ms = (time.time() - send_start) * 1000
            request_id = outputs[0] if isinstance(outputs, tuple) else "unknown"
            
            logger.debug(
                f"[{request_id}] C2W_ZMQ_SEND time={send_duration_ms:.0f}ms "
                f"size={len(output_bytes)/1024:.1f}KB"
            )
            
            if send_duration_ms > 500:
                logger.warning(
                    f"[{request_id}] ZMQ_SEND_SLOW "
                    f"time={send_duration_ms:.0f}ms (>500ms)"
                )

    def _send_healthy(self):
        """Send HEALTHY message to RPCClient."""
        if not self.heartbeat_socket.closed:
            self.heartbeat_socket.send_multipart(HEALTHY_RESPONSE, copy=False)

    def _send_unhealthy(self, error: BaseException):
        """Send UNHEALTHY message to RPCClient."""
        if not self.heartbeat_socket.closed:
            error_bytes = pickle.dumps(error)
            self.heartbeat_socket.send_multipart((error_bytes,), copy=False)

    def _set_errored(self, e: BaseException):
        """Log and set errored status if this is the first issue."""
        with self.error_lock:
            if self._errored_with is None:
                self._errored_with = e

    def _get_errored(self):
        with self.error_lock:
            return self._errored_with

    @staticmethod
    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("Code2WavEngine terminated")

    @staticmethod
    def run_code2wav_engine(
        ipc_path: str,
        model_path: str,
        enable_torch_compile: bool,
        N_code_size: int = 16,
        codec_chunk_size: int = 1,
        codec_left_context: int = 25,
        codec_right_context: int = 0,
        codec_upsample_rate: int = 1920,
        engine_alive: Any = None,
        device: Union[int, str] = "cuda",
        talker_to_code2wav_queue=None,
    ):
        try:
            engine = _MQOmniCode2WavEngine(
                ipc_path,
                model_path,
                enable_torch_compile=enable_torch_compile,
                N_code_size=N_code_size,
                codec_chunk_size=codec_chunk_size,
                codec_left_context=codec_left_context,
                codec_right_context=codec_right_context,
                codec_upsample_rate=codec_upsample_rate,
                device=device,
                talker_to_code2wav_queue=talker_to_code2wav_queue,
            )

            signal.signal(signal.SIGTERM, engine.signal_handler)

            engine.start()
        except BaseException as e:
            logger.exception("Code2WavEngine failed with exception: %s", e)
            engine_alive.value = False
            raise e


class _MQCode2WavEngineClient:

    def __init__(self, ipc_path: str, engine_pid: int, talker_to_code2wav_queue):
        self.context = zmq.asyncio.Context()
        self._errored_with: Optional[BaseException] = None

        # ZMQ High Water Mark for high concurrency scenarios
        # Increase buffer to prevent blocking under load
        ZMQ_HWM = 1000

        # Send RPCGenerateRequest to the MQLLMEngine.
        if VLLM_OMNI_T2C_USE_ZMQ:
            self.input_socket: Socket = self.context.socket(zmq.constants.PUSH)
            self.input_socket.setsockopt(zmq.constants.SNDHWM, ZMQ_HWM)
            self.input_socket.connect(f"{ipc_path}{IPC_INPUT_EXT}")
        else:
            # use queue instread of zmq
            self.talker_to_code2wav_queue = talker_to_code2wav_queue

        # Receive streams of RequestOutput from the MQLLMEngine.
        self.output_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.output_socket.setsockopt(zmq.constants.RCVHWM, ZMQ_HWM)
        self.output_socket.connect(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # IPC path for acking heartbeats.
        self.heartbeat_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.heartbeat_socket.connect(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Stream for requests.
        # Use threading.Queue (not asyncio.Queue) so that
        # _listen_code2wav can .get() directly without the
        # asyncio.run_coroutine_threadsafe().result() bridge
        # that adds ~5-7ms overhead per call.
        self.output_queue: queue.Queue[
            tuple[str, Union[np.ndarray, BaseException]]
        ] = queue.Queue()

        # Loop to handle output of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready so that we can
        # build the Client in an executor to enable clean shutdown.
        self.output_loop: Optional[asyncio.Task] = None

        # Loop to check health of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready.
        self.health_loop: Optional[asyncio.Task] = None
        self._engine_process = psutil.Process(engine_pid)

    @contextmanager
    def get_data_socket(self) -> Iterator[Socket]:
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def run_heartbeat_loop(self, timeout: int):
        """Background loop that continually checks to ensure the engine process
        is still alive.
        """
        try:
            while True:
                # Check if the engine process is running:
                if not self._engine_process.is_running() or (
                    self._engine_process.status() == psutil.STATUS_ZOMBIE
                ):
                    # NB: is_running() returns True for zombies
                    self._set_errored(
                        RuntimeError(
                            f"Engine process (pid {self._engine_process.pid}) " "died."
                        )
                    )
                    break

                if await self.heartbeat_socket.poll(timeout=timeout):
                    # Heartbeat received- check the message
                    await self._check_success(
                        error_message="Heartbeat failed.", socket=self.heartbeat_socket
                    )

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient check health loop.")

        except psutil.NoSuchProcess:
            self._set_errored(
                RuntimeError(f"Engine process (pid {self._engine_process.pid}) died.")
            )

        except Exception as e:
            self._set_errored(e)

    async def run_output_handler_loop(self):
        """Get RequestOutputs from Engine and stream to Request Queues"""
        self.output_handler_seq = 0
        last_iteration_end = time.time()

        try:
            while True:
                iteration_start = time.time()
                loop_gap_ms = (iteration_start - last_iteration_end) * 1000
                self.output_handler_seq += 1
                seq = self.output_handler_seq
                if loop_gap_ms > 500:
                    logger.warning(
                        f"[OutputHandler] EVENT_LOOP_GAP seq={seq} "
                        f"gap={loop_gap_ms:.0f}ms (>500ms)"
                    )
                await self.run_output_handler(seq, iteration_start, loop_gap_ms)
                last_iteration_end = time.time()
        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient output handler.")

    async def run_output_handler(
        self, seq: int = 0, iteration_start: float = None,
        loop_gap_ms: float = 0,
    ):
        """Receive batched audio outputs from Code2Wav Engine.

        The C2W engine packs all batch results into a SINGLE
        ZMQ message.  We unpack and fan out to the threading
        output_queue in one shot — only 1 poll per batch
        instead of N polls.
        """
        if iteration_start is None:
            iteration_start = time.time()

        poll_start = time.time()

        # Poll, checking for ENGINE_DEAD
        while await self.output_socket.poll(
            timeout=VLLM_RPC_TIMEOUT
        ) == 0:
            # If errored, alert all running requests.
            if self.errored:
                logger.error("Engine is errored.")
                return

        poll_end = time.time()
        poll_wait_ms = (poll_end - poll_start) * 1000

        recv_start = time.time()
        message: Frame = await self.output_socket.recv(
            copy=False
        )
        recv_end = time.time()
        recv_duration_ms = (recv_end - recv_start) * 1000

        batch_items = pickle.loads(message.buffer)

        # batch_items is a list of (request_id, audio, output_tokens)
        queue_size = self.output_queue.qsize()
        total_handler_ms = (
            (time.time() - iteration_start) * 1000
        )

        batch_count = len(batch_items)
        if batch_count > 0:
            first_rid = batch_items[0][0]
        else:
            first_rid = "empty"

        logger.debug(
            "ZMQ_RECV_BATCH seq=%d batch=%d "
            "poll=%.0fms recv=%.0fms "
            "total=%.0fms queue=%d first_rid=%s",
            seq,
            batch_count,
            poll_wait_ms,
            recv_duration_ms,
            total_handler_ms,
            queue_size,
            first_rid[:12],
        )

        if poll_wait_ms > 500:
            logger.warning(
                "ZMQ_RECV_BATCH_SLOW poll=%.0fms (>500ms)",
                poll_wait_ms,
            )
        if queue_size > 200:
            logger.warning(
                "QUEUE_BACKLOG size=%d (>200)",
                queue_size,
            )

        # Fan out all items to threading.Queue in one shot
        for item in batch_items:
            self.output_queue.put(item)

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        # Start output_loop
        if self.output_loop is None:
            self.output_loop = asyncio.create_task(self.run_output_handler_loop())

        with self.get_data_socket() as socket:
            # Wait until server is ready.
            response = await self._wait_for_server_rpc(socket)

            self.tracing_flag = response.tracing_enabled

            # Start health_loop.
            if self.health_loop is None:
                self.health_loop = asyncio.create_task(
                    self.run_heartbeat_loop(timeout=VLLM_RPC_TIMEOUT)
                )

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets and terminate the context.
        self.context.destroy(linger=0)

        # Cancel background tasks.
        if self.health_loop is not None:
            self.health_loop.cancel()
        if self.output_loop is not None:
            self.output_loop.cancel()

    def _set_errored(self, e: BaseException):
        logger.exception(repr(e))
        if self._errored_with is None:
            self._errored_with = e

    @staticmethod
    async def _send_get_data_rpc_request(
        request: RPCStartupRequest,
        expected_type: Any,
        error_message: str,
        socket: Socket,
    ) -> Any:
        """Send an RPC request that is expecting data back."""

        # Ping RPCServer with a request.
        await socket.send_multipart((pickle.dumps(request),), copy=False)

        # Make sure the server responds in time.
        if await socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
            raise TimeoutError(
                "RPCServer didn't reply within " f"{VLLM_RPC_TIMEOUT} ms"
            )

        # Await the data from the Server.
        frame = await socket.recv(copy=False)
        data = pickle.loads(frame.buffer)

        if isinstance(data, BaseException):
            raise data
        elif not isinstance(data, expected_type):
            raise ValueError(error_message)

        return data

    # @staticmethod
    # async def _send_one_way_rpc_request(request: RPC_REQUEST_T, socket: Socket):
    #    """Send one-way RPC request to trigger an action."""

    #    if socket.closed:
    #        raise MQClientClosedError()

    #    await socket.send_multipart((pickle.dumps(request),))

    # async def _await_ack(self, error_message: str, socket: Socket):
    #    """Await acknowledgement that a request succeeded."""

    #    if socket.closed:
    #        raise MQClientClosedError()

    #    if await socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
    #        raise TimeoutError(
    #            "MQLLMEngine didn't reply within " f"{VLLM_RPC_TIMEOUT}ms"
    #        )

    #    await self._check_success(error_message, socket)

    @staticmethod
    async def _check_success(error_message: str, socket: Socket):
        """Confirm that socket has a VLLM_RPC_SUCCESS_STR message"""

        if socket.closed:
            raise Exception("socket closed")

        frame = await socket.recv(copy=False)
        response = pickle.loads(frame.buffer)

        # Raise error if unsuccessful
        if isinstance(response, BaseException):
            raise response
        elif not isinstance(response, str) or response != VLLM_RPC_SUCCESS_STR:
            raise ValueError(error_message)

    async def check_health(self):
        """
        The check health loop probes the health status of the
        Engine's health every N seconds and sets _errored_with
        if the engine is unhealthy.
        """
        if self._errored_with is not None:
            raise self._errored_with

    @property
    def is_running(self) -> bool:
        return not self.errored

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    # @property
    # def dead_error(self) -> BaseException:
    #    return ENGINE_DEAD_ERROR(self._errored_with)

    async def _wait_for_server_rpc(self, socket: Socket) -> RPCStartupResponse:
        """Wait for the RPCServer to start up."""

        return await self._send_get_data_rpc_request(
            request=RPCStartupRequest.IS_SERVER_READY,
            expected_type=RPCStartupResponse,
            error_message="Unable to start RPC Server",
            socket=socket,
        )

    async def process_chunk(
        self,
        request_id: str,
        valid_chunk: int,
        codec_list: List[torch.Tensor],
        total_codec_num: int,
        flushed: bool = False,
        finished: bool = False,
    ):
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        # If already dead, error out.
        if self._errored_with is not None:
            raise RuntimeError(
                "Engine loop is not running. Inspect the stacktrace to "
                f"find the original error: {repr(self._errored_with)}."
            )

        if VLLM_OMNI_T2C_USE_ZMQ:
            request_bytes = pickle.dumps(
                (
                    request_id,
                    valid_chunk,
                    codec_list,
                    total_codec_num,
                    flushed,
                    finished,
                )
            )
            logger.debug(
                "[%s] C2W_SEND_ZMQ tokens=%d chunk_size=%d bytes=%d",
                request_id,
                total_codec_num,
                len(codec_list),
                len(request_bytes),
            )
            send_start = time.time()
            await self.input_socket.send_multipart((request_bytes,), copy=False)
            send_duration_ms = (time.time() - send_start) * 1000
            if send_duration_ms > 500:
                logger.warning(
                    f"[{request_id}] ZMQ send request blocked {send_duration_ms:.0f}ms "
                    f"(Code2Wav engine busy, size={len(request_bytes)/1024:.1f}KB)"
                )
        else:
            self.talker_to_code2wav_queue.put(
                (
                    request_id,
                    valid_chunk,
                    codec_list,
                    total_codec_num,
                    flushed,
                    finished,
                )
            )
            logger.debug(
                "[%s] C2W_SEND_QUEUE tokens=%d chunk_size=%d",
                request_id,
                total_codec_num,
                len(codec_list),
            )

    def get_chunk(
        self, timeout: float = None,
    ) -> tuple[str, Union[np.ndarray, BaseException]]:
        """Get the next chunk of audio from the engine.

        Now reads directly from threading.Queue — no asyncio
        bridge needed.  Called from _listen_code2wav thread.
        """
        return self.output_queue.get(timeout=timeout)

    async def abort_request(
        self,
        request_id: str,
    ):
        # use flushed && finished as abort signal
        if VLLM_OMNI_T2C_USE_ZMQ:
            request_bytes = pickle.dumps((request_id, 1, None, 0, True, True))
            logger.info("[%s] C2W_ABORT_ZMQ", request_id)
            await self.input_socket.send_multipart((request_bytes,), copy=False)
        else:
            self.talker_to_code2wav_queue.put((request_id, 1, None, 0, True, True))
            logger.info("[%s] C2W_ABORT_QUEUE", request_id)


class suppress_output_queue_exception(contextlib.AbstractContextManager):
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         with suppress(FileNotFoundError):
             os.remove(somefile)
         # Execution still resumes here if the file was already removed
    """

    def __init__(self):
        self._exceptions = (Exception,)

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        if exctype is not None:
            logger.exception(
                "Exception in output queue: %s, %r, %r", exctype, excinst, exctb
            )
        return exctype is not None and issubclass(exctype, self._exceptions)


class OmniLLMEngine:

    def __init__(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: Union[
            Mapping[str, AsyncEngineArgs], AsyncEngineArgs
        ] = None,
        code2wav_model_path: Optional[str] = None,
        code2wav_data_parallelism: int = 1,
        code2wav_enable_torch_compile: bool = False,
        code2wav_enable_torch_compile_first_chunk: bool = False,
        code2wav_odeint_method: str = "rk4",
        code2wav_odeint_method_relaxed: bool = False,
        code2wav_batched_chunk: int = None,
        code2wav_frequency: str = "50hz",
        code2wav_dit_quant: str = None,
        thinker_visible_devices: Optional[list[int]] = None,
        talker_visible_devices: Optional[list[int]] = None,
        code2wav_visible_devices: Optional[list[int]] = None,
        code2wav_dynamic_batch: Optional[bool] = False,
    ):
        self.thinker_engine_args = thinker_engine_args
        if talker_engine_args is None:
            talker_engine_args = {}
        if not isinstance(talker_engine_args, dict):
            talker_engine_args = {
                "default": talker_engine_args,
            }
        self.talker_engine_args = talker_engine_args
        self.code2wav_model_path = code2wav_model_path
        self.code2wav_enable_torch_compile = code2wav_enable_torch_compile
        self.code2wav_enable_torch_compile_first_chunk = (
            code2wav_enable_torch_compile_first_chunk
        )
        self.code2wav_data_parallelism = code2wav_data_parallelism
        self.code2wav_odeint_method = code2wav_odeint_method
        self.code2wav_odeint_method_relaxed = code2wav_odeint_method_relaxed
        if code2wav_batched_chunk is None:
            if code2wav_frequency == "50hz":
                code2wav_batched_chunk = 2
            else:
                code2wav_batched_chunk = 1
        self.code2wav_batched_chunk = code2wav_batched_chunk
        self.code2wav_frequency = code2wav_frequency
        self.code2wav_dit_quant = code2wav_dit_quant

        self.thinker_visible_devices = thinker_visible_devices
        self.talker_visible_devices = talker_visible_devices
        self.code2wav_visible_devices = code2wav_visible_devices

        # multiprocessing manager for thinker to talker and talker to code2wav.
        self.mp_manager = multiprocessing.Manager()

        if self.thinker_visible_devices and not isinstance(
            self.thinker_visible_devices, list
        ):
            self.thinker_visible_devices = [self.thinker_visible_devices]
        if self.talker_visible_devices and not isinstance(
            self.talker_visible_devices, list
        ):
            self.talker_visible_devices = [self.talker_visible_devices]
        if self.code2wav_visible_devices and not isinstance(
            self.code2wav_visible_devices, list
        ):
            self.code2wav_visible_devices = [self.code2wav_visible_devices]

        # Set VLLM_OMNI_ENABLE_TALKER based on whether talker args are provided.
        # This controls whether thinker needs to handle accept_hidden_states
        # and initialize shm-related configs.
        has_talker = bool(talker_engine_args)
        os.environ["VLLM_OMNI_ENABLE_TALKER"] = "true" if has_talker else "false"

        # Enable hidden state caching for the thinker engine
        # only when talker is present (thinker+talker+code2wav mode).
        if has_talker:
            self.thinker_engine_args.enable_hidden_state_caching = True

        if not code2wav_model_path:
            self.code2wav_data_parallelism = 0

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        self.context = multiprocessing.get_context("spawn")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._async_loop_thread = Thread(target=self._start_async_loop, daemon=True)
        self._async_loop_thread.start()

        # thinker model config
        self.thinker_config = get_config(
            thinker_engine_args.model,
            trust_remote_code=thinker_engine_args.trust_remote_code,
        )
        self.thinker_generation_config = try_get_generation_config(
            thinker_engine_args.model,
            trust_remote_code=thinker_engine_args.trust_remote_code,
        )

        if "Qwen2_5OmniModel" in self.thinker_config.architectures:
            envs.VLLM_QWEN2_VL_USE_SPLIT_QKV = 1

        # Initialize mooncake master and SharedHiddenBuffer for hidden
        # states transfer (only in thinker+talker+code2wav mode).
        self._mooncake_master_process = None
        self._shared_hidden_buffer = None

        if has_talker:
            backend_type = os.environ.get("VLLM_HIDDEN_BUFFER_BACKEND", "")
            is_mooncake_backend = backend_type.lower() == "mooncake"

            if is_mooncake_backend:
                from vllm.engine.shared_hidden_buffer import (
                    DEFAULT_MOONCAKE_MASTER_HOST,
                    DEFAULT_MOONCAKE_MASTER_RPC_PORT,
                )
                master_host = os.environ.get(
                    "MOONCAKE_MASTER_HOST", DEFAULT_MOONCAKE_MASTER_HOST
                )
                master_rpc_port = int(os.environ.get(
                    "MOONCAKE_MASTER_RPC_PORT",
                    str(DEFAULT_MOONCAKE_MASTER_RPC_PORT)
                ))
                self._mooncake_master_process = self._start_mooncake_master(
                    host=master_host,
                )
                master_address = f"{master_host}:{master_rpc_port}"
                os.environ["MOONCAKE_MASTER_SERVER_ADDRESS"] = master_address
                logger.info(
                    "Set MOONCAKE_MASTER_SERVER_ADDRESS=%s for workers",
                    master_address
                )

            from vllm.engine.shared_hidden_buffer import SharedHiddenBuffer

            thinker_hidden_size = None
            if hasattr(self.thinker_config, "text_config") and \
               hasattr(self.thinker_config.text_config, "hidden_size"):
                thinker_hidden_size = (
                    self.thinker_config.text_config.hidden_size
                )
            elif hasattr(self.thinker_config, "hidden_size"):
                thinker_hidden_size = self.thinker_config.hidden_size

            max_num_seqs = getattr(thinker_engine_args, "max_num_seqs", None)
            max_model_len = getattr(
                thinker_engine_args, "max_model_len", None
            )

            logger.info(
                "Creating SharedHiddenBuffer with: max_num_seqs=%s, "
                "max_model_len=%s, hidden_size=%s",
                max_num_seqs, max_model_len, thinker_hidden_size
            )

            self._shared_hidden_buffer = SharedHiddenBuffer.create_global(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                hidden_size=thinker_hidden_size,
            )
        else:
            logger.info(
                "Thinker-only mode: skipping shm and hidden buffer "
                "initialization (no talker args provided)"
            )

        # thinker: async engine
        self.thinker_engine = self._start_thinker_engine(
            thinker_engine_args, self.thinker_visible_devices
        )

        # self.device = self.thinker_engine.engine.device_config.device
        # thinker_vllm_config = asyncio.run(
        #     self.get_vllm_config_async(self.thinker_engine)
        # )
        # self.device = thinker_vllm_config.device_config.device
        self.device = "cuda"

        # thinker state
        # request_id -> prompt_embeds
        self.thinker_prompt_embeds: dict[str, torch.Tensor] = dict()
        # request_id -> listener thread
        self.thinker_listener: dict[str, Thread] = dict()
        # talker type -> (request_id, voice_type, thinker request_output)
        self.thinker_outputs: dict[
            str,
            queue.Queue[tuple[str, str, RequestOutput]],
        ] = collections.defaultdict(queue.Queue)
        self.thinker_finished: Set[str] = set()

        # thinker tokenizer
        self.thinker_tokenizer = get_tokenizer(
            thinker_engine_args.tokenizer,
            tokenizer_mode=thinker_engine_args.tokenizer_mode,
            trust_remote_code=thinker_engine_args.trust_remote_code,
            revision=thinker_engine_args.revision,
            download_dir=thinker_engine_args.download_dir,
        )

        # talker: async mp engine(s)
        self._talker_engine_threads: dict[str, Thread] = {}
        for i, (talker_type, engine_args) in enumerate(talker_engine_args.items()):
            # use thinker as talkers tokenizer
            if thinker_engine_args.tokenizer:
                engine_args.tokenizer = thinker_engine_args.tokenizer
            else:
                engine_args.tokenizer = thinker_engine_args.model
            if thinker_engine_args.processor:
                engine_args.processor = thinker_engine_args.processor
            else:
                engine_args.processor = thinker_engine_args.model
            if not engine_args.hf_overrides:
                engine_args.hf_overrides = {}

            if hasattr(self.thinker_config, "audio_config"):
                engine_args.hf_overrides["audio_config"] = (
                    self.thinker_config.audio_config
                )
            if hasattr(self.thinker_config, "vision_config"):
                engine_args.hf_overrides["vision_config"] = (
                    self.thinker_config.vision_config
                )

            # Adapt the qwen3_omni_model
            engine_args.hf_overrides["architectures"] = ["Qwen3OmniTalkerModel"]

            if self.talker_visible_devices:
                visible_devices = [self.talker_visible_devices[i]]
            else:
                visible_devices = None
            (
                self.talker_engine,
                thread,
            ) = self._start_talker_engine(talker_type, engine_args, visible_devices)
            self._talker_engine_threads[talker_type] = thread
        if talker_engine_args:
            # talker model config
            default_talker_args = next(iter(talker_engine_args.values()))
            self.talker_config = get_config(
                default_talker_args.model,
                trust_remote_code=default_talker_args.trust_remote_code,
            )
            self.talker_generation_config = try_get_generation_config(
                default_talker_args.model,
                trust_remote_code=default_talker_args.trust_remote_code,
            )
            # init embeddings for special tokens
            self._load_thinker_to_talker_projection(default_talker_args)
            self._init_omni3_special_tokens_embeddings(
                thinker_engine_args, default_talker_args
            )

        # talker request state
        self.talker_requests: dict[
            str, tuple[PromptType, SamplingParams, SamplingParams]
        ] = dict()
        self.talker_listener: dict[str, Thread] = dict()
        self.talker_finished: Set[str] = set()
        self.talker_inputs: dict[
            str,
            queue.Queue[Optional[tuple[torch.Tensor, bool]]],
        ] = dict()

        # DAC code2wav model related parameters
        self.N_code_size = 16
        self.codec_chunk_size = VLLM_OMNI_CODE2WAV_CHUNK_SIZE
        self.codec_left_context = 25
        self.codec_right_context = 0
        self.codec_upsample_rate = 1920
        self.max_codec_chunk_size = (
            self.codec_left_context + self.codec_chunk_size + self.codec_right_context
        )

        # code2wav dynamic chunk strategy
        self.dynamic_codec_chunk_size_list = [2, 4, 6, 8]
        self.dynamic_codec_chunk_step_list = [8, 4, 2, 1]
        assert len(self.dynamic_codec_chunk_size_list) == len(
            self.dynamic_codec_chunk_step_list
        )
        self.dynamic_codec_chunks = [
            val
            for val, count in zip(
                self.dynamic_codec_chunk_size_list, self.dynamic_codec_chunk_step_list
            )
            for _ in range(count)
        ]

        # code2wav: async mp engine(s)
        self._code2wav_pids = []
        self._code2wav_processes = []
        self._code2wav_engine_clients: list[_MQCode2WavEngineClient] = []
        for i in range(self.code2wav_data_parallelism):
            assert code2wav_model_path, "code2wav_model_path is required"

            (
                pid,
                process,
                engine_client,
            ) = self._start_code2wav_engine(
                get_open_zmq_ipc_path(),
                code2wav_model_path,
                code2wav_enable_torch_compile,
                N_code_size=self.N_code_size,
                codec_chunk_size=self.codec_chunk_size,
                codec_left_context=self.codec_left_context,
                codec_right_context=self.codec_right_context,
                codec_upsample_rate=self.codec_upsample_rate,
                visible_devices=(
                    [
                        self.code2wav_visible_devices[
                            i % len(self.code2wav_visible_devices)
                        ]
                    ]
                    if self.code2wav_visible_devices
                    else None
                ),
            )
            self._code2wav_pids.append(pid)
            self._code2wav_processes.append(process)
            self._code2wav_engine_clients.append(engine_client)

        # code2wav voice types
        if self.code2wav_model_path:
            self.code2wav_voice_types = self._load_code2wav_voice_types_legacy()
            if not self.code2wav_voice_types:
                self.code2wav_voice_types = self._load_code2wav_voice_types()
        else:
            self.code2wav_voice_types = []
        # for prefix-caching
        if self.code2wav_voice_types:
            self.code2wav_voice_types.append("prefix_caching")
        logger.info("Code2Wav voice types: %s", self.code2wav_voice_types)

        # code2wav result listener
        self._code2wav_listener_thread = []
        for i in range(self.code2wav_data_parallelism):
            self._code2wav_listener_thread.append(
                Thread(target=self._listen_code2wav, args=(i,), daemon=True)
            )
            self._code2wav_listener_thread[i].start()

        # request id -> [union(request_output, (waveform, finished))]
        self.output_queue: dict[
            str, queue.Queue[Union[RequestOutput, tuple[np.ndarray, int]]]
        ] = collections.defaultdict(queue.Queue)

        self.im_start_user_tokens = self.thinker_tokenizer.encode("<|im_start|>user")

    async def get_vllm_config_async(self, engine):
        return await engine.get_vllm_config()

    def _load_code2wav_voice_types_legacy(self):
        return [
            _MQOmniCode2WavEngine.parse_key(os.path.basename(f), "spk_emb.npy")
            for f in sorted(
                glob.glob(
                    os.path.join(self.code2wav_model_path, "inputs", "*spk_emb.npy")
                )
                + glob.glob(
                    os.path.join(
                        self.code2wav_model_path, "inputs_sft4spks", "*spk_emb.npy"
                    )
                )
            )
        ]

    def _load_code2wav_voice_types(self):
        voice_type_mapping = {
            "Cherry": "f245",
            "Ethan": "m02",
            "Serena": "f05",
            "Chelsie": "f030",
        }
        code2wav_voice_types = []

        if not os.path.exists(os.path.join(self.code2wav_model_path, "spk_dict.pt")):
            return code2wav_voice_types

        for key, value in torch.load(
            os.path.join(self.code2wav_model_path, "spk_dict.pt")
        ).items():
            code2wav_voice_types.append(voice_type_mapping.get(key, key))
        return code2wav_voice_types

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_mooncake_master(
        self,
        host: str = "127.0.0.1",
    ) -> Optional[subprocess.Popen]:
        """Start mooncake_master process for distributed hidden states transfer.

        mooncake_master runs two services:
        1. RPC service on default port 50051 - for actual data transfer
        2. HTTP metadata server (optional) - for metadata queries
        """
        import shutil as _shutil
        mooncake_master_path = _shutil.which("mooncake_master")
        if mooncake_master_path is None:
            logger.warning(
                "mooncake_master not found in PATH. "
                "Mooncake backend will not be available."
            )
            return None

        cmd = [mooncake_master_path]
        logger.info("Starting mooncake_master on host=%s", host)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            time.sleep(1.0)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(
                    "mooncake_master failed to start: exit_code=%d, "
                    "stdout=%s, stderr=%s",
                    process.returncode,
                    stdout.decode() if stdout else "",
                    stderr.decode() if stderr else "",
                )
                return None
            logger.info(
                "mooncake_master started successfully: pid=%d, host=%s",
                process.pid, host,
            )
            return process
        except Exception as e:
            logger.error("Failed to start mooncake_master: %s", e)
            return None

    def _stop_mooncake_master(self):
        """Stop the mooncake_master process if running."""
        if self._mooncake_master_process is not None:
            try:
                self._mooncake_master_process.terminate()
                try:
                    self._mooncake_master_process.wait(timeout=5.0)
                    logger.info("mooncake_master stopped gracefully")
                except subprocess.TimeoutExpired:
                    self._mooncake_master_process.kill()
                    self._mooncake_master_process.wait()
                    logger.info("mooncake_master killed forcefully")
            except Exception as e:
                logger.warning("Error stopping mooncake_master: %s", e)
            finally:
                self._mooncake_master_process = None

    def _start_thinker_engine(
        self,
        engine_args: AsyncEngineArgs,
        visible_devices: Optional[list[int]] = None,
    ) -> AsyncLLMEngine:
        if visible_devices:
            original_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if original_devices is not None:
                original_devices = original_devices.split(",")
                value = ",".join(original_devices[device] for device in visible_devices)
            else:
                value = ",".join(str(device) for device in visible_devices)
            device_ctx = envvars("CUDA_VISIBLE_DEVICES", value)
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            return AsyncLLMEngine.from_engine_args(
                engine_args,
                start_engine_loop=True,
            )

    def _start_talker_engine(
        self,
        talker_type: str,
        engine_args: AsyncEngineArgs,
        visible_devices: Optional[list[int]] = None,
    ) -> tuple[AsyncLLMEngine, Thread]:
        if visible_devices:
            original_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if original_devices is not None:
                original_devices = original_devices.split(',')
                value = ','.join(original_devices[device]
                                 for device in visible_devices)
            else:
                value = ','.join(str(device) for device in visible_devices)
            device_ctx = envvars('CUDA_VISIBLE_DEVICES', value)
            if VLLM_OMNI_PREPROCESS_ON_GPU:
                os.environ['PREPROCESS_DEVICE_T'] = 'cuda:0'
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            usage_context = UsageContext.API_SERVER

            talker_engine = AsyncLLMEngine.from_engine_args(
                engine_args=engine_args,
                start_engine_loop=True,
                usage_context=usage_context,
            )

        thread = Thread(
            target=self._thinker_to_talker_loop,
            args=(self.thinker_outputs[talker_type],),
            daemon=True,
        )
        thread.start()

        return talker_engine, thread

    def _start_code2wav_engine(
        self,
        ipc_path: str,
        model_path: str,
        enable_torch_compile: bool,
        N_code_size: int,
        codec_chunk_size: int,
        codec_left_context: int,
        codec_right_context: int,
        codec_upsample_rate: int,
        device: Union[int, str] = 'cuda',
        visible_devices: Optional[list[int]] = None,
    ) -> tuple[int, multiprocessing.Process, _MQCode2WavEngineClient]:
        logger.info(
            'Multiprocessing frontend to use %s for IPC Path for model %s.',
            ipc_path, model_path)

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        # if visible_devices:
        #     device_ctx = envvars(
        #         'CUDA_VISIBLE_DEVICES',
        #         ','.join(str(device) for device in visible_devices))
        # else:
        #     device_ctx = contextlib.nullcontext()

        talker_to_code2wav_queue = None
        logger.info('Start code2wav engine, using zmq to send: %s',
                    VLLM_OMNI_T2C_USE_ZMQ)
        if not VLLM_OMNI_T2C_USE_ZMQ:
            talker_to_code2wav_queue = self.mp_manager.Queue()

        engine_process = self.context.Process(
            target=_MQOmniCode2WavEngine.run_code2wav_engine,
            args=(ipc_path, model_path, enable_torch_compile, N_code_size,
                  codec_chunk_size, codec_left_context, codec_right_context,
                  codec_upsample_rate, engine_alive, visible_devices[0],
                  talker_to_code2wav_queue))
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info('Started engine process with PID %d, device: %s',
                    engine_pid, str(visible_devices))

        mq_engine_client: _MQCode2WavEngineClient = _MQCode2WavEngineClient(
            ipc_path, engine_pid, talker_to_code2wav_queue)

        while True:
            try:
                asyncio.run_coroutine_threadsafe(
                    mq_engine_client.setup(),
                    self._loop,
                ).result()
                break
            except TimeoutError:
                if (not engine_process.is_alive() or not engine_alive.value):
                    raise RuntimeError(
                        "Engine process failed to start. See stack "
                        "trace for the root cause.") from None

        logger.info('Engine process with PID %d has started', engine_pid)
        return engine_pid, engine_process, mq_engine_client

    def _shutdown_mp_engine(
        self,
        engine_pid: int,
        engine_process: multiprocessing.Process,
        mq_engine_client: _MQCode2WavEngineClient,
    ):
        engine_process.terminate()
        mq_engine_client.close()
        engine_process.join(5)
        if engine_process.is_alive():
            logger.warning('Engine process did not terminate in time.')
            engine_process.kill()
        logger.info('Engine process with PID %d has stopped', engine_pid)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self._closed = True
        # shutdown thinker engine
        self.thinker_engine.shutdown()

        # shutdown talker engines
        if hasattr(self, "talker_engine"):
            self.talker_engine.shutdown()

        # shutdown code2wav engines
        for pid, process, engine_client in zip(
            self._code2wav_pids,
            self._code2wav_processes,
            self._code2wav_engine_clients,
        ):
            self._shutdown_mp_engine(pid, process, engine_client)

        # Stop mooncake master if running
        self._stop_mooncake_master()

    def _load_model_embedding(
        self,
        engine_args: EngineArgs,
        is_talker: bool = False,
    ) -> torch.nn.Embedding:
        model_loader = get_model_loader(
            LoadConfig(
                load_format=engine_args.load_format,
                download_dir=engine_args.download_dir,
                model_loader_extra_config=engine_args.model_loader_extra_config,
                ignore_patterns=engine_args.ignore_patterns,
            )
        )
        for key, value in model_loader.get_all_weights(
            ModelConfig(
                model=engine_args.model,
                # task=engine_args.task,
                tokenizer=engine_args.tokenizer,
                tokenizer_mode=engine_args.tokenizer_mode,
                processor=engine_args.processor,
                trust_remote_code=engine_args.trust_remote_code,
                dtype=engine_args.dtype,
                seed=engine_args.seed,
            ),
            model=None,
        ):
            if not is_talker:
                if key in [
                    "model.embed_tokens.weight",
                    "thinker.model.embed_tokens.weight",
                    "talker.model.embed_tokens.weight",
                ]:
                    logger.info(
                        "get tensor %s and %s",
                        key,
                        torch.nn.Embedding.from_pretrained(value).weight.shape,
                    )
                    return torch.nn.Embedding.from_pretrained(value).to(self.device)
            elif key in [
                "talker.model.codec_embedding.weight",
            ]:
                logger.info(
                    "get tensor %s and %s",
                    key,
                    torch.nn.Embedding.from_pretrained(value).weight.shape,
                )
                return torch.nn.Embedding.from_pretrained(value).to(self.device)
        raise ValueError(
            "Embedding weight 'model.embed.weight' not found in model weights."
        )

    def _load_thinker_to_talker_projection(
        self,
        talker_engine_args: AsyncEngineArgs,
    ):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        model_path = talker_engine_args.model
        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(
            talker_hf_config.thinker_hidden_size,
            talker_hf_config.intermediate_size,
            talker_hf_config.hidden_size,
            talker_hf_config.hidden_act,
            bias=True,
        ).to(dtype=torch.bfloat16)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(
            talker_hf_config.thinker_hidden_size,
            talker_hf_config.intermediate_size,
            talker_hf_config.hidden_size,
            talker_hf_config.hidden_act,
            bias=True,
        ).to(dtype=torch.bfloat16)

        safetensors = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for key, value in safetensors_weights_iterator(
            safetensors, use_tqdm_on_load=True
        ):
            if key.startswith("talker.hidden_projection."):
                self.hidden_projection.state_dict()[
                    key.replace("talker.hidden_projection.", "")
                ].copy_(value)

            if key.startswith("talker.text_projection."):
                self.text_projection.state_dict()[
                    key.replace("talker.text_projection.", "")
                ].copy_(value)

        self.text_projection = self.text_projection.to(self.device)
        self.hidden_projection = self.hidden_projection.to(self.device)

    def _init_omni3_special_tokens_embeddings(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: AsyncEngineArgs,
    ):
        # thinker and talker embeddings
        self.thinker_embedding = self._load_model_embedding(thinker_engine_args)
        self.talker_embedding = self._load_model_embedding(talker_engine_args, True)

        talker_hf_config = self.talker_config

        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = (
            self.text_projection(
                self.thinker_embedding(
                    torch.tensor(
                        [
                            talker_hf_config.tts_bos_token_id,
                            talker_hf_config.tts_eos_token_id,
                            talker_hf_config.tts_pad_token_id,
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                )
            ).chunk(3, dim=0)
        )

    def _listen_thinker(
        self,
        request_id: str,
        voice_type: str,
        generator: AsyncGenerator[RequestOutput, None],
        thinker_outputs: queue.Queue[tuple[str, str, RequestOutput]],
    ):
        include_voice = not self._ignore_voice(voice_type)

        def _listen():
            last_output = None
            prefill_hidden_states: torch.Tensor | None = None
            prefill_hidden_states_read = False
            sent_token_count = 0
            for output in SynchronizedGenerator(generator, self._loop):
                if include_voice and self.talker_engine_args:
                    # Read prefill hidden states from SHM once when
                    # the first token arrives.
                    token_ids = output.outputs[0].token_ids
                    num_new_tokens = len(token_ids)
                    if (not prefill_hidden_states_read
                            and num_new_tokens > 0):
                        from vllm.engine.shared_hidden_buffer import (
                            SharedHiddenBuffer,
                        )
                        shm_key = f"{request_id}_prefill"
                        prefill_hidden_states = (
                            SharedHiddenBuffer.read_global_by_request_id(
                                request_id=shm_key,
                                device=self.device,
                                wait=True,
                                timeout=5.0,
                            )
                        )
                        if prefill_hidden_states is None:
                            logger.error(
                                "[%s] Failed to read prefill "
                                "hidden_states from SHM",
                                request_id,
                            )
                        else:
                            logger.info(
                                "[%s] Read prefill hidden_states "
                                "from SHM, shape=%s",
                                request_id,
                                prefill_hidden_states.shape,
                            )
                        prefill_hidden_states_read = True

                    if output.finished:
                        thinker_output = copy.deepcopy(output)
                        if len(output.outputs[0].token_ids) > 1:
                            thinker_output.finished = False
                    else:
                        thinker_output = copy.deepcopy(output)

                    # Attach prefill hidden states to the first output
                    # so that talker can build its input embeddings.
                    if sent_token_count == 0 and prefill_hidden_states is not None:
                        thinker_output.accept_hidden_states = (
                            prefill_hidden_states.cpu()
                        )
                    sent_token_count += 1

                    thinker_outputs.put((request_id, voice_type, thinker_output))
                    logger.info(
                        "[%s] listen thinker to talker output: %s, "
                        "accept_hidden_states=%s",
                        request_id,
                        thinker_output.outputs[0].token_ids,
                        thinker_output.accept_hidden_states.shape
                        if hasattr(thinker_output, 'accept_hidden_states')
                        and thinker_output.accept_hidden_states is not None
                        else None,
                    )
                else:
                    logger.info(
                        "[%s] listen thinker output: %s",
                        request_id,
                        output.outputs[0].token_ids,
                    )

                if last_output is None:
                    logger.info(
                        "rid: %s got first token, text: %s, token_id: %s",
                        request_id,
                        output.outputs[0].text,
                        output.outputs[0].token_ids,
                    )
                last_output = output
                with suppress_output_queue_exception():
                    self.output_queue[request_id].put(output)
            if not include_voice or not self.talker_engine_args:
                with suppress_output_queue_exception():
                    self.output_queue[request_id].put(None)
                    self.output_queue.pop(request_id)
            if last_output is not None:
                logger.info(
                    "[%s] Thinker request finished, reason: (%s, %s), "
                    "outputs: %s, output tokens: %s",
                    request_id,
                    last_output.outputs[0].finish_reason,
                    last_output.outputs[0].stop_reason,
                    repr(last_output.outputs[0].text),
                    last_output.outputs[0].token_ids,
                )
            else:
                logger.info("[%s] Thinker request finished, no outputs", request_id)

            if (
                include_voice
                and self.talker_engine_args
                and len(last_output.outputs[0].token_ids) > 1
            ):
                # add special tokens for talker model.
                special_token_embeds = [
                    self.tts_eos_embed,
                ]
                for i, token_embed in enumerate(special_token_embeds):
                    thinker_outputs.put(
                        (
                            request_id,
                            voice_type,
                            RequestOutput(
                                request_id=request_id,
                                prompt=last_output.prompt,
                                prompt_token_ids=last_output.prompt_token_ids,
                                prompt_logprobs=last_output.prompt_logprobs,
                                outputs=[
                                    CompletionOutput(
                                        index=0,
                                        text="",
                                        token_ids=[],
                                        cumulative_logprob=None,
                                        logprobs=None,
                                        prompt_embeds=token_embed,
                                    )
                                ],
                                finished=i == len(special_token_embeds) - 1,
                            ),
                        )
                    )

            with contextlib.suppress(Exception):
                del self.thinker_listener[request_id]

        self.thinker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.thinker_listener[request_id].start()

    def _listen_talker(
        self,
        request_id: str,
        voice_type: str,
        generator: AsyncGenerator[RequestOutput, None],
        output_queue: queue.Queue[Union[RequestOutput, np.ndarray]],
        talker_input_queue: queue.Queue[Optional[tuple[torch.Tensor, bool]]],
    ):
        include_code2wav = not self._ignore_code2wav(voice_type)

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        def _listen():
            total_codec_num = 0
            codec_ids_list = []
            last_output = None
            for output in SynchronizedGenerator(generator, self._loop):
                last_output = output
                if output.outputs is None and output.finished:
                    logger.info('[%s] got None, maybe aborted', request_id)
                    break
                # use subtalker output to count talker token number
                logger.info(
                    '[%s] talker listener got output, finished: %s, codec num: %s',
                    request_id, output.finished, total_codec_num)
                if not self._code2wav_engine_clients or not include_code2wav:
                    # skip code2wav
                    output_queue.put(output)
                elif output.outputs[0].codec_ids is not None:
                    codec_ids_list.append(output.outputs[0].codec_ids)
                    total_codec_num += 1
                    if (total_codec_num % self.codec_chunk_size == 0):
                        logger.info('[%s] talker to code2wav: %s/%s',
                                    request_id, total_codec_num,
                                    self.codec_chunk_size)
                        self._talker_to_code2wav(output.request_id,
                                                 self.codec_chunk_size,
                                                 codec_ids_list,
                                                 total_codec_num, False, False)
                    if output.finished:
                        # flush codecs not aligned to chunk_size
                        if (total_codec_num % self.codec_chunk_size != 0):
                            logger.info(
                                '[%s] talker to code2wav flush not aligned: %s/%s',
                                request_id, total_codec_num,
                                self.codec_chunk_size)
                            self._talker_to_code2wav(
                                output.request_id,
                                total_codec_num % self.codec_chunk_size,
                                codec_ids_list, total_codec_num, False, False)
                        # flush right context
                        if self.codec_right_context > 0:
                            self._talker_to_code2wav(output.request_id,
                                                     self.codec_chunk_size,
                                                     codec_ids_list,
                                                     total_codec_num, True,
                                                     False)
                        self._talker_to_code2wav(output.request_id,
                                                 self.codec_chunk_size,
                                                 codec_ids_list,
                                                 total_codec_num, False, True)
                elif output.finished:
                    # flush codecs not aligned to chunk_size
                    if (total_codec_num % self.codec_chunk_size != 0):
                        self._talker_to_code2wav(
                            output.request_id,
                            total_codec_num % self.codec_chunk_size,
                            codec_ids_list, total_codec_num, False, False)
                    # flush right context
                    if self.codec_right_context > 0:
                        self._talker_to_code2wav(output.request_id,
                                                 self.codec_chunk_size,
                                                 codec_ids_list,
                                                 total_codec_num, True, False)
                    self._talker_to_code2wav(
                        output.request_id,
                        self.codec_chunk_size,
                        codec_ids_list,
                        total_codec_num, False, True)
                else:
                    # has code2wav, no codec_id and not finished
                    pass
                # prune codec list
                if len(codec_ids_list) > self.max_codec_chunk_size:
                    codec_ids_list = codec_ids_list[-self.
                                                    max_codec_chunk_size:]

            logger.info('[%s] Talker output generator exit', request_id)
            if not self._code2wav_engine_clients or not include_code2wav:
                output_queue.put(None)

            if last_output is None or last_output.outputs is None:
                logger.info('[%s] Talker output None, maybe aborted',
                            request_id)
            else:
                logger.info(
                    '[%s] Talker request finished, reason: (%s, %s), output tokens: %s',
                    request_id, last_output.outputs[0].finish_reason,
                    last_output.outputs[0].stop_reason,
                    last_output.outputs[0].token_ids)

            with contextlib.suppress(Exception):
                logger.info('[%s] del talker listener and talker inputs',
                            request_id)
                del self.talker_listener[request_id]
                del self.talker_inputs[request_id]

        self.talker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.talker_listener[request_id].start()

    def _listen_code2wav(self, code2wav_engine_client_id):
        while True:
            try:
                request_id, audio, output_tokens = (
                    self._code2wav_engine_clients[
                        code2wav_engine_client_id
                    ].get_chunk()
                )
                if audio is None:
                    logger.info('[%s] Code2Wav request finished', request_id)
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(None)
                        self.output_queue.pop(request_id)
                else:
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(
                            (audio, output_tokens))
            except Exception:
                logger.exception('Error in code2wav listener')

    def _thinker_to_talker_loop(
        self,
        thinker_outputs: queue.Queue[tuple[str, str, RequestOutput]],
    ):
        while True:
            try:
                request_id, voice_type, output = thinker_outputs.get()
                self._thinker_to_talker(request_id, voice_type, output)
            except Exception:
                logger.exception("Error in thinker to talker loop")

    def _thinker_to_talker(
        self,
        request_id: str,
        voice_type: str,
        output: RequestOutput,
    ):
        if output is None:
            logger.info("thinker finish")
            if request_id in self.thinker_prompt_embeds:
                self.thinker_prompt_embeds.pop(request_id)
            # cancel talker
            asyncio.run_coroutine_threadsafe(
                self.talker_engine.abort(request_id),
                self._loop,
            ).result()
            if request_id in self.talker_requests:
                self.talker_requests.pop(request_id)
            if request_id in self.talker_inputs:
                self.talker_inputs.pop(request_id)
            with suppress_output_queue_exception():
                self.output_queue[request_id].put(None)
                self.output_queue.pop(request_id)
            return

        config = self.talker_config
        if hasattr(config, "talker_config"):
            talker_hf_config = config.talker_config
        else:
            talker_hf_config = config

        if output.outputs[0].prompt_embeds is not None:
            output_prompt_embeds = output.outputs[0].prompt_embeds.to(self.device)

        if len(output.outputs[0].token_ids) == 1:
            if request_id not in self.talker_requests:
                return
            # prepare talker input ids
            talker_input_ids_with_hash = []
            prompt, sampling_params, subtalker_sampling_params = (
                self.talker_requests.pop(request_id)
            )
            if not envs.VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE:
                if output.prompt is not None:
                    prompt_token_ids = self.thinker_tokenizer.encode(output.prompt)
                elif isinstance(prompt, str):
                    prompt_token_ids = self.thinker_tokenizer.encode(prompt)
                else:
                    if "prompt_token_ids" in prompt:
                        prompt_token_ids = prompt["prompt_token_ids"]
                    else:
                        prompt_token_ids = self.thinker_tokenizer.encode(
                            prompt["prompt"]
                        )
            else:
                # in this stage, the prompt ids is rearange to
                # <im_start><audio_start><audio1><audio2><audio_end><im_end>
                # the following codes only use token ids to create
                # talker input without history and assistant prompts.
                prompt_token_ids = output.prompt_token_ids

            # append thinker output token[0]
            prompt_token_ids += [
                output.outputs[0].token_ids[0],
            ]

            # prepare talker input embeds
            # TODO: maybe we can use thinker prompt embeds instead of recompute.
            prompt_embeds = self.thinker_embedding(
                torch.tensor(prompt_token_ids, device=self.device)
            )
            im_start_indexes = torch.cat(
                (
                    torch.nonzero(
                        torch.tensor(prompt_token_ids) == config.im_start_token_id
                    ).squeeze(),
                    torch.tensor([len(prompt_token_ids)]),
                ),
                dim=-1,
            )
            logger.info("start index %r", im_start_indexes)
            if not envs.VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE:
                # use padding to create mask
                # Shape [n_starts];
                # Take batch 0 since batched inference is not supported here.
                multimodal_mask = (
                    (
                        torch.tensor(prompt_token_ids)
                        == config.thinker_config.audio_token_index
                    )
                    | (
                        torch.tensor(prompt_token_ids)
                        == config.thinker_config.image_token_index
                    )
                    | (
                        torch.tensor(prompt_token_ids)
                        == config.thinker_config.video_token_index
                    )
                ).squeeze()
            else:
                # use audio/vision/image start/end to generated mm mask
                multimodal_mask = [False] * len(prompt_token_ids)
                token_p = 0
                while token_p < len(prompt_token_ids):
                    tmp_token_id = prompt_token_ids[token_p]
                    if tmp_token_id == config.thinker_config.vision_start_token_id:
                        multimodal_mask[token_p] = False
                        if (
                            token_p + 1 < len(prompt_token_ids)
                            and prompt_token_ids[token_p + 1]
                            == config.thinker_config.audio_start_token_id
                        ):
                            # <vision_start><audio_start>...<audio_end><vision_end>
                            # token_p to audio_start
                            token_p += 1
                            remain_token_length = len(prompt_token_ids) - token_p - 1
                            for i in range(remain_token_length):
                                token_p += 1
                                if (
                                    prompt_token_ids[token_p]
                                    == config.thinker_config.audio_end_token_id
                                ):
                                    # cur token is audio_end, token += 2,
                                    # skip audio end and vision end and break
                                    token_p += 2
                                    break
                                multimodal_mask[token_p] = True
                        else:
                            # <vision_start>...<vision_end>
                            # token_p is vision_start
                            remain_token_length = len(prompt_token_ids) - token_p - 1
                            for i in range(remain_token_length):
                                token_p += 1
                                if (
                                    prompt_token_ids[token_p]
                                    == config.thinker_config.vision_end_token_id
                                ):
                                    # cur token is vision_end, skip vision end and break
                                    token_p += 1
                                    break
                                multimodal_mask[token_p] = True
                        continue
                    elif tmp_token_id == config.thinker_config.audio_start_token_id:
                        # <audio_start>...<audio_end>
                        # token_p is audio_start
                        remain_token_length = len(prompt_token_ids) - token_p - 1
                        for i in range(remain_token_length):
                            token_p += 1
                            if (
                                prompt_token_ids[token_p]
                                == config.thinker_config.audio_end_token_id
                            ):
                                # cur token is audio_end, skip audio end and break
                                token_p += 1
                                break
                            multimodal_mask[token_p] = True
                        continue
                    else:
                        token_p += 1
                multimodal_mask = torch.tensor(multimodal_mask)
            talker_input_parts = []
            for i in range(len(im_start_indexes) - 1):
                im_start_index = im_start_indexes[i]
                role_token = prompt_token_ids[im_start_index + 1]
                if role_token == config.system_token_id:
                    continue
                elif role_token == config.user_token_id:
                    segment_end_index = im_start_indexes[i + 1]
                    user_talker_part = torch.empty(
                        (
                            segment_end_index - im_start_index,
                            talker_hf_config.hidden_size,
                        ),
                        device=self.device,
                        dtype=torch.bfloat16,
                    )

                    user_mm_mask = multimodal_mask[im_start_index:segment_end_index]

                    # Multimodal data exists
                    if user_mm_mask.any():
                        user_thinker_hidden_mm = output.accept_hidden_states[
                            im_start_index:segment_end_index
                        ][user_mm_mask].to('cuda')
                        mm_hidden = self.hidden_projection(user_thinker_hidden_mm)
                        user_talker_part[user_mm_mask] = mm_hidden
                    user_thinker_embed = prompt_embeds[
                        im_start_index:segment_end_index
                    ][~user_mm_mask]
                    user_text_hidden = self.text_projection(user_thinker_embed)
                    user_talker_part[~user_mm_mask] = user_text_hidden
                    talker_input_parts.append(user_talker_part)
                    talker_input_ids_with_hash += prompt_token_ids[
                        im_start_index:segment_end_index
                    ]
                elif (
                    role_token == config.assistant_token_id
                    and i == len(im_start_indexes) - 2
                ):
                    assistant_hidden = self.text_projection(
                        prompt_embeds[im_start_index:]
                    )  # [t d]
                    assistant_text_hidden = torch.cat(
                        (
                            assistant_hidden[:3],
                            self.tts_pad_embed.expand(4, -1),
                            self.tts_bos_embed,
                            assistant_hidden[3:4],  # First token
                        ),
                        dim=0,
                    )
                    if self._ignore_code2wav(voice_type):
                        spk_id = next(iter(talker_hf_config.spk_id.values()))
                    else:
                        spk_id = talker_hf_config.spk_id[voice_type]
                    assistant_codec_hidden = torch.cat(
                        (
                            torch.zeros(
                                (3, talker_hf_config.hidden_size),
                                device=self.device,
                                dtype=torch.long,
                            ),
                            self.talker_embedding(
                                torch.tensor(
                                    [
                                        talker_hf_config.codec_nothink_id,
                                        talker_hf_config.codec_think_bos_id,
                                        talker_hf_config.codec_think_eos_id,
                                        spk_id,
                                        talker_hf_config.codec_pad_id,
                                        talker_hf_config.codec_bos_id,
                                    ],
                                    device=self.device,
                                    dtype=torch.long,
                                )
                            ),
                        ),
                        dim=0,
                    )
                    talker_input_parts.append(
                        assistant_text_hidden + assistant_codec_hidden
                    )
                    talker_input_ids_with_hash += [
                        config.tts_pad_token_id
                    ] * assistant_text_hidden.shape[0]
                # History assistant output (ignore for now)
                elif (
                    role_token == config.assistant_token_id
                    and i != len(im_start_indexes) - 2
                ):
                    continue
                else:
                    raise AssertionError("Unreachable")
            # prune history
            talker_input_ids_with_hash, prompt_embeds = self._prune_talker_input(
                talker_input_ids_with_hash, prompt_embeds
            )
            prompt_embeds = torch.cat(talker_input_parts, dim=0).detach().cpu()
            generator = self.talker_engine.generate(
                request_id=output.request_id,
                prompt=TokensPrompt(
                    prompt_token_ids=talker_input_ids_with_hash,
                    prompt_embeds=prompt_embeds,
                    # multi_modal_data=prompt.get('multi_modal_data', None),
                    multi_modal_data=None,
                    mm_processor_kwargs=prompt.get("mm_processor_kwargs", None),
                ),
                sampling_params=sampling_params,
                # subtalker_sampling_params=subtalker_sampling_params,
                using_external_data=not bool(output.finished),
            )
            self.talker_inputs[output.request_id] = queue.Queue()
            self._listen_talker(
                output.request_id,
                voice_type,
                generator,
                self.output_queue[output.request_id],
                self.talker_inputs[output.request_id],
            )
        elif not output.finished:
            prompt_embeds = self.thinker_embedding(
                torch.tensor([output.outputs[0].token_ids[-1]], device=self.device)
            )
            output_prompt_embeds = self.text_projection(prompt_embeds)
            self._add_external_data(
                request_id=output.request_id,
                prompt_embeds=output_prompt_embeds,
                is_padding=False,
            )
        else:
            self._add_external_data(
                request_id=output.request_id,
                prompt_embeds=output_prompt_embeds,
                is_padding=False,
            )

            pad_prompt_embeds = self.tts_pad_embed.clone()
            self._add_external_data(
                request_id=output.request_id,
                prompt_embeds=pad_prompt_embeds,
                is_padding=True,
            )

    def _add_external_data(
        self,
        request_id: str,
        prompt_embeds: torch.Tensor,
        is_padding: bool,
    ):
        embeds_tensor = prompt_embeds.detach().cpu()
        external_data = ExternalData(
            request_id=request_id,
            external_embeds=embeds_tensor,
            is_padding=is_padding,
        )
        asyncio.run_coroutine_threadsafe(
            self.talker_engine.update_request_with_data(external_data),
            self._loop,
        ).result()

    def _find_first_im_start_user(
        self,
        prompt_token_ids: List[int],
    ):
        for i in range(len(prompt_token_ids) - 1):
            if (
                prompt_token_ids[i] == self.im_start_user_tokens[0]
                and prompt_token_ids[i + 1] == self.im_start_user_tokens[1]
            ):
                return i
        return -1

    def _find_last_im_start_user(
        self,
        prompt_token_ids: List[int],
    ):
        for i in range(len(prompt_token_ids) - 1, 1, -1):
            if (
                prompt_token_ids[i] == self.im_start_user_tokens[1]
                and prompt_token_ids[i - 1] == self.im_start_user_tokens[0]
            ):
                return i - 1
        return -1

    def _prune_talker_input(self, prompt_token_ids: list, prompt_embeds: torch.Tensor):
        # prune thinker history prompt embeds and tokens before send to talker

        first_user_start = self._find_first_im_start_user(prompt_token_ids)
        last_user_start = self._find_last_im_start_user(prompt_token_ids)
        if (
            first_user_start == -1
            or last_user_start == -1
            or first_user_start == last_user_start
        ):
            return prompt_token_ids, prompt_embeds

        prompt_embeds = torch.cat(
            [
                prompt_embeds[:first_user_start],
                prompt_embeds[last_user_start:],
            ],
            dim=0,
        )
        prompt_token_ids = (
            prompt_token_ids[:first_user_start] + prompt_token_ids[last_user_start:]
        )
        return prompt_token_ids, prompt_embeds

    def _talker_to_code2wav(
        self,
        request_id: str,
        valid_chunk: int,
        codec_ids_list: List[torch.Tensor],
        total_codec_num: int,
        flushed: bool,
        finished: bool,
    ):
        if finished:
            logger.info(
                "[%s] Talker request finished with %s tokens",
                request_id,
                len(codec_ids_list),
            )
        if len(codec_ids_list) < self.codec_chunk_size + self.codec_right_context:
            return
        code2wav_engine_id = 0
        if self.code2wav_data_parallelism > 0:
            request_hash = hash(request_id) if len(request_id) > 1 else ord(request_id)
            code2wav_engine_id = request_hash % self.code2wav_data_parallelism
        asyncio.run_coroutine_threadsafe(
            self._code2wav_engine_clients[code2wav_engine_id].process_chunk(
                request_id=request_id,
                valid_chunk=valid_chunk,
                codec_list=codec_ids_list,
                total_codec_num=total_codec_num,
                flushed=flushed,
                finished=finished,
            ),
            self._loop,
        ).result()

    def print_gpu_memory(self, tag: str, request_id: str, device=0):
        # 获取当前设备的显存信息
        allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MiB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # MiB
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MiB
        used_percent = (allocated / total) * 100  # 已用百分比

        logger.info(
            "[%s] %s GPU usage in MiB allocated %s "
            "max_allocated %s total %s used_percent %s",
            request_id,
            tag,
            allocated,
            max_allocated,
            total,
            used_percent,
        )

    def add_request(
        self,
        request_id: str,
        prompt: Optional[PromptType] = None,
        params: Optional[Union[SamplingParams, PoolingParams]] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        talker_params: Optional[Union[SamplingParams, PoolingParams]] = None,
        subtalker_params: Optional[SamplingParams] = None,
        *,
        talker_type: str = "default",
        voice_type: str = "default",
    ) -> queue.Queue[Union[RequestOutput, np.ndarray]]:
        # normalize voice_type
        if voice_type:
            voice_type = self._get_voice_code(voice_type)

        if (
            self.talker_engine_args
            and voice_type
            and (not self._ignore_voice(voice_type))
            and self.code2wav_voice_types
            and voice_type not in self.code2wav_voice_types
        ):
            raise ValueError(
                f"Voice type '{voice_type}' not found in available voice types: "
                f"{self.code2wav_voice_types}"
            )

        if voice_type == "default":
            voice_type = self.default_tts_text_spk_type

        if not params:
            params = SamplingParams(
                top_k=self.thinker_generation_config.top_k,
                top_p=self.thinker_generation_config.top_p,
                temperature=self.thinker_generation_config.temperature,
                repetition_penalty=self.thinker_generation_config.repetition_penalty,
                stop_token_ids=self.thinker_config.eos_token_id,
            )
        elif (
            VLLM_ENFORCE_GENERATE_CONFIG and self.thinker_generation_config is not None
        ):
            # 安全检查 thinker_generation_config 是否存在且包含所需字段
            if hasattr(self.thinker_generation_config, "top_k"):
                params.top_k = self.thinker_generation_config.top_k
            if hasattr(self.thinker_generation_config, "top_p"):
                params.top_p = self.thinker_generation_config.top_p
            if hasattr(self.thinker_generation_config, "temperature"):
                params.temperature = self.thinker_generation_config.temperature
            if hasattr(self.thinker_generation_config, "repetition_penalty"):
                params.repetition_penalty = (
                    self.thinker_generation_config.repetition_penalty
                )
            if hasattr(self.thinker_generation_config, "max_new_tokens"):
                params.max_tokens = self.thinker_generation_config.max_new_tokens

        logger.info(
            "[SAMPLING_PARAMS] Request %s - VLLM_ENFORCE_GENERATE_CONFIG: %s "
            "params: %s",
            request_id,
            VLLM_ENFORCE_GENERATE_CONFIG,
            params,
        )

        self.print_gpu_memory("add_request", request_id, 0)
        logger.info(
            "thinker_outputs.size %s thinker_listener.size %s",
            len(self.thinker_outputs),
            len(self.thinker_listener),
        )

        async def run_output_handler_and_add_request():
            self.thinker_engine._run_output_handler()
            return await self.thinker_engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
            )

        generator = asyncio.run_coroutine_threadsafe(
            run_output_handler_and_add_request(),
            self._loop,
        ).result()

        self._listen_thinker(
            request_id, voice_type, generator, self.thinker_outputs[talker_type]
        )

        if self.talker_engine_args and not self._ignore_voice(voice_type):
            talker_hf_config = self.talker_config
            if hasattr(talker_hf_config, "talker_config"):
                talker_hf_config = talker_hf_config.talker_config

            if not talker_params:
                talker_params = SamplingParams(
                    top_k=self.talker_generation_config.top_k,
                    top_p=self.talker_generation_config.top_p,
                    temperature=self.talker_generation_config.temperature,
                    repetition_penalty=self.talker_generation_config.repetition_penalty,
                    max_tokens=8192,
                )
            if not subtalker_params:
                subtalker_params = SamplingParams(
                    top_k=self.talker_generation_config.top_k,
                    top_p=self.talker_generation_config.top_p,
                    temperature=self.talker_generation_config.temperature,
                    repetition_penalty=self.talker_generation_config.repetition_penalty,
                )
            if not talker_params.stop_token_ids:
                talker_params.stop_token_ids = []
            talker_params.stop_token_ids.extend(
                [
                    talker_hf_config.codec_eos_token_id,
                ]
            )

            # talker: chunked output to reduce overhead of zmq
            talker_params.chunked_return = False
            talker_params.chunked_return_first_size = (
                self.codec_chunk_size + self.codec_right_context
            )
            talker_params.chunked_return_size = self.codec_chunk_size

            if params and params.max_tokens:
                factor = 1 if self._ignore_code2wav(voice_type) else 40
                talker_params.max_tokens = min(
                    talker_params.max_tokens, params.max_tokens * factor
                )

            self.talker_requests[request_id] = (prompt, talker_params, subtalker_params)

        return self.output_queue[request_id]

    def stop_repetition(self, request_id: str, talker_type: str = "default"):
        # cancel thinker
        asyncio.run_coroutine_threadsafe(
            self.thinker_engine.abort(request_id),
            self._loop,
        ).result()

        cmp_out = CompletionOutput(0, "", [0], None, None, "stop_repetition")
        fake_output = RequestOutput(
            request_id,
            None,
            [],
            None,
            outputs=[cmp_out],
            finished=True,
        )
        self.thinker_outputs[talker_type].put((request_id, None, fake_output))

    def abort_request(self, request_id: str, talker_type: str = "default"):
        self.print_gpu_memory("abort_request", request_id, 0)
        # cancel thinker
        asyncio.run_coroutine_threadsafe(
            self.thinker_engine.abort(request_id),
            self._loop,
        ).result()

        # Cleanup shared memory for prefill hidden states that may
        # have been written but not yet read
        if self._shared_hidden_buffer is not None:
            shm_key = f"{request_id}_prefill"
            self._shared_hidden_buffer.cleanup_stale(shm_key)

        # # temporarily disable this since code2wave is not supported
        # for code2wav_index in range(len(self._code2wav_engine_clients)):
        #     asyncio.run_coroutine_threadsafe(
        #         self._code2wav_engine_clients[code2wav_index].abort_request(
        #             request_id=request_id,
        #         ),
        #         self._loop,
        #     ).result()
        # # cancel talker
        # self.thinker_outputs[talker_type].put((request_id, None, None))

    def _ignore_voice(self, voice_type: str = "default") -> bool:
        return not voice_type or voice_type == "null" or voice_type == "none"

    def _ignore_code2wav(self, voice_type: str = "default") -> bool:
        return voice_type and voice_type.lower() in ["prefix_caching"]

    def _get_voice_code(self, voice_name) -> str:
        if not voice_name:
            return voice_name

        voice_to_spk_mapping = {
            "cherry": "f245",
            "serena": "f05",
            "ethan": "m02",
            "chelsie": "f030",
            "sunny": "f568_04",
            "jada": "f21",
            "dylan": "m325_75",
            "momo": "f30",
            "rhodes": "m05",
            "vivian": "f943_006",
            "moon": "m15",
            "maia": "f20",
            "kai": "m55",
            "nofish": "m31",
            "bella": "f104_83",
            "jennifer": "f04",
            "ryan": "m36",
            "katerina": "f37",
            "aiden": "m11",
            "bodega": "m10",
            "alek": "m03",
            "dolce": "m04",
            "ivan": "m01",
            "sohee": "f02",
            "one anna": "f3001",
            "lenn": "m06",
            "inna": "f39",
            "sonrisa": "f3002",
            "emilien": "m1001",
            "andre": "m40",
            "radio gol": "m034_23",
            "li": "m680",
            "marcus": "m987",
            "roy": "m534_02",
            "peter": "m952",
            "christy": "fjm770",
            "eric": "m0002",
            "rocky": "jm555",
            "kiki": "jm774",
            "eldric sage": "br_m003",
            "mia": "br_f001",
            "mochi": "br_m002",
            "bellona": "br_f004",
            "vincent": "br_m005",
            "bunny": "br_f021",
            "neil": "br_m017",
            "elias": "br_f019",
            "arthur": "br_m024",
            "nini": "br_f008",
            "ebona": "br_f026",
            "seren": "br_f016",
            "pip": "br_m054",
            "stella": "br_f056",
        }

        # support case insensitive
        voice_name_lower = voice_name.lower().strip()

        if voice_name_lower in voice_to_spk_mapping:
            return voice_to_spk_mapping[voice_name_lower]

        # support zeroshot and sft voice name
        return voice_name
