#!/usr/bin/env python3
import asyncio
import collections
import concurrent.futures
import contextlib
import glob
import logging
import os
import queue
import time
from threading import Thread
from typing import Any, AsyncGenerator, Generator, Mapping, Optional, Union

import numpy as np
import torch

from vllm.config import LoadConfig, ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.shared_hidden_buffer import SharedHiddenBuffer
from vllm import envs
from vllm.envs import VLLM_ENFORCE_GENERATE_CONFIG
from vllm.inputs import PromptType
from vllm.inputs.data import EmbedsPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config, try_get_generation_config
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, ExternalData
from vllm.v1.engine.async_llm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine
from vllm.v1.engine.output_processor import RequestOutputCollector

POLLING_TIMEOUT_MS = 10000

logger = init_logger(__name__)
if os.getenv("LOG_LEVEL") == "INFO":
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.WARNING)


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
                return asyncio.run_coroutine_threadsafe(
                    self._collector.get(),
                    self._loop,
                ).result()
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


class OmniDRCA2Engine:
    def __init__(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: Union[
            Mapping[str, AsyncEngineArgs], AsyncEngineArgs
        ] = None,
        thinker_visible_devices: Optional[list[int]] = None,
        talker_visible_devices: Optional[list[int]] = None,
        time_profile: bool = False,
    ):
        self.thinker_engine_args = thinker_engine_args
        if talker_engine_args is None:
            talker_engine_args = {}
        if not isinstance(talker_engine_args, dict):
            talker_engine_args = {
                "default": talker_engine_args,
            }
        self.talker_engine_args = talker_engine_args
        self.has_talker = bool(self.talker_engine_args)

        self.thinker_visible_devices = thinker_visible_devices
        self.talker_visible_devices = talker_visible_devices

        if self.thinker_visible_devices and not isinstance(
            self.thinker_visible_devices, list
        ):
            self.thinker_visible_devices = [self.thinker_visible_devices]
        if self.talker_visible_devices and not isinstance(
            self.talker_visible_devices, list
        ):
            self.talker_visible_devices = [self.talker_visible_devices]

        if self.has_talker:
            # Enable hidden state caching for the thinker engine
            # only when talker is present
            self.thinker_engine_args.enable_hidden_state_caching = True
            os.environ["VLLM_OMNI_ENABLE_TALKER"] = "true"
            os.environ["VLLM_HIDDEN_BUFFER_FAST_TRANSFER"] = "1"

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
        # Initialize SharedMemory hidden buffer for prefill hidden states transfer
        # Calculate shm size based on service configuration:
        # max_num_seqs * max_model_len * hidden_size * dtype_size * eviction_factor
        if self.has_talker:
            # Get hidden_size from thinker config
            # For Qwen3 Omni models, hidden_size is in text_config
            thinker_hidden_size = None
            if hasattr(self.thinker_config, "text_config") and hasattr(
                self.thinker_config.text_config, "hidden_size"
            ):
                thinker_hidden_size = self.thinker_config.text_config.hidden_size
            elif hasattr(self.thinker_config, "hidden_size"):
                thinker_hidden_size = self.thinker_config.hidden_size

            # Get max_num_seqs and max_model_len from engine_args
            max_num_seqs = getattr(thinker_engine_args, "max_num_seqs", None)
            max_model_len = getattr(thinker_engine_args, "max_model_len", None)

            logger.info(
                "Creating SharedHiddenBuffer with: max_num_seqs=%s, "
                "max_model_len=%s, hidden_size=%s",
                max_num_seqs,
                max_model_len,
                thinker_hidden_size,
            )

            self._shared_hidden_buffer = SharedHiddenBuffer.create_global(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                hidden_size=thinker_hidden_size,
            )
        else:
            self._shared_hidden_buffer = None
            logger.info(
                "Thinker-only mode: skipping shm and hidden buffer "
                "initialization (no talker args provided)"
            )
        # thinker: async engine
        self.thinker_engine = self._start_thinker_engine(
            thinker_engine_args, self.thinker_visible_devices
        )
        self.thinker_tokenizer = get_tokenizer(
            thinker_engine_args.tokenizer or thinker_engine_args.model,
            tokenizer_mode=thinker_engine_args.tokenizer_mode,
            trust_remote_code=thinker_engine_args.trust_remote_code,
            revision=thinker_engine_args.revision,
            download_dir=thinker_engine_args.download_dir,
        )

        # self.device = self.thinker_engine.engine.device_config.device
        # thinker_vllm_config = asyncio.run(
        #     self.get_vllm_config_async(self.thinker_engine)
        # )
        # self.device = thinker_vllm_config.device_config.device
        self.device = "cuda"

        # thinker state
        # request_id -> generate_speech
        self.thinker_generate_speech: dict[str, bool] = dict()
        # request_id -> total output token length
        self.thinker_output_len: dict[str, int] = dict()
        # request_id -> listener thread
        self.thinker_listener: dict[str, Thread] = dict()
        self.request_prompts: dict[str, PromptType] = dict()
        self.cosyvoice_talker_hidden: dict[str, torch.Tensor] = dict()
        self.cosyvoice_talker_hidden_cursor: dict[str, int] = dict()
        self.cosyvoice_talker_debug_step: dict[str, int] = dict()
        self.cosyvoice_talker_fm_hidden: dict[str, list[torch.Tensor]] = dict()
        self.cosyvoice_completed_fm_hidden: dict[str, torch.Tensor] = dict()
        #  (request_id, thinker request_output)
        self.thinker_outputs: queue.Queue[tuple[str, RequestOutput]] = queue.Queue()

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

            # Adapt the qwen_omnidrca2_model
            engine_args.hf_overrides = {
                "architectures": ["OmniDRCA2TalkerModel"],
                "text_config": {"vocab_size": 6565},
            }

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
            if self._is_cosyvoice_codex_talker_config(self.talker_config):
                hidden_size = int(
                    getattr(self.talker_config.audio_config, "hidden_size", 896)
                )
                self.embed_talker_bos_token = torch.zeros(
                    1, hidden_size, dtype=torch.float32, device="cpu"
                )
            else:
                self._init_omni3_special_tokens_embeddings(
                    thinker_engine_args, default_talker_args
                )

        # talker request state
        self.talker_requests: dict[str, SamplingParams] = dict()
        self.talker_listener: dict[str, Thread] = dict()
        self.talker_speech_ids: dict[str, list[int]] = dict()
        # request id -> [union(request_output, (speech_token_ids))]
        self.output_queue: dict[str, queue.Queue[Union[RequestOutput, list[int]]]] = (
            collections.defaultdict(queue.Queue)
        )
        # request id -> [float]
        self.time_profile = time_profile
        self.output_profile_time: dict[str, list[float]] = dict()

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_thinker_engine(
        self,
        engine_args: AsyncEngineArgs,
        visible_devices: Optional[list[int]] = None,
    ) -> AsyncLLMEngine:
        if visible_devices:
            original_devices = os.environ.get("VLLM_CUDA_VISIBLE_DEVICES", None)
            if original_devices is not None:
                original_devices = original_devices.split(",")
                value = ",".join(original_devices[device] for device in visible_devices)
            else:
                value = ",".join(str(device) for device in visible_devices)
            device_ctx = envvars("VLLM_CUDA_VISIBLE_DEVICES", value)
        else:
            device_ctx = contextlib.nullcontext()
        if engine_args.quantization is not None:
            os.environ["VLLM_QUANTIZATION_LAYER_WISE"] = "1"

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
            original_devices = os.environ.get("VLLM_CUDA_VISIBLE_DEVICES", None)
            if original_devices is not None:
                original_devices = original_devices.split(",")
                value = ",".join(original_devices[device] for device in visible_devices)
            else:
                value = ",".join(str(device) for device in visible_devices)
            device_ctx = envvars("VLLM_CUDA_VISIBLE_DEVICES", value)
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
            args=(self.thinker_outputs,),
            daemon=True,
        )
        thread.start()

        return talker_engine, thread

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self._closed = True
        # shutdown thinker engine
        self.thinker_engine.shutdown()

        # shutdown talker engines
        if hasattr(self, "talker_engine"):
            self.talker_engine.shutdown()

    def pop_cosyvoice_fm_hidden(self, request_id: str) -> torch.Tensor | None:
        """Wait for Talker finalization, then return its unified-FM states.

        The codec consumer can observe EOS from ``output_queue`` just before
        the Talker listener stacks and publishes the collected states.  An
        immediate pop therefore races with listener finalization.
        """
        listener = self.talker_listener.get(request_id)
        if listener is not None:
            listener.join(timeout=30.0)
            if listener.is_alive():
                logger.error(
                    "[%s] timed out waiting for Talker hidden finalization",
                    request_id,
                )
        return self.cosyvoice_completed_fm_hidden.pop(request_id, None)

    def _load_model_embedding(
        self,
        engine_args: EngineArgs,
        embedding_name: str,
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
                tokenizer=engine_args.tokenizer,
                tokenizer_mode=engine_args.tokenizer_mode,
                processor=engine_args.processor,
                trust_remote_code=engine_args.trust_remote_code,
                dtype=engine_args.dtype,
                seed=engine_args.seed,
            ),
            model=None,
        ):
            if key in [
                embedding_name,
            ]:
                logger.debug("get embedding %s value %s", key, value)
                return torch.nn.Embedding.from_pretrained(value).to(self.device)
        raise ValueError(
            "Embedding weight 'model.embed.weight' not found in model weights."
        )

    @staticmethod
    def _is_cosyvoice_codex_talker_config(config) -> bool:
        cosyvoice_codex = getattr(config, "cosyvoice_codex", None)
        if isinstance(cosyvoice_codex, dict):
            return bool(cosyvoice_codex.get("enabled", False))
        return bool(getattr(cosyvoice_codex, "enabled", False))

    @staticmethod
    def _cosyvoice_codex_cfg_get(config, key: str, default=None):
        cosyvoice_codex = getattr(config, "cosyvoice_codex", None)
        if isinstance(cosyvoice_codex, dict):
            return cosyvoice_codex.get(key, default)
        return getattr(cosyvoice_codex, key, default)

    def _get_talker_hf_config(self):
        config = self.talker_config
        if self._is_cosyvoice_codex_talker_config(config):
            if not hasattr(config, "group_size"):
                config.group_size = int(
                    os.environ.get("OMNIDRCA2_TALKER_GROUP_SIZE", "1")
                )
            if not hasattr(config, "hidden_size"):
                audio_config = getattr(config, "audio_config", None)
                config.hidden_size = int(
                    self._cosyvoice_codex_cfg_get(
                        config,
                        "llm_hidden_size",
                        getattr(audio_config, "hidden_size", 896),
                    )
                )
            if not hasattr(config, "codec_unit"):
                config.codec_unit = int(
                    self._cosyvoice_codex_cfg_get(
                        config,
                        "codec_unit",
                        self._cosyvoice_codex_cfg_get(
                            config, "codec_vocab_size", 0
                        ),
                    )
                )
            if not hasattr(config, "bos_token_id"):
                config.bos_token_id = 0
            if not hasattr(config, "eos_token_id"):
                config.eos_token_id = int(os.environ.get("CODEX_CODEC_EOS_TOKEN_ID", "59050"))
            return config
        if hasattr(config, "audio_config"):
            return config.audio_config
        return config

    def _init_omni3_special_tokens_embeddings(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: AsyncEngineArgs,
    ):
        # thinker and talker embeddings
        self.thinker_embedding = self._load_model_embedding(
            thinker_engine_args, "language_model.model.embed_tokens.weight"
        )

        self.talker_embedding = self._load_model_embedding(
            talker_engine_args, "audio_tower.embed_tokens.weight"
        )

        talker_hf_config = self._get_talker_hf_config()
        self.embed_talker_bos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.bos_token_id],
                dtype=torch.long,
                device=self.device,
            )
        ).cpu()

    def _listen_thinker(
        self,
        request_id: str,
        generator: AsyncGenerator[RequestOutput, None],
        thinker_outputs: queue.Queue[tuple[str, RequestOutput]],
    ):
        def _listen():
            last_output = None
            total_text_token_num = 0
            for output in SynchronizedGenerator(generator, self._loop):
                # may get same RequestOutput, so skip
                if total_text_token_num == len(output.outputs[0].token_ids):
                    if output.finished:
                        logger.info("[%s] got same token num, maybe aborted",
                                    request_id)
                        break
                    logger.warning(
                        "[%s] Duplicate RequestOutput in listen_thinker. Skipping.",
                        request_id,
                    )
                    continue
                if (self.has_talker
                    and (total_text_token_num + 1 < len(output.outputs[0].token_ids))
                ):
                    logger.error(
                        "[%s] skip too much RequestOutput!"
                        "It will effect text output len.",
                        request_id,
                    )

                is_first_thinker_output = last_output is None
                total_text_token_num = len(output.outputs[0].token_ids)
                if is_first_thinker_output:
                    logger.info(
                        "rid: %s got first token, text: %s, token_id: %s",
                        request_id,
                        output.outputs[0].text,
                        output.outputs[0].token_ids,
                    )
                    if self.time_profile:
                        self.output_profile_time[request_id].append(time.perf_counter())
                last_output = output
                with suppress_output_queue_exception():
                    self.output_queue[request_id].put(output)
                if (
                    self.has_talker
                    and not self._is_cosyvoice_codex_talker_config(self.talker_config)
                ):
                    token_ids_for_talker = output.outputs[0].token_ids
                    if is_first_thinker_output:
                        # The async collector can coalesce multiple generated
                        # tokens before this listener is scheduled. The first
                        # talker event must still use the prefill hidden states
                        # and must look like the first token event to start the
                        # talker request.
                        token_ids_for_talker = output.outputs[0].token_ids[:1]
                        # Prefill hidden states - read once when first token is received
                        shm_key = f"{request_id}_prefill"
                        token_hidden_states = (
                            SharedHiddenBuffer.read_global_by_request_id(
                                request_id=shm_key,
                                device="cpu",
                                wait=True,
                                timeout=30.0,
                            )
                        )
                        if token_hidden_states is None:
                            logger.warning(
                                "[%s] Failed to read prefill hidden_states", request_id
                            )
                        # add request generate_speech flag
                        audio_bos_index = getattr(
                            self.thinker_config.text_config,
                            "audio_bos_index",
                            None,
                        )
                        self.thinker_generate_speech[request_id] = (
                            True
                            if audio_bos_index is None
                            and self._is_cosyvoice_codex_talker_config(
                                self.talker_config
                            )
                            else output.outputs[0].token_ids[0] == audio_bos_index
                        )
                    elif len(output.outputs[0].token_ids) > 1:
                        # Decode hidden states
                        shm_key = f"{request_id}_decode"
                        token_hidden_states = (
                            SharedHiddenBuffer.read_global_by_request_id(
                                request_id=shm_key,
                                device="cpu",
                                wait=True,
                                timeout=30.0,
                            )
                        )
                        if token_hidden_states is None:
                            logger.warning(
                                "[%s] Failed to read decode hidden_states", request_id
                            )
                        # update request generate_speech flag
                        audio_bos_index = getattr(
                            self.thinker_config.text_config,
                            "audio_bos_index",
                            None,
                        )
                        self.thinker_generate_speech[request_id] |= (
                            True
                            if audio_bos_index is None
                            and self._is_cosyvoice_codex_talker_config(
                                self.talker_config
                            )
                            else output.outputs[0].token_ids[-1] == audio_bos_index
                        )
                    else:
                        logger.warning(
                            "[%s] Failed to generate any decode tokens", request_id
                        )
                    if token_hidden_states is None:
                        logger.warning(
                            "[%s] Skip talker update because hidden_states is missing",
                            request_id,
                        )
                        continue
                    thinker_output = RequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",  # Text is not needed for talker
                                token_ids=token_ids_for_talker,
                                cumulative_logprob=None,
                                logprobs=None,
                            )
                        ],
                        finished=False,  # Will be set properly at the end
                        accept_hidden_states=token_hidden_states,
                    )
                    thinker_outputs.put((request_id, thinker_output))
                    logger.info(
                        "[%s] listen thinker to talker output: %s,"
                        " thinker output token number: %s",
                        request_id,
                        thinker_output.outputs[0].token_ids,
                        len(thinker_output.outputs[0].token_ids),
                    )
                else:
                    logger.info(
                        "[%s] listen thinker output: %s",
                        request_id,
                        output.outputs[0].token_ids,
                    )

                if output.finished:
                    break

            if not self.has_talker:
                with suppress_output_queue_exception():
                    self.output_queue[request_id].put(None)
                    self.output_queue.pop(request_id)
            if last_output is not None:
                if self.has_talker:
                    self.thinker_output_len[request_id] = len(
                        last_output.outputs[0].token_ids
                    )

                logger.info(
                    "[%s] Thinker request finished, reason: (%s, %s), "
                    "outputs: %s, output len is: %s, tokens are: %s",
                    request_id,
                    last_output.outputs[0].finish_reason,
                    last_output.outputs[0].stop_reason,
                    repr(last_output.outputs[0].text),
                    len(last_output.outputs[0].token_ids),
                    last_output.outputs[0].token_ids,
                )
            else:
                logger.info("[%s] Thinker request finished, no outputs", request_id)
            if (
                self.has_talker
                and last_output is not None
                and self._is_cosyvoice_codex_talker_config(self.talker_config)
            ):
                try:
                    self._start_cosyvoice_aligned_talker(request_id, last_output)
                except Exception:
                    logger.exception(
                        "[%s] Failed to run CosyVoice aligned talker",
                        request_id,
                    )
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(None)
                        self.output_queue.pop(request_id)
            with contextlib.suppress(Exception):
                del self.thinker_listener[request_id]

        self.thinker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.thinker_listener[request_id].start()

    def _listen_talker(
        self,
        request_id: str,
        generator: AsyncGenerator[RequestOutput, None],
        output_queue: queue.Queue[Union[RequestOutput, list[int]]],
    ):
        talker_hf_config = self._get_talker_hf_config()
        is_cosyvoice_codex = self._is_cosyvoice_codex_talker_config(
            self.talker_config
        )

        def _listen():
            total_speech_token_num = 0
            speech_token_id_list = []
            self.talker_speech_ids[request_id] = []
            # Use the same registered process-level switch as the Talker
            # worker. The nested Talker HF override can drop custom top-level
            # config fields, so it must not be used as the runtime gate.
            if (
                is_cosyvoice_codex
                and envs.VLLM_COSYVOICE_EXPORT_LLM_HIDDEN_FOR_FM
            ):
                self.cosyvoice_talker_fm_hidden[request_id] = []
            last_output = None
            for output in SynchronizedGenerator(generator, self._loop):
                last_output = output
                if output.outputs is None:
                    logger.info("[%s] got None, maybe aborted", request_id)
                    break
                # may get same RequestOutput, so skip
                if total_speech_token_num == len(output.outputs[0].token_ids):
                    logger.warning(
                        "[%s] Duplicate RequestOutput in listen_talker. Skipping.",
                        request_id,
                    )
                    continue

                previous_speech_token_num = total_speech_token_num
                current_speech_token_num = len(output.outputs[0].token_ids)
                if (
                    previous_speech_token_num + talker_hf_config.group_size
                    < current_speech_token_num
                ):
                    logger.info(
                        "[%s] talker listener received coalesced output, "
                        "speech token num: %s -> %s.",
                        request_id,
                        previous_speech_token_num,
                        current_speech_token_num,
                    )
                total_speech_token_num = current_speech_token_num
                max_codec_text_ratio = float(os.environ.get(
                    "CODEX_TALKER_MAX_CODEC_TEXT_RATIO", "16"))
                if (
                    max_codec_text_ratio > 0
                    and request_id in self.thinker_output_len
                    and total_speech_token_num >= int(
                        self.thinker_output_len[request_id]
                        * max_codec_text_ratio
                    )
                ):
                    logger.warning(
                        "[%s] Talker output length exceeds thinker text ratio "
                        "limit: speech_tokens=%s thinker_tokens=%s "
                        "group_size=%s ratio=%.2f. Abort talker request.",
                        request_id,
                        total_speech_token_num,
                        self.thinker_output_len[request_id],
                        talker_hf_config.group_size,
                        max_codec_text_ratio,
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.talker_engine.abort(request_id),
                        self._loop,
                    ).result()
                    break
                if self.time_profile and previous_speech_token_num == 0:
                    self.output_profile_time[request_id].append(time.perf_counter())
                logger.info(
                    "[%s] talker listener got output, speech token num: %s",
                    request_id,
                    total_speech_token_num,
                    )
                last_complete_group_end = (
                    current_speech_token_num // talker_hf_config.group_size
                ) * talker_hf_config.group_size
                next_group_end = (
                    (previous_speech_token_num // talker_hf_config.group_size) + 1
                ) * talker_hf_config.group_size
                while next_group_end <= last_complete_group_end:
                    speech_token_id_list = output.outputs[0].token_ids[
                        next_group_end
                        - talker_hf_config.group_size : next_group_end
                    ]
                    if (
                        is_cosyvoice_codex
                        or next_group_end / talker_hf_config.group_size > 1
                    ):
                        if self.thinker_generate_speech[request_id]:
                            self.talker_speech_ids[request_id].extend(
                                speech_token_id_list
                            )
                            if is_cosyvoice_codex:
                                if request_id in self.cosyvoice_talker_fm_hidden:
                                    step_hidden = SharedHiddenBuffer.read_global_by_request_id(
                                        request_id=f"{request_id}_codex_fm_hidden",
                                        device="cpu",
                                        wait=True,
                                        timeout=30.0,
                                    )
                                    if step_hidden is None:
                                        raise RuntimeError(
                                            f"[{request_id}] missing codec-LLM hidden for FM"
                                        )
                                    self.cosyvoice_talker_fm_hidden[request_id].append(
                                        step_hidden.reshape(-1, step_hidden.shape[-1])[0].clone()
                                    )
                                debug_steps = int(os.environ.get(
                                    "CODEX_DEBUG_STEPS", "0"))
                                if debug_steps > 0:
                                    token_offset = (
                                        next_group_end
                                        - talker_hf_config.group_size
                                    )
                                    step_index = (
                                        token_offset
                                        // talker_hf_config.group_size
                                    )
                                    if step_index < debug_steps:
                                        logger.warning(
                                            "[%s] CODEX_STEP_DEBUG output "
                                            "step=%d token_ids=%s all_len=%d",
                                            request_id,
                                            step_index,
                                            list(speech_token_id_list),
                                            len(self.talker_speech_ids[request_id]),
                                        )
                            with suppress_output_queue_exception():
                                output_queue.put(self.talker_speech_ids[request_id])
                            finish_speech = any(
                                speech_id
                                == getattr(
                                    talker_hf_config,
                                    "eos_token_id",
                                    getattr(talker_hf_config, "codec_unit", -1),
                                )
                                for speech_id in speech_token_id_list
                            )
                            self.thinker_generate_speech[request_id] = not finish_speech
                            if is_cosyvoice_codex and not finish_speech:
                                self._feed_next_cosyvoice_talker_hidden(request_id)

                    next_group_end += talker_hf_config.group_size

                if total_speech_token_num % talker_hf_config.group_size == 0:
                    if (
                        not is_cosyvoice_codex
                        and
                        request_id in self.thinker_output_len
                        and total_speech_token_num / talker_hf_config.group_size
                        == self.thinker_output_len[request_id]
                    ):
                        # engine abort talker request when thinker finished,
                        # not depend on talker request finished flag.
                        logger.info(
                            "[%s] Talker output length equals thinker output len [%s].",
                            request_id,
                            self.thinker_output_len[request_id]
                            * talker_hf_config.group_size,
                        )
                        asyncio.run_coroutine_threadsafe(
                            self.talker_engine.abort(request_id),
                            self._loop,
                        ).result()
                        break

                    logger.info(
                        "[%s] talker to thinker: %s/%s,"
                        " current step generate speech %s.",
                        request_id,
                        total_speech_token_num,
                        talker_hf_config.group_size,
                        speech_token_id_list,
                    )
                    if not is_cosyvoice_codex:
                        self._talker_to_thinker(output.request_id)
    
                if output.finished:
                    logger.info("[%s] Talker finished", request_id)
                    break

            logger.info("[%s] Talker output generator exit", request_id)
            if last_output is None or last_output.outputs is None:
                logger.info("[%s] Talker output None, maybe aborted", request_id)
            else:
                if (
                    not is_cosyvoice_codex
                    and (
                        request_id not in self.thinker_output_len
                        or total_speech_token_num / talker_hf_config.group_size
                        != self.thinker_output_len[request_id]
                    )
                ):
                    logger.error(
                        "[%s] talker sampling parameters are not right,"
                        "talker should not finished when thinker still running",
                        request_id,
                    )
                if total_speech_token_num % talker_hf_config.group_size != 0:
                    logger.error(
                        "[Req %s] Speech token count not divisible by group_size; "
                        "talker ended while thinker still running.",
                        request_id,
                    )

                logger.info(
                    "[%s] Talker request finished, reason: (%s, %s), "
                    "output len is: %s, output tokens are: %s",
                    request_id,
                    last_output.outputs[0].finish_reason,
                    last_output.outputs[0].stop_reason,
                    len(last_output.outputs[0].token_ids),
                    last_output.outputs[0].token_ids,
                )

            try:
                fm_hidden = self.cosyvoice_talker_fm_hidden.pop(request_id, None)
                if fm_hidden:
                    # Drop EOS and any other out-of-codebook token so this is
                    # aligned one-to-one with the filtered codec saved by eval.
                    valid_count = sum(
                        int(token) >= 0
                        and int(token) < int(getattr(talker_hf_config, "codec_unit", -1))
                        for token in self.talker_speech_ids.get(request_id, [])
                    )
                    self.cosyvoice_completed_fm_hidden[request_id] = torch.stack(
                        fm_hidden[:valid_count]
                    )
                elif (
                    is_cosyvoice_codex
                    and envs.VLLM_COSYVOICE_EXPORT_LLM_HIDDEN_FOR_FM
                ):
                    logger.error(
                        "[%s] codec-LLM hidden export was enabled but the "
                        "Talker listener collected no states",
                        request_id,
                    )
                if self.time_profile:
                    time_interval = [
                        (
                            self.output_profile_time[request_id][i]
                            - self.output_profile_time[request_id][i - 1]
                        )
                        * 1000
                        for i in range(1, len(self.output_profile_time[request_id]))
                    ]
                    del self.output_profile_time[request_id]
                    assert len(time_interval) == 3
                    print(
                        f"request {request_id} "
                        f"get first text token costs {time_interval[0]} ms, "
                        f"thinker2talker costs {time_interval[1]} ms, "
                        f"talker prefill costs {time_interval[2]} ms"
                    )
                logger.info("[%s] del talker listener and talker inputs", request_id)
                del self.talker_listener[request_id]
                del self.thinker_output_len[request_id]
                del self.thinker_generate_speech[request_id]
                del self.talker_speech_ids[request_id]
                self.request_prompts.pop(request_id, None)
                self.cosyvoice_talker_hidden.pop(request_id, None)
                self.cosyvoice_talker_hidden_cursor.pop(request_id, None)
                self.cosyvoice_talker_debug_step.pop(request_id, None)
            except Exception:
                logger.exception(
                    "[%s] failed to finalize codec-LLM hidden states for FM",
                    request_id,
                )

            with suppress_output_queue_exception():
                self.output_queue[request_id].put(None)
                self.output_queue.pop(request_id)

        self.talker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.talker_listener[request_id].start()

    def _thinker_to_talker_loop(
        self,
        thinker_outputs: queue.Queue[tuple[str, RequestOutput]],
    ):
        while True:
            try:
                request_id, output = thinker_outputs.get()
                self._thinker_to_talker(request_id, output)
            except Exception:
                logger.exception("Error in thinker to talker loop")

    def _thinker_to_talker(
        self,
        request_id: str,
        output: RequestOutput,
    ):
        if output is None:
            logger.info("thinker finish")
            if request_id in self.thinker_generate_speech:
                self.thinker_generate_speech.pop(request_id)
            if request_id in self.thinker_output_len:
                self.thinker_output_len.pop(request_id)
            # cancel talker
            asyncio.run_coroutine_threadsafe(
                self.talker_engine.abort(request_id),
                self._loop,
            ).result()
            if request_id in self.talker_requests:
                self.talker_requests.pop(request_id)
            if request_id in self.talker_speech_ids:
                self.talker_speech_ids.pop(request_id)
            if request_id in self.output_profile_time:
                self.output_profile_time.pop(request_id)
            self.request_prompts.pop(request_id, None)
            self.cosyvoice_talker_hidden.pop(request_id, None)
            self.cosyvoice_talker_hidden_cursor.pop(request_id, None)
            self.cosyvoice_talker_debug_step.pop(request_id, None)
            with suppress_output_queue_exception():
                self.output_queue[request_id].put(None)
                self.output_queue.pop(request_id)
            return

        talker_hf_config = self._get_talker_hf_config()
        if output.accept_hidden_states is None:
            logger.warning(
                "[%s] Missing accept_hidden_states for talker; "
                "skip this thinker output.",
                request_id,
            )
            return
        talker_input_hidden_states = output.accept_hidden_states
        seq_len = talker_input_hidden_states.size(0)
        talker_input_hidden_states = talker_input_hidden_states.reshape(
            int(seq_len * talker_hf_config.group_size), talker_hf_config.hidden_size
        )
        is_cosyvoice_codex = self._is_cosyvoice_codex_talker_config(self.talker_config)
        if len(output.outputs[0].token_ids) == 1:
            if request_id not in self.talker_requests:
                logger.warning(
                    "[%s] does not in the self.talker_requests",
                    request_id,
                )
                return
            sampling_params = self.talker_requests.pop(request_id)

            for i in range(talker_hf_config.group_size):
                if i == 0:
                    prompt_embeds = talker_input_hidden_states[
                        : seq_len * talker_hf_config.group_size
                        - (talker_hf_config.group_size - 1)
                    ]
                    if not is_cosyvoice_codex:
                        prompt_embeds = prompt_embeds + self.embed_talker_bos_token.to(
                            device=talker_input_hidden_states.device,
                            dtype=talker_input_hidden_states.dtype,
                        )
                    prompt_embeds = self._compact_cpu_tensor(prompt_embeds)
                    prompt_embeds_key = f"{request_id}_talker_prompt"
                    self._write_hidden_to_shm(prompt_embeds, prompt_embeds_key)
                    use_external_data = True
                    talker_prompt = EngineCoreRequest(
                        request_id=output.request_id,
                        prompt_token_ids=[0] * int(prompt_embeds.shape[0]),
                        mm_features=None,
                        sampling_params=sampling_params,
                        pooling_params=None,
                        eos_token_id=None,
                        arrival_time=time.time(),
                        lora_request=None,
                        cache_salt=None,
                        data_parallel_rank=None,
                        prompt_embeds=None,
                        using_external_data=use_external_data,
                        prompt_embeds_shm_key=prompt_embeds_key,
                    )
                    if self.time_profile:
                        self.output_profile_time[request_id].append(time.perf_counter())
                    generator = self.talker_engine.generate(
                        request_id=output.request_id,
                        prompt=talker_prompt,
                        sampling_params=sampling_params,
                        using_external_data=use_external_data,
                    )

                else:
                    prompt_embeds = talker_input_hidden_states[
                        seq_len * talker_hf_config.group_size
                        - (talker_hf_config.group_size - i) : seq_len
                        * talker_hf_config.group_size
                        - (talker_hf_config.group_size - i)
                        + 1
                    ]
                    self._add_talker_external_data(
                        request_id=output.request_id,
                        prompt_embeds=prompt_embeds,
                        is_padding=False,
                        add_token_embedding=True,
                    )
            self._listen_talker(
                output.request_id,
                generator,
                self.output_queue[output.request_id],
            )
        else:
            for i in range(talker_hf_config.group_size):
                if i == 0:
                    prompt_embeds = talker_input_hidden_states[
                        : seq_len * talker_hf_config.group_size
                        - (talker_hf_config.group_size - 1)
                    ]
                else:
                    prompt_embeds = talker_input_hidden_states[
                        seq_len * talker_hf_config.group_size
                        - (talker_hf_config.group_size - i) : seq_len
                        * talker_hf_config.group_size
                        - (talker_hf_config.group_size - i)
                        + 1
                    ]
                self._add_talker_external_data(
                    request_id=output.request_id,
                    prompt_embeds=prompt_embeds,
                    is_padding=False,
                    add_token_embedding=True,
                )

    def _start_cosyvoice_aligned_talker(
        self,
        request_id: str,
        thinker_output: RequestOutput,
    ) -> None:
        if request_id not in self.talker_requests:
            logger.warning("[%s] Missing talker sampling params", request_id)
            return

        sampling_params = self.talker_requests.pop(request_id)
        codec_prefix_ids = [
            int(token_id)
            for token_id in self.cosyvoice_codec_prefix_ids.pop(request_id, [])
        ]
        prompt_token_ids = list(thinker_output.prompt_token_ids or [])
        generated_token_ids = list(thinker_output.outputs[0].token_ids or [])
        generated_token_ids = self._strip_thinker_stop_tokens(generated_token_ids)

        max_codec_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
        if max_codec_tokens <= 0:
            max_codec_tokens = 8192
        blank_count = self._get_cosyvoice_blank_count()
        audio_align_len = blank_count + max_codec_tokens
        target_segment_len = max(len(generated_token_ids), audio_align_len)
        original_prompt = self.request_prompts.get(request_id)
        base_prompt_token_ids = self._get_alignment_base_prompt_token_ids(
            original_prompt,
            prompt_token_ids,
        )
        align_prompt: dict[str, Any]
        use_token_ids = os.environ.get(
            "CODEX_ALIGNMENT_USE_TOKEN_IDS",
            "1",
        ).lower() not in ("0", "false", "no")
        mask_token_id = int(os.environ.get("CODEX_ALIGNMENT_MASK_TOKEN_ID", "-60"))
        if use_token_ids and base_prompt_token_ids:
            insert_blank_prefix = os.environ.get(
                "CODEX_ALIGNMENT_INSERT_BLANK_PREFIX", "0"
            ).lower() in ("1", "true", "yes", "on")
            if insert_blank_prefix:
                # Training prefixes the target codec stream with blank audio
                # frames. Materialize those positions in the Thinker hidden
                # timeline as well; otherwise prompt_frames skips the first
                # `blank_count` generated text states and Talker starts from
                # generated text token blank_count + 1.
                aligned_segment = (
                    [mask_token_id] * blank_count
                    + generated_token_ids
                    + [mask_token_id] * max(
                        0,
                        target_segment_len - blank_count - len(generated_token_ids),
                    )
                )
            else:
                aligned_segment = (
                    generated_token_ids
                    + [mask_token_id]
                    * (target_segment_len - len(generated_token_ids))
                )
            aligned_prompt_token_ids = base_prompt_token_ids + aligned_segment
            align_prompt = {
                "prompt_token_ids": aligned_prompt_token_ids,
            }
            if isinstance(original_prompt, Mapping):
                if original_prompt.get("multi_modal_data") is not None:
                    align_prompt["multi_modal_data"] = original_prompt.get(
                        "multi_modal_data"
                    )
                if original_prompt.get("mm_processor_kwargs") is not None:
                    align_prompt["mm_processor_kwargs"] = original_prompt.get(
                        "mm_processor_kwargs"
                    )
        elif isinstance(original_prompt, Mapping):
            prompt_text = original_prompt.get("prompt")
            if isinstance(prompt_text, str):
                pad_count = target_segment_len - len(generated_token_ids)
                pad_text = os.environ.get(
                    "CODEX_ALIGNMENT_PAD_TEXT",
                    "<|endoftext|>",
                )
                align_prompt = {
                    "prompt": (
                        prompt_text
                        + str(thinker_output.outputs[0].text or "")
                        + pad_text * pad_count
                    ),
                }
            else:
                pad_token_id = self._get_thinker_pad_token_id()
                aligned_prompt_token_ids = (
                    base_prompt_token_ids
                    + generated_token_ids
                    + [pad_token_id] * (target_segment_len - len(generated_token_ids))
                )
                align_prompt = {
                    "prompt_token_ids": aligned_prompt_token_ids,
                }
            if original_prompt.get("multi_modal_data") is not None:
                align_prompt["multi_modal_data"] = original_prompt.get(
                    "multi_modal_data"
                )
            if original_prompt.get("mm_processor_kwargs") is not None:
                align_prompt["mm_processor_kwargs"] = original_prompt.get(
                    "mm_processor_kwargs"
                )
        else:
            pad_token_id = self._get_thinker_pad_token_id()
            aligned_prompt_token_ids = (
                base_prompt_token_ids
                + generated_token_ids
                + [pad_token_id] * (target_segment_len - len(generated_token_ids))
            )
            align_prompt = {
                "prompt_token_ids": aligned_prompt_token_ids,
            }

        align_request_id = f"{request_id}_codec_align"
        align_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=1,
            detokenize=False,
        )
        logger.info(
            "[%s] Build CosyVoice aligned thinker prompt: base=%d text=%d "
            "audio_align=%d blank=%d total=%d token_ids=%s mask=%d",
            request_id,
            len(base_prompt_token_ids),
            len(generated_token_ids),
            audio_align_len,
            blank_count,
            len(base_prompt_token_ids) + target_segment_len,
            bool(use_token_ids and base_prompt_token_ids),
            mask_token_id,
        )

        generator = self.thinker_engine.generate(
            request_id=align_request_id,
            prompt=align_prompt,
            sampling_params=align_params,
            using_external_data=False,
        )

        align_hidden_states = None
        for output in SynchronizedGenerator(generator, self._loop):
            if align_hidden_states is None:
                align_hidden_states = SharedHiddenBuffer.read_global_by_request_id(
                    request_id=f"{align_request_id}_prefill",
                    device="cpu",
                    wait=True,
                    timeout=30.0,
                )
            if output.finished:
                break

        if align_hidden_states is None:
            logger.warning("[%s] Failed to read CosyVoice aligned hidden", request_id)
            return

        prefix_len = max(0, int(align_hidden_states.size(0)) - target_segment_len)
        segment_start = 0
        segment_end = min(prefix_len + audio_align_len, align_hidden_states.size(0))
        if segment_end <= prefix_len:
            logger.warning(
                "[%s] Empty CosyVoice aligned hidden: prefix=%d start=%d end=%d "
                "shape=%s",
                request_id,
                prefix_len,
                segment_start,
                segment_end,
                tuple(align_hidden_states.shape),
            )
            return

        segment_hidden = align_hidden_states[:segment_end]
        prompt_frame_count = min(
            prefix_len + blank_count + len(codec_prefix_ids),
            int(segment_hidden.size(0)),
        )
        logger.info(
            "[%s] CosyVoice aligned hidden slice: shape=%s start=%d end=%d "
            "prefix=%d prompt_frames=%d",
            request_id,
            tuple(segment_hidden.shape),
            segment_start,
            segment_end,
            prefix_len,
            prompt_frame_count,
        )
        debug_steps = int(os.environ.get("CODEX_DEBUG_STEPS", "0"))
        if debug_steps > 0:
            with contextlib.suppress(Exception):
                debug_frame_count = min(
                    int(segment_hidden.size(0)),
                    prefix_len + blank_count + debug_steps,
                )
                debug_hidden = segment_hidden[:debug_frame_count].float()
                debug_norms = torch.linalg.vector_norm(
                    debug_hidden, dim=-1
                ).detach().cpu().tolist()
                blank_start = prefix_len
                blank_end = min(prefix_len + blank_count, len(debug_norms))
                next_start = blank_end
                next_end = min(blank_end + debug_steps, len(debug_norms))
                logger.warning(
                    "[%s] CODEX_STEP_DEBUG align prefix=%d blank=%d "
                    "segment_frames=%d prompt_frames=%d blank_norms=%s "
                    "next_norms=%s",
                    request_id,
                    prefix_len,
                    blank_count,
                    int(segment_hidden.size(0)),
                    prompt_frame_count,
                    debug_norms[blank_start:blank_end],
                    debug_norms[next_start:next_end],
                )
        self.thinker_generate_speech[request_id] = True
        self._start_cosyvoice_talker_from_hidden(
            request_id=request_id,
            hidden_states=segment_hidden,
            sampling_params=sampling_params,
            blank_count=blank_count,
            prompt_frame_count=prompt_frame_count,
            codec_prefix_ids=codec_prefix_ids,
        )

    def _start_cosyvoice_talker_from_hidden(
        self,
        request_id: str,
        hidden_states: torch.Tensor,
        sampling_params: SamplingParams,
        blank_count: int,
        prompt_frame_count: int | None = None,
        codec_prefix_ids: list[int] | None = None,
    ) -> None:
        talker_hf_config = self._get_talker_hf_config()
        seq_len = hidden_states.size(0)
        flat_hidden = hidden_states.reshape(
            int(seq_len * talker_hf_config.group_size),
            talker_hf_config.hidden_size,
        )
        prompt_frames = (
            int(prompt_frame_count)
            if prompt_frame_count is not None
            else int(blank_count)
        )
        prompt_len = min(
            max(1, prompt_frames * int(talker_hf_config.group_size)),
            int(flat_hidden.size(0)),
        )
        debug_steps = int(os.environ.get("CODEX_DEBUG_STEPS", "0"))
        if debug_steps > 0:
            with contextlib.suppress(Exception):
                frame_group = int(talker_hf_config.group_size)
                prompt_frames_actual = prompt_len // max(1, frame_group)
                blank_start_frame = max(0, prompt_frames_actual - int(blank_count))
                blank_start = blank_start_frame * frame_group
                blank_end = prompt_len
                blank_norms = torch.linalg.vector_norm(
                    flat_hidden[blank_start:blank_end].float(), dim=-1
                ).detach().cpu().tolist()
                logger.warning(
                    "[%s] CODEX_STEP_DEBUG talker_prefill seq_len=%d "
                    "flat_len=%d group=%d prompt_frames=%d prompt_len=%d "
                    "blank_input_token_ids_are_external_only=true "
                    "blank_flat_norms=%s",
                    request_id,
                    int(seq_len),
                    int(flat_hidden.size(0)),
                    frame_group,
                    prompt_frames_actual,
                    prompt_len,
                    blank_norms,
                )
        prompt_embeds = self._compact_cpu_tensor(flat_hidden[:prompt_len])
        prompt_embeds_key = f"{request_id}_talker_prompt"
        self._write_hidden_to_shm(prompt_embeds, prompt_embeds_key)
        codec_prefix_ids = [int(token_id) for token_id in (codec_prefix_ids or [])]
        prompt_token_ids = [0] * int(prompt_embeds.shape[0])
        if codec_prefix_ids:
            prefix_count = min(len(codec_prefix_ids), len(prompt_token_ids))
            prompt_token_ids[-prefix_count:] = codec_prefix_ids[-prefix_count:]
        talker_prompt = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            prompt_embeds=None,
            using_external_data=True,
            prompt_embeds_shm_key=prompt_embeds_key,
        )
        generator = self.talker_engine.generate(
            request_id=request_id,
            prompt=talker_prompt,
            sampling_params=sampling_params,
            using_external_data=True,
        )

        self.cosyvoice_talker_hidden[request_id] = flat_hidden
        self.cosyvoice_talker_hidden_cursor[request_id] = prompt_len
        self.cosyvoice_talker_debug_step[request_id] = 0

        self._listen_talker(
            request_id,
            generator,
            self.output_queue[request_id],
        )

    def _feed_next_cosyvoice_talker_hidden(self, request_id: str) -> None:
        hidden = self.cosyvoice_talker_hidden.get(request_id)
        if hidden is None or hidden.size(0) == 0:
            return
        cursor = self.cosyvoice_talker_hidden_cursor.get(request_id, 0)
        if cursor < hidden.size(0):
            prompt_embeds = hidden[cursor : cursor + 1]
            self.cosyvoice_talker_hidden_cursor[request_id] = cursor + 1
            is_padding = False
        else:
            prompt_embeds = hidden[-1:]
            self.cosyvoice_talker_hidden_cursor[request_id] = int(hidden.size(0)) + 1
            is_padding = True
        debug_steps = int(os.environ.get(
            "CODEX_CURSOR_TRACE_STEPS",
            os.environ.get("CODEX_DEBUG_STEPS", "0"),
        ))
        debug_step = self.cosyvoice_talker_debug_step.get(request_id, 0)
        if debug_steps > 0 and debug_step < debug_steps:
            with contextlib.suppress(Exception):
                hidden_norm = float(torch.linalg.vector_norm(
                    prompt_embeds.float(), dim=-1
                ).detach().cpu().reshape(-1)[0])
                logger.warning(
                    "[%s] CODEX_STEP_DEBUG feed_next step=%d cursor=%d "
                    "hidden_len=%d is_padding=%s hidden_norm=%.6f "
                    "add_token_embedding=true",
                    request_id,
                    debug_step + 1,
                    cursor,
                    int(hidden.size(0)),
                    is_padding,
                    hidden_norm,
                )
            self.cosyvoice_talker_debug_step[request_id] = debug_step + 1
        self._add_talker_external_data(
            request_id=request_id,
            prompt_embeds=prompt_embeds,
            is_padding=is_padding,
            add_token_embedding=True,
        )

    def _get_cosyvoice_blank_count(self) -> int:
        return int(os.environ.get("CODEX_AUDIO_BLANK_TOKEN_COUNT", "6"))

    def _get_thinker_pad_token_id(self) -> int:
        config = getattr(self.thinker_config, "text_config", self.thinker_config)
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(config, attr, None)
            if isinstance(token_id, (list, tuple)):
                token_id = token_id[0] if token_id else None
            if token_id is not None:
                return int(token_id)
        return 0

    def _get_alignment_base_prompt_token_ids(
        self,
        original_prompt: PromptType | None,
        fallback_prompt_token_ids: list[int],
    ) -> list[int]:
        if isinstance(original_prompt, Mapping):
            prompt_text = original_prompt.get("prompt")
            if isinstance(prompt_text, str):
                return list(
                    self.thinker_tokenizer.encode(
                        prompt_text,
                        add_special_tokens=False,
                    )
                )
            prompt_token_ids = original_prompt.get("prompt_token_ids")
            if isinstance(prompt_token_ids, list):
                return list(prompt_token_ids)
        return list(fallback_prompt_token_ids)

    def _strip_thinker_stop_tokens(self, token_ids: list[int]) -> list[int]:
        config = getattr(self.thinker_config, "text_config", self.thinker_config)
        stop_ids: set[int] = set()
        for attr in ("eos_token_id", "pad_token_id"):
            value = getattr(config, attr, None)
            if isinstance(value, (list, tuple, set)):
                stop_ids.update(int(v) for v in value if v is not None)
            elif value is not None:
                stop_ids.add(int(value))
        while token_ids and token_ids[-1] in stop_ids:
            token_ids.pop()
        return token_ids

    def _talker_to_thinker(
        self,
        request_id: str,
    ):
        talker_hf_config = self._get_talker_hf_config()
        self._add_thinker_external_data(
            request_id=request_id,
            prompt_embeds=None,
            is_padding=False,
            add_token_embedding=False,
            external_token_ids=self.talker_speech_ids[request_id][
                -talker_hf_config.group_size :
            ],
            generate_speech=self.thinker_generate_speech[request_id],
        )

    def _add_talker_external_data(
        self,
        request_id: str,
        prompt_embeds: torch.Tensor,
        is_padding: bool,
        add_token_embedding: bool,
    ):
        embeds_tensor = self._compact_cpu_tensor(prompt_embeds)
        embeds_key = f"{request_id}_talker_external_{time.time_ns()}"
        self._write_hidden_to_shm(embeds_tensor, embeds_key)
        external_data = ExternalData(
            request_id=request_id,
            external_embeds=None,
            external_embeds_shm_key=embeds_key,
            is_padding=is_padding,
            add_token_embedding=add_token_embedding,
        )
        asyncio.run_coroutine_threadsafe(
            self.talker_engine.update_request_with_data(external_data),
            self._loop,
        ).result()

    @staticmethod
    def _compact_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Detach tensor views from large shared-memory backing storage."""
        return tensor.detach().cpu().clone().contiguous()

    @staticmethod
    def _write_hidden_to_shm(tensor: torch.Tensor, key: str) -> None:
        if not SharedHiddenBuffer.write_global(
            tensor,
            request_id=key,
            timeout=30.0,
            blocking=True,
        ):
            raise RuntimeError(f"Failed to write hidden tensor to shared buffer: {key}")

    def _add_thinker_external_data(
        self,
        request_id: str,
        prompt_embeds: torch.Tensor,
        is_padding: bool,
        add_token_embedding: bool,
        external_token_ids: list[int],
        generate_speech: bool,
    ):
        external_data = ExternalData(
            request_id=request_id,
            external_embeds=prompt_embeds,
            is_padding=is_padding,
            add_token_embedding=add_token_embedding,
            external_token_ids=external_token_ids,
            generate_speech=generate_speech,
        )
        asyncio.run_coroutine_threadsafe(
            self.thinker_engine.update_request_with_data(external_data),
            self._loop,
        ).result()

    def print_gpu_memory(self, tag: str, request_id: str, device=0):
        # 获取当前设备的显存信息
        allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MiB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # MiB
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MiB
        used_percent = (allocated / total) * 100  # 已用百分比

        logger.info(
            "%s %s GPU usage in MiB allocated %d max_allocated %d"
            " total %d used_percent %d",
            tag,
            request_id,
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
    ) -> queue.Queue[Union[RequestOutput, np.ndarray]]:
        if self.time_profile:
            self.output_profile_time[request_id] = [time.perf_counter()]
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
        self.request_prompts[request_id] = prompt

        self.print_gpu_memory("add_request", request_id, 0)
        logger.info(
            "thinker_outputs.size %s thinker_listener.size %s",
            self.thinker_outputs.qsize(),
            len(self.thinker_listener),
        )

        is_cosyvoice_codex = (
            self.has_talker
            and self._is_cosyvoice_codex_talker_config(self.talker_config)
        )
        thinker_uses_external_data = self.has_talker and not is_cosyvoice_codex

        async def run_output_handler_and_add_request():
            self.thinker_engine._run_output_handler()
            return await self.thinker_engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                using_external_data=thinker_uses_external_data,
            )

        generator = asyncio.run_coroutine_threadsafe(
            run_output_handler_and_add_request(),
            self._loop,
        ).result()

        # Pin the queue reference BEFORE starting the listener thread.
        # The listener pops the key from self.output_queue, self.output_queue.pop(request_id).
        # if the thread finishes before we reach the return statement,
        # defaultdict would create a new (empty) queue on access — causing
        # the caller to wait on a queue that never receives anything.
        # 这次就是这个情况——由于高并发下多个线程竞争 GIL，listener 线程被延迟了 1.7 秒才启动，而推理 464ms 就完成了。等 listener 终于跑起来，collector 里已经有了全部 13 个合并好的 token，4ms 就走完了整个生命周期（包括 finally 的 pop）。

        # 所以这个 bug 的触发需要两个条件同时满足：

        # 推理结果在 listener 启动前就已完成（高并发 + 短请求）
        # GIL 调度让 listener 在 return 之前抢到执行权（线程竞争激烈时更容易发生）

        output_q = self.output_queue[request_id]

        self._listen_thinker(request_id, generator, self.thinker_outputs)

        if self.has_talker:
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

            self.talker_requests[request_id] = talker_params

        return output_q

    def abort_request(self, request_id: str):
        self.print_gpu_memory("abort_request", request_id, 0)
        # cancel thinker
        asyncio.run_coroutine_threadsafe(
            self.thinker_engine.abort(request_id),
            self._loop,
        ).result()

        # cancel talker
        self.thinker_outputs.put((request_id, None))
        self.request_prompts.pop(request_id, None)
        self.cosyvoice_talker_hidden.pop(request_id, None)
        self.cosyvoice_talker_hidden_cursor.pop(request_id, None)
        self.cosyvoice_talker_debug_step.pop(request_id, None)
