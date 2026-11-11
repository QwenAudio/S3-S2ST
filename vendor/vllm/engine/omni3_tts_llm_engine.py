#!/usr/bin/env python3

import asyncio
import collections
import contextlib
import copy
import glob
import json
import multiprocessing
import os
import pickle
import queue
import time
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import emoji
import numpy as np
import regex as re
import torch
from fast_langdetect import LangDetectConfig, LangDetector
from transformers.models.qwen3_tts_moe.modeling_qwen3_tts_moe import (
    Qwen3TTSMoeTalkerResizeMLP,
)

from vllm.config import LoadConfig, ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.omni3_llm_engine import (
    SynchronizedGenerator,
    _MQCode2WavEngineClient,
    _MQOmniCode2WavEngine,
    envvars,
    suppress_output_queue_exception,
)
from vllm.envs import (
    VLLM_OMNI_CODE2WAV_CHUNK_SIZE,
    VLLM_OMNI_CODE2WAV_DYNAMIC_CHUNK_SIZE_LIST,
    VLLM_OMNI_CODE2WAV_ENABLE_DYNAMIC_CHUNK,
    VLLM_OMNI_ENABLE_TURN_BASE,
    VLLM_OMNI_PREPROCESS_ON_GPU,
    VLLM_OMNI_T2C_USE_ZMQ,
    VLLM_OMNI_XVECTOR_CACHE_SIZE,
    VLLM_QWEN_TTS_GENERATE_MODE,
)
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config, try_get_generation_config
from vllm.transformers_utils.processors.qwen3_tts_moe import Qwen3TTSMoeProcessor
from vllm.usage.usage_lib import UsageContext
from vllm.utils import LRUCache
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.engine import ExternalData
from vllm.v1.engine.async_llm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine

logger = init_logger(__name__)

language_code_mapping = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "en-au": "English",
    "en-bz": "English",
    "en-ca": "English",
    "en-en": "English",
    "en-gb": "English",
    "en-ie": "English",
    "en-in": "English",
    "en-nz": "English",
    "en-uk": "English",
    "en-ke": "English",
    "en-tt": "English",
    "en-us": "English",
    "en-pi": "English",
    "en-za": "English",
    "en-jp": "English",
    "en-sea": "English",
    "en-ph": "English",
    "en-gh": "English",
    "en-nl": "English",
    "en-jm": "English",
    "en-es": "English",
    "en-aus": "English",
    "en-zw": "English",
    "en-eng": "English",
    "en-ng": "English",
    "en-tz": "English",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nan": "Hokkien",
    "nl": "Dutch",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "wuu": "Chinese_wu",
    "yue": "Cantonese",
    "zh-yue": "Cantonese",
    "zh-us": "Cantonese",
    "zh-hant-hk": "Cantonese",
    "zh": "Chinese",
    "zh-zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hk": "Chinese",
    "zh-hans": "Chinese",
    "zh-tw": "Chinese_tw",
    "zh-hant": "Chinese_tw",
    "dialect-zh": "Chinese_dialect",
    "es": "Spanish",
    "es-419": "Spanish",
    "fr": "French",
    "tr": "Turkish",
    "pl": "Polish",
    "bg": "Bulgarian",
    "ro": "Romanian",
    "iw": "Hebrew",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "sr": "Serbian",
    "sv": "Swedish",
    "cs": "Czech",
    "no": "Norwegian",
    "da": "Danish",
    "cmn": "Chinese",
    "eng": "English",
    "deu": "German",
    "spa": "Spanish",
    "kor": "Korean",
    "fra": "French",
    "jpn": "Japanese",
    "ita": "Italian",
    "rus": "Russian",
    "por": "Portuguese",
    "ara": "Arabic",
    "tur": "Turkish",
    "pol": "Polish",
    "bul": "Bulgarian",
    "hin": "Hindi",
    "tha": "Thai",
    "ron": "Romanian",
    "ind": "Indonesian",
    "heb": "Hebrew",
    "ukr": "Ukrainian",
    "srp": "Serbian",
    "swe": "Swedish",
    "nld": "Dutch",
    "ces": "Czech",
    "vie": "Vietnamese",
}

# TTS Generate Mode Configuration
_TTS_GENERATE_MODE_CONFIG = {
    "default": {
        "ignore_instructions": True,
        "force_voice_type": None,  # None means keep user's value
        "force_non_streaming": None,
    },
    "voice_design": {
        "ignore_instructions": False,
        "force_voice_type": "",  # Empty string for instruct-only
        "force_non_streaming": True,
    },
    "instructions": {
        "ignore_instructions": False,
        "force_voice_type": None,  # Keep user's voice_type (must be specified)
        "force_non_streaming": None,
    },
}

class OmniTTSLLMEngine:
    def __init__(
        self,
        talker_engine_args: Union[
            Mapping[str, AsyncEngineArgs], AsyncEngineArgs
        ] = None,
        code2wav_model_path: Optional[str] = None,
        code2wav_data_parallelism: int = 1,
        code2wav_enable_torch_compile: bool = False,
        code2wav_enable_torch_compile_first_chunk: bool = False,
        code2wav_enable_cudagraph_capture: bool = False,
        code2wav_cudagraph_capture_max_batch: int = 8,
        code2wav_odeint_method: str = "rk4",
        code2wav_odeint_method_relaxed: bool = False,
        code2wav_batched_chunk: int = 2,
        code2wav_frequency: str = "50hz",
        code2wav_dit_quant: str = None,
        talker_visible_devices: Optional[List[int]] = None,
        code2wav_visible_devices: Optional[List[int]] = None,
        code2wav_dynamic_batch: Optional[bool] = False,
        use_dynamic_chunk: Optional[bool] = True,
        dynamic_chunk_progress: Optional[int] = None,
        dynamic_chunk_cnt: Optional[int] = None,
        code2wav_padding_batch: Optional[bool] = False,
        lang_detect_model_path: Optional[str] = None,
    ):
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
        self.code2wav_enable_cudagraph_capture = code2wav_enable_cudagraph_capture
        self.code2wav_cudagraph_capture_max_batch = code2wav_cudagraph_capture_max_batch
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

        self.talker_visible_devices = talker_visible_devices
        self.code2wav_visible_devices = code2wav_visible_devices

        # multiprocessing manager for thinker to talker and talker to code2wav.
        self.mp_manager = multiprocessing.Manager()
        self.producer_conn, self.comsumer_conn = multiprocessing.Pipe()

        if self.talker_visible_devices and not isinstance(
            self.talker_visible_devices, list
        ):
            self.talker_visible_devices = [self.talker_visible_devices]
        if self.code2wav_visible_devices and not isinstance(
            self.code2wav_visible_devices, list
        ):
            self.code2wav_visible_devices = [self.code2wav_visible_devices]

        if not code2wav_model_path:
            self.code2wav_data_parallelism = 0
        self.code2wav_idx = 0
        # self.task_2_code2wav: list[ThreadSafeList] = []
        # for i in range(self.code2wav_data_parallelism):
        #     self.task_2_code2wav.append(ThreadSafeList())

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        self.context = multiprocessing.get_context("spawn")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._async_loop_thread = Thread(target=self._start_async_loop, daemon=True)
        self._async_loop_thread.start()

        for i, (talker_type, engine_args) in enumerate(talker_engine_args.items()):
            self.processor = Qwen3TTSMoeProcessor.from_pretrained(engine_args.model)
            if engine_args.limit_mm_per_prompt is None:
                engine_args.limit_mm_per_prompt = {
                    "audio": 32768,
                    "image": 32768,
                    "video": 32768,
                }

            # if engine_args.device == 'auto' or engine_args.device == 'cuda':
            #     engine_args.device = "cuda:" + ','.join(
            #         str(device) for device in self.talker_visible_devices)
            self.device = torch.device("cuda:0")

            engine_args.tokenizer = engine_args.model
            engine_args.processor = engine_args.model
            if not engine_args.hf_overrides:
                engine_args.hf_overrides = {}

            default_talker_args = next(iter(talker_engine_args.values()))
            self.talker_config = get_config(
                default_talker_args.model,
                trust_remote_code=default_talker_args.trust_remote_code,
            )
            self.talker_generation_config = None
            generate_config_path = os.path.join(
                default_talker_args.model, "generation_config.json"
            )
            generate_config_path = str(generate_config_path)
            if os.path.exists(generate_config_path):
                self.talker_generation_config = try_get_generation_config(
                    default_talker_args.model,
                    trust_remote_code=default_talker_args.trust_remote_code,
                )

            # Adapt the qwen3_omni_model
            engine_args.hf_overrides["architectures"] = [
                "Qwen3TTSMoeForConditionalGeneration"
            ]

            if self.talker_visible_devices:
                visible_devices = [self.talker_visible_devices[i]]
            else:
                visible_devices = None
            logger.info(
                "self.talker_config.architectures %s visible_devices %s",
                self.talker_config.architectures,
                visible_devices,
            )

            self.talker_engine = self._start_talker_engine(
                talker_type,
                engine_args,
                visible_devices,
            )

            # init embeddings for special tokens
            self._load_talker_projection(default_talker_args)
            self._init_omni3_special_tokens_embeddings(None, default_talker_args)

        # talker request state
        self.talker_requests: Dict[str, Tuple[PromptType, SamplingParams]] = dict()
        self.talker_listener: Dict[str, Thread] = dict()
        self.talker_resumer: dict[str, Thread] = dict()
        self.talker_finished: Set[str] = set()
        self.talker_inputs: Dict[
            str,
            queue.Queue[Tuple[Any, bool]],
        ] = dict()
        self.requests_str: Dict[str, str] = dict()
        self.input_tokens: Dict[str, int] = dict()
        # Track request start time for TTFA metrics
        self.request_start_time: Dict[str, float] = dict()
        # Tuple fields:
        #   (request_id,        # str  - which request this chunk belongs to
        #    valid_chunk,       # int  - codec tokens in this chunk
        #    codec_ids_list,    # list[Tensor] - accumulated codec tensors
        #    total_codec_num,   # int  - cumulative codec token count
        #    flushed,           # bool - right-context flush signal
        #    finished)          # bool - end-of-stream signal
        self.talker_outputs: queue.Queue[
            tuple[str, int, list[torch.Tensor], int, bool, bool]
        ] = queue.Queue(maxsize=10000)

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
        # chunk size will change to dynamic_codec_chunk_size_list[i]
        # after dynamic_codec_chunk_step_list[i] chunks
        self.dynamic_codec_chunk_size_list = VLLM_OMNI_CODE2WAV_DYNAMIC_CHUNK_SIZE_LIST
        self.dynamic_codec_chunk_step_list = [16, 16, 1]
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
        if VLLM_OMNI_CODE2WAV_ENABLE_DYNAMIC_CHUNK:
            self.codec_chunk_size = max(self.dynamic_codec_chunk_size_list)

        # code2wav: async mp engine(s)
        self._code2wav_pids = []
        self._code2wav_processes = []
        self._code2wav_engine_clients: List[_MQCode2WavEngineClient] = []
        for i in range(self.code2wav_data_parallelism):
            assert code2wav_model_path, "code2wav_model_path is required"

            (
                pid,
                process,
                engine_client,
                thread,
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
            self.code2wav_engine_thread = thread

        # code2wav voice types
        if self.code2wav_model_path:
            self.code2wav_voice_types = self._load_code2wav_voice_types()
        else:
            self.code2wav_voice_types = []
        logger.info("Code2Wav voice types: %s", self.code2wav_voice_types)

        # code2wav result listener
        self._code2wav_listener_thread = []
        for i in range(self.code2wav_data_parallelism):
            self._code2wav_listener_thread.append(
                Thread(target=self._listen_code2wav, args=(i,), daemon=True)
            )
            self._code2wav_listener_thread[i].start()

        # request id -> [union(request_output, (waveform, finished))]
        self.output_queue: Dict[
            str, queue.Queue[Union[RequestOutput, Tuple[np.ndarray, int]]]
        ] = collections.defaultdict(queue.Queue)

        # request id -> xvector_info for zeroshot TTS
        self.request_xvector_info: Dict[str, Optional[dict]] = {}

        # LRU cache for xvector features to reduce loading overhead
        self.xvector_cache = LRUCache(capacity=VLLM_OMNI_XVECTOR_CACHE_SIZE)
        logger.info(
            "Initialized xvector LRU cache with capacity %d "
            "(configurable via VLLM_OMNI_XVECTOR_CACHE_SIZE)",
            VLLM_OMNI_XVECTOR_CACHE_SIZE,
        )
        logger.info(
            "Turn-base mode switch: %s (enabled=%s)",
            "ENABLED" if VLLM_OMNI_ENABLE_TURN_BASE else "DISABLED",
            VLLM_OMNI_ENABLE_TURN_BASE,
        )

        self.text_lang_detector: LangDetector = None

        ## load text language detector
        if lang_detect_model_path is not None:
            config = LangDetectConfig(
                cache_dir=lang_detect_model_path,  # Custom model cache directory
                allow_fallback=True,  # Enable fallback to small model
            )
            self.text_lang_detector = LangDetector(config)
        else:
            self.text_lang_detector = None

    def _load_code2wav_voice_types(self):
        code2wav_voice_types = []
        if self.talker_config.talker_config.spk_id is not None:
            code2wav_voice_types = self.talker_config.talker_config.spk_id.keys()
        # code2wav_voice_types = ['f245', 'm36', 'beijing']
        return code2wav_voice_types

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_talker_engine(
        self,
        talker_type: str,
        engine_args: AsyncEngineArgs,
        visible_devices: Optional[List[int]] = None,
    ) -> AsyncLLMEngine:
        if visible_devices:
            original_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if original_devices is not None:
                original_devices = original_devices.split(",")
                value = ",".join(original_devices[device] for device in visible_devices)
            else:
                value = ",".join(str(device) for device in visible_devices)
            device_ctx = envvars("CUDA_VISIBLE_DEVICES", value)
            if VLLM_OMNI_PREPROCESS_ON_GPU:
                os.environ["PREPROCESS_DEVICE_T"] = "cuda:0"
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            usage_context = UsageContext.API_SERVER
            talker_engine = AsyncLLMEngine.from_engine_args(
                engine_args=engine_args,
                start_engine_loop=True,
                usage_context=usage_context,
            )

        return talker_engine

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
        device: Union[int, str] = "cuda",
        visible_devices: Optional[list[int]] = None,
    ) -> tuple[int, multiprocessing.Process, _MQCode2WavEngineClient, Thread]:
        logger.info(
            "Multiprocessing frontend to use %s for IPC Path for model %s.",
            ipc_path,
            model_path,
        )

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value("b", True, lock=False)
        # if visible_devices:
        #     device_ctx = envvars(
        #         'CUDA_VISIBLE_DEVICES',
        #         ','.join(str(device) for device in visible_devices))
        # else:
        #     device_ctx = contextlib.nullcontext()

        talker_to_code2wav_queue = None
        logger.info(
            "Start code2wav engine, using zmq to send: %s", VLLM_OMNI_T2C_USE_ZMQ
        )
        if not VLLM_OMNI_T2C_USE_ZMQ:
            talker_to_code2wav_queue = self.mp_manager.Queue()

        engine_process = self.context.Process(
            target=_MQOmniCode2WavEngine.run_code2wav_engine,
            args=(
                ipc_path,
                model_path,
                enable_torch_compile,
                N_code_size,
                codec_chunk_size,
                codec_left_context,
                codec_right_context,
                codec_upsample_rate,
                engine_alive,
                visible_devices[0],
                talker_to_code2wav_queue,
            ),
        )
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info(
            "Started engine process with PID %d, device: %s",
            engine_pid,
            str(visible_devices),
        )

        mq_engine_client: _MQCode2WavEngineClient = _MQCode2WavEngineClient(
            ipc_path, engine_pid, talker_to_code2wav_queue
        )

        while True:
            try:
                asyncio.run_coroutine_threadsafe(
                    mq_engine_client.setup(),
                    self._loop,
                ).result()
                break
            except TimeoutError:
                if not engine_process.is_alive() or not engine_alive.value:
                    raise RuntimeError(
                        "Engine process failed to start. See stack "
                        "trace for the root cause."
                    ) from None
                
        thread = Thread(
            target=self._talker_to_code2wav_loop,
            daemon=True,
        )
        thread.start()

        logger.info("Engine process with PID %d has started", engine_pid)
        return engine_pid, engine_process, mq_engine_client, thread

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
            logger.warning("Engine process did not terminate in time.")
            engine_process.kill()
        logger.info("Engine process with PID %d has stopped", engine_pid)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self._closed = True
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
            # logger.debug('key: %s value %s', key, value.shape)
            if key in [
                embedding_name,
            ]:
                logger.debug("get embedding %s value %s", key, value)
                return torch.nn.Embedding.from_pretrained(value).to(self.device)
        raise ValueError(
            "Embedding weight 'model.embed.weight' not found in model weights."
        )

    def _load_model_linear(
        self,
        engine_args: EngineArgs,
        weight_name: str,
        bias_name: str = None,
    ) -> torch.nn.Linear:
        """Load linear layer weights and bias from model checkpoint."""
        model_loader = get_model_loader(
            LoadConfig(
                load_format=engine_args.load_format,
                download_dir=engine_args.download_dir,
                model_loader_extra_config=engine_args.model_loader_extra_config,
                ignore_patterns=engine_args.ignore_patterns,
            )
        )

        weight_tensor = None
        bias_tensor = None

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
            logger.debug("key: %s value %s", key, value.shape)
            if key == weight_name:
                logger.debug("get linear weight %s value %s", key, value)
                weight_tensor = value
            elif bias_name and key == bias_name:
                logger.debug("get linear bias %s value %s", key, value)
                bias_tensor = value

        if weight_tensor is None:
            raise ValueError(
                f"Linear weight '{weight_name}' not found in model weights."
            )

        # Create linear layer from weight tensor
        out_features, in_features = weight_tensor.shape
        linear_layer = torch.nn.Linear(
            in_features, out_features, bias=(bias_tensor is not None)
        )

        # Load the weights
        linear_layer.weight.data = weight_tensor.to(self.device)
        if bias_tensor is not None:
            linear_layer.bias.data = bias_tensor.to(self.device)

        return linear_layer.to(self.device)

    def _load_talker_projection(
        self,
        talker_engine_args: AsyncEngineArgs,
    ):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        model_path = talker_engine_args.model
        self.text_projection = Qwen3TTSMoeTalkerResizeMLP(
            talker_hf_config.text_hidden_size,
            talker_hf_config.intermediate_size,
            talker_hf_config.hidden_size,
            talker_hf_config.hidden_act,
            bias=True,
        ).to(dtype=torch.bfloat16)

        safetensors = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for key, value in safetensors_weights_iterator(
            safetensors, use_tqdm_on_load=True
        ):
            if key.startswith("talker.text_projection."):
                self.text_projection.state_dict()[
                    key.replace("talker.text_projection.", "")
                ].copy_(value)

        self.text_projection = self.text_projection.to(self.device)

    def _init_omni3_special_tokens_embeddings(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: AsyncEngineArgs,
    ):
        # thinker and talker embeddings
        self.codec_embedding = self._load_model_embedding(
            talker_engine_args, "talker.model.codec_embedding.weight"
        )
        self.text_embedding = self._load_model_embedding(
            talker_engine_args, "talker.model.text_embedding.weight"
        )

        # Load code_predictor embeddings (for layers 1-15 of codec)
        # code_predictor.model.codec_embedding is a ModuleList with 15 embeddings
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        num_code_groups = getattr(
            talker_hf_config.code_predictor_config, "num_code_groups", 16
        )
        self.code_predictor_embeddings = []
        for i in range(num_code_groups - 1):  # 15 embeddings for layers 1-15
            embedding = self._load_model_embedding(
                talker_engine_args,
                f"talker.code_predictor.model.codec_embedding.{i}.weight",
            )
            self.code_predictor_embeddings.append(embedding)

        logger.info(
            "Loaded %d code_predictor embeddings for zero-shot",
            len(self.code_predictor_embeddings),
        )

        talker_hf_config = self.talker_config

        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = (
            self.text_projection(
                self.text_embedding(
                    torch.tensor(
                        [
                            [
                                talker_hf_config.tts_bos_token_id,
                                talker_hf_config.tts_eos_token_id,
                                talker_hf_config.tts_pad_token_id,
                            ]
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                )
            ).chunk(3, dim=1)
        )

    def _get_embed_text_spk_token(self, voice_type: str):
        if voice_type not in self.embed_text_spk_tokens:
            return self.embed_text_bos_token
        return self.embed_text_spk_tokens[voice_type]

    def _get_text_spk_token_id(self, voice_type: str):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        if voice_type not in self.tts_text_spk_token_ids:
            return talker_hf_config.tts_text_start_token_id
        return self.tts_text_spk_token_ids[voice_type]

    def _get_codec_embeddings(self):
        return self.codec_embedding

    def _get_text_embeddings(self):
        return self.text_embedding

    # Threshold for slow-path warning in _add_external_data (ms)
    _ADD_DATA_SLOW_MS = 50

    def _add_external_data(
        self,
        request_id: str,
        prompt_embeds: torch.Tensor,
        is_padding: bool,
    ):
        import time as _time
        _t0 = _time.monotonic()
        embeds_tensor = prompt_embeds.detach().cpu()
        external_data = ExternalData(
            request_id=request_id,
            external_embeds=embeds_tensor,
            is_padding=is_padding,
        )
        asyncio.run_coroutine_threadsafe(
            self.talker_engine.update_request_with_data(external_data),
            self._loop,
        )
        elapsed_ms = (_time.monotonic() - _t0) * 1000
        if elapsed_ms > self._ADD_DATA_SLOW_MS:
            logger.warning(
                "[%s] ADD_EXTERNAL_DATA_SLOW elapsed=%.0fms is_padding=%s",
                request_id, elapsed_ms, is_padding,
            )

    def _listen_talker(
        self,
        request_id: str,
        voice_type: str,
        generator: AsyncGenerator[RequestOutput, None],
        output_queue: queue.Queue[Union[RequestOutput, np.ndarray]],
        talker_input_queue: queue.Queue[Tuple[Any, bool]],
        code2wav_index: int,
    ):
        include_code2wav = not self._ignore_code2wav(voice_type)

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        def _listen():
            import time as _time
            _listen_start = _time.monotonic()
            _first_output_time = None
            last_output = None
            total_codec_num = 0
            last_codec_interval = 0
            dynamic_codec_chunk_index = 0
            codec_ids_list = []
            _last_chunk_time = _time.monotonic()
            for output in SynchronizedGenerator(generator, self._loop):
                if _first_output_time is None:
                    _first_output_time = _time.monotonic()
                    logger.info(
                        "[%s] TALKER_FIRST_TOKEN ttft=%.0fms",
                        request_id,
                        (_first_output_time - _listen_start) * 1000,
                    )
                if VLLM_OMNI_CODE2WAV_ENABLE_DYNAMIC_CHUNK:
                    if dynamic_codec_chunk_index < len(self.dynamic_codec_chunks):
                        current_codec_chunk_size = self.dynamic_codec_chunks[
                            dynamic_codec_chunk_index
                        ]
                    else:
                        current_codec_chunk_size = self.dynamic_codec_chunks[-1]
                else:
                    current_codec_chunk_size = self.codec_chunk_size
                if output.outputs is None and output.finished:
                    logger.info("[%s] TALKER_ABORT", request_id)
                    last_output = None
                    output_queue.put(None)
                    break
                # Per-token logging removed — use T2C_CHUNK for perf tracking
                if not self._code2wav_engine_clients or not include_code2wav:
                    output_queue.put(output)
                elif output.outputs[0].codec_ids is not None:
                    last_output = output
                    codec_ids_list.append(output.outputs[0].codec_ids)
                    total_codec_num += 1
                    last_codec_interval += 1
                    if last_codec_interval % current_codec_chunk_size == 0:
                        _now = _time.monotonic()
                        _delta_ms = (_now - _last_chunk_time) * 1000
                        _last_chunk_time = _now
                        logger.debug(
                            "[%s] T2C_CHUNK idx=%d tokens=%d "
                            "chunk_size=%d delta=%.0fms",
                            request_id,
                            dynamic_codec_chunk_index,
                            total_codec_num,
                            current_codec_chunk_size,
                            _delta_ms,
                        )
                        if _delta_ms > 500:
                            logger.warning(
                                "[%s] T2C_CHUNK_SLOW idx=%d "
                                "delta=%.0fms (>500ms)",
                                request_id,
                                dynamic_codec_chunk_index,
                                _delta_ms,
                            )
                        self.talker_outputs.put(
                            (
                                output.request_id,
                                current_codec_chunk_size,
                                list(codec_ids_list),
                                total_codec_num,
                                False,
                                False, 
                            )
                        )
                        last_codec_interval = 0
                        dynamic_codec_chunk_index += 1
                    if output.finished:
                        # flush codecs not aligned to chunk_size
                        if last_codec_interval % current_codec_chunk_size != 0:
                            self.talker_outputs.put(
                                (
                                    output.request_id,
                                    last_codec_interval % current_codec_chunk_size,
                                    list(codec_ids_list),
                                    total_codec_num,
                                    False,
                                    False, 
                                )
                            )
                        # flush right context
                        if self.codec_right_context > 0:
                            logger.debug(
                                "[%s] T2C_FLUSH chunk_idx=%d tokens=%d",
                                request_id,
                                dynamic_codec_chunk_index,
                                total_codec_num,
                            )
                            self.talker_outputs.put(
                                (
                                    output.request_id,
                                    current_codec_chunk_size,
                                    list(codec_ids_list),
                                    total_codec_num,
                                    True,
                                    False, 
                                )
                            )
                        self.talker_outputs.put(
                            (
                                output.request_id,
                                current_codec_chunk_size,
                                list(codec_ids_list),
                                total_codec_num,
                                False,
                                True,
                            )
                        )
                elif output.finished:
                    # flush codecs not aligned to chunk_size
                    if last_codec_interval % current_codec_chunk_size != 0:
                        self.talker_outputs.put(
                            (
                                output.request_id,
                                last_codec_interval % current_codec_chunk_size,
                                list(codec_ids_list),
                                total_codec_num,
                                False,
                                False,
                            )
                        )
                    # flush right context
                    if self.codec_right_context > 0:
                        self.talker_outputs.put(
                            (
                                output.request_id,
                                current_codec_chunk_size,
                                list(codec_ids_list),
                                total_codec_num,
                                True,
                                False,
                            )
                        )
                    self.talker_outputs.put(
                        (
                            output.request_id,
                            current_codec_chunk_size,
                            list(codec_ids_list),
                            total_codec_num,
                            False,
                            True,
                        )
                    )
                else:
                    # has code2wav, no codec_id and not finished
                    pass

                # prune codec list
                if len(codec_ids_list) > self.max_codec_chunk_size:
                    codec_ids_list = codec_ids_list[-self.max_codec_chunk_size :]

            logger.debug("[%s] TALKER_GENERATOR_EXIT", request_id)
            if not self._code2wav_engine_clients or not include_code2wav:
                output_queue.put(None)

            # Clean up xvector_info when request is finished
            if request_id in self.request_xvector_info:
                del self.request_xvector_info[request_id]

            if last_output is None or last_output.outputs is None:
                logger.info("[%s] TALKER_FINISH reason=aborted", request_id)
            else:
                logger.info(
                    "[%s] TALKER_FINISH reason=(%s,%s) tokens=%s",
                    request_id,
                    last_output.outputs[0].finish_reason,
                    last_output.outputs[0].stop_reason,
                    (
                        len(last_output.outputs[0].token_ids) 
                        if last_output.outputs[0].token_ids else 0,
                    )
                )

            with contextlib.suppress(Exception):
                # for i in range(self.code2wav_data_parallelism):
                #     if self.task_2_code2wav[i].has(request_id):
                #         self.task_2_code2wav[i].remove(request_id)
                #         break
                logger.debug("[%s] TALKER_CLEANUP", request_id)
                del self.talker_listener[request_id]
                del self.talker_inputs[request_id]
                del self.requests_str[request_id]

        def _talker_resume():
            import time as _time
            _resume_start = _time.monotonic()
            finished = False
            token_count = 0
            total_queue_wait = 0.0
            total_embed_time = 0.0
            total_add_data_time = 0.0
            while not finished:
                _tq0 = _time.monotonic()
                text_id, finished = talker_input_queue.get()
                _tq1 = _time.monotonic()
                total_queue_wait += (_tq1 - _tq0)
                if text_id is not None:
                    token_count += 1
                    self.input_tokens[request_id] = self.input_tokens[request_id] + 1
                    _te0 = _time.monotonic()
                    prompt_embeds = self._get_text_embeddings()(
                        torch.tensor([text_id]).to(self.device)
                    )
                    prompt_embeds = self.text_projection(prompt_embeds)
                    _te1 = _time.monotonic()
                    total_embed_time += (_te1 - _te0)

                    _ta0 = _time.monotonic()
                    self._add_external_data(
                        request_id=request_id,
                        prompt_embeds=prompt_embeds,
                        is_padding=False,
                    )
                    _ta1 = _time.monotonic()
                    total_add_data_time += (_ta1 - _ta0)

            # input finished, add a padding embeds
            _tp0 = _time.monotonic()
            prompt_embeds = self.tts_pad_embed.squeeze(0)
            self._add_external_data(
                request_id=request_id,
                prompt_embeds=prompt_embeds,
                is_padding=True,
            )
            _tp1 = _time.monotonic()
            _resume_end = _time.monotonic()
            logger.info(
                "[%s] resume_request finished %s, tokens=%d, "
                "queue_wait=%.3fms, embed=%.3fms, add_data=%.3fms, "
                "padding=%.3fms, total=%.3fms",
                request_id, finished, token_count,
                total_queue_wait * 1000,
                total_embed_time * 1000,
                total_add_data_time * 1000,
                (_tp1 - _tp0) * 1000,
                (_resume_end - _resume_start) * 1000,
            )
            with contextlib.suppress(Exception):
                logger.debug("[%s] TALKER_RESUMER_CLEANUP", request_id)
                del self.talker_resumer[request_id]

        logger.info("[%s] TALKER_START", request_id)
        self.talker_resumer[request_id] = Thread(target=_talker_resume, daemon=True)
        self.talker_resumer[request_id].start()
        self.talker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.talker_listener[request_id].start()

    # Threshold for slow C2W chunk delta (ms)
    _C2W_CHUNK_SLOW_MS = 500

    def _listen_code2wav(self, code2wav_engine_client_id):
        import time as _time
        request_first_audio_time: dict[str, float] = {}
        request_chunk_count: dict[str, int] = {}
        request_last_chunk_time: dict[str, float] = {}

        while True:
            try:
                # Read directly from threading.Queue —
                # no asyncio bridge overhead.
                get_start = _time.monotonic()
                request_id, audio, output_tokens = (
                    self._code2wav_engine_clients[
                        code2wav_engine_client_id
                    ].get_chunk()
                )
                get_ms = (
                    (_time.monotonic() - get_start) * 1000
                )
                if get_ms > 200:
                    logger.warning(
                        "[%s] C2W_GET_CHUNK_SLOW "
                        "wait=%.0fms (>200ms)",
                        request_id,
                        get_ms,
                    )
                if audio is None:
                    chunk_count = request_chunk_count.pop(request_id, 0)
                    logger.info(
                        "[%s] CODE2WAV_FINISH chunks=%d",
                        request_id,
                        chunk_count,
                    )
                    request_first_audio_time.pop(request_id, None)
                    request_last_chunk_time.pop(request_id, None)
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(None)
                        self.output_queue.pop(request_id)
                        del self.input_tokens[request_id]
                else:
                    # Increment chunk count
                    if request_id not in request_chunk_count:
                        request_chunk_count[request_id] = 0
                    request_chunk_count[request_id] += 1
                    chunk_idx = request_chunk_count[request_id]
                    
                    if request_id not in request_first_audio_time:
                        request_first_audio_time[request_id] = _time.monotonic()
                        # Get request start time if tracked
                        start_time = self.request_start_time.get(request_id)
                        if start_time is not None:
                            now = request_first_audio_time[request_id]
                            ttfa_ms = (now - start_time) * 1000
                            logger.info(
                                "[%s] CODE2WAV_FIRST_AUDIO ttfa=%.0fms",
                                request_id,
                                ttfa_ms,
                            )
                        else:
                            logger.info("[%s] CODE2WAV_FIRST_AUDIO", request_id)
                    
                    # Per-chunk with delta timing
                    _now = _time.monotonic()
                    _prev = request_last_chunk_time.get(
                        request_id, request_first_audio_time[request_id]
                    )
                    _delta_ms = (_now - _prev) * 1000
                    request_last_chunk_time[request_id] = _now
                    logger.debug(
                        "[%s] CODE2WAV_CHUNK idx=%d "
                        "audio_len=%d delta=%.0fms",
                        request_id,
                        chunk_idx,
                        audio.shape[-1],
                        _delta_ms,
                    )
                    if _delta_ms > self._C2W_CHUNK_SLOW_MS:
                        logger.warning(
                            "[%s] CODE2WAV_CHUNK_SLOW idx=%d "
                            "delta=%.0fms (>%dms)",
                            request_id,
                            chunk_idx,
                            _delta_ms,
                            self._C2W_CHUNK_SLOW_MS,
                        )
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(
                            (audio, output_tokens, self.input_tokens[request_id])
                        )
            except Exception:
                logger.exception("[%s] Error in code2wav listener", request_id)

    def _talker_to_code2wav_loop(self):
        while True:
            try:
                (
                    request_id,
                    valid_chunk,
                    codec_ids_list,
                    total_codec_num,
                    flushed,
                    finished,
                ) = self.talker_outputs.get()
                self._talker_to_code2wav(
                    request_id=request_id,
                    valid_chunk=valid_chunk,
                    codec_ids_list=codec_ids_list,
                    total_codec_num=total_codec_num,
                    flushed=flushed,
                    finished=finished,
                )
            except Exception as e:
                logger.exception(
                    "[%s] sender_loop error in _talker_to_code2wav: %s",
                    request_id, e
                )

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
            logger.debug(
                "[%s] T2C_SEND finished tokens=%s",
                request_id,
                len(codec_ids_list),
            )
        else:
            if len(codec_ids_list) < valid_chunk + self.codec_right_context:
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
        )

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

    def post_process(self, text):
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`(.*?`)", "", text, flags=re.DOTALL)
        text = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.DOTALL)
        text = re.sub(r"\$(.*?)\$", "", text, flags=re.DOTALL)
        text = text.replace("、", "，")
        text = text.replace("(", ",")
        text = text.replace(")", ",")
        text = text.replace("（", "，")
        text = text.replace("）", "，")
        text = text.replace("-", " ")
        text = text.replace("\n", " ")
        text = re.sub(r"[\n\r\t]", "", text)
        text = re.sub(r"[#*_`~]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(\.|\:)\s+", r"\1 ", text)

        if re.search(r"[\u4e00-\u9fa5]", text):
            text = re.sub(r"(\d+)\.\s*([\u4e00-\u9fa5A-Za-z])", r"\1：\2", text)
        else:
            text = re.sub(r"(\d+)\.\s*([\w])", r"\1:\2", text)
        text = re.sub(r'[^\w\s,.;:?!\'"、。，：；？！“”‘’]', "", text)
        punctuations = r",.!?;:，。！？；：、"
        text = re.sub(r"([" + re.escape(punctuations) + r"]){2,}", r"\1", text)
        if text and text[-1] not in ["。", "？", "！", ".", "?", "!"]:
            if text[-1] in [",", "，", ";", "；", ":", "：", "、"]:
                text = text[:-1] + "。"
            else:
                text += "。"

        return text

    def get_input_text_language(
        self,
        text: str,
    ) -> str:
        if self.text_lang_detector is None:
            logger.error("no text_lang_detector initialized ...")
            return "unkown"
        # 文本正则化
        cleaned_text = self.post_process(text)

        # 文本语种检测
        text_lang_id = self.text_lang_detector.detect(cleaned_text, low_memory=False)[
            "lang"
        ]

        text_lang_id = language_code_mapping.get(text_lang_id, "unknown")
        return text_lang_id

    def load_zeroshot_xvector(
        self,
        request_id: str,
        xvector_path: str,
    ) -> dict:
        """
        从本地文件加载 zero-shot 特征

        按照 qwen3-tts-zeroshot.md 文档格式，从两个文件加载：
        - feat.pkl: 包含 tensor 数据（xvector, prompt_code）
        - info.json: 包含元信息（prompt_text, language_type, version）

        使用 LRU 缓存来减少重复加载的开销。

        Args:
            request_id: 请求 ID
            xvector_path: 特征目录路径（包含 feat.pkl 和 info.json）

        Returns:
            dict: 合并后的特征字典，包含：
                - xvector: torch.Tensor [1024]
                - prompt_code: torch.Tensor [seq_len, 16]
                - prompt_text: str
                - language_type: str
                - version: str
        """
        # Check cache first
        if xvector_path in self.xvector_cache:
            logger.info("[%s] Using cached xvector from %s", request_id, xvector_path)
            # Return deep copy to prevent external modifications from affecting cache
            return copy.deepcopy(self.xvector_cache[xvector_path])

        if not os.path.exists(xvector_path):
            raise FileNotFoundError(
                f"[{request_id}] XVector directory not found: {xvector_path}"
            )

        feat_path = os.path.join(xvector_path, "feat.pkl")
        info_path = os.path.join(xvector_path, "info.json")

        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"[{request_id}] feat.pkl not found in {xvector_path}"
            )
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"[{request_id}] info.json not found in {xvector_path}"
            )

        logger.info("[%s] Loading zero-shot features from %s", request_id, xvector_path)

        try:
            # 1. 加载 feat.pkl
            with open(feat_path, "rb") as file:
                xvector_info = pickle.load(file)

            # 2. 加载 info.json
            with open(info_path, encoding="utf-8") as file:
                info = json.load(file)
                xvector_info["prompt_text"] = info.get("prompt_text", None)
                xvector_info["language_type"] = info.get("language_type", None)
                xvector_info["version"] = info.get("version", None)

            # 3. 类型转换：numpy.ndarray → torch.Tensor
            if "xvector" in xvector_info:
                if isinstance(xvector_info["xvector"], np.ndarray):
                    xvector_info["xvector"] = torch.from_numpy(
                        xvector_info["xvector"]
                    ).float()
                elif isinstance(xvector_info["xvector"], torch.Tensor):
                    xvector_info["xvector"] = xvector_info["xvector"].float()

            if "prompt_code" in xvector_info:
                if isinstance(xvector_info["prompt_code"], np.ndarray):
                    xvector_info["prompt_code"] = torch.from_numpy(
                        xvector_info["prompt_code"]
                    ).long()
                elif isinstance(xvector_info["prompt_code"], torch.Tensor):
                    xvector_info["prompt_code"] = xvector_info["prompt_code"].long()

            # 兼容旧的 ref_code 字段名（如果存在）
            if "ref_code" in xvector_info and "prompt_code" not in xvector_info:
                if isinstance(xvector_info["ref_code"], np.ndarray):
                    xvector_info["ref_code"] = torch.from_numpy(
                        xvector_info["ref_code"]
                    ).long()
                elif isinstance(xvector_info["ref_code"], torch.Tensor):
                    xvector_info["ref_code"] = xvector_info["ref_code"].long()
                xvector_info["prompt_code"] = xvector_info["ref_code"]

            logger.info(
                "[%s] Loaded features: xvector=%s, "
                "prompt_code=%s, language=%s, version=%s",
                request_id,
                xvector_info["xvector"].shape,
                xvector_info["prompt_code"].shape,
                xvector_info.get("language_type", "N/A"),
                xvector_info.get("version", "N/A"),
            )

            # Store in cache for future requests
            self.xvector_cache[xvector_path] = xvector_info
            logger.info(
                "[%s] Cached xvector at %s (cache size: %d)",
                request_id,
                xvector_path,
                len(self.xvector_cache),
            )

            return xvector_info

        except Exception as e:
            logger.exception(
                "[%s] Failed to load xvector from %s", request_id, xvector_path
            )
            raise RuntimeError(
                f"[{request_id}] Failed to load xvector from {xvector_path}: {e}"
            ) from e

    def add_request(
        self,
        request_id: str,
        prompt: Optional[Union[PromptType, str]] = None,
        params: Optional[Union[SamplingParams, PoolingParams]] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        talker_params: Optional[Union[SamplingParams, PoolingParams]] = None,
        subtalker_params: Optional[SamplingParams] = None,
        *,
        inputs: Optional[PromptType] = None,  # DEPRECATED
        talker_type: str = "default",
        voice_type: str = "default",
        finish: Optional[bool] = False,
        xvector_info: Optional[Union[dict, str]] = None,  # xvec data(dict) or path(str)
        language: Optional[str] = None,
        instructions: Optional[str] = None,
        non_streaming: Optional[
            bool
        ] = False,  # Non-streaming mode: full text embedding at once
    ) -> queue.Queue[Union[RequestOutput, np.ndarray]]:
        # Track request start time for TTFA metrics
        import time as _time
        self.request_start_time[request_id] = _time.monotonic()
        
        # Step 0: Apply TTS generate mode settings
        voice_type, non_streaming, instructions = self._apply_tts_generate_mode(
            request_id, voice_type, non_streaming, instructions
        )

        # Step 1: Process xvector for zero-shot mode
        xvector_ctx = self._process_xvector_info(request_id, xvector_info)
        if xvector_ctx is not None:
            self.request_xvector_info[request_id] = xvector_ctx

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
                f"Voice type '{voice_type}' not found in"
                f"available voice types: {self.code2wav_voice_types}"
            )
        # If instruct is provided,
        # voice is controlled by instruct (no speaker_embed needed)
        use_instruct_mode = instructions is not None and len(instructions.strip()) > 0
        if use_instruct_mode:
            voice_type = ""
            logger.debug(
                "[%s] INSTRUCT_MODE voice_type cleared",
                request_id,
            )
        elif voice_type == "default" or voice_type is None or voice_type == "":
            if self.code2wav_voice_types:
                voice_type = self.code2wav_voice_types[0]

        # Log REQUEST_START with key parameters
        prompt_len = len(prompt) if isinstance(prompt, str) else 0
        logger.info(
            "[%s] REQUEST_START voice=%s prompt_len=%d",
            request_id,
            voice_type,
            prompt_len,
        )

        # Remove GPU memory log from hot path (moved to DEBUG)
        logger.debug("[%s] GPU memory at add_request", request_id)

        # 2. Instruct mode (voice controlled by instruct)
        if self.talker_engine_args and (
            not self._ignore_voice(voice_type) or use_instruct_mode
        ):
            talker_hf_config = self.talker_config
            if hasattr(talker_hf_config, "talker_config"):
                talker_hf_config = talker_hf_config.talker_config

            # Initialize talker_params with defaults if not provided
            if not talker_params:
                if self.talker_generation_config is not None:
                    talker_params = SamplingParams(
                        top_k=self.talker_generation_config.top_k,
                        top_p=self.talker_generation_config.top_p,
                        temperature=self.talker_generation_config.temperature,
                        repetition_penalty=self.talker_generation_config.repetition_penalty,
                        max_tokens=8192,
                    )
                else:
                    talker_params = SamplingParams(
                        top_k=50,
                        top_p=0.9,
                        temperature=1.0,
                        repetition_penalty=1.05,
                        max_tokens=8192,
                    )

            if not subtalker_params:
                subtalker_params = SamplingParams(
                    top_k=50, top_p=1.0, temperature=0.9, detokenize=False
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
            logger.debug(
                "[%s] TALKER_PARAMS %s",
                request_id,
                talker_params,
            )

            if isinstance(prompt, str):
                logger.debug("[%s] REQUEST_PROMPT str len=%d", request_id, len(prompt))
                if finish and len(prompt) == 0:
                    logger.warning("[%s] EMPTY_INPUT", request_id)
                    return
                conversation = [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    },
                ]
                self.requests_str[request_id] = prompt

                prompt = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                    padding=True,
                )
                input_ids = inputs["input_ids"]
            else:
                logger.debug(
                    "[%s] REQUEST_PROMPT tokenized", request_id
                )
                inputs = prompt
            input_ids = inputs["input_ids"].to(self.device)

            # ============ Zero-shot Prompt Construction (Turn-base Mode) ============
            zero_shot_prompt_embed = None
            use_zeroshot = request_id in self.request_xvector_info

            if use_zeroshot:
                try:
                    xvector_ctx = self.request_xvector_info[request_id]
                    x_vector_only_mode = xvector_ctx["x_vector_only_mode"]
                    ref_spk_embedding = xvector_ctx["ref_spk_embedding"]

                    # Turn-base mode:
                    #   construct zero-shot prompt with ref_code and ref_text
                    if not x_vector_only_mode:
                        ref_code = xvector_ctx["ref_code"]
                        ref_language = xvector_ctx["ref_language"]
                        ref_text = xvector_ctx["ref_text"]

                        logger.info(
                            "[%s] Constructing zero-shot prompt (Turn-base): "
                            'ref_text="%s", ref_language=%s',
                            request_id,
                            ref_text[:30] + "..." if len(ref_text) > 30 else ref_text,
                            ref_language,
                        )

                        # Tokenize ref_text
                        ref_conversation = [
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": ref_text}],
                            }
                        ]
                        ref_prompt = self.processor.apply_chat_template(
                            ref_conversation, add_generation_prompt=True, tokenize=False
                        )
                        ref_inputs = self.processor(
                            text=ref_prompt, return_tensors="pt", padding=True
                        )
                        ref_ids = ref_inputs["input_ids"].to(self.device)

                        # 1. Role Start: <|im_start|>assistant\n
                        _prompt_embed_role_start = self.text_projection(
                            self._get_text_embeddings()(ref_ids[:, :3])
                        )

                        # 2. Tag Embeddings: language_tag + speaker_embedding
                        # 根据 transformers 实现，如果 ref_language 为 None
                        # 或 "auto" 或不在 config 中，使用 nothing_id
                        if ref_language is None or ref_language.lower() == "auto":
                            ref_language_id = None
                        else:
                            ref_lang = ref_language.lower()
                            if ref_lang not in talker_hf_config.codec_language_id:
                                logger.warning(
                                    '[%s] ref_language "%s" not in codec_language_id, '
                                    "setting to None",
                                    request_id,
                                    ref_language,
                                )
                                ref_language_id = None
                            else:
                                ref_language_id = talker_hf_config.codec_language_id[
                                    ref_lang
                                ]

                        if ref_language_id is None:
                            ref_codec_prefill = [
                                talker_hf_config.codec_nothink_id,
                                talker_hf_config.codec_think_bos_id,
                                talker_hf_config.codec_think_eos_id,
                            ]
                        else:
                            ref_codec_prefill = [
                                talker_hf_config.codec_think_id,
                                talker_hf_config.codec_think_bos_id,
                                ref_language_id,
                                talker_hf_config.codec_think_eos_id,
                            ]

                        codec_input_embed = self._get_codec_embeddings()(
                            torch.tensor(
                                [ref_codec_prefill],
                                device=self.device,
                                dtype=torch.long,
                            )
                        )
                        codec_input_embed = torch.cat(
                            [
                                codec_input_embed,
                                ref_spk_embedding.view(1, 1, -1).to(self.device),
                            ],
                            dim=1,
                        )

                        _prompt_embed_tag = (
                            self.tts_pad_embed.expand(
                                -1, codec_input_embed.shape[1], -1
                            )
                            + codec_input_embed
                        )

                        # 3. Code Embeddings: ref_code + ref_text alignment
                        text_embed = torch.cat(
                            [
                                self.tts_bos_embed,
                                self.text_projection(
                                    self._get_text_embeddings()(ref_ids[:, 3:-2])
                                ),
                                self.tts_eos_embed,
                            ],
                            dim=1,
                        )

                        # Codec embedding: sum across 16 layers
                        # Layer 0: use talker's codec_embedding
                        # Layers 1-15: use code_predictor's embeddings
                        # ref_code shape: [seq_len, 16]
                        num_code_groups = ref_code.shape[1]  # Should be 16
                        codec_embed_list = []
                        for i in range(num_code_groups):
                            if i == 0:
                                # Layer 0: talker's codec_embedding
                                codec_embed_list.append(
                                    self._get_codec_embeddings()(
                                        ref_code[:, i : i + 1].to(self.device)
                                    )
                                )
                            else:
                                # Layers 1-15: code_predictor's embeddings
                                codec_embed_list.append(
                                    self.code_predictor_embeddings[i - 1](
                                        ref_code[:, i : i + 1].to(self.device)
                                    )
                                )
                        # Concatenate and sum across the layer dimension
                        codec_embed = (
                            torch.cat(codec_embed_list, dim=1).sum(1).unsqueeze(0)
                        )

                        codec_embed = torch.cat(
                            [
                                self._get_codec_embeddings()(
                                    torch.tensor(
                                        [
                                            [
                                                talker_hf_config.codec_pad_id,
                                                talker_hf_config.codec_bos_id,
                                            ]
                                        ],
                                        device=self.device,
                                        dtype=torch.long,
                                    )
                                ),
                                codec_embed,
                                self._get_codec_embeddings()(
                                    torch.tensor(
                                        [[talker_hf_config.codec_eos_token_id]],
                                        device=self.device,
                                        dtype=torch.long,
                                    )
                                ),
                            ],
                            dim=1,
                        )

                        # Align lengths and add
                        if codec_embed.shape[1] >= text_embed.shape[1]:
                            padding_num = codec_embed.shape[1] - text_embed.shape[1]
                            _prompt_embed_code = codec_embed + torch.cat(
                                [
                                    text_embed,
                                    self.tts_pad_embed.expand(-1, padding_num, -1),
                                ],
                                dim=1,
                            )
                        else:
                            codec_mask_num = text_embed.shape[1] - codec_embed.shape[1]
                            _prompt_embed_code = text_embed + torch.cat(
                                [
                                    codec_embed,
                                    torch.zeros(
                                        1,
                                        codec_mask_num,
                                        codec_embed.shape[2],
                                        device=self.device,
                                    ),
                                ],
                                dim=1,
                            )

                        # 4. Role End: <|im_end|>\n
                        _prompt_embed_role_end = self.text_projection(
                            self._get_text_embeddings()(ref_ids[:, -2:])
                        )

                        # Concatenate full zero-shot prompt
                        zero_shot_prompt_embed = torch.cat(
                            [
                                _prompt_embed_role_start,
                                _prompt_embed_tag,
                                _prompt_embed_code,
                                _prompt_embed_role_end,
                            ],
                            dim=1,
                        )

                        logger.info(
                            "[%s] Zero-shot prompt shape: %s",
                            request_id,
                            zero_shot_prompt_embed.shape,
                        )

                except Exception as e:
                    logger.exception(
                        "[%s] Failed to construct zero-shot prompt: %s. "
                        "Falling back to XVector-only mode.",
                        request_id,
                        str(e),
                    )

                    # Fallback to XVector-only mode if prompt construction fails
                    zero_shot_prompt_embed = None
                    if request_id in self.request_xvector_info:
                        self.request_xvector_info[request_id]["x_vector_only_mode"] = (
                            True
                        )

            # ============ Target Text Talker Input Construction ============
            # Get speaker embedding
            # Priority: Instruct mode > Zero-shot mode > SFT mode
            speaker_embed = None
            if use_instruct_mode:
                # Instruct mode: no speaker embedding needed,
                # voice is controlled by instruct
                speaker_embed = None
                logger.info(
                    "[%s] Instruct mode: no speaker embedding, "
                    "voice controlled by instruct",
                    request_id,
                )
            elif use_zeroshot:
                # Zero-shot mode: use ref_spk_embedding from xvector
                xvector_ctx = self.request_xvector_info[request_id]
                if xvector_ctx["x_vector_only_mode"]:
                    # XVector-only mode: use speaker embedding
                    speaker_embed = xvector_ctx["ref_spk_embedding"].to(self.device)
                    logger.info(
                        "[%s] Zero-shot mode (XVector-only): "
                        "using speaker embedding from xvector, shape=%s",
                        request_id,
                        speaker_embed.shape,
                    )
                else:
                    # Turn-base mode: speaker_embed will be None
                    # (already in zero-shot prompt)
                    speaker_embed = None
                    logger.info(
                        "[%s] Zero-shot mode (Turn-base): "
                        "speaker_embed set to None (already included in prompt)",
                        request_id,
                    )
            else:
                # SFT mode: use spk_id from voice_type
                if voice_type.lower() not in talker_hf_config.spk_id:
                    raise NotImplementedError(f"Speaker {voice_type} not implemented")
                spk_id = talker_hf_config.spk_id[voice_type.lower()]
                speaker_embed = self._get_codec_embeddings()(
                    torch.tensor([spk_id], device=self.device, dtype=torch.long)
                )

            # Language boost for talker
            language_tag = None
            if language:
                language = language.lower()
                # For zero-shot mode, skip dialect check (only apply to SFT mode)
                if use_zeroshot:
                    # Zero-shot: only skip language tag for 'auto'
                    if language == "auto":
                        language_tag = None
                    else:
                        if language not in talker_hf_config.codec_language_id:
                            raise ValueError(f"Language {language} not implemented")
                        else:
                            language_tag = talker_hf_config.codec_language_id[language]
                else:
                    # SFT mode: check dialect
                    if (
                        language == "chinese"
                        and talker_hf_config.spk_is_dialect.get(voice_type, False)
                    ) or language == "auto":
                        language_tag = None
                    else:
                        if language not in talker_hf_config.codec_language_id:
                            raise ValueError(f"Language {language} not implemented")
                        else:
                            language_tag = talker_hf_config.codec_language_id[language]
            logger.info(
                "[%s] generate talker input language: %s, %s, %s",
                request_id,
                voice_type,
                language,
                language_tag,
            )
            if language_tag is None:
                codec_prefill_list = [
                    talker_hf_config.codec_nothink_id,
                    talker_hf_config.codec_think_bos_id,
                    talker_hf_config.codec_think_eos_id,
                ]
            else:
                codec_prefill_list = [
                    talker_hf_config.codec_think_id,
                    talker_hf_config.codec_think_bos_id,
                    language_tag,
                    talker_hf_config.codec_think_eos_id,
                ]

            codec_input_embed_0 = self._get_codec_embeddings()(
                torch.tensor([codec_prefill_list], device=self.device, dtype=torch.long)
            )
            codec_input_embed_1 = self._get_codec_embeddings()(
                torch.tensor(
                    [[talker_hf_config.codec_pad_id, talker_hf_config.codec_bos_id]],
                    device=self.device,
                    dtype=torch.long,
                )
            )

            # Build codec_input_embedding with or without speaker_embed
            if speaker_embed is not None:
                # Normal mode: include speaker embedding
                codec_input_embedding = torch.cat(
                    [
                        codec_input_embed_0,
                        speaker_embed.view(1, 1, -1),
                        codec_input_embed_1,
                    ],
                    dim=1,
                )
            else:
                # Instruct-only mode: no speaker embedding
                codec_input_embedding = torch.cat(
                    [codec_input_embed_0, codec_input_embed_1], dim=1
                )

            _talker_input_embed_role = self.text_projection(
                self._get_text_embeddings()(input_ids[:, :3])
            )

            if non_streaming:
                # Non-streaming mode: full text embedding at once
                # Structure: tts_pad*N + full_text_embed + tts_eos + tts_pad + codec_bos
                full_text_ids = input_ids[
                    :, 3:-5
                ]  # Full text tokens (excluding role and trailing)
                full_text_embed = self.text_projection(
                    self._get_text_embeddings()(full_text_ids)
                )
                full_text_len = full_text_embed.shape[1]

                # codec_pad for full text
                codec_pad_embed = self._get_codec_embeddings()(
                    torch.tensor(
                        [[talker_hf_config.codec_pad_id] * full_text_len],
                        device=self.device,
                        dtype=torch.long,
                    )
                )

                # codec_bos for the final position
                codec_bos_embed = self._get_codec_embeddings()(
                    torch.tensor(
                        [[talker_hf_config.codec_bos_id]],
                        device=self.device,
                        dtype=torch.long,
                    )
                )

                _talker_input_embed = (
                    torch.cat(
                        [
                            self.tts_pad_embed.expand(
                                -1, codec_input_embedding.shape[1] - 2, -1
                            ),
                            self.tts_bos_embed,
                        ],
                        dim=1,
                    )
                    + codec_input_embedding[:, :-1, :]
                )  # Remove last position (will be replaced)

                # Add full text + tts_eos + final position
                _talker_input_embed = torch.cat(
                    [
                        _talker_input_embed,
                        full_text_embed + codec_pad_embed,  # full text with codec_pad
                        self.tts_eos_embed,  # tts_eos
                        self.tts_pad_embed + codec_bos_embed,  # tts_pad + codec_bos
                    ],
                    dim=1,
                )

                logger.info(
                    "[%s] Non-streaming mode: full_text_len=%d, talker_input_shape=%s",
                    request_id,
                    full_text_len,
                    _talker_input_embed.shape,
                )
            else:
                # Streaming mode: only first text token, rest via queue
                # Structure: tts_pad*N + tts_bos + tts_text_first_token
                _talker_input_embed = (
                    torch.cat(
                        [
                            self.tts_pad_embed.expand(
                                -1, codec_input_embedding.shape[1] - 2, -1
                            ),
                            self.tts_bos_embed,
                            self.text_projection(
                                self._get_text_embeddings()(input_ids[:, 3:4])
                            ),
                        ],
                        dim=1,
                    )
                    + codec_input_embedding
                )

            talker_input_embed = torch.cat(
                [_talker_input_embed_role, _talker_input_embed], dim=1
            )

            # ============ Merge Instruct Prompt (Instruct TTS) ============
            if instructions is not None and len(instructions.strip()) > 0:
                try:
                    instruct_text = (
                        f"<|im_start|>user\n{instructions.strip()}<|im_end|>\n"
                    )
                    instruct_inputs = self.processor(
                        text=instruct_text, return_tensors="pt", padding=True
                    )
                    instruct_ids = instruct_inputs["input_ids"].to(self.device)
                    instruct_text_embed = self._get_text_embeddings()(instruct_ids)
                    instruct_embed = self.text_projection(instruct_text_embed)
                    talker_input_embed = torch.cat(
                        [instruct_embed, talker_input_embed], dim=1
                    )
                except Exception as e:
                    logger.error(  # noqa: G201
                        "[%s] Failed to process instruct text: %s",
                        request_id,
                        str(e),
                        exc_info=True,
                    )

            # ============ Merge Zero-shot Prompt (Turn-base only) ============
            if zero_shot_prompt_embed is not None:
                # [zero_shot_prompt] + [instruct (if any)] + [role + tts_text]
                talker_input_embed = torch.cat(
                    [zero_shot_prompt_embed, talker_input_embed], dim=1
                )
                logger.info(
                    "[%s] Final talker input shape (with zero-shot): %s",
                    request_id,
                    talker_input_embed.shape,
                )

            # # 叫通义千问，是阿里云的开源大模型。
            # trailing_text_hidden = torch.cat(
            #     (self.text_projection(self._get_text_embeddings()(
            #         input_ids[:, 4:-5])), self.tts_eos_embed),
            #     dim=1)

            talker_input_id = (
                torch.ones([1, talker_input_embed.shape[1]]).long().flatten().tolist()
            )
            # logger.debug(f"{len(talker_input_id)=} {talker_input_embed.shape=}")

            ## 输入talker
            talker_input_embed = talker_input_embed.cpu()
            tokens_prompt = TokensPrompt(
                prompt_token_ids=talker_input_id,
                prompt_embeds=talker_input_embed.squeeze(0),
            )

            self.input_tokens[request_id] = 1
            generator = self.talker_engine.generate(
                request_id=request_id,
                prompt=tokens_prompt,
                sampling_params=talker_params,
                # subtalker_sampling_params=subtalker_params,
                using_external_data=True,
            )
            self.talker_inputs[request_id] = queue.Queue()
            min_idx = 0
            # for i in range(self.code2wav_data_parallelism):
            #     if self.task_2_code2wav[i].length(
            #     ) < self.task_2_code2wav[min_idx].length():
            #         min_idx = i
            # self.task_2_code2wav[min_idx].append(request_id)

            if non_streaming:
                # Non-streaming mode: all text already in talker_input_embed,
                # just signal finish
                if finish:
                    self.talker_inputs[request_id].put(
                        (self.talker_config.tts_eos_token_id, True)
                    )
                logger.info(
                    "[%s] Non-streaming mode: text already embedded, finish=%s",
                    request_id,
                    finish,
                )
            else:
                # Streaming mode: put text tokens to queue one by one
                # TODO: put hidden directly
                for text_id in input_ids[:, 4:-5].flatten().tolist():
                    self.talker_inputs[request_id].put((text_id, False))
                if finish:
                    self.talker_inputs[request_id].put(
                        (self.talker_config.tts_eos_token_id, True)
                    )
                logger.info(
                    "[%s] Streaming mode: text tokens to queue one by one, finish=%s",
                    request_id,
                    finish,
                )

            self._listen_talker(
                request_id,
                voice_type,
                generator,
                self.output_queue[request_id],
                self.talker_inputs[request_id],
                min_idx,
            )

        return self.output_queue[request_id]

    def append_input(
        self,
        request_id: str,
        text_ids: Optional[List[int]] = None,
        texts: Optional[str] = None,
        finish: Optional[bool] = False,
    ):
        logger.info(
            "[%s] append_input input texts %s text_ids %s finish %s",
            request_id,
            texts,
            text_ids,
            finish,
        )
        if request_id not in self.talker_inputs:
            raise Exception(f"request not added {request_id}")
        if texts is not None and len(texts) > 0:
            conversation = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": texts,
                        }
                    ],
                },
            ]

            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"]
            for text_id in input_ids[:, 3:-5].flatten().tolist():
                self.talker_inputs[request_id].put((text_id, False))
        elif text_ids is not None and len(text_ids) > 0:
            for text_id in text_ids:
                self.talker_inputs[request_id].put((text_id, False))
        if finish:
            self.talker_inputs[request_id].put(
                (self.talker_config.tts_eos_token_id, True)
            )
        return

    def abort_request(self, request_id: str):
        logger.info("[%s] REQUEST_ABORT", request_id)
        self.clear_talker_engine(request_id)
        self.clear_talker_inputs(request_id)
        self.clear_talker_resume_thread(request_id)
        self.clear_request_status(request_id)
        self.clear_code2wav_engine(request_id)

    def clear_talker_engine(self, request_id: str) -> None:
        try:
            # cancel talker engine request
            asyncio.run_coroutine_threadsafe(
                self.talker_engine.abort(request_id),
                self._loop,
            ).result(timeout=5.0)
        except TimeoutError:
            logger.error("[%s] Clear talker engine timeout", request_id)
        except Exception as e:
            logger.error(f"[%s] Clear talker engine failed: {e}", request_id)

    def clear_talker_inputs(self, request_id: str) -> None:
        try:
            # clear input queue(forbid resume thread block)
            if request_id in self.talker_inputs:
                input_queue = self.talker_inputs[request_id]
                cleared_count = 0
                while not input_queue.empty():
                    try:
                        input_queue.get_nowait()
                        cleared_count+=1
                    except queue.Empty:
                        break
                del self.talker_inputs[request_id]
                logger.debug(
                    "[%s] CLEAR_INPUT_QUEUE count=%d", 
                    request_id,
                    cleared_count,
                )
        except Exception as e:
            logger.error(f"[%s] Clear input queue failed: {e}", request_id)
    
    def clear_talker_resume_thread(self, request_id: str) -> None:
        try:
            if request_id in self.talker_resumer:
                self.talker_resumer[request_id].join(timeout=5.0)
                del self.talker_resumer[request_id]
        except Exception as e:
            logger.error(f"[%s] Clear talker resume thread failed: {e}", request_id)
    
    def clear_request_status(self, request_id: str) -> None:
        try:
            if request_id in self.talker_requests:
                del self.talker_requests[request_id]
            if request_id in self.talker_finished:
                self.talker_finished.remove(request_id)
            if request_id in self.request_xvector_info:
                del self.request_xvector_info[request_id]
            # Clean up TTFA tracking
            if request_id in self.request_start_time:
                del self.request_start_time[request_id]
        except Exception as e:
            logger.error(f"[%s] Error clearing request status: {e}", request_id)
    
    def clear_code2wav_engine(self, request_id: str) -> None:
        try:
            for code2wav_index in range(len(self._code2wav_engine_clients)):
                asyncio.run_coroutine_threadsafe(
                    self._code2wav_engine_clients[code2wav_index].abort_request(
                        request_id=request_id,
                    ),
                    self._loop,
                ).result()
        except Exception as e:
            logger.error(f"[%s] Clear code2wav engine failed: {e}", request_id)

    def _apply_tts_generate_mode(
        self,
        request_id: str,
        voice_type: str,
        non_streaming: bool,
        instructions: Optional[str],
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Apply TTS generate mode settings based on VLLM_QWEN_TTS_GENERATE_MODE env.

        Returns:
            Tuple of (voice_type, non_streaming, instructions) after mode applied.
        """
        mode = VLLM_QWEN_TTS_GENERATE_MODE
        config = _TTS_GENERATE_MODE_CONFIG.get(
            mode, _TTS_GENERATE_MODE_CONFIG["default"]
        )

        # Apply forced voice_type if configured
        if config["force_voice_type"] is not None:
            voice_type = config["force_voice_type"]

        # Apply forced non_streaming if configured
        if config["force_non_streaming"] is not None:
            non_streaming = config["force_non_streaming"]

        # Ignore instructions if configured
        if config["ignore_instructions"]:
            if instructions is not None and len(instructions.strip()) > 0:
                logger.info(
                    "[%s] TTS mode=%s: instructions ignored (len=%d)",
                    request_id,
                    mode,
                    len(instructions.strip()),
                )
            instructions = None
        else:
            logger.info(
                "[%s] TTS mode=%s: voice_type=%s, non_streaming=%s",
                request_id,
                mode,
                voice_type if voice_type else '""',
                non_streaming,
            )

        return voice_type, non_streaming, instructions

    def _process_xvector_info(
        self,
        request_id: str,
        xvector_info: Optional[Union[dict, str]],
    ) -> Optional[dict]:
        """
        Process xvector_info for zero-shot TTS.

        Args:
            request_id: Request identifier for logging
            xvector_info: Either a dict with xvector data or a path string

        Returns:
            None if not using zero-shot, otherwise a dict with:
            - use_zeroshot: bool
            - x_vector_only_mode: bool
            - ref_spk_embedding: torch.Tensor
            - ref_code: Optional[torch.Tensor]
            - ref_language: Optional[str]
            - ref_text: Optional[str]
        """
        if xvector_info is None:
            return None

        # Load from path if string
        if isinstance(xvector_info, str):
            xvector_info = self.load_zeroshot_xvector(request_id, xvector_info)

        ref_spk_embedding = xvector_info["xvector"]
        ref_code = xvector_info.get("prompt_code")
        ref_language = xvector_info.get("language_type")
        ref_text = xvector_info.get("prompt_text")

        # Determine mode: Turn-base vs XVector-only
        x_vector_only_mode = True
        if ref_code is not None:
            has_valid_ref_text = ref_text is not None and len(ref_text.strip()) > 0
            if ref_language is not None and has_valid_ref_text:
                x_vector_only_mode = False
                logger.info(
                    "[%s] Zero-shot mode: Turn-base | ref_language=%s, ref_text_len=%d",
                    request_id,
                    ref_language,
                    len(ref_text),
                )
            else:
                logger.warning(
                    "[%s] Zero-shot mode: Turn-base incomplete, fallback to XVector",
                    request_id,
                )

            # Convert numpy to torch if needed
            if isinstance(ref_code, np.ndarray):
                ref_code = torch.from_numpy(ref_code).long()
        else:
            logger.info(
                "[%s] Zero-shot mode: XVector-only | xvector: %s",
                request_id,
                ref_spk_embedding.shape,
            )

        return {
            "use_zeroshot": True,
            "x_vector_only_mode": x_vector_only_mode,
            "ref_spk_embedding": ref_spk_embedding,
            "ref_code": ref_code,
            "ref_language": ref_language,
            "ref_text": ref_text,
        }

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
