r"""
批量音频评测脚本：支持 VAD 语音活动检测切分、多线程并发推理，以及完整的 OmniDRCA2Engine 双路引擎高级配置。

用法:
    python batch_eval.py \
        /path/to/model_ckpt \                  # [位置参数1] ckpt_dir: 模型 checkpoint 目录
        /path/to/input_jsonl_list.txt \        # [位置参数2] input_jsonl_list: 包含多个输入 jsonl 文件路径的列表文件
        /path/to/output_dir_list.txt \         # [位置参数3] output_dir_list: 包含对应输出目录路径的列表文件
        --njobs 20 \                           # [并发配置] 行级别的多线程并发数，默认 20
        --vad-mode farfield \                  # [VAD配置] vad 模式，可选 farfield/nearfield/opensource/none，默认 farfield
        --use-oss-cache \                      # [路径配置] 开启后会将 oss 路径替换为本地缓存路径
        --thinker-only \                       # [引擎配置] 仅启用 Thinker 引擎 (不初始化 Talker)
        --thinker-devices '[0,1,2,3,4,5,6,7]' \# [引擎配置] Thinker 引擎使用的 GPU 卡号列表
        --talker-devices '[1]' \               # [引擎配置] Talker 引擎使用的 GPU 卡号列表 (非 thinker-only 时生效)
        --enable-chunked-prefill \             # [引擎配置] 开启 Chunked Prefill 特性
        --enable-prefix-caching \              # [引擎配置] 开启 Prefix Caching 缓存支持
        --enforce-eager \                      # [引擎配置] 全局强制启用 Eager 模式 (也可分别用 --thinker-enforce-eager 或 --talker-enforce-eager)
        --dtype bfloat16 \                     # [引擎配置] 数据类型选择，如 auto, half, bfloat16 等
        --block-size 16 \                      # [引擎配置] KV Cache 的 Block Size
        --thinker-quantization fp8 \           # [引擎配置] Thinker 引擎的量化方法 (可选)
        --talker-quantization fp8 \            # [引擎配置] Talker 引擎的量化方法 (可选)
        --max-tokens 2048 \                    # [采样配置] Thinker 生成的最大 token 数 (Talker 会按配置自动扩大 5 倍)
        --seed 0                               # [采样配置] 随机种子，用于固定输出
"""
import argparse
import asyncio
import os
import sys
import json
import traceback
import re
import uuid
import time
import logging
import types
from collections.abc import Mapping
from pathlib import Path

from s2s_key_utils import make_output_key

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer

# Keep the system vLLM build (which includes the 0803 S2S hidden-state export)
# and override only its OmniDRCA2 thinker model with our dense-4B-compatible
# implementation.  This path manipulation is repeated in spawned workers
# because they re-import this entrypoint.
import vllm.model_executor.models as _vllm_models
_override_models = Path(__file__).resolve().parent / "vllm_overrides"
if str(_override_models) not in _vllm_models.__path__:
    _vllm_models.__path__.insert(0, str(_override_models))
_vllm_models.ModelRegistry.register_model(
    "Qwen3_5ForCausalLM",
    "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
)

from vllm.config import CompilationConfig
from vllm.config.compilation import PassConfig
from vllm.engine.omnidrca2_engine import OmniDRCA2Engine, SynchronizedGenerator
from vllm.engine.shared_hidden_buffer import (
    SharedHiddenBuffer,
    SharedMemoryHiddenBuffer,
)
from vllm.inputs import TextPrompt
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, ExternalData
from vllm.v1.engine.async_llm import AsyncEngineArgs
from vllm.utils.load_audio_utils import load_audio_text_image_video
from vllm.logger import init_logger


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = init_logger("vllm.batch_eval")
logger.setLevel(logging.INFO)


# The 0803 vLLM fast SHM writer retains every producer-side SharedMemory
# handle in _created_shm until engine shutdown. The consumer unlinks the
# name after reading, but cannot close the producer's file descriptor, so a
# long dataset eventually reaches RLIMIT_NOFILE and deadlocks both engines.
# Closing the producer handle after the completed write is safe: POSIX SHM
# remains available by name until the consumer unlinks it.
_original_fast_hidden_write = SharedMemoryHiddenBuffer._fast_write


def _fast_hidden_write_without_fd_leak(
    self, tensor, request_id, timeout=None
):
    ok = _original_fast_hidden_write(self, tensor, request_id, timeout)
    if ok:
        with self._lock:
            shm = self._created_shm.pop(request_id, None)
        if shm is not None:
            shm.close()
    return ok


SharedMemoryHiddenBuffer._fast_write = _fast_hidden_write_without_fd_leak


def _is_int_sequence(value):
    if isinstance(value, np.ndarray):
        return value.size > 0 and np.issubdtype(value.dtype, np.integer)
    if torch.is_tensor(value):
        return value.numel() > 0 and value.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8,
        )
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(x, (int, np.integer)) for x in value)
    return False


def _to_codec_array(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        value = np.asarray(value, dtype=np.int64)
    elif not isinstance(value, np.ndarray):
        return None
    value = np.asarray(value)
    if value.ndim == 0:
        return None
    if value.ndim > 1:
        value = value.reshape(-1)
    return value.astype(np.int64, copy=False)


def _summarize_output_object(obj, depth=0, max_depth=2):
    if obj is None or depth > max_depth:
        return repr(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if torch.is_tensor(obj):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, np.ndarray):
        return {
            "type": "np.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, (list, tuple)):
        return [_summarize_output_object(x, depth + 1, max_depth) for x in obj[:8]]
    if isinstance(obj, dict):
        return {
            str(k): _summarize_output_object(v, depth + 1, max_depth)
            for k, v in list(obj.items())[:32]
        }
    data = {"type": type(obj).__name__}
    try:
        data["attrs"] = {
            k: _summarize_output_object(v, depth + 1, max_depth)
            for k, v in list(vars(obj).items())[:32]
        }
    except TypeError:
        data["repr"] = repr(obj)
    return data


def _extract_codec_from_object(obj, allow_token_ids_fallback=False, depth=0, max_depth=4):
    if obj is None or depth > max_depth:
        return None
    if _is_int_sequence(obj):
        return _to_codec_array(obj)

    preferred_names = (
        "codec", "codex", "codec_ids", "codex_ids", "audio_tokens",
        "audio_token_ids", "talker_token_ids", "talker_tokens",
        "speech_tokens", "speech_token_ids",
    )
    fallback_names = ("token_ids", "output_token_ids") if allow_token_ids_fallback else ()

    if isinstance(obj, dict):
        for name in preferred_names + fallback_names:
            if name in obj:
                arr = _extract_codec_from_object(
                    obj[name], allow_token_ids_fallback=allow_token_ids_fallback,
                    depth=depth + 1, max_depth=max_depth,
                )
                if arr is not None:
                    return arr
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            arr = _extract_codec_from_object(
                value, allow_token_ids_fallback=allow_token_ids_fallback,
                depth=depth + 1, max_depth=max_depth,
            )
            if arr is not None:
                return arr
        return None

    try:
        attrs = vars(obj)
    except TypeError:
        attrs = {}
    for name in preferred_names + fallback_names:
        if name in attrs:
            arr = _extract_codec_from_object(
                attrs[name], allow_token_ids_fallback=allow_token_ids_fallback,
                depth=depth + 1, max_depth=max_depth,
            )
            if arr is not None:
                return arr
    return None


def _codec_quality(text, codec, min_char_ratio, max_char_ratio, max_codec_len):
    if codec is None:
        return {
            "ok": False,
            "reason": "missing_codec",
            "text_chars": len((text or "").strip()),
            "codec_len": 0,
            "codec_char_ratio": 0.0,
        }
    codec_len = int(np.asarray(codec).reshape(-1).shape[0])
    text_chars = len((text or "").strip())
    ratio = codec_len / max(text_chars, 1)
    if max_codec_len > 0 and codec_len > max_codec_len:
        ok = False
        reason = "codec_too_long"
    elif text_chars != 0 and ratio < min_char_ratio:
        ok = False
        reason = "codec_too_short_for_text"
    elif text_chars != 0 and max_char_ratio > 0 and ratio > max_char_ratio:
        ok = False
        reason = "codec_too_long_for_text"
    else:
        ok = True
        reason = "ok"
    return {
        "ok": ok,
        "reason": reason,
        "text_chars": text_chars,
        "codec_len": codec_len,
        "codec_char_ratio": ratio,
        "min_codec_char_ratio": min_char_ratio,
        "max_codec_char_ratio": max_char_ratio,
        "max_codec_len": max_codec_len,
    }

def starts_with_language_letter(s):
    # 数字
    number = r"[0-9]"
    # 英文字母 (a-zA-Z)
    english = r"[a-zA-Z]"
    # 法语字母（包括带重音符号的字母）
    french = r"[a-zA-Zàâçéèêëîïôûùüÿæœ]"
    # 韩语字母
    korean = r"[가-힣]"
    # 德语字母（包括Umlaut和scharfes S)
    german = r"[a-zA-Zäöüß]"
    # 俄语字母
    russian = r"[а-яА-ЯёЁ]"
    # 意大利语字母
    italian = r"[a-zA-Zàèéìíîòóùú]"
    # 西班牙语字母（包括带重音符号的字母和ñ)
    spanish = r"[a-zA-Záéíñóúü]"

    # 合并所有语言的正则表达式
    pattern = f"^{number}|{english}|{french}|{korean}|{german}|{russian}|{italian}|{spanish}"
    return bool(re.match(pattern, s))

def join_strings_with_space(strings, use_hist='none', add_vad_delimiter=False):
    result = ""

    for i, s in enumerate(strings):
        if i != len(strings) - 1 and (use_hist == "punc" or use_hist == "punc+hist"):
            s = re.sub(f"{PUNC}$", "", s)
        if i > 0 and len(result.rstrip()) > 0:
            prev_char = result.rstrip()[-1]
            if add_vad_delimiter:
                result += "|"
            if starts_with_language_letter(s):
                if not prev_char.isspace():
                    result += " "
                s = s[0].lower() + s[1:]
            if len(s) >= 4 and s[0] in ".?!":
                if s[1].isspace():
                    s = s[0] + s[1] + s[2].upper() + s[3:]
                else:
                    s = s[0] + " " + s[1].upper() + s[2:]
            else:
                if len(s) >= 2 and s[0] in ",;":
                    if not s[1].isspace():
                        s = s[0] + " " + s[1:]
        result += s

    return result


class VAD_Processor:
    def __init__(self, vad_mode="farfield"):
        self.vad_mode = vad_mode
        self.lock = threading.Lock()  # 调C++ lib, 添加一个互斥锁

        if self.vad_mode == "farfield":
            from python_vad.nls_vad import AlsVad
            print("Using farfield VAD model.")
            self.model_dir = '/path/to/20201118'
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            # self.model_dir = os.path.join(current_dir, "python_vad/model/20201118")
            self.frm_len = 20  #单位是毫秒，只在需要流式调用时设置，可以是10毫秒的任意倍数
            self.is_streaming_mode = False
            #算法初始化
            self.vad = AlsVad()
            self.vad.create_vad_instance(self.model_dir)

        elif self.vad_mode == "nearfield":
            from python_vad.nls_vad import AlsVad
            print("Using nearfield VAD model.")
            self.model_dir = '/path/to/20240329_V2'
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            # self.model_dir = os.path.join(current_dir, "python_vad/model/20240329_V2")
            self.frm_len = 20  #单位是毫秒，只在需要流式调用时设置，可以是10毫秒的任意倍数
            self.is_streaming_mode = False
            #算法初始化
            self.vad = AlsVad()
            self.vad.create_vad_instance(self.model_dir)

        elif self.vad_mode == "opensource":
            print("Using opensource VAD model.")
            vad_model = AutoModel(model=os.environ.get("VAD_MODEL"), model_revision="v2.0.4")
            self.vad = vad_model

        else:
            raise ValueError(f"Invalid vad_mode:{self.vad_mode}. Choose from ['farfield', 'nearfield', 'opensource'].")
        print("VAD initialized.")

    def __del__(self):
        if self.vad_mode in ["farfield", "nearfield"]:
            self.vad.destroy_vad_instance()
            print("VAD instance destroyed.")

    def vad_engine_online(self, waveform, sample_rate):
        print("Start VAD processing...")
        #音频文件加载并转成int16格式
        # waveform, sample_rate = torchaudio.load(audio_path)
        waveform = (waveform.squeeze(dim=0).numpy() * (2**15)).astype(np.int16)

        #vad处理
        self.vad.begin_utterance() #每处理一个音频文件调用一次
        speech_timestamps = [] #vad结果，包含语音段起尾点信息，单位是毫秒
        if self.is_streaming_mode:
            pcm_len = int(self.frm_len*self.sample_rate/1000)
            for i in range(0, len(waveform), pcm_len):
                cur_wav = waveform[i: i + pcm_len].tobytes()
                is_end = (i + pcm_len) >= len(waveform)
                self.vad.set_data(cur_wav, is_end)
                cur_timestamps = self.vad.do_detect()
                if len(cur_timestamps) > 0:
                    speech_timestamps.extend(cur_timestamps)
        else:
            self.vad.set_data(waveform.tobytes(), True)
            cur_timestamps = self.vad.do_detect()
            if len(cur_timestamps) > 0:
                speech_timestamps.extend(cur_timestamps)
        self.vad.end_utterance() #每处理一个音频文件调用一次

        #算法注销
        # self.vad.destroy_vad_instance()

        vad_res = []
        for i in range(len(speech_timestamps)):
            vad_res.append([speech_timestamps[i]['start'], speech_timestamps[i]['end']])
        print("VAD result (in ms):", vad_res)

        return vad_res

    def vad_engine_torch(self, waveform):
        # vad处理
        speech_timestamps = self.vad.generate(input=waveform)
        vad_res = speech_timestamps[0]['value']
        return vad_res

    def get_vad_result(self, waveform, sample_rate):
        with self.lock:
            print("Get VAD result...")
            if self.vad_mode in ["farfield", "nearfield"]:
                vad_res = self.vad_engine_online(waveform, sample_rate)
            elif self.vad_mode == "opensource":
                vad_res = self.vad_engine_torch(waveform)
            else:
                raise ValueError(f"Invalid vad_mode:{self.vad_mode}. Choose from ['farfield', 'nearfield', 'opensource'].")

        return vad_res

class Inference:
    def __init__(self, args):
        self.args = args

        # Worker processes are created with the engines below, so publish the
        # conditioning mode before spawning them.  Hidden-state conditioning is
        # the training-aligned default; explicitly clear any inherited opt-in so
        # a parent shell cannot silently switch the Talker to text-token input.
        if args.use_thinker_text_for_talker:
            os.environ["VLLM_CODEX_USE_TEXT_TOKEN_INPUT"] = "1"
            logger.info("Talker conditioning: CosyVoice text-token embedding (explicit opt-in)")
        else:
            os.environ.pop("VLLM_CODEX_USE_TEXT_TOKEN_INPUT", None)
            logger.info(
                "Talker conditioning: projected Thinker hidden states "
                "(training-aligned; includes expanded prompt/audio positions)"
            )

        thinker_engine_args = AsyncEngineArgs(
            model=args.ckpt_dir,
            trust_remote_code=True,
            gpu_memory_utilization=args.thinker_gpu_memory_utilization,
            enable_chunked_prefill=args.enable_chunked_prefill,
            tensor_parallel_size=len(args.thinker_devices),
            enforce_eager=args.enforce_eager or args.thinker_enforce_eager,
            compilation_config=CompilationConfig(
                cudagraph_mode="PIECEWISE", use_inductor=True, custom_ops=["all"]
            ),
            distributed_executor_backend="mp",
            limit_mm_per_prompt={"audio": 16},
            max_model_len=32768,
            max_num_batched_tokens=32768 if args.enable_chunked_prefill else None,
            max_num_seqs=args.njobs,
            block_size=args.block_size,
            quantization=args.thinker_quantization,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_prompt_embeds=False,
        )
        talker_engine_args = AsyncEngineArgs(
            model=args.ckpt_dir,
            trust_remote_code=True,
            gpu_memory_utilization=args.talker_gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=args.enforce_eager or args.talker_enforce_eager,
            compilation_config=CompilationConfig(
                cudagraph_mode="PIECEWISE", use_inductor=True, custom_ops=["all"],
                pass_config=PassConfig(fuse_norm_quant=False),
            ),
            distributed_executor_backend="mp",
            max_model_len=32768,
            max_num_seqs=args.njobs,
            block_size=args.block_size,
            quantization=args.talker_quantization,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_prompt_embeds=True,
            dtype=args.dtype,
        )

        if args.thinker_only:
            self.omni = OmniDRCA2Engine(
                thinker_engine_args,
                thinker_visible_devices=args.thinker_devices,
            )
        else:
            self.omni = OmniDRCA2Engine(
                thinker_engine_args, talker_engine_args,
                thinker_visible_devices=args.thinker_devices,
                talker_visible_devices=args.talker_devices,
            )
            # The local aligned-Talker vLLM path consumes an optional codec
            # history map, but some installed revisions omitted its __init__
            # assignment.  Keep the compatibility state on this engine
            # instance instead of modifying the system vLLM package.
            if not hasattr(self.omni, "cosyvoice_codec_prefix_ids"):
                self.omni.cosyvoice_codec_prefix_ids = {}
        if args.use_thinker_text_for_talker:
            self._enable_thinker_text_for_talker()
        logger.info("OmniDRCA2Engine initialized.")

        self.vad_processor = None
        if args.vad_mode != "none":
            self.vad_processor = VAD_Processor(vad_mode=args.vad_mode)

    def _enable_thinker_text_for_talker(self):
        """Feed the Thinker's translated text to the codec LLM text embedding.

        This mirrors the text-token codec training input.  It is installed on
        the engine instance only, so the original hidden-state aligned path is
        completely unchanged when the switch is off.
        """
        config = self.omni.talker_config
        if not self.omni._is_cosyvoice_codex_talker_config(config):
            raise ValueError(
                "--use-thinker-text-for-talker requires a CosyVoice codex talker"
            )

        llm_init_path = self.omni._cosyvoice_codex_cfg_get(
            config, "llm_init_param_path", None
        )
        if not llm_init_path:
            raise ValueError(
                "cosyvoice_codex.llm_init_param_path is required for thinker-text input"
            )
        self.omni.codex_text_tokenizer = AutoTokenizer.from_pretrained(
            llm_init_path, trust_remote_code=True
        )
        self.omni.codex_text_vocab_size = int(
            AutoConfig.from_pretrained(
                llm_init_path, trust_remote_code=True
            ).vocab_size
        )

        def start_from_thinker_text(engine, request_id, thinker_output):
            if request_id not in engine.talker_requests:
                logger.warning("[%s] Missing talker sampling params", request_id)
                return
            sampling_params = engine.talker_requests.pop(request_id)
            text = str(thinker_output.outputs[0].text or "")
            text_ids = engine.codex_text_tokenizer.encode(
                text, add_special_tokens=False
            )
            text_vocab_size = engine.codex_text_vocab_size
            invalid_ids = [token_id for token_id in text_ids
                           if token_id < 0 or token_id >= text_vocab_size]
            if invalid_ids:
                raise ValueError(
                    f"Codec tokenizer produced out-of-vocabulary ids: {invalid_ids[:8]}"
                )

            # Negative ids are an internal transport encoding interpreted by
            # OmniDRCA2Talker inside the worker.  Keep the original aligned
            # Talker schedule: the base prompt is followed by a small blank
            # look-ahead, then one text frame is added for every codec step.
            encoded_text_ids = [-(int(token_id) + 1) for token_id in text_ids]
            zero_sentinel = -(text_vocab_size + 1)

            thinker_prompt_ids = list(thinker_output.prompt_token_ids or [])
            generated_token_ids = list(thinker_output.outputs[0].token_ids or [])
            generated_token_ids = engine._strip_thinker_stop_tokens(
                generated_token_ids
            )
            original_prompt = engine.request_prompts.get(request_id)
            base_prompt_ids = engine._get_alignment_base_prompt_token_ids(
                original_prompt, thinker_prompt_ids
            )
            blank_count = engine._get_cosyvoice_blank_count()
            max_codec_tokens = int(getattr(sampling_params, "max_tokens", 0) or 8192)
            audio_align_len = blank_count + max_codec_tokens
            target_segment_len = max(len(generated_token_ids), audio_align_len)

            # Multimodal preprocessing expands the single <audio_pad> token to
            # thousands of model positions.  Run the same one-token alignment
            # prefill as the legacy hidden-state path to recover that exact
            # expanded prefix length.  Only this one prefill tensor crosses
            # shared memory; per-step text transport below remains token-only.
            mask_token_id = int(os.environ.get(
                "CODEX_ALIGNMENT_MASK_TOKEN_ID", "-60"
            ))
            aligned_prompt_ids = (
                base_prompt_ids
                + generated_token_ids
                + [mask_token_id] * (
                    target_segment_len - len(generated_token_ids)
                )
            )
            align_prompt = {"prompt_token_ids": aligned_prompt_ids}
            if isinstance(original_prompt, Mapping):
                if original_prompt.get("multi_modal_data") is not None:
                    align_prompt["multi_modal_data"] = original_prompt.get(
                        "multi_modal_data"
                    )
                if original_prompt.get("mm_processor_kwargs") is not None:
                    align_prompt["mm_processor_kwargs"] = original_prompt.get(
                        "mm_processor_kwargs"
                    )
            align_request_id = f"{request_id}_codec_text_align"
            align_generator = engine.thinker_engine.generate(
                request_id=align_request_id,
                prompt=align_prompt,
                sampling_params=SamplingParams(
                    temperature=0.0, top_k=1, max_tokens=1,
                    detokenize=False,
                ),
                using_external_data=False,
            )
            align_hidden = None
            for align_output in SynchronizedGenerator(
                    align_generator, engine._loop):
                if align_hidden is None:
                    align_hidden = SharedHiddenBuffer.read_global_by_request_id(
                        request_id=f"{align_request_id}_prefill",
                        device="cpu", wait=True, timeout=30.0,
                    )
                if align_output.finished:
                    break
            if align_hidden is None:
                raise RuntimeError(
                    f"[{request_id}] failed to obtain aligned prefix length"
                )
            prefix_len = max(0, int(align_hidden.size(0)) - target_segment_len)
            del align_hidden

            aligned_text = (
                [zero_sentinel] * prefix_len
                + encoded_text_ids
                + [zero_sentinel] * max_codec_tokens
            )
            prompt_len = min(
                len(aligned_text), max(1, prefix_len + blank_count)
            )
            # The expanded source prefix is all-zero on both codec-LLM input
            # streams during training.  The sentinel preserves that zero;
            # actual text frames add text_embed(id) + codec_embed(0), matching
            # the aligned blank bootstrap used by the legacy path.
            prompt_token_ids = aligned_text[:prompt_len]

            logger.info(
                "[%s] Start aligned text-token Talker: chars=%d "
                "text_tokens=%d prefix=%d blank=%d prompt_frames=%d",
                request_id, len(text), len(text_ids), prefix_len,
                blank_count, prompt_len,
            )
            engine.thinker_generate_speech[request_id] = True
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
                using_external_data=True,
            )
            generator = engine.talker_engine.generate(
                request_id=request_id,
                prompt=talker_prompt,
                sampling_params=sampling_params,
                using_external_data=True,
            )
            engine.cosyvoice_talker_text_ids[request_id] = aligned_text
            engine.cosyvoice_talker_text_cursor[request_id] = prompt_len
            engine._listen_talker(
                request_id,
                generator,
                engine.output_queue[request_id],
            )

        def feed_codec_token_only(engine, request_id):
            text_stream = engine.cosyvoice_talker_text_ids.get(request_id)
            if not text_stream:
                return
            cursor = engine.cosyvoice_talker_text_cursor.get(request_id, 0)
            if cursor < len(text_stream):
                encoded_text_id = text_stream[cursor]
                engine.cosyvoice_talker_text_cursor[request_id] = cursor + 1
                is_padding = False
            else:
                encoded_text_id = -(engine.codex_text_vocab_size + 1)
                is_padding = True
            external_data = ExternalData(
                request_id=request_id,
                external_token_ids=[int(encoded_text_id)],
                is_padding=is_padding,
                add_token_embedding=True,
            )
            asyncio.run_coroutine_threadsafe(
                engine.talker_engine.update_request_with_data(external_data),
                engine._loop,
            ).result()

        self.omni._start_cosyvoice_aligned_talker = types.MethodType(
            start_from_thinker_text, self.omni
        )
        self.omni._feed_next_cosyvoice_talker_hidden = types.MethodType(
            feed_codec_token_only, self.omni
        )
        self.omni.cosyvoice_talker_text_ids = {}
        self.omni.cosyvoice_talker_text_cursor = {}
        logger.info(
            "Enabled Thinker translation -> CosyVoice text-token talker input "
            "(tokenizer=%s)", llm_init_path,
        )

    def prepend_transcript_to_user_prompt(self, user_prompt, transcript):
        if not self.args.prepend_transcript_to_user_prompt:
            return user_prompt
        if transcript is None:
            transcript = ""
        transcript = str(transcript).strip()
        prefix = self.args.transcript_prompt_template.format(transcript=transcript)
        separator = "\n" if user_prompt else ""
        return f"{prefix}{separator}{user_prompt}"

    def infer_audio_segment(self, wav_seg, user_prompt="语音转写："):
        system_prompt = "You are a helpful assistant."
        #user_prompt = "Translate into English:"
        prompt_text = (
            f"<|im_start|>system\n"
            f"{system_prompt}"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_prompt}"
            f"<|audio_start|><|audio_pad|><|audio_end|>"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        #print("prompt_text", prompt_text)
        prompt = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": wav_seg},
            mm_processor_kwargs={},
        )
        thinker_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=self.args.max_tokens,
            detokenize=True,
            seed=self.args.seed,
        )
        codec_stop_extra_args = {}
        if self.args.codec_repeat_stop_len > 0:
            codec_stop_extra_args["stop_on_repeated_token_len"] = self.args.codec_repeat_stop_len
        if self.args.codec_ngram_repeat_stop_len > 0:
            codec_stop_extra_args["stop_on_repeated_ngram_len"] = self.args.codec_ngram_repeat_stop_len
            codec_stop_extra_args["stop_on_repeated_ngram_min"] = self.args.codec_ngram_repeat_min
            codec_stop_extra_args["stop_on_repeated_ngram_max"] = self.args.codec_ngram_repeat_max

        talker_max_tokens = self.args.max_tokens * 5
        if self.args.codec_decode_max_len > 0:
            talker_max_tokens = min(talker_max_tokens, self.args.codec_decode_max_len)

        talker_params = SamplingParams(
            # Greedy codec decoding collapses into short repeated-token loops.
            # Match CosyVoice's default nucleus candidate set; the existing
            # repetition guards remain as a final safety net.
            temperature=self.args.codec_temperature,
            top_p=self.args.codec_top_p,
            top_k=self.args.codec_top_k,
            max_tokens=talker_max_tokens,
            ignore_eos=False,
            stop_token_ids=[
                int(os.environ.get(
                    "CODEX_CODEC_EOS_TOKEN_ID",
                    "59050",
                ))
            ],
            detokenize=False,
            seed=self.args.seed,
            extra_args=codec_stop_extra_args or None,
        )
        request_id = str(uuid.uuid4())
        output_queue = self.omni.add_request(
            request_id, prompt, thinker_params,
            **({"talker_params": talker_params} if not self.args.thinker_only else {}),
        )

        last_output = None
        seen_outputs = []
        talker_codec = None
        while True:
            output = output_queue.get()
            if output is None:
                break
            seen_outputs.append(output)
            if isinstance(output, RequestOutput):
                last_output = output
            else:
                codec_candidate = _to_codec_array(output)
                if codec_candidate is not None:
                    talker_codec = codec_candidate.copy()

        text = last_output.outputs[0].text if last_output else ""
        codec = talker_codec
        if codec is None:
            codec = _extract_codec_from_object(
                last_output,
                allow_token_ids_fallback=False,
            )
        if codec is None:
            codec = _extract_codec_from_object(
                seen_outputs,
                allow_token_ids_fallback=False,
            )
        if codec is None and self.args.allow_token_ids_codec_fallback:
            logger.warning(
                "No explicit talker codec found for request %s; "
                "not falling back to text token_ids for S2S output.",
                request_id,
            )
        codec_eos_token = int(os.environ.get("CODEX_CODEC_EOS_TOKEN_ID", "59050"))
        codec_flat = np.asarray(codec).reshape(-1) if codec is not None else np.asarray([])
        codec_ends_with_eos = bool(
            codec_flat.size > 0 and int(codec_flat[-1]) == codec_eos_token
        )
        debug = _summarize_output_object(seen_outputs[-1] if seen_outputs else last_output)
        codec_llm_hidden = None
        if self.args.save_codec_llm_hidden:
            codec_llm_hidden = self.omni.pop_cosyvoice_fm_hidden(request_id)
            if codec_llm_hidden is None:
                raise RuntimeError(
                    f"No codec-LLM hidden states were exported for request {request_id}"
                )
            codec_llm_hidden = codec_llm_hidden.float().cpu().numpy()
            codec_eos = int(os.environ.get("CODEX_CODEC_EOS_TOKEN_ID", "59050"))
            if (
                codec is not None
                and len(codec) == len(codec_llm_hidden) + 1
                and int(np.asarray(codec).reshape(-1)[-1]) == codec_eos
            ):
                # The Talker output includes its terminal EOS, while the FM
                # hidden export intentionally contains acoustic tokens only.
                codec = np.asarray(codec).reshape(-1)[:-1].copy()
        return {
            "text": text,
            "codec": codec,
            "codec_ends_with_eos": codec_ends_with_eos,
            "codec_llm_hidden": codec_llm_hidden,
            "debug": debug,
        }

    def process_single_line(self, line, do_vad=True):
        wav_pattern = re.compile(r"<\|startofspeech\|>!(.*)<\|endofspeech\|>")
        try:
            data_dict = json.loads(line.strip())
            messages = data_dict["messages"]
            # Packed training data uses ``text`` while reference historical
            # evaluation data uses ``content``.  Accept both so that a packed
            # sample can be replayed directly during train/infer smoke tests.
            user_content = messages[1].get("content", messages[1].get("text"))
            target_content = messages[2].get("content", messages[2].get("text", ""))
            if user_content is None:
                raise ValueError("user message contains neither 'content' nor 'text'")
            user_prompt = user_content.split('<|startofspeech|>', 1)[0]
            target_out = target_content.replace('\n', ' ')
            source = data_dict.get("transcript", "")
            user_prompt = self.prepend_transcript_to_user_prompt(user_prompt, source)
            wav_match = wav_pattern.search(user_content)
            if wav_match is None:
                raise ValueError("user message does not contain a !<wav-path> speech span")
            wav_path = wav_match.group(1)

            try:
                duration = (float(messages[1]["start_time"]), float(messages[1]["end_time"]))
            except:
                duration = (0, None)

            original_key = wav_path.replace(' ', '')
            if duration[1] is not None:
                original_key = f'{wav_path}_{duration[0]}_{duration[1]}'
            key = make_output_key(original_key)
            #print("key: ", key)

            if self.args.use_oss_cache:
                wav_path = wav_path.replace("oss://example-storage-4", "/nls_spider_sh_cache", 1)
                wav_path = wav_path.replace("oss://example-storage-1", "/fun_resource_sh_cache", 1)

            wav_file = load_audio_text_image_video(wav_path, duration=duration, use_ffmpeg=False)

            if do_vad and self.vad_processor is not None:
                vad_res = self.vad_processor.get_vad_result(wav_file, 16000)
                wav_segments = [wav_file[st * 16 : ed * 16] for st, ed in vad_res]
            else:
                wav_segments = [wav_file]
        except Exception as e:
            traceback.print_exc()
            print(f"[ERROR] {type(e)}: {e}", flush=True)
            return None

        res_cur_wav = []
        codec_segments = []
        codec_eos_segments = []
        codec_llm_hidden_segments = []
        debug_obj = None
        for wav_seg in wav_segments:
            try:
                result = self.infer_audio_segment(wav_seg, user_prompt)
            except Exception as e:
                traceback.print_exc()
                return None
            #print(f"label: {target_out}, result: {result}", flush=True)
            res_cur_wav.append(result["text"])
            if result.get("codec") is not None:
                codec_segments.append(result["codec"])
                codec_eos_segments.append(bool(result.get("codec_ends_with_eos")))
            if result.get("codec_llm_hidden") is not None:
                codec_llm_hidden_segments.append(result["codec_llm_hidden"])
            if debug_obj is None:
                debug_obj = result.get("debug")

        res_str = join_strings_with_space(res_cur_wav, add_vad_delimiter=True)
        final_res = res_str.replace("\n", "")
        codec = np.concatenate(codec_segments, axis=0) if codec_segments else None
        codec_llm_hidden = (
            np.concatenate(codec_llm_hidden_segments, axis=0)
            if codec_llm_hidden_segments
            else None
        )

        res_dict = {
            "key": key,
            "original_key": original_key,
            "source": source,
            "label": target_out,
            "text": final_res,
            "codec": codec,
            "codec_ends_with_eos": bool(codec_eos_segments) and all(codec_eos_segments),
            "codec_llm_hidden": codec_llm_hidden,
            "debug": debug_obj,
        }
        if self.args.print_each_result:
            print(res_dict, flush=True)
        return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频路径 -> OmniDRCA2Engine 转写")
    parser.add_argument("ckpt_dir", help="模型 checkpoint 目录")
    parser.add_argument("input_jsonl_list", help="输入 jsonl 路径列表文件")
    parser.add_argument("output_dir_list", help="输出路径列表文件")
    parser.add_argument("--vad-mode", type=str, default='farfield', help="vad 模式：farfield/nearfield/opensource/none")
    parser.add_argument("--njobs", type=int, default=20, help="并发数")
    parser.add_argument("--use-oss-cache", action="store_true")

    # 新增的推理引擎相关参数
    parser.add_argument("--thinker-only", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--thinker-enforce-eager", action="store_true")
    parser.add_argument("--talker-enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--thinker-quantization", type=str, choices=QUANTIZATION_METHODS)
    parser.add_argument("--talker-quantization", type=str, choices=QUANTIZATION_METHODS)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--thinker-devices", type=json.loads, default="[0,1,2,3,4,5,6,7]")
    parser.add_argument("--talker-devices", type=json.loads, default="[1]")
    parser.add_argument("--thinker-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--talker-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--prepend-transcript-to-user-prompt", action="store_true")
    parser.add_argument(
        "--use-thinker-text-for-talker",
        action="store_true",
        help=(
            "Retokenize the Thinker's final translation with the CosyVoice "
            "tokenizer and use the codec LLM text embedding as Talker input. "
            "Disabled by default to preserve the original decode path."
        ),
    )
    parser.add_argument("--transcript-prompt-template", default="transcripts: {transcript}")
    parser.add_argument(
        "--allow-token-ids-codec-fallback",
        action="store_true",
        help="If the vLLM output object has no explicit codec/codex field, save output token_ids as codec candidates.",
    )
    parser.add_argument(
        "--save-codec-llm-hidden",
        action="store_true",
        help=("Save the codec LLM state used to predict each codec token for "
              "unified FM inference. Disabled by default."),
    )
    parser.add_argument(
        "--min-codec-char-ratio",
        type=float,
        default=float(os.environ.get("S2S_MIN_CODEC_CHAR_RATIO", "0.35")),
        help="Mark generated codec as suspicious when codec_len / output_text_chars is below this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-short-codec",
        action="store_true",
        default=os.environ.get("S2S_SAVE_SHORT_CODEC", "").lower()
        in ("1", "true", "yes", "on"),
        help="Save codec files even when the codec/text length sanity check fails.",
    )
    parser.add_argument(
        "--max-codec-char-ratio",
        type=float,
        default=float(os.environ.get("S2S_MAX_CODEC_CHAR_RATIO", "20.0")),
        help="Mark generated codec as suspicious when codec_len / output_text_chars is above this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-codec-len",
        type=int,
        default=int(os.environ.get("S2S_MAX_CODEC_LEN", "3750")),
        help="Mark generated codec as suspicious when flattened codec length exceeds this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--codec-decode-max-len",
        type=int,
        default=int(os.environ.get("S2S_CODEC_DECODE_MAX_LEN", "1024")),
        help="Hard cap for talker codec decoding length. Use 0 to keep the original max_tokens * 5 cap.",
    )
    parser.add_argument(
        "--codec-temperature",
        type=float,
        default=float(os.environ.get("S2S_CODEC_TEMPERATURE", "1.0")),
        help=(
            "Talker sampling temperature. Defaults to 1.0; set 0 to restore "
            "legacy greedy codec decoding."
        ),
    )
    parser.add_argument(
        "--codec-top-p",
        type=float,
        default=float(os.environ.get("S2S_CODEC_TOP_P", "0.8")),
        help="Talker nucleus sampling probability (CosyVoice default: 0.8).",
    )
    parser.add_argument(
        "--codec-top-k",
        type=int,
        default=int(os.environ.get("S2S_CODEC_TOP_K", "25")),
        help="Talker sampling candidate count (CosyVoice default: 25).",
    )
    parser.add_argument(
        "--codec-repeat-stop-len",
        type=int,
        default=int(os.environ.get("S2S_CODEC_REPEAT_STOP_LEN", "16")),
        help=(
            "Stop talker decoding when the same output token repeats this many "
            "times consecutively. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--codec-ngram-repeat-stop-len",
        type=int,
        default=int(os.environ.get("S2S_CODEC_NGRAM_REPEAT_STOP_LEN", "32")),
        help=(
            "Stop talker decoding when a short token ngram loop has repeated "
            "for this many output tokens. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--codec-ngram-repeat-min",
        type=int,
        default=int(os.environ.get("S2S_CODEC_NGRAM_REPEAT_MIN", "2")),
        help="Minimum repeated ngram size for codec loop stop.",
    )
    parser.add_argument(
        "--codec-ngram-repeat-max",
        type=int,
        default=int(os.environ.get("S2S_CODEC_NGRAM_REPEAT_MAX", "8")),
        help="Maximum repeated ngram size for codec loop stop.",
    )
    parser.add_argument(
        "--print-each-result",
        action="store_true",
        default=os.environ.get("S2S_PRINT_EACH_RESULT", "").lower()
        in ("1", "true", "yes", "on"),
        help="Print every decoded result dictionary, including codec arrays. Disabled by default to reduce log IO.",
    )
    args = parser.parse_args()

    args.vad_mode = "none"
    do_vad = False if args.vad_mode == "none" else True

    in_jsonl_list = [line.strip() for line in open(args.input_jsonl_list)]
    out_dir_list = [line.strip() for line in open(args.output_dir_list)]
    assert len(in_jsonl_list) == len(out_dir_list)


    inference = Inference(args)
    logger.info("##### model loaded")

    total_start_time = time.perf_counter()
    total_samples = 0

    for jsonl, out_dir in zip(in_jsonl_list, out_dir_list):
        lines = open(jsonl).readlines()
        setName = jsonl.split("/")[-1].split(".jsonl")[0]

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting dataset: {setName}")
        logger.info(f"  Input:  {jsonl}")
        logger.info(f"{'='*60}")

        dataset_start_time = time.perf_counter()


        outfile_path = os.path.join(out_dir, "1best_recog", "source")
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
        outfile_source = open(outfile_path, "w")

        outfile_path = os.path.join(out_dir, "1best_recog", "text")
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
        outfile_text = open(outfile_path, "w")

        outfile_path = os.path.join(out_dir, "1best_recog", "label")
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
        outfile_label = open(outfile_path, "w")
        key_map_path = os.path.join(out_dir, "1best_recog", "key_map.scp")
        outfile_key_map = open(key_map_path, "w")
        codec_dir = os.path.join(out_dir, "1best_recog", "codec")
        os.makedirs(codec_dir, exist_ok=True)
        codec_llm_hidden_dir = os.path.join(
            out_dir, "1best_recog", "codec_llm_hidden"
        )
        if args.save_codec_llm_hidden:
            os.makedirs(codec_llm_hidden_dir, exist_ok=True)
        debug_path = os.path.join(out_dir, "1best_recog", "s2s_output_debug.json")
        debug_written = os.path.exists(debug_path)
        codec_quality_path = os.path.join(out_dir, "1best_recog", "codec_quality.jsonl")
        outfile_codec_quality = open(codec_quality_path, "w")

        success_samples = 0
        with ThreadPoolExecutor(max_workers=args.njobs) as executor:
            future_to_line = {executor.submit(inference.process_single_line, line, do_vad): line for line in lines}

            for future in tqdm(as_completed(future_to_line), total=len(lines), desc=f"Processing {jsonl}"):
                res = future.result()
                if res is None: continue
                key = res["key"]
                original_key = res["original_key"]
                source = res["source"]
                text = res["text"]
                label = res["label"]
                codec = res.get("codec")
                codec_ends_with_eos = bool(res.get("codec_ends_with_eos"))
                codec_llm_hidden = res.get("codec_llm_hidden")

                outfile_source.write(f"{key}\t{source}\n")
                outfile_source.flush()
                outfile_text.write(f"{key}\t{text}\n")
                outfile_text.flush()
                outfile_label.write(f"{key}\t{label}\n")
                outfile_label.flush()
                outfile_key_map.write(f"{key}\t{original_key}\n")
                outfile_key_map.flush()
                quality = _codec_quality(
                    text,
                    codec,
                    args.min_codec_char_ratio,
                    args.max_codec_char_ratio,
                    args.max_codec_len,
                )
                quality["key"] = key
                quality["codec_ends_with_eos"] = codec_ends_with_eos
                quality["termination"] = (
                    "codec_eos" if codec_ends_with_eos
                    else ("missing_codec" if codec is None else "no_codec_eos")
                )
                outfile_codec_quality.write(json.dumps(quality, ensure_ascii=False) + "\n")
                outfile_codec_quality.flush()
                save_codec = codec is not None
                if not quality["ok"] and not args.save_short_codec:
                    save_codec = False
                    logger.warning(
                            "Suspicious S2S codec for %s: reason=%s, codec_len=%s, "
                            "text_chars=%s, ratio=%.4f, expected ratio range=[%.4f, %.4f], "
                            "max_codec_len=%s. Skipping codec save; this usually means "
                            "the synced vLLM talker output did not stop at codec EOS.",
                            key,
                            quality["reason"],
                            quality["codec_len"],
                            quality["text_chars"],
                            quality["codec_char_ratio"],
                            quality["min_codec_char_ratio"],
                            quality["max_codec_char_ratio"],
                            quality["max_codec_len"],
                    )
                if save_codec:
                    np.save(os.path.join(codec_dir, f"{key}.npy"), codec)
                    if args.save_codec_llm_hidden:
                        if codec_llm_hidden is None:
                            raise RuntimeError(f"Missing codec LLM hidden for {key}")
                        if len(codec_llm_hidden) != len(codec):
                            raise RuntimeError(
                                f"Codec/hidden length mismatch for {key}: "
                                f"codec={len(codec)}, hidden={len(codec_llm_hidden)}"
                            )
                        np.save(
                            os.path.join(codec_llm_hidden_dir, f"{key}.npy"),
                            codec_llm_hidden,
                        )
                if not debug_written and res.get("debug") is not None:
                    with open(debug_path, "w") as f:
                        json.dump(res["debug"], f, ensure_ascii=False, indent=2)
                    debug_written = True
                success_samples += 1

        outfile_source.close()
        outfile_text.close()
        outfile_label.close()
        outfile_key_map.close()
        outfile_codec_quality.close()

        # 数据集统计信息
        dataset_elapsed = time.perf_counter() - dataset_start_time
        dataset_samples = len(lines)
        total_samples += success_samples
        throughput = success_samples / dataset_elapsed if dataset_elapsed > 0 else 0
        logger.info(f"Dataset [{setName}] done:")
        logger.info(f"  Total samples: {dataset_samples}")
        logger.info(f"  Successful samples: {success_samples}")
        logger.info(f"  Time: {dataset_elapsed:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} samples/s")

    # 全局统计信息
    total_elapsed = time.perf_counter() - total_start_time
    logger.info(f"\n{'='*60}")
    logger.info("All datasets done!")
    logger.info(f"  Total samples processed: {total_samples}")
    logger.info(f"  Total time: {total_elapsed:.2f}s")
    if total_elapsed > 0:
        logger.info(f"  Overall throughput: {total_samples / total_elapsed:.2f} samples/s")
    logger.info(f"{'='*60}")

    inference.omni.shutdown()
