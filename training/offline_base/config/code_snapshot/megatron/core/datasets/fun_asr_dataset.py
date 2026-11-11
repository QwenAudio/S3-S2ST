# megatron-series/Qwen-public/megatron/core/datasets/fun_asr_dataset.py

import os
import json
import math
import random
import contextlib
import re
import time
import torch
import numpy as np
import bisect
import logging
import traceback
from collections import defaultdict, OrderedDict
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Optional
from dataclasses import dataclass

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.tokenizer.tokenizer import AbstractTokenizer
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.utils import Split

# 导入FunASR的ASR处理函数
try:
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
    from funasr.frontends.whisper_frontend import WhisperFrontend
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("Warning: FunASR not available, ASR processing will be limited")


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class JsonlASRDataset(torch.utils.data.Dataset):
    """Low-level JSONL dataset for ASR data"""
    def __init__(self, path, **kwargs):
        super().__init__()
        
        # 添加配置参数，与 MinMoSense 保持一致
        self.max_source_length = kwargs.get("max_source_length", 8000)
        self.min_source_length = kwargs.get("min_source_length", 10)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.min_audio_samples_for_fbank = int(kwargs.get("min_audio_samples_for_fbank", 400))
        asr_downsample_rate = int(kwargs.get("asr_downsample_rate", 8))
        
        # 支持文件列表和单文件
        is_training = kwargs.get("is_training", True)
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)
            
            if not is_training:
                data_split_num = 1
                data_split_i = 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()
                
                num_per_slice = (len(file_list_all) - 1) // data_split_num + 1
                file_list = file_list_all[
                            data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
                            ]
                print(f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}")
        else:
            file_list = [path]
        
        self.contents = []
        total_whrs = 0.0
        total_token_for_llm_B = 0.0
        
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    try:
                        data_dict = json.loads(line.strip())
                    except Exception as e:
                        print(f"drop it, json error: {e}, line: {line}, file_json: {file_json}")
                        continue
                    
                    # 检查是否有 messages 字段
                    if "messages" not in data_dict:
                        print(f"drop it, no messages field: {data_dict}")
                        continue
                    
                    data = data_dict["messages"]
                    
                    # 处理 speech_length 和 text_length
                    if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
                        speech_length = int(data_dict.get("speech_length", [0])[0])
                        text_length = int(data_dict.get("text_length", 0)[0])
                    else:
                        speech_length = int(data_dict.get("speech_length", 0))
                        text_length = int(data_dict.get("text_length", 0))
                    speech_length = int(speech_length)
                    text_length = int(text_length)
                    
                    # 数据过滤逻辑，与 MinMoSense 保持一致
                    if speech_length > 0 and speech_length < 1:
                        continue
                    
                    if text_length < 1:
                        print(f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}")
                        if len(data) > 2:
                            text_length = len(data[2]['content'])
                        continue
                    
                    if speech_length > self.max_source_length:
                        continue
                    
                    if speech_length < self.min_source_length:
                        continue
                    
                    if text_length > self.max_target_length:
                        continue
                    
                    # 解析 messages 数据
                    system, user, assistant = [], [], []
                    duration = []
                    
                    for i, item in enumerate(data):
                        try:
                            role = item["role"]
                            content = item["content"]
                        except KeyError:
                            print(f"drop it, KeyError: {item}, file_json: {file_json}")
                            continue
                        
                        if role == "system":
                            system.append(content)
                        elif role == "user":
                            user.append(content)
                            duration.append([item.get("start_time", 0), item.get("end_time", None)])
                        elif role == "assistant":
                            if "prev_content" in item:
                                prev_content = item["prev_content"]
                                assistant.append([content, {"prev_content": prev_content}])
                            else:
                                assistant.append(content)
                    
                    if len(system) == 0:
                        system = ["You are a helpful assistant."]
                    system = system * len(user)
                    
                    # 构建内容项
                    contents_i = {
                        "system": system,
                        "user": user,
                        "assistant": assistant,
                        "source_len": speech_length + text_length,
                        "duration": duration,
                    }
                    
                    # 添加 key 字段
                    if "key" in data_dict:
                        contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("key", "key_01234"),
                                                                               (list, tuple)) else data_dict["key"][0]
                    
                    # 添加其他元数据字段，与 MinMoSense 保持一致
                    if "hist_context" in data_dict:
                        contents_i["hist_context"] = data_dict["hist_context"]
                    if "hotwords" in data_dict:
                        contents_i["hotwords"] = data_dict["hotwords"]
                    if "asr_hotwords" in data_dict:
                        contents_i["asr_hotwords"] = data_dict["asr_hotwords"]
                    if "vad_segs" in data_dict:
                        contents_i["vad_segs"] = data_dict["vad_segs"]
                    if "word_list" in data_dict:
                        contents_i["word_list"] = data_dict["word_list"]
                    if "one_pass_result" in data_dict:
                        contents_i["one_pass_result"] = data_dict["one_pass_result"]
                    if "one_pass_wer" in data_dict:
                        contents_i["one_pass_wer"] = data_dict["one_pass_wer"]
                    if "noised" in data_dict:
                        contents_i["noised"] = data_dict["noised"]
                    
                    if kwargs.get("save_meta", False):
                        contents_i["meta"] = data_dict
                    
                    # 统计信息
                    total_whrs += speech_length / 100.0 / 3600 / 10000 * asr_downsample_rate
                    total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
                    self.contents.append(contents_i)
        
        print(f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index]
    
    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", -1)
        if source_len < 0:
            source_len = len(data_dict["system"]) + len(data_dict["user"])
        return source_len
    
    def get_target_len(self, data_dict):
        return 0


def _codex_extract_audio_path_from_item(item):
    try:
        users = item.get("user", []) if isinstance(item, dict) else []
        if not users:
            return None
        text = str(users[0])
        start = text.find("<|startofspeech|>")
        end = text.find("<|endofspeech|>")
        if start < 0 or end <= start:
            return None
        return text[start + len("<|startofspeech|>"):end]
    except Exception:
        return None


def _codex_list_stats(values):
    if values is None:
        return {"len": 0}
    if isinstance(values, torch.Tensor):
        try:
            data = values.detach().cpu().reshape(-1)
            if data.numel() == 0:
                return {"shape": tuple(values.shape), "len": 0}
            nums = [float(x) for x in data.tolist()]
            return {
                "shape": tuple(values.shape),
                "len": len(nums),
                "min": min(nums),
                "max": max(nums),
                "mean": sum(nums) / len(nums),
            }
        except Exception as exc:
            return {"type": type(values).__name__, "error": repr(exc)}
    nums = []
    for value in values:
        try:
            if isinstance(value, torch.Tensor):
                nums.extend([float(x) for x in value.detach().cpu().reshape(-1).tolist()])
            elif isinstance(value, (int, float)):
                nums.append(float(value))
        except Exception:
            continue
    if not nums:
        return {"len": len(values or [])}
    return {"len": len(values or []), "min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)}


def _codex_log_slow_item(
    *,
    elapsed_s,
    idx,
    index_cur,
    ds_idx,
    retry_idx,
    item,
    input_ids=None,
    speech_lengths=None,
    fake_token_len=None,
):
    threshold_s = float(os.environ.get("CODEX_SLOW_ITEM_LOG_SEC", "5"))
    if elapsed_s < threshold_s:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else -1
    except Exception:
        rank = -1
    worker = torch.utils.data.get_worker_info()
    worker_id = worker.id if worker is not None else None
    audio_path = _codex_extract_audio_path_from_item(item)
    user = item.get("user", []) if isinstance(item, dict) else []
    assistant = item.get("assistant", []) if isinstance(item, dict) else []
    summary = {
        "rank": rank,
        "worker_id": worker_id,
        "elapsed_s": round(elapsed_s, 3),
        "idx": int(idx) if isinstance(idx, (int, np.integer)) else idx,
        "index_cur": int(index_cur) if isinstance(index_cur, (int, np.integer)) else index_cur,
        "ds_idx": int(ds_idx) if isinstance(ds_idx, (int, np.integer)) else ds_idx,
        "retry_idx": retry_idx,
        "key": item.get("key") if isinstance(item, dict) else None,
        "source_len": item.get("source_len") if isinstance(item, dict) else None,
        "duration": item.get("duration") if isinstance(item, dict) else None,
        "audio": audio_path[:220] if audio_path else None,
        "user_prefix": str(user[0])[:160] if user else None,
        "assistant_len": len(str(assistant[0])) if assistant else None,
        "input_len": len(input_ids) if input_ids is not None else None,
        "speech_lengths": _codex_list_stats(speech_lengths),
        "fake_token_len": _codex_list_stats(fake_token_len),
    }
    # print(f"[CODEX_SLOW_ITEM] {summary}", flush=True)


def _codex_collate_tensor_stats(value):
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    stats = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
    try:
        data = value.detach().cpu().reshape(-1)
        stats["numel"] = int(data.numel())
        if data.numel() > 0 and value.dtype in (
            torch.float16, torch.float32, torch.float64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool,
        ):
            data_f = data.float()
            stats.update({
                "min": round(float(data_f.min().item()), 4),
                "max": round(float(data_f.max().item()), 4),
                "mean": round(float(data_f.mean().item()), 4),
            })
    except Exception as exc:
        stats["error"] = repr(exc)
    return stats


def _codex_collate_sample_summary(samples, limit=3):
    summary = []
    for sample in (samples or [])[:limit]:
        if not isinstance(sample, dict):
            summary.append({"type": type(sample).__name__})
            continue
        item = sample.get("item")
        if isinstance(item, list) and item:
            item = item[0]
        if isinstance(item, dict):
            users = item.get("user", [])
            assistants = item.get("assistant", [])
            summary.append({
                "key": item.get("key"),
                "source_len": item.get("source_len"),
                "duration": item.get("duration"),
                "user_prefix": str(users[0])[:120] if users else None,
                "assistant_len": len(str(assistants[0])) if assistants else None,
            })
        else:
            summary.append({
                "keys": sorted(sample.keys())[:12],
                "input_ids": _codex_collate_tensor_stats(sample.get("input_ids")) if "input_ids" in sample else None,
                "speech_lengths": _codex_collate_tensor_stats(sample.get("speech_lengths")) if "speech_lengths" in sample else None,
            })
    return summary


def _codex_log_slow_dataset_collate(name, elapsed_s, samples, outputs=None, pad_times=None):
    threshold_s = float(os.environ.get("CODEX_SLOW_COLLATE_LOG_SEC", "5"))
    if elapsed_s < threshold_s:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else -1
    except Exception:
        rank = -1
    worker = torch.utils.data.get_worker_info()
    summary = {
        "name": name,
        "rank": rank,
        "worker_id": worker.id if worker is not None else None,
        "elapsed_s": round(elapsed_s, 3),
        "num_samples": len(samples or []),
        "samples": _codex_collate_sample_summary(samples),
        "pad_times": pad_times or {},
    }
    if outputs:
        summary["outputs"] = {
            key: (_codex_collate_tensor_stats(value) if isinstance(value, torch.Tensor) else {"type": type(value).__name__, "len": len(value) if hasattr(value, "__len__") else None})
            for key, value in outputs.items()
        }
    # print(f"[CODEX_SLOW_DATASET_COLLATE] {summary}", flush=True)


@dataclass
class FunASRDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for FunASR datasets"""
    
    # ASR processing parameters
    asr_fs: int = 16000
    max_source_length: int = 8000
    min_source_length: int = 10
    max_target_length: int = 2048
    max_token_length: int = 1500
    
    # Frontend configuration
    frontend_type: str = "WhisperFrontend"
    n_mels: int = 128
    do_pad_trim: bool = False
    
    # ASR preprocessing
    preprocessor_speech: Optional[str] = None
    preprocessor_speech_conf: Optional[dict] = None
    preprocessor_text: Optional[str] = None
    preprocessor_text_conf: Optional[dict] = None
    preprocessor_noise: Optional[str] = None
    preprocessor_noise_conf: Optional[dict] = None
    
    # Prompt configuration
    prompt_classes: Optional[str] = None
    prompt_conf: Optional[dict] = None
    
    # Training parameters
    retry: int = 100
    batch_size_scale_ratio_max: float = 1.5
    batch_size_token_max: int = 4000
    multiturn_num_max: int = 5
    do_think: bool = True
    sys_prompt: bool = True
    
    # Dynamic output alignment
    use_dynamic_output_ratio: float = 0.0
    min_output_mask_token_len: int = 1
    min_output_non_mask_token_len: int = 6

    prompt_classes=None
    prompt_conf=None

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()
        assert self.tokenizer is not None


class FunASRDataset(MegatronDataset):
    """Base class for FunASR ASR datasets in Megatron"""
    
    def __init__(
        self,
        indexed_dataset: JsonlASRDataset,
        dataset_path: Optional[str],
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        config: FunASRDatasetConfig,
    ):
        self.args = get_args()
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        
        # Extract source information
        if dataset_path.endswith("final.jsonl"):
            self.source = '/'.join('/'.join(dataset_path.split('/')[-3:-1]).split('.')[:1])
        else:
            self.source = '/'.join('/'.join(dataset_path.split('/')[-2:]).split('.')[:1])

        indexed_indices = np.arange(indexed_indices[0], indexed_indices[1])
        assert np.min(indexed_indices) >= 0, indexed_indices
        assert np.max(indexed_indices) <= len(indexed_dataset)

        # Create shuffle indices
        iter_cnt = int(math.ceil(num_samples / len(indexed_indices)))
        shuffle_idx = [indexed_indices.copy() for _ in range(iter_cnt)]
        if index_split == Split.train:
            with numpy_seed(self.config.random_seed):
                for i in range(iter_cnt):
                    np.random.shuffle(shuffle_idx[i])
        self.shuffle_idx = np.array(shuffle_idx).flatten()

        self.tokenizer = get_tokenizer()
        self.eod = self.tokenizer.eod
        
        # Initialize frontend
        self.frontend = self._init_frontend()
        self.fs = self.config.asr_fs
        self.data_type = "sound"
        
        # Initialize preprocessors
        self.preprocessor_speech = self._init_preprocessor_speech()
        self.preprocessor_text = self._init_preprocessor_text()
        self.preprocessor_noise = self._init_preprocessor_noise()
        self.prompt_classes = self._init_prompt_classes()
        
        # ASR processing parameters
        self.max_source_length = self.config.max_source_length
        self.min_source_length = self.config.min_source_length
        self.max_target_length = self.config.max_target_length
        self.max_token_length = self.config.max_token_length
        self.min_audio_samples_for_fbank = getattr(
            indexed_dataset, "min_audio_samples_for_fbank", 400
        )
        self.retry = self.config.retry
        
        # Pattern for speech detection
        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        
        # Check if frontend requires permutation
        self.permute = isinstance(self.frontend, WhisperFrontend) if FUNASR_AVAILABLE else False

    def _init_frontend(self):
        """Initialize ASR frontend"""
        if not FUNASR_AVAILABLE:
            return None
            
        if self.config.frontend_type == "WhisperFrontend":
            print (f'INIT FRONTEND: {self.config.frontend_type}; fs: {self.config.asr_fs}, n_mels: {self.config.n_mels}, do_pad_trim: {self.config.do_pad_trim}, filters_path: {getattr(self.args, "mel_filters_path", None)}')
            return WhisperFrontend(
                fs=self.config.asr_fs,
                n_mels=self.config.n_mels,
                do_pad_trim=self.config.do_pad_trim,
                filters_path=getattr(self.args, 'mel_filters_path', None)
            )
        return None

    def _init_preprocessor_speech(self):
        """Initialize speech preprocessor"""
        if not FUNASR_AVAILABLE or not self.config.preprocessor_speech:
            return None
            
        # Import and initialize preprocessor
        try:
            from funasr.register import tables
            preprocessor_class = tables.preprocessor_classes.get(self.config.preprocessor_speech)
            if preprocessor_class:
                return preprocessor_class(**self.config.preprocessor_speech_conf or {})
        except ImportError:
            pass
        return None

    def _init_preprocessor_text(self):
        """Initialize text preprocessor"""
        if not FUNASR_AVAILABLE or not self.config.preprocessor_text:
            return None
            
        try:
            from funasr.register import tables
            preprocessor_class = tables.preprocessor_classes.get(self.config.preprocessor_text)
            if preprocessor_class:
                return preprocessor_class(**self.config.preprocessor_text_conf or {})
        except ImportError:
            pass
        return None

    def _init_preprocessor_noise(self):
        """Initialize noise preprocessor"""
        if not FUNASR_AVAILABLE or not self.config.preprocessor_noise:
            return None
            
        try:
            from funasr.register import tables
            preprocessor_class = tables.preprocessor_classes.get(self.config.preprocessor_noise)
            if preprocessor_class:
                return preprocessor_class(**self.config.preprocessor_noise_conf or {})
        except ImportError:
            pass
        return None

    def _init_prompt_classes(self):
        """Initialize prompt classes"""
        if not FUNASR_AVAILABLE or not self.config.prompt_classes:
            print (f'NO PROMPT CLASS: FUNASR_AVAILABLE: {FUNASR_AVAILABLE}, self.config.prompt_classes: {self.config.prompt_classes}')
            return None
            
        try:
            from funasr.register import tables
            prompt_class = tables.prompt_classes.get(self.config.prompt_classes)
            if prompt_class:
                return prompt_class(**self.config.prompt_conf or {})
        except ImportError:
            print (f'IMPORT ERROR PROMPT CLASS: FUNASR_AVAILABLE: {FUNASR_AVAILABLE}, self.config.prompt_classes: {self.config.prompt_classes}')
            pass
        return None

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: JsonlASRDataset) -> int:
        """Return the number of elements in the underlying low level dataset"""
        return len(low_level_dataset)
    
    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: FunASRDatasetConfig) -> JsonlASRDataset:
        """Build the low level dataset"""
        return JsonlASRDataset(dataset_path)

    def __len__(self):
        return self.num_samples

    def load_audio(self, audio_path: str):
        """Load audio from path"""
        if not FUNASR_AVAILABLE:
            # Fallback to torchaudio
            import torchaudio
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != self.fs:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.fs)
                    waveform = resampler(waveform)
                return waveform.squeeze(0)  # Remove channel dimension
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                return None
        
        try:
            data_src = load_audio_text_image_video(audio_path, fs=self.fs)
            return data_src
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None

    def extract_asr_features(self, audio_data):
        """Extract ASR features using frontend"""
        if not FUNASR_AVAILABLE or self.frontend is None:
            # Fallback: return raw audio
            return audio_data, torch.tensor([len(audio_data)], dtype=torch.int32)
        
        try:
            speech, speech_lengths = extract_fbank(
                audio_data,
                data_type=self.data_type,
                frontend=self.frontend,
                is_final=True,
            )
            if self.permute:
                speech = speech.permute(0, 2, 1)
            return speech, speech_lengths
        except Exception as e:
            print(f"Error extracting features: {e}")
            return audio_data, torch.tensor([len(audio_data)], dtype=torch.int32)

    def _audio_num_samples(self, audio_data):
        if audio_data is None:
            return 0
        if hasattr(audio_data, "numel"):
            return int(audio_data.numel())
        return len(audio_data)

    def _duration_too_short_for_fbank(self, duration):
        if duration is None or len(duration) < 2 or duration[1] is None:
            return False
        try:
            num_samples = int(round((float(duration[1]) - float(duration[0])) * self.fs))
        except Exception:
            return False
        return num_samples < self.min_audio_samples_for_fbank

    def process_asr_segment(self, audio_path: str):
        """Process a single ASR segment"""
        # Load audio
        audio_data = self.load_audio(audio_path)
        if audio_data is None:
            return None, None, None, None
        if self._audio_num_samples(audio_data) < self.min_audio_samples_for_fbank:
            return None, None, None, None
        
        # Apply speech preprocessor if available
        if self.preprocessor_speech:
            try:
                audio_data = self.preprocessor_speech(audio_data, fs=self.fs)
            except Exception as e:
                print(f"Error in speech preprocessing: {e}")
        
        # Extract features
        speech, speech_lengths = self.extract_asr_features(audio_data)
        
        # Check length constraints
        if speech_lengths[0].item() > self.max_source_length:
            print(f"Audio too long: {speech_lengths[0].item()} > {self.max_source_length}")
            return None, None, None, None
        
        # Calculate fake token length for speech
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len = (olens - 1) // 2 + 1
        
        return speech, speech_lengths, fake_token_len, len(speech[0]) if len(speech.shape) > 1 else 0

    def parse_asr_text(self, content):
        """Parse content with audio and text"""
        if isinstance(content, list):
            return self._parse_list_format(content)
        elif isinstance(content, str):
            return self._parse_string_format(content)
        else:
            raise ValueError(f"Unknown content type: {type(content)}")

    def _parse_list_format(self, content):
        """Parse list format content"""
        tokens, loss_mask, audio_data = [], [], []
        
        for ele in content:
            if ele is None:
                tokens.append(self.eod)
                loss_mask.append(1)
                continue

            if isinstance(ele, str):
                ele = {'text': ele}

            if 'audio' in ele or 'speech' in ele:
                # Process audio
                audio_path = ele.get('audio') or ele.get('speech')
                speech, speech_lengths, fake_token_len, seq_len = self.process_asr_segment(audio_path)
                
                if speech is not None:
                    # Add fake tokens for audio
                    fake_tokens = [0] * fake_token_len
                    tokens.extend(fake_tokens)
                    loss_mask.extend([0] * len(fake_tokens))  # Don't compute loss on audio tokens
                    
                    audio_data.append({
                        'speech': speech[0] if len(speech.shape) > 1 else speech,
                        'speech_lengths': speech_lengths,
                        'fake_token_len': fake_token_len,
                        'seq_len': seq_len
                    })
                else:
                    # Skip this audio segment
                    continue
                    
            elif 'text' in ele:
                # Process text
                text = ele['text']
                if self.preprocessor_text:
                    text = self.preprocessor_text(text)
                
                text_tokens = self.tokenizer.tokenize(text)
                tokens.extend(text_tokens)
                loss_mask.extend([1] * len(text_tokens))
                
            elif 'prompt' in ele:
                # Process prompt (usually not included in loss)
                prompt_tokens = self.tokenizer.tokenize(ele['prompt'])
                tokens.extend(prompt_tokens)
                loss_mask.extend([0] * len(prompt_tokens))
            else:
                raise ValueError(f"Unknown element type: {ele}")

        if len(tokens) > 0 and tokens[-1] != self.eod:
            tokens.append(self.eod)
            loss_mask.append(1)

        return torch.tensor(tokens), torch.tensor(loss_mask, dtype=torch.float32), audio_data

    def _parse_string_format(self, content):
        """Parse string format content"""
        # For string format, we need to detect audio segments
        splits = self.pattern.split(content)
        tokens, loss_mask, audio_data = [], [], []
        
        for sub_str in splits:
            if not sub_str.startswith("<|startofspeech|>"):
                # Regular text
                text_tokens = self.tokenizer.tokenize(sub_str)
                tokens.extend(text_tokens)
                loss_mask.extend([1] * len(text_tokens))
            else:
                # Audio segment
                audio_path = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                if audio_path.startswith("!"):
                    audio_path = audio_path[1:]  # Remove the '!' prefix
                    
                speech, speech_lengths, fake_token_len, seq_len = self.process_asr_segment(audio_path)
                if speech is not None:
                    fake_tokens = [0] * fake_token_len
                    tokens.extend(fake_tokens)
                    loss_mask.extend([0] * len(fake_tokens))
                    
                    audio_data.append({
                        'speech': speech[0] if len(speech.shape) > 1 else speech,
                        'speech_lengths': speech_lengths,
                        'fake_token_len': fake_token_len,
                        'seq_len': seq_len
                    })

        tokens = torch.tensor(tokens + [self.eod])
        loss_mask = torch.tensor(loss_mask + [1], dtype=torch.float32)
        
        return tokens, loss_mask, audio_data

    def __getitem__(self, idx: int) -> dict:
        """Get item from dataset"""
        try:
            ds_idx = self.shuffle_idx[idx]
            content = self.dataset[ds_idx]
            
            # Parse content
            tokens, loss_mask, audio_data = self.parse_asr_text(content)
            
            # Truncate if necessary
            trunc_pos = self.config.sequence_length + 1
            tokens = tokens[:trunc_pos]
            loss_mask = loss_mask[:trunc_pos]
            
            if len(audio_data) == 0:
                return {
                    "text": tokens,
                    "loss_mask": loss_mask,
                    "source": [self.source, 'asr'],
                }
            
            # Prepare audio data
            audio_patches = []
            audio_lengths = []
            fake_token_lens = []
            
            for audio_item in audio_data:
                audio_patches.append(audio_item['speech'])
                audio_lengths.append(audio_item['speech_lengths'])
                fake_token_lens.append(audio_item['fake_token_len'])
            
            return {
                "text": tokens,
                "loss_mask": loss_mask,
                "audio": audio_patches,
                "audio_lengths": audio_lengths,
                "fake_token_lens": fake_token_lens,
                "raw_text": content,
                "source": [self.source, 'asr'],
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a fallback item
            return self.__getitem__(idx - 1 if idx > 0 else idx + 1)


class FunASRSFTDataset(FunASRDataset):
    """SFT version of FunASR dataset with chat format support"""
    
    def __init__(
        self,
        indexed_dataset: JsonlASRDataset,
        dataset_path: Optional[str],
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        config: FunASRDatasetConfig,
    ):
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        
        # SFT-specific tokens - 与 MinMoSense 保持一致
        self.sos = getattr(self.args, 'sos', "<|startoftranscript|>")
        self.eos = getattr(self.args, 'eos', "<|endoftext|>")
        
        # 尝试获取 im_start 和 im_end token IDs
        try:
            self.im_start_token = self.tokenizer.im_start_id if hasattr(self.tokenizer, 'im_start_id') else self.tokenizer.tokenize("<|im_start|>")
            self.im_end_token = self.tokenizer.im_end_id if hasattr(self.tokenizer, 'im_end_id') else self.tokenizer.tokenize("<|im_end|>")
        except:
            self.im_start_token = self.tokenizer.tokenize("<|im_start|>")
            self.im_end_token = self.tokenizer.tokenize("<|im_end|>")
        
        self.nl_token = self.tokenizer.tokenize("\n")
        
        # SFT parameters - 与 MinMoSense 保持一致
        self.mask_special_token = True
        self.mask_prompt = getattr(self.args, 'mask_prompt', True)
        self.mask_chatml_role = getattr(self.args, 'mask_chatml_role', False)
        self.mask_asr_response = getattr(self.args, 'mask_asr_response', False)
        
        # 添加 MinMoSense 中的参数
        self.batch_size = getattr(self.args, 'batch_size', None)
        self.batch_type = getattr(self.args, 'batch_type', None)
        self.prompt_ids_len = 0
        self.sys_prompt = getattr(self.args, 'sys_prompt', True)
        self.do_think = getattr(self.args, 'do_think', True)
        
        # 动态输出对齐参数
        self.use_dynamic_output_ratio = getattr(self.args, 'use_dynamic_output_ratio', 0.0)
        self.min_output_mask_token_len = getattr(self.args, 'min_mask_token_len', 1)
        self.min_output_non_mask_token_len = getattr(self.args, 'min_non_mask_token_len', 6)
        
        # 添加 MinMoSense 中的关键参数
        self.batch_size_scale_ratio_max = getattr(self.args, 'batch_size_scale_ratio_max', 1.5)
        self.batch_size_token_max = getattr(self.args, 'batch_size_token_max', 4000)
        self.multiturn_num_max = getattr(self.args, 'multiturn_num_max', 5)
        self.max_source_length = getattr(self.args, 'max_source_length', 8000)
        self.max_target_length = getattr(self.args, 'max_target_length', 2048)

    def get_random_user_prompt(self, item, user_prompt):
        """随机选择用户提示，与 MinMoSense 保持一致"""
        tasks = ["语音转写：", "Speech transcription:"]
        language = item.get("language", None)
        # LID in distill data is fake
        language = None
        if language is not None:
            if language.lower() == "zh":
                tasks.append("语音转写成中文：")
                tasks.append("Transcribe speech into Chinese:")
            elif language.lower() == "en":
                tasks.append("语音转写成英文：")
                tasks.append("Transcribe speech into English:")
        if len(tasks) == 2:
            task = random.choice(tasks)
        elif len(tasks) == 4:
            task = random.choices(tasks, weights=[0.4, 0.4, 0.1, 0.1])[0]
        if "语音转写：<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("语音转写：<|startofspeech|>", task + "<|startofspeech|>")
        elif "Speech transcription:<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("Speech transcription:<|startofspeech|>", task + "<|startofspeech|>")
        return user_prompt

    def parse_asr_text(self, data: dict, add_generation_prompt: bool = False):
        """Parse SFT format data with audio and text - 与 MinMoSense 逻辑保持一致"""
        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [], [], [], [], [], [], []
        )
        
        # 检查数据格式：优先处理转换后的格式
        if 'system' in data and 'user' in data and 'assistant' in data:
            # 处理 JsonlASRDataset 转换后的数据格式
            system = data['system']
            user = data['user']
            assistant = data['assistant']
            is_noised = data.get("noised", False)
            duration = data.get("duration", [[0, None]] * len(user))
        elif 'messages' in data:
            # 处理原始 messages 格式
            messages = data['messages']
            system, user, assistant = [], [], []
            duration = []
            
            for item in messages:
                role = item["role"]
                content = item["content"]
                if role == "system":
                    system.append(content)
                elif role == "user":
                    user.append(content)
                    duration.append([item.get("start_time", 0), item.get("end_time", None)])
                elif role == "assistant":
                    if "prev_content" in item:
                        prev_content = item["prev_content"]
                        assistant.append([content, {"prev_content": prev_content}])
                    else:
                        assistant.append(content)
            
            if len(system) == 0:
                system = ["You are a helpful assistant."]
            system = system * len(user)
            is_noised = data.get("noised", False)
        else:
            # Fallback to regular parsing
            return super().parse_asr_text(data)
        
        # 数据验证
        if len(user) < 1 or len(assistant) < 1:
            logging.warning(f"item is error: {data}")
            return [], [], [], [], [], [], []
        
        valid_turn_count = 0
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= self.multiturn_num_max:
                break
            if len(input_ids) > self.max_token_length:
                logging.info(f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {data}")
                break
            
            # 处理用户提示
            # user_prompt = self.get_random_user_prompt(data, user_prompt)
            
            # 语言检测和提示词处理
            if self.prompt_classes is not None:
                asr_prompt = user_prompt.split("<|startofspeech|>")[0]
                language = self.prompt_classes.detect_language(asr_prompt)
                user_prompt_all_context = self.prompt_classes.get_prompt(data, language)
            else:
                user_prompt_all_context = ""
            
            # 构建输入
            if valid_turn_count == 0:
                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                if not self.sys_prompt:
                    source_input = f"<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 添加think标签
            if not self.do_think:
                source_input += "<think>\n\n</think>\n\n"
            
            # 处理音频和文本
            splits = self.pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech = []
            speech_lengths = []
            audio_failed = False
            audio_failure_reason = None
            
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = self.tokenizer.tokenize(sub_str)
                    source_ids += sub_token
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                    if sub_str.startswith("!"):
                        try:
                            # 使用 duration 信息
                            duration_i = duration[i] if i < len(duration) else [0, None]
                            if self._duration_too_short_for_fbank(duration_i):
                                audio_failed = True
                                audio_failure_reason = f"duration too short: {duration_i}"
                                break
                            
                            data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs, duration=duration_i)
                            if self._audio_num_samples(data_src) < self.min_audio_samples_for_fbank:
                                audio_failed = True
                                audio_failure_reason = (
                                    f"audio too short after loading: "
                                    f"{self._audio_num_samples(data_src)} < {self.min_audio_samples_for_fbank}"
                                )
                                break
                            
                            # 应用噪声预处理器
                            if self.preprocessor_noise is not None and not is_noised:
                                try:
                                    data_src = self.preprocessor_noise(data_src.numpy())
                                except Exception as e:
                                    logging.error(f"Generate noise audio failed: {e}")

                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )
                            
                            if speech_lengths[0].item() > self.max_source_length:
                                logging.info(f"speech_lengths > max_source_length: {speech_lengths[0].item()}>{self.max_source_length}, {data}")
                                audio_failed = True
                                audio_failure_reason = (
                                    f"speech too long: {speech_lengths[0].item()} > {self.max_source_length}"
                                )
                                break
                                
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                                
                        except Exception as e:
                            logging.warning(f"Loading wav failed! {str(e)}, {traceback.format_exc()}\n{data}")
                            audio_failed = True
                            audio_failure_reason = f"audio loading failed: {e}"
                            break
                            
                        # 计算 fake token 长度
                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token

            # A required audio segment is part of the supervision contract for
            # this turn.  Never keep its target text when the audio is absent or
            # invalid, otherwise the sample silently degrades into text-only SFT.
            if audio_failed:
                logging.warning(
                    "Drop SFT turn because required audio is invalid: turn=%s, reason=%s",
                    i,
                    audio_failure_reason,
                )
                continue

            # 处理文本预处理器和重复token检查
            if self.preprocessor_text is not None:
                has_repeat_tokens = self.preprocessor_text(target_out)
                if has_repeat_tokens:
                    logging.info(f"item has_repeat_tokens: {data}")
                    continue

            # 记录音频信息
            if fbank_beg_i > 0:
                fbank_beg.append(fbank_beg_i + len(input_ids))
                fake_token_len.append(fake_token_len_i)
            else:
                fbank_beg.append(-1)
                fake_token_len.append(0)

            # 构建标签
            source_mask = [-100] * len(source_ids)
            
            # 处理目标输出
            if target_out is not None and any(isinstance(item, dict) and "prev_content" in item for item in target_out):
                prev_value = next(
                    (item["prev_content"] for item in target_out if isinstance(item, dict) and "prev_content" in item),
                    None
                )
                if prev_value:
                    source_ids += self.tokenizer.tokenize(prev_value)
                    source_mask += [-100] * len(self.tokenizer.tokenize(prev_value))
                target_out = target_out[0] if isinstance(target_out[0], str) else str(target_out[0])
            
            target_out = f"{target_out}<|im_end|>"
            target_ids = self.tokenizer.tokenize(target_out)
            
            if len(target_ids) > self.max_target_length:
                logging.info(f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {data}")
                continue
            
            # 模拟动态输出对齐
            if np.random.rand() < self.use_dynamic_output_ratio:
                max_len = len(target_ids)
                min_output_mask_token_len = min(self.min_output_mask_token_len, max_len)
                min_output_non_mask_token_len = min(self.min_output_non_mask_token_len, max_len)
                if max_len - min_output_non_mask_token_len > min_output_mask_token_len:
                    end_index = np.random.randint(min_output_mask_token_len,
                                                  max_len - min_output_non_mask_token_len)
                else:
                    end_index = max_len - min_output_non_mask_token_len
                if end_index > 0:
                    target_ids[:end_index] = [-100] * end_index

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)
            valid_turn_count += 1

        # if add_generation_prompt:
        #     input_ids = input_ids + self.im_start_token + self.tokenizer.tokenize("assistant") + self.nl_token
        #     labels = []
        #     return input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len
        # else:
        #     input_ids = input_ids + [self.eod]
        #     labels = labels + [0]

        return input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len

    def __getitem__(self, idx: int) -> dict:
        """Get item from SFT dataset - 与 MinMoSense 输出格式保持一致"""
        # item_start = time.time()
        output = None
        index_cur = idx
        ds_idx = None
        content = None
        last_retry_idx = 0
        
        for retry_idx in range(self.retry):
            last_retry_idx = retry_idx
            if retry_idx > 0:
                logging.info(f"retry: {retry_idx}")
            
            badcase_flag = False
            if retry_idx == 0:
                index_cur = idx
            else:
                index_cur = torch.randint(0, len(self.dataset), ()).item()

            ds_idx = self.shuffle_idx[index_cur]
            content = self.dataset[ds_idx]
            
            # 解析 SFT 内容
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = self.parse_asr_text(content)

            # If all turns were dropped (for example because every required
            # audio failed), retry another dataset item instead of returning an
            # empty sample to the collator.
            if len(input_ids) == 0 or len(labels) == 0:
                logging.warning(
                    "No valid SFT turn remains after parsing; retry another item: ds_idx=%s",
                    ds_idx,
                )
                continue
            
            if badcase_flag:
                print(f"badcase_flag: {badcase_flag}")
                continue
                
            # 转换为 tensor
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = content
            
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
                
            if len(input_ids) > self.max_token_length:
                logging.warning(f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {content}")
                print(f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {content}")
                continue

            break
        if output is None:
            # CODEX slow-item debug logging disabled.
            # _codex_log_slow_item(
            #     elapsed_s=time.time() - item_start,
            #     idx=idx,
            #     index_cur=index_cur,
            #     ds_idx=ds_idx,
            #     retry_idx=last_retry_idx,
            #     item=content or {},
            # )
            # logging.warning(f"output is None, item: {content}")
            # print(f"output is None, item: {content}")
            return None
        else:
            # print(f"output is not None, item: {content}")
            # 加一个判断，确保output每个字段都不为None
            for key, value in output.items():
                if value is None:
                    # logging.warning(f"output[{key}] is None, item: {content}")
                    # print(f"output[{key}] is None, item: {content}")
                    # CODEX slow-item debug logging disabled.
                    # _codex_log_slow_item(
                    #     elapsed_s=time.time() - item_start,
                    #     idx=idx,
                    #     index_cur=index_cur,
                    #     ds_idx=ds_idx,
                    #     retry_idx=last_retry_idx,
                    #     item=content or {},
                    #     input_ids=output.get("input_ids"),
                    #     speech_lengths=output.get("speech_lengths"),
                    #     fake_token_len=output.get("fake_token_len"),
                    # )
                    return None
        # CODEX slow-item debug logging disabled.
        # _codex_log_slow_item(
        #     elapsed_s=time.time() - item_start,
        #     idx=idx,
        #     index_cur=index_cur,
        #     ds_idx=ds_idx,
        #     retry_idx=last_retry_idx,
        #     item=content or {},
        #     input_ids=output.get("input_ids"),
        #     speech_lengths=output.get("speech_lengths"),
        #     fake_token_len=output.get("fake_token_len"),
        # )
        return output

    def collator(self, samples: list = None):
        """Collator function - 与 MinMoSense 保持一致"""
        # collate_start = time.time()
        # pad_times = {}
        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        pad_value = -1  # int_pad_value
                    else:
                        pad_value = 0.0  # float_pad_value

                    # pad_start = time.time()
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
                    # pad_times[key] = round(time.time() - pad_start, 3)

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data")
                    samples = samples[:-1]
                    continue

            break

        # CODEX slow-collate debug logging disabled.
        # _codex_log_slow_dataset_collate(
        #     "FunASRDataset.collator",
        #     time.time() - collate_start,
        #     samples,
        #     outputs,
        #     pad_times,
        # )
        return outputs
