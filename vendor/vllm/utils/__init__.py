# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import functools
import hashlib
import io
import os
import uuid
import warnings
from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image

import vllm.envs as envs
from vllm.envs import VLLM_USE_FILE_PATH_MMD_CACHE
from vllm.logger import init_logger
from vllm.utils.cache import LRUCache

_DEPRECATED_MAPPINGS = {
    "cprofile": "profiling",
    "cprofile_context": "profiling",
    # Used by lm-eval
    "get_open_port": "network_utils",
}


def __getattr__(name: str) -> Any:  # noqa: D401 - short deprecation docstring
    """Module-level getattr to handle deprecated utilities."""
    if name in _DEPRECATED_MAPPINGS:
        submodule_name = _DEPRECATED_MAPPINGS[name]
        warnings.warn(
            f"vllm.utils.{name} is deprecated and will be removed in a future version. "
            f"Use vllm.utils.{submodule_name}.{name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(f"vllm.utils.{submodule_name}", fromlist=[submodule_name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # expose deprecated names in dir() for better UX/tab-completion
    return sorted(list(globals().keys()) + list(_DEPRECATED_MAPPINGS.keys()))


logger = init_logger(__name__)

# Constants related to forcing the attention backend selection

# String name of register which may be set in order to
# force auto-selection of attention backend by Attention
# wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values of STR_BACKEND_ENV_VAR
# register, corresponding to possible backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_XFORMERS_ATTN_VAL: str = "XFORMERS"
STR_BLADNN_ATTN_VAL: str = "BLADNN_ATTN"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_FLASHINFER_VAL: str = "FLASHINFER"
STR_DUAL_CHUNK_FLASH_ATTN_VAL: str = "DUAL_CHUNK_FLASH_ATTN"
STR_SPARSE_FLASH_ATTN_VAL: str = "SPARSE_FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"


MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


_abort_request_id_cache: LRUCache[str, Any] = LRUCache(capacity=1024)


def set_abort_request_id(request_id: str) -> None:
    _abort_request_id_cache.put(request_id, 1)


def is_abort_request_id(request_id: str) -> bool:
    return _abort_request_id_cache.get(request_id) is not None


def get_audio_md5_digest_list(audio_object: np.ndarray) -> list:
    if not isinstance(audio_object, np.ndarray):
        return []
    audio_bytes = audio_object.tobytes()
    return list(hashlib.md5(audio_bytes).digest())


def get_image_md5_digest_list(image_object: Image.Image) -> list:
    if not isinstance(image_object, Image.Image):
        return []

    image_bytes = None
    if hasattr(image_object, "getexif"):
        image_exif = image_object.getexif()
        if (
            image_exif.__contains__("image_path")
            and isinstance(image_exif["image_path"], str)
            and len(image_exif["image_path"]) > 0
        ):
            image_bytes = image_exif["image_path"].encode("utf-8")

    if image_bytes is None:
        image_bytes = image_object.tobytes()

    return list(hashlib.md5(image_bytes).digest())


def locate_parent(root, full_path):
    parent = root
    path = full_path.split(".")
    for p in path[:-1]:
        parent = getattr(parent, p)
    return parent, path[-1]


def hash_video_tensor(tensor: torch.Tensor, num_samples: int = 1024) -> list:
    flat_tensor = tensor.reshape(-1)

    total_elements = flat_tensor.size(0)

    if total_elements <= num_samples:
        sampled = flat_tensor
    else:
        step = total_elements / num_samples
        indices = torch.arange(num_samples) * step
        indices = indices.floor().long()
        sampled = flat_tensor[indices]

    if sampled.device.type != "cpu":
        sampled = sampled.cpu()
    numpy_array = sampled.numpy()

    hash_object = hashlib.md5()
    hash_object.update(numpy_array.tobytes())
    hash_object.update(str(total_elements).encode("utf-8"))

    return list(hash_object.digest())


def extract_video_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, list):
        tensors = []
        for item in obj:
            tensors.extend(extract_video_tensors(item))
        return tensors
    return []


def get_video_md5_digest_list(video_object) -> list:
    if VLLM_USE_FILE_PATH_MMD_CACHE and isinstance(video_object[0][0], str):
        urls = "_".join(video_object[0])
        return list(hashlib.md5(urls.encode()).digest())
    if isinstance(video_object[0], Image.Image):
        batch_io = io.BytesIO()
        for image in video_object:
            bytes_io = io.BytesIO()
            image.save(bytes_io, format="JPEG")
            batch_io.write(bytes_io.getvalue())
        batch_io.seek(0)
        combined_data = batch_io.read()
        return list(hashlib.md5(combined_data).digest())
    video_list = extract_video_tensors(video_object)
    if not video_list:
        return []

    _video_bytes = b"".join(bytes(hash_video_tensor(video)) for video in video_list)

    return list(hashlib.md5(_video_bytes).digest())


def get_random_md5_digest_list() -> list:
    return list(hashlib.md5(os.urandom(16)).digest())


class EventPool:
    def __init__(self, reserve_num_events: int, device: torch.device):
        self.reserve_num_events = reserve_num_events
        self.event_queue: deque[torch.cuda.Event] = deque()
        self.device = device
        with torch.cuda.device(device):
            for i in range(reserve_num_events):
                event = torch.cuda.Event()
                # create the detail new event
                event.record()
                event.synchronize()
                self.event_queue.append(event)

    def get_event(self) -> torch.cuda.Event:
        if len(self.event_queue) == 0:
            with torch.cuda.device(self.device):
                event = torch.cuda.Event()
                # create the detail new event
                event.record()
                event.synchronize()
                self.event_queue.append(event)
        return self.event_queue.popleft()

    def put_event(self, event: torch.cuda.Event):
        self.event_queue.append(event)

    def get_events(self, num_events: int) -> list[torch.cuda.Event]:
        ret = []
        for i in range(num_events):
            ret.append(self.get_event())
        return ret

    def put_events(self, events: list[torch.cuda.Event]):
        for event in events:
            self.event_queue.append(event)


def serialize_tensor(tensor):
    # 使用io.BytesIO保存张量到字节流
    byte_stream = io.BytesIO()
    torch.save(tensor, byte_stream)
    tensor_bytes = byte_stream.getvalue()
    tensor_size = len(tensor_bytes)
    return tensor_bytes, tensor_size


def convert_blocks_to_write_to_tensor(blocks_to_write):
    # Blocks to write is a list of
    # [user_id, CPU block number, content hash, token_ids, udc_ttl].
    # token_ids is a fixed size(block_size) list, so flat it
    if blocks_to_write:
        flat_data = []
        for user_id, block_id, content_hash, token_ids, udc_ttl in blocks_to_write:
            # 将 user_id, block_id, content_hash, token_ids..., udc_ttl 拼接成一行
            row = [user_id, block_id, content_hash, *token_ids, udc_ttl]
            flat_data.append(row)

        blocks_to_write_tensor = torch.tensor(
            flat_data, device="cuda", dtype=torch.int64
        )
    else:
        blocks_to_write_tensor = torch.empty(0, 0, device="cuda", dtype=torch.int64)
    return blocks_to_write_tensor


def format_content_hash(content_hash: int) -> str:
    """
    将 content_hash 转换为无符号 64-bit 十六进制字符串，
    避免负号或高位截断影响路径。

    Args:
        content_hash (int): 内容哈希值，可能是负数或超出 32-bit 范围

    Returns:
        str: 无符号十六进制表示，适合用作路径或缓存 key
    """
    unsigned_hash = content_hash & 0xFFFFFFFFFFFFFFFF  # 转为无符号64位整数
    return f"{unsigned_hash:016x}"  # 固定输出 16 位十六进制（64 bit）


def set_bladnn_quantization_tuning_config_file() -> None:
    if not envs.VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX:
        return

    import bladnn

    from vllm.distributed import get_tensor_model_parallel_rank

    file_full_path = (
        f"{envs.VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX}"
        f".rank{get_tensor_model_parallel_rank()}"
    )
    if os.path.exists(file_full_path):
        logger.info("bladnn set_tuning_config_load_path %s", file_full_path)
        bladnn.set_tuning_config_load_path(
            get_tensor_model_parallel_rank(), envs.VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX
        )
    else:
        logger.info("bladnn set_tuning_config_save_path %s", file_full_path)
        bladnn.set_tuning_config_save_path(
            get_tensor_model_parallel_rank(), envs.VLLM_BLADNN_TUNING_CONFIG_FILE_PREFIX
        )
