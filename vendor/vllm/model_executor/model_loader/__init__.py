# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, List, Literal, Optional

from torch import nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.bitsandbytes_loader import BitsAndBytesModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader

try:
    from vllm.model_executor.model_loader.quick_model_loader import QuickLoader
except ImportError:
    QuickLoader = None
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader,
)
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm.model_executor.model_loader.tensorizer_loader import TensorizerLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
    get_model_cls,
)

logger = init_logger(__name__)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "hf",
    "bitsandbytes",
    "dummy",
    "fastsafetensors",
    "gguf",
    "mistral",
    "npcache",
    "pt",
    "runai_streamer",
    "runai_streamer_sharded",
    "safetensors",
    "sharded_state",
    "tensorizer",
    "mows_safetensors",
]
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "fastsafetensors": DefaultModelLoader,
    "gguf": GGUFModelLoader,
    "mistral": DefaultModelLoader,
    "npcache": DefaultModelLoader,
    "pt": DefaultModelLoader,
    "runai_streamer": RunaiModelStreamerLoader,
    "runai_streamer_sharded": ShardedStateLoader,
    "safetensors": DefaultModelLoader,
    "sharded_state": ShardedStateLoader,
    "tensorizer": TensorizerLoader,
    "mows_safetensors": DefaultModelLoader,
}


def register_model_loader(load_format: str):
    """Register a customized vllm model loader.

    When a load format is not supported by vllm, you can register a customized
    model loader to support it.

    Args:
        load_format (str): The model loader format name.

    Examples:
        >>> from vllm.config.load import LoadConfig
        >>> from vllm.model_executor.model_loader import (
        ...     get_model_loader,
        ...     register_model_loader,
        ... )
        >>> from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        >>>
        >>> @register_model_loader("my_loader")
        ... class MyModelLoader(BaseModelLoader):
        ...     def download_model(self):
        ...         pass
        ...
        ...     def load_weights(self):
        ...         pass
        >>>
        >>> load_config = LoadConfig(load_format="my_loader")
        >>> type(get_model_loader(load_config))
        <class 'MyModelLoader'>
    """  # noqa: E501

    def _wrapper(model_loader_cls):
        if load_format in _LOAD_FORMAT_TO_MODEL_LOADER:
            logger.warning(
                "Load format `%s` is already registered, and will be "
                "overwritten by the new loader class `%s`.",
                load_format,
                model_loader_cls,
            )
        if not issubclass(model_loader_cls, BaseModelLoader):
            raise ValueError(
                "The model loader must be a subclass of `BaseModelLoader`."
            )
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = model_loader_cls
        logger.info(
            "Registered model loader `%s` with load format `%s`",
            model_loader_cls,
            load_format,
        )
        return model_loader_cls

    return _wrapper


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    tensor_queue_union: Any | List[Any] | None = None,
    enable_spec_decoding: bool = False,
) -> nn.Module:
    if envs.VLLM_USE_SWAP_ENGINE:
        try:
            logger.info("using quick model loader to load model")
            loader = QuickLoader(
                vllm_config.load_config, tensor_queue_union=tensor_queue_union
            )
            return loader.load_model(
                vllm_config=vllm_config,
                model_config=model_config,
                enable_spec_decoding=enable_spec_decoding,
            )
        except Exception as e:
            logger.warning("using quick model loader fails: %s", e)
            envs.VLLM_USE_QUICK_LOADER = False
            rank = get_tensor_model_parallel_rank()
            if rank == 0 and isinstance(tensor_queue_union, List):
                # except the last tensor queue, put False to all queues
                for tensor_queue in tensor_queue_union[:-1]:
                    tensor_queue.put(False)
    logger.info("using vllm's default model loader to load model")
    loader = get_model_loader(vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(vllm_config=vllm_config, model_config=model_config)


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
    "QuickLoader",
]
