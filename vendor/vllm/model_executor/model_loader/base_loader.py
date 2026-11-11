# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""

        # MoWS requires some preparation before loading weights.
        if (
            self.load_config.load_format == "mows_safetensors"
            and get_tensor_model_parallel_rank() == 0
        ):
            logger.info("MoWS try to set ModelConfig")
            # yapf: disable
            from model_weight_service.utils import ModelConfig as MoWSModelConfig
            from model_weight_service.utils import set_model_config

            # yapf: enable
            if not os.path.isdir(model_config.model):  # model is not local
                raise ValueError(f"{model_config.model} is not local")
            mows_model_config = MoWSModelConfig()
            mows_model_config.set_tensor_parallel_num(
                vllm_config.parallel_config.tensor_parallel_size
            )
            try:
                set_model_config(model_config.model, mows_model_config)
            except Exception as e:
                logger.warning("MoWS try to set ModelConfig failed, msg: %s", e)

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()
