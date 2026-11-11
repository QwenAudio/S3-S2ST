# SPDX-License-Identifier: Apache-2.0

import os
from multiprocessing.queues import Queue
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import (
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)

try:
    from quick_model_loader.handle import ShardingHandle
    from quick_model_loader.meta import DynamicMinMaxConfig

    HAS_QUICK_MODEL_LOADER = True
except ImportError:
    HAS_QUICK_MODEL_LOADER = False

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
    set_default_torch_dtype,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.s3_utils import glob as s3_glob
from vllm.transformers_utils.utils import is_s3

logger = init_logger(__name__)


class QuickLoader(BaseModelLoader):
    """Model loader that uses quick model loader to load model weights."""

    def __init__(
        self,
        load_config: LoadConfig,
        tensor_queue_union: Union[None, Any, List[Any]],
        load_timeout_seconds: Optional[int] = 7200,
    ):
        assert HAS_QUICK_MODEL_LOADER, "Quick model loader is not imported."
        super().__init__(load_config)
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.model_loader.quick_model_loader_utils import (
            get_loaded_model,
            get_model_handle,
            get_model_handle_queue,
            join_thread_for_loading_model,
            set_loaded_model,
            set_model_handle,
        )

        self.rank = get_tensor_model_parallel_rank()
        if self.rank == 0:
            # the last queue is the driver queue
            driver_queue = tensor_queue_union[-1]
            # wait for model handle to be ready
            if driver_queue:
                model_handle = driver_queue.get(timeout=load_timeout_seconds)
                model = model_handle.to_model()
                set_model_handle(model_handle)
                set_loaded_model(model)

        if self.rank == 0:
            self.loaded_model = get_loaded_model()
            self.model_handle = get_model_handle()

        if self.rank == 0:
            self.tensor_queues = tensor_queue_union
        else:
            self.tensor_queue = tensor_queue_union

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        return None

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: Optional[ModelConfig] = None,
        enable_spec_decoding: bool = False,
    ) -> nn.Module:
        parallel_config = vllm_config.parallel_config

        self.configure_parallelism(parallel_config)
        self.check_params()

        model = self.prepare_model(vllm_config=vllm_config, model_config=model_config)

        if self.rank == 0:
            index = 0
            if enable_spec_decoding:
                index = self.parallel_size
            self.send_shardings(index)
            self.set_model_weights(model, self.loaded_model.to_tensors(0 + index))

        else:
            sharidng_content = self.recv_sharding()
            self.set_model_weights(model, sharidng_content.to_tensors())

        return model.eval()

    def configure_parallelism(self, parallel_config: ParallelConfig):
        tensor_parallel_size = parallel_config.tensor_parallel_size
        pipeline_parallel_size = parallel_config.pipeline_parallel_size

        if tensor_parallel_size > 1 and pipeline_parallel_size > 1:
            raise RuntimeError("both tp and pp are > 1, conflicts!")
        if pipeline_parallel_size > 1:
            parallel_type = "pp"
            parallel_size = pipeline_parallel_size
        else:
            parallel_type = "tp"
            parallel_size = tensor_parallel_size

        self.parallel_type = parallel_type
        self.parallel_size = parallel_size

    def check_params(self):
        if self.rank == 0:
            if self.parallel_size == 1:
                assert len(self.tensor_queues) == 1, (
                    "tp=1 should only have driver tensor queue for worker 0"
                )

            elif not isinstance(self.tensor_queues, List):
                raise RuntimeError(
                    "master worker should get a list of tensor queues to "
                    "dispatch tensors"
                )
            else:
                assert len(self.tensor_queues) == self.parallel_size, (
                    "tensor queues size doesn't match parallel size,"
                    f"queues size: {len(self.tensor_queues)}, "
                    f"parallel size: {self.parallel_size}"
                )

            if self.loaded_model is None or self.model_handle is None:
                for i in range(1, self.parallel_size):
                    self.tensor_queues[i - 1].put(False)
                raise RuntimeWarning(
                    "loaded model or handle is missing in master worker, "
                    "fallback to vllm's default loader"
                )
            else:
                for i in range(1, self.parallel_size):
                    self.tensor_queues[i - 1].put(True)

        else:
            assert self.parallel_size > 1
            assert self.tensor_queue is not None, (
                "slave worker should get a queue to receive tensors from master"
            )

            is_quick_loader_enabled = self.tensor_queue.get()
            if not is_quick_loader_enabled:
                raise RuntimeWarning(
                    "loaded model or handle is missing in master worker, "
                    "fallback to vllm's default loader"
                )

    def prepare_model(
        self,
        vllm_config: VllmConfig,
        model_config: Optional[ModelConfig] = None,
    ):
        envs.VLLM_USE_QUICK_LOADER = True

        model = self.init_model_structure(
            vllm_config=vllm_config, model_config=model_config
        )
        return model

    def init_model_structure(
        self,
        vllm_config: VllmConfig,
        model_config: Optional[ModelConfig] = None,
    ):
        device_config = vllm_config.device_config
        if model_config is None:
            model_config = vllm_config.model_config
        with (
            set_default_torch_dtype(model_config.dtype),
            torch.device(device_config.device),
        ):
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
            return model

    def send_shardings(self, index: int = 0):
        logger.info("quick loader: master worker dispatch tensors to slaves")
        for i in range(1, self.parallel_size):
            sharding_handle = self.model_handle.sharding_handles[i + index]
            self.tensor_queues[i - 1].put(sharding_handle)

    def recv_sharding(self):
        logger.info("quick loader: slave worker receive tensors from master")
        sharding_handle: ShardingHandle = self.tensor_queue.get()
        shariding_content = sharding_handle.to_sharding_content()

        return shariding_content

    def set_model_weights(
        self,
        model: nn.Module,
        tensors_generator: Generator[Tuple[str, torch.Tensor], None, None],
    ):
        logger.info("linking weights to model......")
        params_dict = dict(model.named_parameters())
        for name, param in params_dict.items():
            param.detach_()

        count = 0
        for name, weight in tensors_generator:
            if name.endswith(":t_"):
                name = name[: -len(":t_")]
                weight = weight.t()
            else:
                weight = weight
            if name not in params_dict:
                logger.warning("parameter %s not found in model, skip loading it", name)
                continue
            param = params_dict[name]
            if param.dtype != weight.dtype:
                param.data = param.data.to(dtype=weight.dtype)
            param.set_(source=weight)
            count += 1

        assert count == len(params_dict), (
            f"missing tensors to load! loaded: {count}, expected: {len(params_dict)}"
        )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        parallel_num: int,
        parallel_type: str = "tp",  # currently only support tp
        quantization_type: Optional[str] = None,
        dynamic_minmax_config: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        params_dict = dict(model.named_parameters())

        if parallel_num == 1:
            rank = 0
        else:
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()

        logger.info("save model start... rank:%s", rank)

        if dynamic_minmax_config and DynamicMinMaxConfig:
            dynamic_minmax_config = DynamicMinMaxConfig(dynamic_minmax_config)
            dynamic_minmax_config_str = dynamic_minmax_config.to_str()
        else:
            dynamic_minmax_config_str = ""

        model_dir = ""
        if quantization_type is None:
            model_dir = f"{parallel_type}{parallel_num}"
        else:
            if dynamic_minmax_config_str:
                model_dir = f"{quantization_type}_{dynamic_minmax_config_str}_{parallel_type}{parallel_num}"  # noqa: E501
            else:
                model_dir = f"{quantization_type}_{parallel_type}{parallel_num}"
        save_path = os.path.join(path, model_dir, f"rank{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        from safetensors.torch import save_file

        if max_size is None:
            file_name = f"model-rank-{rank}-part-0.safetensors"
            file_path = os.path.join(save_path, file_name)
            save_file(params_dict, file_path)
        else:
            part_idx = 0
            total_size = 0
            params_dict_part = {}

            for key, tensor in params_dict.items():
                if not tensor.is_contiguous():
                    key += ":t_"
                    tensor = tensor.t()
                param_size = tensor.nelement() * tensor.element_size()
                if total_size + param_size > max_size:
                    file_name = f"model-rank-{rank}-part-{part_idx}.safetensors"
                    file_path = os.path.join(save_path, file_name)
                    save_file(params_dict_part, file_path)
                    part_idx += 1
                    total_size = 0
                    params_dict_part = {}
                params_dict_part[key] = tensor
                total_size += param_size

            if len(params_dict_part) > 0:
                file_name = f"model-rank-{rank}-part-{part_idx}.safetensors"
                file_path = os.path.join(save_path, file_name)
                save_file(params_dict_part, file_path)

        logger.info("save model finish! rank:%s", rank)
