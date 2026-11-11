import os
from abc import ABC
from operator import attrgetter
from tokenize import group
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    overload,
)

import torch
from torch.func import functional_call

import vllm
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.utils.platform_utils import (
    is_pin_memory_available,
    is_uva_available,
)

logger = init_logger(__name__)
_SubmoduleAccessor = Callable[[torch.nn.Module], torch.nn.Module]
_WhitelistParamNamesCreator = Callable[[torch.nn.Module], List[str]]
class BaseOffloader:
    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return list(all_modules_generator)

    def post_init(self):
        pass

    @property
    def forbid_copy_engine_usage(self):
        return False

class _BaseParamOffloader:
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        return {
            "cpu": _CpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, module, param_name):
        self._module = module
        self._param_name = param_name
        self._param = attrgetter(self._param_name)(self._module)

    def post_init(self):
        pass

    def create_device_tensor(self):
        raise NotImplementedError

class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        cpu_data = torch.empty_strided(
            size=self._param.size(),
            stride=self._param.stride(),
            dtype=self._param.dtype,
            layout=self._param.layout,
            device="cpu",
            pin_memory=True,
        )
        cpu_data.copy_(self._param)
        self._param.data = cpu_data

    def create_device_tensor(self):
        return self._param.to("cuda", non_blocking=True)

class _ModuleOffloader:
    def __init__(
        self,
        mode: str,
        module: torch.nn.Module,
        alt_stream: torch.cuda.Stream,
        whitelist_param_names: List[str],
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream

        assert self.device != torch.device(
            "cpu"
        ), "not handled device=cpu case yet (should skip this tensor)"

        self._device_tensors = None
        self._load_event = None

        param_dict = dict(self.module.named_parameters())
        assert all(
            name in param_dict for name in whitelist_param_names
        ), f"{whitelist_param_names=} {list(param_dict.keys())=}"

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for name, param_offloader in self._param_offloaders.items():
            param_offloader.post_init()

    def start_onload(self):
        self.alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.alt_stream):
            self._device_tensors = self._create_device_tensors()
            self._load_event = torch.cuda.Event()
            self._load_event.record()  # type: ignore[attr-defined]

    def offload(self):
        self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self):
        assert self._device_tensors is not None
        self._load_event.wait()
        return self._device_tensors

    def _create_device_tensors(self):
        return {k: v.create_device_tensor() for k, v in self._param_offloaders.items()}

class NoopOffloader(BaseOffloader):
    pass

# For simplicity use singleton, but can surely support multi instance
_instance: Optional[BaseOffloader] = NoopOffloader()

def get_offloader():
    assert _instance is not None
    return _instance


def set_offloader(instance: BaseOffloader):
    global _instance
    _instance = instance

def create_offloader_from_server_args(
    vllm_config: VllmConfig,
):
    cache_config = vllm_config.cache_config
    if offloading_config:= vllm_config.cpu_offloading_config:
        if offloading_config.offloading_method == "v1":
            if (
                vllm_config.cache_config.cpu_offload_gb > 0
                and
                offloading_config.cpu_offload_gb > 0
            ):
                if (cache_config.cpu_offload_gb != offloading_config.cpu_offload_gb):
                    logger.warning_once(
                        "Both 'cache_config.cpu_offload_gb' and "
                        "'cpu_offloading_config.cpu_offload_gb' are set; "
                        "'cpu_offloading_config.cpu_offload_gb' will be effective."
                    )
            return OffloaderV1(
                cpu_offload_max_bytes=offloading_config.cpu_offload_gb * 1024**3
            )
        elif offloading_config.offloading_method == "v2":
            return OffloaderV2(
                group_size=offloading_config.offload_group_size,
                num_in_group=offloading_config.offload_num_in_group,
                prefetch_step=offloading_config.offload_prefetch_step,
                mode="cpu",
            )
    elif vllm_config.cache_config.cpu_offload_gb > 0:
        return OffloaderV1(
            cpu_offload_max_bytes=int(vllm_config.cache_config.cpu_offload_gb * 1024**3)
        )
    else:
        return NoopOffloader()


class OffloaderV1(BaseOffloader):
    def __init__(self, cpu_offload_max_bytes: int):
        self._cpu_offload_bytes = 0
        self._cpu_offload_max_bytes = cpu_offload_max_bytes

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return [self.maybe_offload_to_cpu(module) for module in all_modules_generator]

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        # note: uva is not_supported here.
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
            return module

        pin_memory = is_pin_memory_available()
        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for p in module.parameters():
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            p.data = cpu_data
            self._cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device, it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }
                output = functional_call(module, device_state, args=args, kwargs=kwargs)
                module.forward = forward
                return output

            module.forward = forward

        return module

class OffloaderV2(BaseOffloader):
    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        mode: str,
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.mode = mode

        self.offloaders: list[_ModuleOffloader] = []

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):

        
        
        assert len(self.offloaders) == 0, "should only call wrap_modules once"

        alt_stream = torch.cuda.Stream()

        all_modules = []
        offload_submodules = []
        for module_index, module in enumerate(all_modules_generator):
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                assert whitelist_param_names_creator is not None
                whitelist_param_names = whitelist_param_names_creator(module)
                logger.info(
                    f"offload module_index={module_index} "
                    f"module={type(module)} "
                    f"params={whitelist_param_names} "
                    f"cuda_memory_allocated={torch.cuda.memory_allocated()}"
                )
                offload_submodules.append(module)
                self.offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=module,
                        alt_stream=alt_stream,
                        whitelist_param_names=whitelist_param_names,
                    )
                )

        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index,
                module=module,
                offloaders=self.offloaders,
                prefetch_step=self.prefetch_step,
            )

        return all_modules

    def post_init(self):
        for offloader in self.offloaders:
            offloader.post_init()

        for i in range(self.prefetch_step):
            self.offloaders[i].start_onload()

    @property
    def forbid_copy_engine_usage(self):
        return self.mode == "cpu"

def _hook_module_forward_for_offloader(index, module, offloaders, prefetch_step):
    def _on_forward_end():
        offloaders[(index + prefetch_step) % len(offloaders)].start_onload()
        offloaders[index].offload()

    _hook_module_forward_raw(
        module,
        on_forward_end=_on_forward_end,
        get_parameter_and_buffer_dicts=lambda: offloaders[
            index
        ].wait_and_get_device_tensors(),
    )

def _hook_module_forward_raw(module, on_forward_end, get_parameter_and_buffer_dicts):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        output = functional_call(
            module, get_parameter_and_buffer_dicts(), args=args, kwargs=kwargs
        )
        on_forward_end()
        module.forward = forward
        return output

    module.forward = forward