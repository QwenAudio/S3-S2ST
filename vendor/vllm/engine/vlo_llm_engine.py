import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import inspect
import math
import multiprocessing
import os
import pickle
import queue
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import zmq
import zmq.asyncio
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

import vllm.envs as envs
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
from vllm.engine.multiprocessing import (
    ENGINE_DEAD_ERROR,
    IPC_DATA_EXT,
    IPC_HEALTH_EXT,
    IPC_INPUT_EXT,
    IPC_OUTPUT_EXT,
    REQUEST_OUTPUTS_T,
    RPC_REQUEST_T,
    VLLM_RPC_SUCCESS_STR,
    RPCError,
    RPCStartupRequest,
    RPCStartupResponse,
)
from vllm.engine.multiprocessing.client import MQClientClosedError, MQLLMEngineClient
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.rotary_embedding import get_vlo_rope_index
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config, try_get_generation_config
from vllm.transformers_utils.qwen2_code2wav_dit.model.utils import except_quantized_fp
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.transformers_utils.vlo.fm_euler_ar_dit import (
    FlowMatchEulerDiscreteSchedulerARDiT,
)
from vllm.transformers_utils.vlo.wan_model import WanModel
from vllm.transformers_utils.vlo.wanx_vae import VideoVAE
from vllm.utils.network_utils import LRUCache, get_open_zmq_ipc_path

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR),)
VLLM_VLO_V2V_USE_ZMQ = False

logger = init_logger(__name__)


@dataclass
class Visual_Gen_Params:
    prompt_token_ids: torch.Tensor
    image_grid_thw: torch.Tensor
    latent_size_h: int
    latent_size_w: int
    attention_mask: torch.Tensor
    num_samples: int
    cfg_img: Optional[float]
    cfg_num: int
    partial_image_num: int


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
        generator: AsyncGenerator[RequestOutput, None],
        loop: asyncio.AbstractEventLoop,
    ):
        self._generator = generator
        self._loop = loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return asyncio.run_coroutine_threadsafe(
                self._generator.__anext__(),
                self._loop,
            ).result()
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


class _VisualGenEngine:
    def __init__(
        self,
        ipc_path: str,
        visual_gen_ckpt: str,
        vae_ckpt: str,
        device: Union[int, str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        visual_gen_quantization: Optional[str] = None,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=3584,
        num_heads=12,
        num_layers=30,
        qk_norm=True,
        cross_attn_norm=True,
        use_mrope=True,
        vl_to_visualgen_queue=None,
    ):
        # some hyperparameters for the visual generation engine
        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample_shift = 1.0
        self.num_inference_steps = 25
        self.num_train_timesteps = 1000
        self.r = (self.num_train_timesteps - 1) / (self.num_inference_steps - 1)
        self.cfg_text = 4.0
        self.cfg_interval = [0.4, 1]
        self.ar_step_accelerate_scale = 0.2
        self.interpolate_mode = "1D"

        # construct the models
        self.visual_gen = WanModel(
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            use_mrope=use_mrope,
        )
        self.visual_gen.eval()
        ckpt = self.visual_gen.get_checkpoint(visual_gen_ckpt)
        self.visual_gen.load_state_dict(ckpt)
        self.visual_gen = self.visual_gen.to(device)
        if visual_gen_quantization is not None:
            self.visual_gen.quantize(visual_gen_quantization)
        with except_quantized_fp():
            self.visual_gen = self.visual_gen.to(dtype)

        self.vae = VideoVAE(vae_pth=vae_ckpt, dtype=torch.bfloat16, device=device)

        # create the scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=self.sample_shift
        )
        self.scheduler_ar = FlowMatchEulerDiscreteSchedulerARDiT(
            num_train_timesteps=self.num_train_timesteps, shift=self.sample_shift
        )

        self.timesteps, self.num_inference_steps = self.retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device
        )
        self.timesteps_ar, self.num_inference_steps_ar = self.retrieve_timesteps(
            self.scheduler_ar, self.num_train_timesteps, self.device
        )

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # use queue instead of zmq
        self.vl_to_visualgen_queue = vl_to_visualgen_queue

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # Error state.
        self._errored_with: Optional[BaseException] = None

    @property
    def dead_error(self) -> BaseException:
        if self._errored_with is not None:
            return ENGINE_DEAD_ERROR(self._errored_with)
        else:
            return ENGINE_DEAD_ERROR()

    def start(self):
        try:
            try:
                logger.debug("Starting Engine Loop.")
                self.run_engine_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug("Shutting down MQOmniCode2WavEngine.")
        finally:
            logger.debug("MQOmniCode2WavEngine is shut down.")
            self.cleanup()

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.visual_gen
        del self.scheduler
        del self.scheduler_ar

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

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if VLLM_VLO_V2V_USE_ZMQ:
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    # When there's no work, check on engine health and send
                    # health status back to client
                    self._health_check()
                # Handle any input from the client.
                logger.info("visual gen engine receive requeset in run_engine_loop")
                self.handle_new_input(None)
            else:
                # # Poll until there is work to do.
                last_health_check_time = time.time()
                new_frame = None
                while new_frame is None:
                    with contextlib.suppress(queue.Empty):
                        new_frame = self.vl_to_visualgen_queue.get(0.1)
                    # When there's no work, check on engine health and send
                    # health status back to client
                    if time.time() - last_health_check_time > POLLING_TIMEOUT_MS / 1000:
                        logger.info(
                            "[code2wav engine][multiprocessing Queue] do health check"
                        )
                        last_health_check_time = time.time()

                # Handle any input from the client.
                self.handle_new_input(new_frame)

    def handle_new_input(
        self, new_frame
    ) -> Optional[Tuple[str, np.ndarray, bool, int]]:
        """Handle new input from the socket"""
        try:
            (
                request_id,
                hidden_states,
                position_ids,
                rope_deltas,
                num_samples,
                cfg_img,
                cfg_num,
                latent_size_h,
                latent_size_w,
                attention_mask,
                partial_image_num,
            ) = new_frame
            hidden_states = hidden_states.to(self.device)
            position_ids = position_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            logger.info(
                "process vl output receive codes from queue, rid: %s", request_id
            )
            try:
                for cur_image_num in range(partial_image_num):
                    finished = cur_image_num + 1 == partial_image_num
                    if cur_image_num == 0:
                        latent = None
                    request_output = self.step(
                        request_id,
                        hidden_states,
                        position_ids,
                        num_samples,
                        cfg_img,
                        cfg_num,
                        latent_size_h,
                        latent_size_w,
                        attention_mask,
                        partial_image_num,
                        cur_image_num,
                        rope_deltas,
                        latent,
                        finished,
                    )
                    if isinstance(request_output, Tuple):
                        request_id, latent, images = request_output
                        from vllm.outputs import CompletionOutput

                        finish_reason = "stop" if finished else None
                        completion_output = CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[],
                            cumulative_logprob=None,
                            logprobs=None,
                            image=images,
                            finish_reason=finish_reason,
                        )
                        output: RequestOutput = RequestOutput(
                            request_id=request_id,
                            prompt=None,
                            prompt_token_ids=[],
                            prompt_logprobs=None,
                            outputs=[completion_output],
                            finished=finished,
                        )
                        self._send_outputs([output])
                    else:
                        logger.warning("Invalid request output: %r", request_output)
                        break

            except SystemExit:
                raise
            except BaseException as e:
                self._set_errored(e)
                rpc_err = RPCError(
                    request_id=request_id, is_engine_errored=True, exception=e
                )
                self._send_outputs(rpc_err)
                raise e
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)
            raise e

    def step(
        self,
        request_id: str,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        num_samples: int,
        cfg_img: Optional[float],
        cfg_num: int,
        latent_size_h: int = None,
        latent_size_w: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        partial_image_num: int = 1,
        cur_image_num: int = 0,
        rope_deltas: torch.Tensor = None,
        latent: Optional[torch.Tensor] = None,
        finished: bool = False,
    ):
        if latent_size_h is None or latent_size_w is None:
            latent_size_h = 32  # set default gen_size to be [512, 512]
            latent_size_w = 32  # resulting in latent size in [512 // 16, 512 //16]
        if self.interpolate_mode == "1D":
            ar_inference_steps = (
                math.ceil(
                    (
                        int(
                            latent_size_h
                            * latent_size_w
                            * self.ar_step_accelerate_scale
                        )
                        + self.num_train_timesteps
                    )
                    / self.r
                )
                + 1
            )
        elif self.interpolate_mode == "2D":
            scale = self.ar_step_accelerate_scale
            ar_inference_steps = (
                math.ceil(
                    (
                        int(latent_size_h * scale) * int(latent_size_w * scale)
                        + self.num_train_timesteps
                    )
                    / self.r
                )
                + 1
            )
        else:
            raise NotImplementedError("Unsupported interpolation mode.")

        ar_inference_start_step = (
            ar_inference_steps // partial_image_num * cur_image_num
        )
        ar_inference_end_step = (
            ar_inference_steps // partial_image_num * (cur_image_num + 1)
        )
        if finished:
            ar_inference_end_step = (
                ar_inference_end_step + ar_inference_steps % partial_image_num
            )

        latent, images = self.forward(
            vl_hidden=hidden_states,
            position_ids=position_ids,
            num_samples=num_samples,
            cfg_img=cfg_img,
            cfg_num=cfg_num,
            ar_inference_start_step=ar_inference_start_step,
            ar_inference_end_step=ar_inference_end_step,
            latent_size_h=latent_size_h,
            latent_size_w=latent_size_w,
            attention_mask=attention_mask,
            latent=latent,
        )
        images = images.detach().cpu()
        return request_id, latent, images

    def _send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes,), copy=False)

    def _health_check(self):
        # Send unhealthy if engine has already errored
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)
        try:
            self._send_healthy()
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)

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
        if self._errored_with is None:
            self._errored_with = e

    @staticmethod
    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("Code2WavEngine terminated")

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
    ):
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def create_timestep_table(self, h, w, nt, down_scale=1.0):
        if self.interpolate_mode == "1D":
            down_hw = int(h * w * down_scale)
            indices = torch.arange(down_hw, device=self.device)
            curr_timestep = indices - (nt * self.r - self.num_train_timesteps)
            curr_timestep = torch.clamp(
                curr_timestep, min=0, max=self.num_train_timesteps
            )
            curr_timestep = F.interpolate(
                curr_timestep.view(1, 1, down_hw).float(), size=(h * w), mode="linear"
            )
            curr_timestep = curr_timestep.view(h * w).int()
        elif self.interpolate_mode == "2D":
            down_scale = down_scale**0.5
            down_h, down_w = int(h * down_scale), int(w * down_scale)
            indices = torch.arange(down_h * down_w, device=self.device)
            curr_timestep = indices - (nt * self.r - self.num_train_timesteps)
            curr_timestep = torch.clamp(
                curr_timestep, min=0, max=self.num_train_timesteps
            )
            curr_timestep = F.interpolate(
                curr_timestep.view(1, 1, down_h, down_w).float(),
                size=(h, w),
                mode="bilinear",
            )
            curr_timestep = curr_timestep.view(h * w).int()
        else:
            raise ValueError(f"Unsupported interpolate mode: {self.interpolate_mode}")
        return curr_timestep

    def forward_with_cfg(
        self,
        latent,
        t,
        time_table,
        context,
        latent_size_h,
        latent_size_w,
        seq_len,
        position_ids,
        cfg_num,
        cfg_text,
        cfg_img,
        attention_mask,
    ):
        half = latent.chunk(cfg_num, dim=0)[0]
        combined = half.repeat(cfg_num, *tuple([1] * (half.ndim - 1)))
        # TODO dtype modify
        noise_pred = self.visual_gen(
            x=combined,
            t=t.unsqueeze(0),
            timestep_table=time_table,
            context=context,
            seq_len=seq_len,
            position_ids=position_ids,
            k_attn_mask=attention_mask,
        )

        noise_pred = torch.stack(noise_pred)

        timestep_ratio = (
            time_table.reshape(-1, latent_size_h, latent_size_w)
            / self.num_train_timesteps
        )
        cfg_mask = (timestep_ratio > self.cfg_interval[0]) & (
            timestep_ratio < self.cfg_interval[1]
        )
        cfg_mask = F.interpolate(
            cfg_mask.unsqueeze(1).to(noise_pred.dtype), scale_factor=2, mode="nearest"
        ).to(noise_pred.dtype)
        cfg_mask = cfg_mask.unsqueeze(2)
        cfg_text = cfg_text * cfg_mask
        cfg_text[cfg_mask == 0] = 1
        if cfg_img is not None:
            cfg_img = cfg_img * cfg_mask
            cfg_img[cfg_mask == 0] = 1

        if cfg_img is not None:  # three latent
            cond_eps, null_text_eps, null_all_eps = noise_pred.chunk(3, dim=0)
            half_eps = (
                null_all_eps
                + cfg_text * (cond_eps - null_text_eps)
                + cfg_img * (null_text_eps - null_all_eps)
            )
            eps = torch.cat([half_eps, half_eps, half_eps], dim=0)
        else:  # two latent
            cond_eps, uncond_eps = noise_pred.chunk(2, dim=0)
            half_eps = uncond_eps + cfg_text * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
        return eps

    @torch.inference_mode()
    def forward(
        self,
        vl_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        num_samples: int,
        cfg_img: Optional[float],
        cfg_num: int,
        ar_inference_start_step: int,
        ar_inference_end_step: int,
        latent_size_h: int,
        latent_size_w: int,
        attention_mask: Optional[torch.Tensor] = None,
        disable_tqdm: bool = False,
        latent: Optional[torch.Tensor] = None,
    ):
        if latent is None:
            latent = torch.randn(
                num_samples,
                16,
                1,
                latent_size_h * 2,
                latent_size_w * 2,
                device=self.device,
                dtype=self.dtype,
            )
            latent = latent.repeat(cfg_num, 1, 1, 1, 1)
        pre_time_table = torch.full(
            (latent_size_h * latent_size_w,),
            self.num_train_timesteps,
            dtype=torch.float32,
            device=latent.device,
        )
        for nt in tqdm(
            range(ar_inference_start_step, ar_inference_end_step), disable=disable_tqdm
        ):
            latent_model_input = latent
            t = self.timesteps[min(nt, self.num_inference_steps - 1)]
            # t is in timesteps
            time_table = self.create_timestep_table(
                latent_size_h,
                latent_size_w,
                nt,
                down_scale=self.ar_step_accelerate_scale,
            )
            noise_pred = self.forward_with_cfg(
                latent=latent_model_input,
                t=t,
                time_table=time_table,
                context=vl_hidden,
                latent_size_h=latent_size_h,
                latent_size_w=latent_size_w,
                seq_len=time_table.shape[0],
                position_ids=position_ids,
                cfg_num=cfg_num,
                cfg_text=self.cfg_text,
                cfg_img=cfg_img,
                attention_mask=attention_mask,
            )
            latent, dt, x0 = self.scheduler_ar.step(
                noise_pred.squeeze(2),
                self.timesteps_ar[0],
                latent.squeeze(2),
                per_timesteps=pre_time_table.repeat(num_samples * cfg_num),
                next_timesteps=time_table.repeat(num_samples * cfg_num),
                num_inference_steps_ar_dit=self.num_inference_steps_ar,
                batch_size=len(latent),
                patch_size=2,
            )
            latent = latent.prev_sample.unsqueeze(2)
            pre_time_table = time_table
        sample = torch.stack(self.vae.decode(latent[:num_samples])).squeeze(2)
        return latent, sample

    @staticmethod
    def run_visual_gen_engine(
        ipc_path: str,
        visual_gen_ckpt: str,
        vae_ckpt: str,
        device: Union[int, str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        visual_gen_quantization: Optional[str] = None,
        vl_to_visualgen_queue=None,
        engine_alive: Any = None,
    ):
        try:
            engine = _VisualGenEngine(
                ipc_path,
                visual_gen_ckpt,
                vae_ckpt,
                device,
                dtype=dtype,
                visual_gen_quantization=visual_gen_quantization,
                vl_to_visualgen_queue=vl_to_visualgen_queue,
            )
            signal.signal(signal.SIGTERM, engine.signal_handler)
            engine.start()
        except BaseException as e:
            logger.exception("VisualGenEngine failed with exception: %s", e)
            engine_alive.value = False
            raise e


class VloLLMEngine(MQLLMEngine):
    def __init__(
        self,
        ipc_path: str,
        thinker_to_talker_queue,
        use_async_sockets: bool,
        *args,
        log_requests: bool = True,
        **kwargs,
    ) -> None:
        visual_gen_dtype = kwargs.get("visual_gen_dtype", "bfloat16")
        visual_gen_data_parallelism = kwargs.get("visual_gen_data_parallelism", 1)
        visual_gen_quantization = kwargs.get("visual_gen_quantization")
        vl_visible_devices = kwargs.get("vl_visible_devices")
        visual_gen_visible_devices = kwargs.get("visual_gen_visible_devices")
        model_config = kwargs["vllm_config"].model_config

        if vl_visible_devices and not isinstance(vl_visible_devices, list):
            vl_visible_devices = [vl_visible_devices]
        if visual_gen_visible_devices and not isinstance(
            visual_gen_visible_devices, list
        ):
            visual_gen_visible_devices = [visual_gen_visible_devices]

        # multiprocessing manager for qwen-vl to visual_gen.
        self.mp_manager = multiprocessing.Manager()

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        self.context = multiprocessing.get_context("spawn")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._async_loop_thread = Thread(target=self._start_async_loop, daemon=True)
        self._async_loop_thread.start()

        model_path = model_config.model
        self.vl_config = get_config(
            model_path,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.vl_generation_config = try_get_generation_config(
            model_path,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=False)

        if vl_visible_devices:
            device_ctx = envvars(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(device) for device in vl_visible_devices),
            )
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            self.vl_engine = AsyncLLMEngine(*args, **kwargs)
            # cause vlo need to return hidden_states, we hardcode here
            # TODO(xiale.zy)maybe need a more proper way to handle this
            self.vl_engine.engine.model_executor.driver_worker.worker.model_runner.return_full_hidden_states = True
            self.engine = self.vl_engine.engine

        self.vl_listener: dict[str, Thread] = dict()

        tokenizer_path = kwargs["vllm_config"].model_config.tokenizer
        self.vl_engine_tokenizer = get_tokenizer(
            tokenizer_path,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")
        # Send heartbeats back to client.
        self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
        self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")
        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"
        # Error state.
        self._errored_with: Optional[BaseException] = None
        self.ipc_path = ipc_path

        # async mp engine(s)
        map_to_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.visual_gen_dtype = map_to_dtype[visual_gen_dtype]
        self.visual_gen_data_parallelism = visual_gen_data_parallelism
        self._visual_gen_pids = []
        self._visual_gen_processes = []
        visual_gen_ckpt = os.path.join(model_path, "saved_ckpt_wan.pt")
        vae_ckpt = os.path.join(model_path, "wanx2_1_t2v_vae.pth")
        for i in range(self.visual_gen_data_parallelism):
            (
                pid,
                process,
            ) = self._start_visual_gen_engine(
                self.ipc_path,
                visual_gen_ckpt,
                vae_ckpt,
                visible_devices=(
                    [visual_gen_visible_devices[i % len(visual_gen_visible_devices)]]
                    if visual_gen_visible_devices
                    else None
                ),
                dtype=self.visual_gen_dtype,
                visual_gen_quantization=visual_gen_quantization,
            )
            self._visual_gen_pids.append(pid)
            self._visual_gen_processes.append(process)

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_vl_engine(
        self, *args, visible_devices: Optional[list[int]] = None, **kwargs
    ) -> AsyncLLMEngine:
        if visible_devices:
            device_ctx = envvars(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(device) for device in visible_devices),
            )
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            engine = AsyncLLMEngine(*args, **kwargs)
        # cause vlo need to return hidden_states, we hardcode here
        # TODO(xiale.zy)maybe need a more proper way to handle this
        engine.engine.model_executor.driver_worker.worker.model_runner.return_full_hidden_states = True
        return engine

    def _start_visual_gen_engine(
        self,
        ipc_path: str,
        visual_gen_ckpt: str,
        vae_ckpt: str,
        visible_devices: Optional[list[int]] = None,
        device: Union[int, str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        visual_gen_quantization: Optional[str] = None,
    ):
        logger.info(
            "Multiprocessing frontend to use %s for IPC Path for model %s, %s",
            ipc_path,
            visual_gen_ckpt,
            vae_ckpt,
        )

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value("b", True, lock=False)
        if visible_devices:
            device_ctx = envvars(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(device) for device in visible_devices),
            )
        else:
            device_ctx = contextlib.nullcontext()

        self.vl_to_visualgen_queue = self.mp_manager.Queue()
        logger.info("Start visual gen engine, using vl_to_visualgen_queue to send")

        with device_ctx:
            engine_process = self.context.Process(
                target=_VisualGenEngine.run_visual_gen_engine,
                args=(
                    ipc_path,
                    visual_gen_ckpt,
                    vae_ckpt,
                    device,
                    dtype,
                    visual_gen_quantization,
                    self.vl_to_visualgen_queue,
                    engine_alive,
                ),
            )
            engine_process.start()

        self.engine_process = engine_process
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        return engine_pid, engine_process

    def _shutdown_mp_engine(
        self,
        engine_pid: int,
        engine_process: multiprocessing.Process,
        mq_engine_client: MQLLMEngineClient,
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
        # shutdown qwen-vl engine
        self.vl_engine.shutdown_background_loop()
        # shutdown visual_gen engines
        for pid, process, engine_client in zip(
            self._visual_gen_pids,
            self._visual_gen_processes,
        ):
            self._shutdown_mp_engine(pid, process, engine_client)

    def print_gpu_memory(self, tag: str, request_id: str, device=0):
        # 获取当前设备的显存信息
        allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MiB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # MiB
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MiB
        used_percent = (allocated / total) * 100  # 已用百分比

        logger.info(
            "%s %s GPU usage in MiB allocated %s max_allocated %s total %s used_percent %s",
            tag,
            request_id,
            allocated,
            max_allocated,
            total,
            used_percent,
        )

    def _listen_vl(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        latent_size_h: int,
        latent_size_w: int,
        attention_mask: torch.Tensor,
        num_samples: int,
        cfg_img: Optional[float],
        cfg_num: int,
        partial_image_num: int,
        generator_list: list[asyncio.Future[RequestOutput]],
    ):
        def _listen():
            output = None
            total_hidden_states = []
            for generator in generator_list:
                for output in SynchronizedGenerator(generator, self._loop):
                    if output.finished:
                        hidden_states = output.outputs[0].hidden_state
                        total_hidden_states.append(hidden_states)
                        break
            hidden_states = torch.stack(total_hidden_states, dim=0)
            position_ids, rope_deltas = get_vlo_rope_index(
                input_ids, image_grid_thw, None, None, attention_mask
            )
            self.vl_to_visualgen_queue.put(
                (
                    request_id,
                    hidden_states.cpu(),
                    position_ids,
                    rope_deltas,
                    num_samples,
                    cfg_img,
                    cfg_num,
                    latent_size_h,
                    latent_size_w,
                    attention_mask,
                    partial_image_num,
                )
            )
            with contextlib.suppress(Exception):
                del self.vl_listener[request_id]

        _listen()

    def remove_consecutive_image_token(
        self, token_ids: torch.Tensor, image_token_id: int
    ) -> list[int]:
        result_token_ids = []
        last = -1
        for token_id in token_ids:
            if token_id != image_token_id:
                result_token_ids.append(int(token_id))
            else:
                if last == image_token_id:
                    continue
                else:
                    result_token_ids.append(int(token_id))
            last = token_id
        return result_token_ids

    def add_request(self, request) -> Optional[Tuple[str, np.ndarray, bool, int]]:
        """Handle new input from the socket"""
        if request.params is None:
            request.params = SamplingParams(
                stop_token_ids=self.vl_generation_config.eos_token_id, max_tokens=1
            )
        else:
            # enforce max_tokens to 1 for vl prefill
            request.params.max_tokens = 1
        # change token_ids to text
        text = []
        for token_ids in request.prompt.get("prompt_token_ids", []):
            text.append(self.vl_engine_tokenizer.decode(token_ids))

        # what info needed by visual gen engine
        if not hasattr(request, "visual_gen_params"):
            inputs = self.processor(
                text=text,
                images=request.prompt["multi_modal_data"].get("image", None),
                videos=request.prompt["multi_modal_data"].get("video", None),
                padding=True,
                return_tensors="pt",
            )
            if request.params.output_image_size is not None:
                gen_size = request.params.output_image_size
            elif "image_grid_thw" in inputs:
                gen_size = inputs.image_grid_thw[0][1:] * 14
            else:
                gen_size = [512, 512]
            gen_h, gen_w = gen_size
            latent_size_h, latent_size_w = int(gen_h // 16), int(gen_w // 16)
            if request.prompt["multi_modal_data"].get("image", None) is not None:
                num_samples = len(request.prompt["multi_modal_data"]["image"]) // 2
            else:
                num_samples = len(text) // 2
            partial_image_num = request.params.partial_images
            visual_gen_params = Visual_Gen_Params(
                prompt_token_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"]
                if request.prompt["multi_modal_data"].get("image", None) is not None
                else None,
                latent_size_h=latent_size_h,
                latent_size_w=latent_size_w,
                attention_mask=inputs["attention_mask"],
                num_samples=num_samples,
                cfg_img=2.0
                if request.prompt["multi_modal_data"].get("image", None) is not None
                else None,
                cfg_num=len(text) // num_samples,
                partial_image_num=partial_image_num,
            )

        self.print_gpu_memory("add_request", request.request_id, 0)
        generator_list = []
        multi_modal_data = request.prompt.get("multi_modal_data", {})
        for idx in range(len(text)):
            temp_request_id = str(uuid.uuid4())
            temp_prompt = copy.deepcopy(request.prompt)
            temp_prompt["prompt_token_ids"] = self.remove_consecutive_image_token(
                inputs["input_ids"][idx], self.vl_config.image_token_id
            )

            if idx < len(multi_modal_data.get("image", [])):
                temp_prompt["multi_modal_data"]["image"] = [
                    multi_modal_data["image"][idx]
                ]
            else:
                temp_prompt["multi_modal_data"]["image"] = []

            generator = asyncio.run_coroutine_threadsafe(
                self.vl_engine.add_request(
                    request_id=temp_request_id,
                    prompt=temp_prompt,
                    params=request.params,
                    lora_request=request.lora_request,
                    trace_headers=request.trace_headers,
                    prompt_adapter_request=request.prompt_adapter_request,
                    priority=request.priority,
                ),
                self._loop,
            ).result()
            generator_list.append(generator)
        return (request.request_id, generator_list, visual_gen_params)

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if not self.vl_engine.engine.has_unfinished_requests():
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    # When there's no work, check on engine health and send
                    # health status back to client
                    self._health_check()
                    self.vl_engine.engine.do_log_stats()
                    logger.debug("Waiting for new requests in engine loop.")

            # Handle any input from the client.
            try:
                while self.input_socket.poll(timeout=0) != 0:
                    frames = self.input_socket.recv_multipart(copy=False)
                    request = pickle.loads(frames[0].buffer)
            except Exception as e:
                self._set_errored(e)
                self._send_unhealthy(e)
                raise e from None

            preprocessed_input = self.add_request(request)
            (request_id, generator_list, visual_gen_params) = preprocessed_input

            self._listen_vl(
                request_id,
                visual_gen_params.prompt_token_ids,
                visual_gen_params.image_grid_thw,
                visual_gen_params.latent_size_h,
                visual_gen_params.latent_size_w,
                visual_gen_params.attention_mask,
                visual_gen_params.num_samples,
                visual_gen_params.cfg_img,
                visual_gen_params.cfg_num,
                visual_gen_params.partial_image_num,
                generator_list,
            )
