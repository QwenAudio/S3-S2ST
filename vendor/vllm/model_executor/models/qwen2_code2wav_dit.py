from __future__ import annotations
import contextlib
from typing import Callable, Dict, List, Tuple, Union
import copy

import torch

from torchdiffeq import odeint

from vllm.transformers_utils.qwen2_code2wav_dit.modeling_qwen2_code2wav import (
    Qwen2Code2wavBigvgan,
)
from vllm.transformers_utils.qwen2_code2wav_dit.model.t2w_cfm import CodecCFM
from vllm.transformers_utils.qwen2_code2wav_dit.model.dit import DiT
from vllm.transformers_utils.qwen2_code2wav_dit.model.utils import (
    load_checkpoint,
    exists,
    except_quantized_fp,
)


def reshape_by_blocks(matrix, a, b=1):
    m, n = matrix.shape
    assert m % a == 0 and n % b == 0, "矩阵维度必须能被块大小整除"

    # 将矩阵重塑为(m//a, a, n//b, b)形状
    reshaped = matrix.reshape(m // a, a, n // b, b)

    # 交换中间两个维度，从(m//a, a, n//b, b)变为(m//a, n//b, a, b)
    permuted = reshaped.permute(0, 2, 1, 3)

    # 重塑为最终形状(-1, a*b)
    final = permuted.reshape(-1, a * b)

    return final


def inverse_reshape_by_blocks(blocked_matrix, m, n, a, b=1):
    assert m % a == 0 and n % b == 0, "矩阵维度必须能被块大小整除"

    # 从(-1, a*b)重塑为(m//a, n//b, a, b)
    reshaped = blocked_matrix.reshape(m // a, n // b, a, b)

    # 交换中间两个维度，从(m//a, n//b, a, b)变为(m//a, a, n//b, b)
    permuted = reshaped.permute(0, 2, 1, 3)

    # 重塑为原始形状(m, n)
    final = permuted.reshape(m, n)

    return final


class CudaGraphRunner:
    def __init__(self, fn, device):
        """
        initialize CUDA Graph Wrapper.

        Args:
            original_fast_forward (callable): original fast_forward method.
            device (torch.device): CUDA device.
        """
        torch._dynamo.config.cache_size_limit = 64
        self.fn_compile = torch.compile(
            fn,
            mode="default",
            fullgraph=False,
        )
        self.cuda_graph: dict[tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.input_buffers: dict[tuple[int, int], dict[str, torch.Tensor]] = dict()
        self.output_buffers: dict[tuple[int, int], torch.Tensor] = dict()

        self.device = device

        # Create customized CUDA stream for Cuda Graph capture
        self.capture_stream = torch.cuda.Stream(device=self.device)

    def capture_cuda_graph(self, x, cond, spk, text, time, mask):
        """
        Capture CUDA Graph。

        Args:
            x (torch.Tensor): nosied input audio.
            cond (torch.Tensor): masked cond audio.
            spk (torch.Tensor): spk embedding.
            text (torch.Tensor): text.
            time (torch.Tensor): time step.
            mask (torch.Tensor or None): mask.
        """
        # Move the input data to the specified device and detach it from the computation graph.

        size = (text.size(0), text.size(1))

        with torch.no_grad():
            if size not in self.input_buffers:
                self.input_buffers[size] = {
                    "x": x.to(self.device, non_blocking=True).clone().detach(),
                    "cond": cond.to(self.device, non_blocking=True).clone().detach(),
                    "spk": spk.to(self.device, non_blocking=True).clone().detach(),
                    "text": text.to(self.device, non_blocking=True).clone().detach(),
                    "time": time.to(self.device, non_blocking=True).clone().detach(),
                    "mask": mask.to(self.device, non_blocking=True).clone().detach()
                    if mask is not None
                    else None,
                }

        # Determine the output shape through a single forward pass and pre-allocate the output buffer.
        with torch.no_grad():
            if size not in self.output_buffers:
                generated = self.fn_compile(
                    x=self.input_buffers[size]["x"],
                    cond=self.input_buffers[size]["cond"],
                    spk=self.input_buffers[size]["spk"],
                    text=self.input_buffers[size]["text"],
                    time=self.input_buffers[size]["time"],
                    mask=self.input_buffers[size]["mask"],
                )
                self.output_buffers[size] = torch.empty_like(
                    generated, device=self.device
                )

        # Ensure that all previous operations are complete.
        torch.cuda.synchronize(self.device)

        # Begin to capture CUDA Graph
        self.cuda_graph[size] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph[size], stream=self.capture_stream):
            # Perform the forward pass and copy the results to the pre-allocated buffer.
            generated = self.fn_compile(
                x=self.input_buffers[size]["x"],
                cond=self.input_buffers[size]["cond"],
                spk=self.input_buffers[size]["spk"],
                text=self.input_buffers[size]["text"],
                time=self.input_buffers[size]["time"],
                mask=self.input_buffers[size]["mask"],
            )
            self.output_buffers[size].copy_(generated)

        # Make sure capture complete
        torch.cuda.synchronize(self.device)

    def __call__(self, x, cond, spk, text, time, mask=None):
        """
        Args:
            x (torch.Tensor): nosied input audio.
            cond (torch.Tensor): masked cond audio.
            spk (torch.Tensor): spk embedding.
            text (torch.Tensor): text.
            time (torch.Tensor): time step.
            mask (torch.Tensor or None): mask.

        Returns:
            torch.Tensor: generated
        """

        size = (text.size(0), text.size(1))

        if size not in self.cuda_graph:
            # Capture the CUDA Graph on the first call.
            self.capture_cuda_graph(x, cond, spk, text, time, mask)

        # Update input to buffer.
        self.input_buffers[size]["x"].copy_(x.to(self.device, non_blocking=True))
        self.input_buffers[size]["cond"].copy_(cond.to(self.device, non_blocking=True))
        self.input_buffers[size]["spk"].copy_(spk.to(self.device, non_blocking=True))
        self.input_buffers[size]["text"].copy_(text.to(self.device, non_blocking=True))
        self.input_buffers[size]["time"].copy_(time.to(self.device, non_blocking=True))
        if self.input_buffers[size]["mask"] is not None and mask is not None:
            self.input_buffers[size]["mask"].copy_(
                mask.to(self.device, non_blocking=True)
            )
        elif mask is None:
            self.input_buffers[size]["mask"] = None
        # Replay CUDA Graph
        self.cuda_graph[size].replay()
        return self.output_buffers[size]


class BigvGanCudaGraphRunner:
    def __init__(self, fn, device):
        """
        initialize CUDA Graph Wrapper.

        Args:
            original_fast_forward (callable): original fast_forward method.
            device (torch.device): CUDA device.
        """
        torch._dynamo.config.cache_size_limit = 64
        self.fn_compile = torch.compile(
            fn,
            mode="default",
            fullgraph=False,
        )

        self.cuda_graph: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.input_buffers: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = dict()
        self.output_buffers: Dict[Tuple[int, int], torch.Tensor] = dict()

        self.device = device

        # Create customized CUDA stream for Cuda Graph capture
        self.capture_stream = torch.cuda.Stream(device=self.device)

    def capture_cuda_graph(self, mel):
        """
        Capture CUDA Graph。

        Args:
            mel (torch.Tensor): nosied input audio.
        """
        # Move the input data to the specified device and detach it from the computation graph.

        size = (mel.size(0), mel.size(1), mel.size(2))

        with torch.no_grad():
            if size not in self.input_buffers:
                self.input_buffers[size] = {
                    "mel": mel.to(self.device, non_blocking=True).clone().detach(),
                }

        # Determine the output shape through a single forward pass and pre-allocate the output buffer.
        with torch.no_grad():
            if size not in self.output_buffers:
                generated = self.fn_compile(
                    mel=self.input_buffers[size]["mel"],
                )
                self.output_buffers[size] = torch.empty_like(
                    generated, device=self.device
                )

        # Ensure that all previous operations are complete.
        torch.cuda.synchronize(self.device)

        # Begin to capture CUDA Graph
        self.cuda_graph[size] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph[size], stream=self.capture_stream):
            # Perform the forward pass and copy the results to the pre-allocated buffer.
            generated = self.fn_compile(
                mel=self.input_buffers[size]["mel"],
            )
            self.output_buffers[size].copy_(generated)

        # Make sure capture complete
        torch.cuda.synchronize(self.device)

    def __call__(self, mel):
        """
        Args:
            x (torch.Tensor): nosied input audio.

        Returns:
            torch.Tensor: generated
        """

        size = (mel.size(0), mel.size(1), mel.size(2))

        if size not in self.cuda_graph:
            # Capture the CUDA Graph on the first call.
            self.capture_cuda_graph(mel)

        # Update input to buffer.
        self.input_buffers[size]["mel"].copy_(mel.to(self.device, non_blocking=True))
        # Replay CUDA Graph
        self.cuda_graph[size].replay()
        return copy.deepcopy(self.output_buffers[size])


class BatchCodecCFM(CodecCFM):
    @torch.no_grad()
    def fast_block_sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        codec: int["b nc dc"],
        ref_mel: float["b n d"],  # noqa: F722
        y0: float["b n d"],
        lens: int[b] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        max_duration = y0.shape[1]
        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()
            ref_mel = ref_mel.half()
            y0 = y0.half()
            # print(next(self.parameters()).dtype)
        elif next(self.parameters()).dtype == torch.bfloat16:
            cond = cond.bfloat16()
            ref_mel = ref_mel.bfloat16()
            y0 = y0.bfloat16()
            # print(next(self.parameters()).dtype)

        # raw wave

        cond = cond.unsqueeze(1).repeat(1, max_duration, 1)
        batch, cond_seq_len, device = *ref_mel.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            out_put = self.transformer.fast_forward(
                x=x,
                text=codec,
                spk=cond,
                cond=ref_mel,
                time=t,
                mask=mask,
            )
            pred, null_pred = torch.chunk(out_put, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=ref_mel.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, ref_mel, out)
        return out, trajectory


class Qwen2Code2wavDit(torch.nn.Module):
    def __init__(
        self, ckpt, frequency: str = "50hz", dit_quant: str = None, device="cpu"
    ):
        super().__init__()
        self.frequency = frequency
        self.device = device
        self.dit = DiT(
            dim=1024,
            depth=22 if frequency == "50hz" else 32,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            use_codec=True,
            repeats=2 if frequency == "50hz" else 4,
            attn_processor="stream_block_sr"
            if frequency == "50hz"
            else "stream_block_8_L_4",
            text_num_embeds=8193 if frequency == "50hz" else 32769,
            mel_dim=80,
        )
        self.mel_spec_kwargs = dict(
            target_sample_rate=16000,
            n_mel_channels=80,
            hop_length=160,
        )
        self.odeint_kwargs = dict(
            method="euler",
        )
        self.cfm_model = BatchCodecCFM(
            transformer=self.dit,
            mel_spec_kwargs=self.mel_spec_kwargs,
            odeint_kwargs=self.odeint_kwargs,
        ).to(device)
        self.cfm_model = load_checkpoint(self.cfm_model, ckpt, device, use_ema=True)

        # do online quantize
        if dit_quant is not None:
            assert dit_quant == "fp8", (
                f"only fp8 is supported, but got {dit_quant} instead"
            )
            self.online_quantize()

    def sample(
        self, cond, ref_mel, codec, steps=10, cfg_strength=0.5, sway_sampling_coef=-1.0
    ):
        y_all = torch.randn([1, 30000, 80], device=self.device, dtype=ref_mel.dtype)
        expect_y_len = codec.shape[1] * (2 if self.frequency == "50hz" else 4)
        y0 = y_all[:, :expect_y_len]
        with torch.inference_mode():
            generated, _ = self.cfm_model.sample(
                cond=cond,
                ref_mel=ref_mel,
                codec=codec,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                y0=y0,
            )
        generated = generated.to(torch.float32)
        generated_mel_spec = generated.permute(0, 2, 1)
        return generated_mel_spec

    def online_quantize(self):
        for name, params in self.dit.named_parameters():
            if "transformer_blocks" in name and "weight" in name:
                new_data, new_data_scale = self.fp8_quantize(params.data)
                params.data = new_data
                module = self.locate_parent(self.dit, name)
                setattr(module, "weight_scale_inv", new_data_scale)

    def locate_parent(self, root: torch.nn.Module, full_path: str):
        parent = root
        path = full_path.split(".")
        for p in path[:-1]:
            parent = getattr(parent, p)
        return parent

    def fp8_quantize(self, v):
        # pre group
        qmax = torch.finfo(torch.float8_e4m3fn).max
        vtype = v.dtype
        shape = v.shape

        # per group
        reshaped_v = reshape_by_blocks(v, 128, 128)
        scales = reshaped_v.abs().amax(dim=-1, keepdim=True) / qmax
        reshaped_v = reshaped_v / scales
        scales = inverse_reshape_by_blocks(
            scales, shape[0] // 128, shape[1] // 128, 1, 1
        )
        reshaped_v = inverse_reshape_by_blocks(reshaped_v, shape[0], shape[1], 128, 128)

        reshaped_v = reshaped_v.to(torch.float8_e4m3fn)
        scales = scales.to(torch.float32)

        return reshaped_v, scales

    def fast_block_sample(
        self,
        cond,
        codec,
        ref_mel,
        y0,
        steps=10,
        cfg_strength=0.5,
        sway_sampling_coef=-1.0,
    ):
        return self.cfm_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )


class Qwen2Code2wav(torch.nn.Module):
    def __init__(
        self,
        dit_ckpt,
        bigvgan_ckpt,
        steps: int = 10,
        bs_mel: int = 24,
        odeint_method: str = "euler",
        odeint_method_relaxed: bool = False,
        batched_chunk: int = 3,
        frequency: str = "50hz",
        dit_quant: str = None,
        device="cpu",
        with_weight_norm: bool = True,
    ):
        super().__init__()
        self.frequency = frequency
        self.code2wav_dit_model = Qwen2Code2wavDit(
            ckpt=dit_ckpt, frequency=frequency, dit_quant=dit_quant, device=device
        )
        # dtype the model to bfloat16
        with except_quantized_fp():
            self.code2wav_dit_model.to(dtype=torch.bfloat16)  # type: ignore [arg-type]

        self.code2wav_bigvgan_model = Qwen2Code2wavBigvgan(
            ckpt=bigvgan_ckpt,
            frequency=frequency,
            device=device,
            with_weight_norm=with_weight_norm,
        )
        self.device = device

        # odeint method: use ruler for first and last chunk to optimize performance
        self.odeint_method_relaxed = odeint_method_relaxed

        # cfm model: override the odeint method
        self.code2wav_dit_model.cfm_model.odeint_kwargs["method"] = odeint_method

        self.dit_forward = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_compiled = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_compiled_first_chunk = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_cudagraph_first = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_cudagraph_intermediate = (
            self.code2wav_dit_model.dit.fast_forward
        )
        self.dit_forward_cudagraph_last = self.code2wav_dit_model.dit.fast_forward

        self.bigvgan_forward = self.code2wav_bigvgan_model.forward
        self.bigvgan_forward_compiled = self.code2wav_bigvgan_model.forward
        self.bigvgan_cuda_graph_first = self.code2wav_bigvgan_model.forward
        self.bigvgan_cuda_graph_intermediate = self.code2wav_bigvgan_model.forward
        self.bigvgan_cuda_graph_last = self.code2wav_bigvgan_model.forward

        self.torch_compile_first_chunk = False

        self.factor = 2 if frequency == "50hz" else 4
        self.steps = steps
        self.bs_mel = bs_mel
        self.bs_codec = bs_mel // self.factor
        self.past_cache_size = bs_mel * self.factor
        self.future_cache_size = bs_mel * 1
        self.chunk_size = bs_mel * batched_chunk
        self.future_size = 20 if self.frequency == "50hz" else 13
        self.past_size = 20 if self.frequency == "50hz" else 51

        text_embed = self.code2wav_dit_model.dit.text_embed
        if hasattr(text_embed, "codec_embed"):
            self.codec_embed_size = text_embed.codec_embed.weight.size(0)
        elif hasattr(text_embed, "text_embed"):
            self.codec_embed_size = text_embed.text_embed.weight.size(0)
        else:
            self.codec_embed_size = -1

    @contextlib.contextmanager
    def relax_odeint_method(self, relax: bool = False):
        if relax and self.odeint_method_relaxed:
            odeint_method = self.code2wav_dit_model.cfm_model.odeint_kwargs["method"]
            self.code2wav_dit_model.cfm_model.odeint_kwargs["method"] = "euler"
        yield
        if relax and self.odeint_method_relaxed:
            self.code2wav_dit_model.cfm_model.odeint_kwargs["method"] = odeint_method

    def enable_torch_compile(self, compile_first_chunk: bool = False):
        self.torch_compile_first_chunk = compile_first_chunk

        self.dit_forward_compiled = torch.compile(
            self.code2wav_dit_model.dit.fast_forward,
            # mode="default",
            mode="reduce-overhead",
            fullgraph=False,
        )
        self.dit_forward_cudagraph_first = CudaGraphRunner(
            self.code2wav_dit_model.dit.fast_forward, self.device
        )
        self.dit_forward_cudagraph_intermediate = CudaGraphRunner(
            self.code2wav_dit_model.dit.fast_forward, self.device
        )
        self.dit_forward_cudagraph_last = CudaGraphRunner(
            self.code2wav_dit_model.dit.fast_forward, self.device
        )

        self.bigvgan_cuda_graph_first = BigvGanCudaGraphRunner(
            self.code2wav_bigvgan_model.forward, self.device
        )
        self.bigvgan_cuda_graph_intermediate = BigvGanCudaGraphRunner(
            self.code2wav_bigvgan_model.forward, self.device
        )
        self.bigvgan_cuda_graph_last = BigvGanCudaGraphRunner(
            self.code2wav_bigvgan_model.forward, self.device
        )

    @torch.inference_mode()
    def forward(self, cond, ref_mel, codec):
        generated_mel = self.code2wav_dit_model.sample(cond, ref_mel, codec)
        generated_mel = generated_mel.permute(0, 2, 1)
        waveform = self.code2wav_bigvgan_model(generated_mel)
        return waveform

    @torch.inference_mode()
    def process_chunk_dit_batch(
        self,
        cond,
        ref_mel,
        codec,
        y0,
        steps,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.codec_embed_size > 0:
            codec[codec >= self.codec_embed_size] = 0

        self.code2wav_dit_model.dit.fast_forward = (
            self.dit_forward_cudagraph_intermediate
        )
        generated, _ = self.code2wav_dit_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=0.5,
            sway_sampling_coef=-1.0,
        )
        return generated

    @torch.inference_mode()
    def process_chunk_bigvgan_batch(self, mel_batch):
        return self.code2wav_bigvgan_model(mel_batch)

    @torch.inference_mode()
    def process_little_chunk(
        self,
        cond,
        ref_mel,
        codec_all,
        y_all,
        i,
        steps,
        prev_generated: torch.Tensor,
        finished: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # mask to prevent codec from being out of range (the eos token)
        if self.codec_embed_size > 0:
            codec_all[codec_all >= self.codec_embed_size] = 0

        return None, self.forward(cond, ref_mel, codec_all)

    @torch.inference_mode()
    def process_chunk(
        self,
        cond,
        ref_mel,
        codec_all,
        y_all,
        i,
        steps,
        prev_generated: Union[torch.Tensor, list[torch.Tensor]],
        finished: bool = False,
    ) -> tuple[Union[torch.Tensor, list[torch.Tensor]], torch.Tensor]:
        start_index = max(i * self.chunk_size - self.past_cache_size, 0)
        end_index = min(
            (i + 1) * self.chunk_size + self.future_cache_size,
            codec_all.shape[1] * self.factor,
        )
        y0 = y_all[:, start_index:end_index].reshape(1, -1, 80).contiguous()
        codec = (
            codec_all[:, start_index // self.factor : end_index // self.factor]
            .reshape(1, -1)
            .contiguous()
        )

        # mask to prevent codec from being out of range (the eos token)
        if self.codec_embed_size > 0:
            codec[codec >= self.codec_embed_size] = 0

        # N.B. when using cuda graph ("reduce-overhead" mode), don't compile
        # shape for the first and the last chunk, as it will affect the performance
        # for normal chunks.
        #
        # The reason is not clear yet. The default torch.compile() mode is not affected.
        if i == 0:
            if self.torch_compile_first_chunk:
                self.code2wav_dit_model.dit.fast_forward = self.dit_forward_compiled
            else:
                self.code2wav_dit_model.dit.fast_forward = (
                    self.dit_forward_cudagraph_first
                )
        elif finished:
            self.code2wav_dit_model.dit.fast_forward = self.dit_forward_cudagraph_last
        else:
            self.code2wav_dit_model.dit.fast_forward = (
                self.dit_forward_cudagraph_intermediate
            )

        with self.relax_odeint_method(relax=i == 0 or finished):
            generated, _ = self.code2wav_dit_model.fast_block_sample(
                cond=cond,
                codec=codec,
                ref_mel=ref_mel,
                y0=y0,
                steps=steps,
                cfg_strength=0.5,
                sway_sampling_coef=-1.0,
            )

        if self.frequency == "50hz":
            return self.process_chunk_for_50hz(
                i,
                start_index,
                end_index,
                finished,
                prev_generated,
                generated,
            )
        elif self.frequency == "25hz":
            return self.process_chunk_for_25hz(
                i,
                start_index,
                end_index,
                finished,
                prev_generated,
                generated,
            )
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")

    def process_chunk_for_50hz(
        self,
        i: int,
        start_index: int,
        end_index: int,
        finished: bool,
        prev_generated: torch.Tensor,
        generated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if i == 0:
            generated = generated.to(torch.float32)[:, : self.chunk_size, :]
            self.code2wav_bigvgan_model.forward = self.bigvgan_cuda_graph_first
            mel = generated
        elif finished:
            generated = generated.to(torch.float32)[:, self.past_cache_size :, :]
            self.code2wav_bigvgan_model.forward = self.bigvgan_cuda_graph_last
            mel = torch.cat(
                [prev_generated[:, -self.future_size * 2 :, :], generated], dim=1
            )
        else:
            # Note that self.chunk_size == self.past_cache_size, so the following branch
            # can be simplified. But for clearness, we keep it as it is in transformers.
            if start_index == 0:
                generated = generated.to(torch.float32)[
                    :, i * self.chunk_size : -self.future_cache_size, :
                ]
            else:
                generated = generated.to(torch.float32)[
                    :, self.past_cache_size : -self.future_cache_size, :
                ]
            self.code2wav_bigvgan_model.forward = self.bigvgan_cuda_graph_intermediate
            mel = torch.cat(
                [prev_generated[:, -self.future_size * 2 :, :], generated], dim=1
            )

        audio = self.code2wav_bigvgan_model(mel)
        if i == 0:
            audio_output = audio[: -self.future_size * 240]
        elif finished:
            audio_output = audio[self.future_size * 240 :]
        else:
            audio_output = audio[self.future_size * 240 : -self.future_size * 240]
        return generated, audio_output

    def process_chunk_for_25hz(
        self,
        i: int,
        start_index: int,
        end_index: int,
        finished: bool,
        prev_generated: list[torch.Tensor],
        generated: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if start_index == 0:
            if i == 0:
                prev_generated = [generated.to(torch.float32)[:, : self.chunk_size, :]]
            else:
                prev_generated.append(
                    generated.to(torch.float32)[
                        :, i * self.chunk_size : -self.future_cache_size, :
                    ]
                )
            self.code2wav_bigvgan_model.forward = self.bigvgan_cuda_graph_first
        elif finished:
            prev_generated.append(
                generated.to(torch.float32)[:, self.past_cache_size :, :]
            )
            self.code2wav_bigvgan_model.forward = self.bigvgan_cuda_graph_last
        else:
            prev_generated.append(
                generated.to(torch.float32)[
                    :, self.past_cache_size : -self.future_cache_size, :
                ]
            )
            self.code2wav_bigvgan_model.forward = self.bigvgan_cuda_graph_intermediate
        mel = torch.cat(prev_generated, dim=1)

        audio = self.code2wav_bigvgan_model(mel)

        if i == 0:
            audio_output = audio[: -self.future_size * 240]
        elif finished:
            if len(prev_generated) <= 2:
                audio_output = audio[(self.past_size - self.chunk_size) * 240 :]
            else:
                audio_output = audio[self.past_size * 240 :]
        else:
            if len(prev_generated) <= 2:
                audio_output = audio[
                    (self.past_size - self.chunk_size) * 240 : -self.future_size * 240
                ]
            else:
                audio_output = audio[self.past_size * 240 : -self.future_size * 240]
        return prev_generated[-2:], audio_output
