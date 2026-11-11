# Copyright 2023 KaiHu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""BigVGAN"""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple, List, Union
import typing as tp

import numpy as np
from scipy.signal import get_window
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils import weight_norm, remove_weight_norm
from funasr.train_utils.device_funcs import force_gatherable
import logging
from funasr.utils.hinter import hint_once
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
import os
from torch.nn.utils.rnn import pad_sequence
from funasr.models.cosyvoice.utils import dtype_map
from funasr.models.cosyvoice.hifigan_module import get_padding, init_weights
from funasr.models.cosyvoice.hifigan_module.activations import Snake
from funasr.models.cosyvoice.alias_free_activation.torch.act import Activation1d as TorchActivation1d
from funasr.models.cosyvoice.causal_hifigan import LookRightConv1d, LookLeftConvTranspose1d, LookLeftConv1dWithDilation, LookLeftConv1d
from funasr.models.cosyvoice.hifigan_module import activations


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AMPBlock1_Causal(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    LookLeftConv1dWithDilation(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    LookLeftConv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from funasr.models.cosyvoice.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x, cache=None):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        # NOTE 不知道为什么如果cache默认参数是{}，会保留上一次的结果
        if cache is None:
            cache = {}
        for i, (c1, c2, a1, a2) in enumerate(zip(self.convs1, self.convs2, acts1, acts2)):
            cache_i = cache.get(i, {})
            xt = a1(x)
            xt, cache_i[c1] = c1(xt, cache=cache_i.get(c1, None))
            # xt, _ = c1(xt)
            xt = a2(xt)
            xt, cache_i[c2] = c2(xt, cache=cache_i.get(c2, None))
            # xt, _ = c2(xt)
            x = xt + x
            cache[i] = cache_i
        return x, cache

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2_Causal(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    LookLeftConv1dWithDilation(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from cosyvoice.bigvgan.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt, _ = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class CausalBigVGanGenerator(torch.nn.Module):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from cosyvoice.bigvgan.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            LookRightConv1d(h.num_mels, h.upsample_initial_channel, h.conv_pre_look_right + 1, 1)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1_Causal
        elif h.resblock == "2":
            resblock_class = AMPBlock2_Causal
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            LookLeftConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            LookLeftConv1d(ch, 1, 7, 1, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def inference(self, speech_feat: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        # Pre-conv
        x = speech_feat

        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, :-self.h.conv_pre_look_right], x[:, :, -self.h.conv_pre_look_right:])

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x, _ = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                new_xs, _ = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = new_xs
                else:
                    xs = xs + new_xs
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x, _ = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x.squeeze(dim=1)

    def inference_cache(self, speech_feat: torch.Tensor, finalize: bool = True, cache: Dict = {}) -> torch.Tensor:
        # Pre-conv
        x = speech_feat

        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, :-self.h.conv_pre_look_right], x[:, :, -self.h.conv_pre_look_right:])

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x, cache[self.ups[i][i_up]] = self.ups[i][i_up](x, cache=cache.get(self.ups[i][i_up], None))
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                new_xs, cache[self.resblocks[i * self.num_kernels + j]] = self.resblocks[i * self.num_kernels + j](x, cache=cache.get(self.resblocks[i * self.num_kernels + j], None))
                if xs is None:
                    xs = new_xs
                else:
                    xs = xs + new_xs
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x, cache[self.conv_post] = self.conv_post(x, cache.get(self.conv_post, None))
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x.squeeze(dim=1), cache

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass


@tables.register("model_classes", "CausalBigVGan")
class CausalBigVGan(nn.Module):

    def __init__(
            self,
            CausalBigVGanGenerator_conf: dict = {},
            sample_rate: float = 24000,
            **kwargs
    ):
        super().__init__()
        self.generator = CausalBigVGanGenerator(AttrDict(CausalBigVGanGenerator_conf))
        self.sample_rate = sample_rate

    def inference_prepare(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        feat_list = []
        feat_len_list = []
        for i, feat in enumerate(data_in):
            if isinstance(feat, str) and os.path.exists(feat):
                feat = np.load(feat)
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)

            feat_list.append(feat)
            feat_len_list.append(feat.shape[0])

        batch = {
            "x": pad_sequence(feat_list, batch_first=True),
            "x_lengths": torch.tensor(feat_len_list, dtype=torch.int64),
        }
        batch = to_device(batch, kwargs["device"])

        return batch

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        finalize: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run inference.

        Args:
            x (torch.Tensor): input representation, B x T x C

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (B, T_wav).

        """
        uttid = key[0]
        batch = self.inference_prepare(data_in, data_lengths, key, **kwargs)
        voc_dtype = dtype_map[kwargs.get("voc_dtype", "fp32")]
        x = batch["x"].to(voc_dtype)
        with torch.amp.autocast(device_type="cuda", enabled=True if kwargs['voc_dtype'] != "fp32" else False, dtype=dtype_map[kwargs['voc_dtype']]):
            recon_speech = self.generator.inference(x.transpose(1, 2), finalize=finalize)
        recon_speech = recon_speech.float()
        logging.info(f"{uttid}: wav lengths {recon_speech.shape[1]}")

        output_sr = kwargs.get("output_sr", None)
        if output_sr is not None and output_sr != self.sample_rate:
            recon_speech = torchaudio.functional.resample(
                recon_speech,
                orig_freq=self.sample_rate,
                new_freq=output_sr
            )

        return recon_speech


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from tqdm import tqdm
    with torch.no_grad():
        device = 'cuda'
        h = AttrDict(activation='snakebeta',
                    num_mels=80,
                    resblock='1',
                    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    resblock_kernel_sizes=[3, 7, 11],
                    snake_logscale=True,
                    upsample_initial_channel=1536,
                    upsample_kernel_sizes=[12, 10, 4, 4, 4, 4],
                    upsample_rates=[6, 5, 2, 2, 2, 2],
                    use_bias_at_final=False,
                    use_tanh_at_final=False,
                    conv_pre_look_right=8)
        generator = CausalBigVGanGenerator(h=h, use_cuda_kernel=False).eval().to(device)
        state_dict = torch.load('/path/to/epoch_75_whole.pt', 'cpu')
        state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items() if 'generator.' in k}
        generator.load_state_dict(state_dict)
        chunk_size, context_size = 30, h.conv_pre_look_right
        mel = torch.load('/path/to/mel.pt', 'cpu').to(device).transpose(1, 2)
        pred_gt = generator.inference(mel, finalize=True).cpu()
        # 验证因果
        pred_stream = torch.zeros(1, 0).to(pred_gt)
        for i in tqdm(range(0, mel.shape[2], chunk_size)):
            finalize = True if i + chunk_size + context_size >= mel.shape[2] else False
            pred_chunk = generator.inference(mel[:, :, :i + chunk_size + context_size], finalize=finalize).cpu()
            print((pred_gt[:, :pred_chunk.shape[1]] - pred_chunk).abs().max().item())
            pred_stream = torch.cat([pred_stream, pred_chunk[:, pred_stream.shape[1]:]], dim=1)
        # 验证cache
        pred_stream_cache, cache = torch.zeros(1, 0).to(pred_gt), {}
        for i in tqdm(range(0, mel.shape[2], chunk_size)):
            finalize = True if i + chunk_size + context_size >= mel.shape[2] else False
            pred_chunk, cache = generator.inference_cache(mel[:, :, i:i + chunk_size + context_size], finalize=finalize, cache=cache)
            pred_chunk = pred_chunk.cpu()
            print((pred_gt[:, pred_stream_cache.shape[1]:pred_stream_cache.shape[1] + pred_chunk.shape[1]] - pred_chunk).abs().max().item())
            pred_stream_cache = torch.cat([pred_stream_cache, pred_chunk], dim=1)
        import torchaudio
        torchaudio.save('causal_bigvgan_offline.wav', pred_gt, sample_rate=24000)
        torchaudio.save('causal_bigvgan_stream.wav', pred_stream, sample_rate=24000)
        torchaudio.save('causal_bigvgan_stream_cache.wav', pred_stream_cache, sample_rate=24000)
