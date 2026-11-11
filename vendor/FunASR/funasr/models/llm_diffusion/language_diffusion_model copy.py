import logging
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
import numpy as np
from funasr.models.cosyvoice.utils import dtype_map
from funasr.utils.hinter import hint_once
import math
from typing import Dict, Optional
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from .utils.mask import make_pad_mask
from einops import repeat
import random
def print_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2     # MB
    print(f"{prefix} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

class WeightedBCELoss(nn.Module):
    def __init__(
        self,
        size,
        padding_idx,
        weight: list = [1, 1],
    ):
        super(WeightedBCELoss, self).__init__()
        self.size = size
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weight), ignore_index=padding_idx)

    def forward(self, x, target):
        x = x.contiguous().view(-1, self.size)
        target = target.contiguous().view(-1)
        return self.ce(x, target)

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        center=False,
        device='cuda',
        feat_type="power_log",
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float().to(device)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.center = center
        self.feat_type = feat_type

    def forward(self, audioin):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        if self.feat_type == "mag_log10":
            power_spec = torch.sqrt(torch.pow(fft.imag, 2) + torch.pow(fft.real, 2))
            mel_output = torch.matmul(self.mel_basis, power_spec)
            return torch.log10(torch.clamp(mel_output, min=1e-5))
        power_spec = torch.pow(fft.imag, 2) + torch.pow(fft.real, 2)
        mel_spec = torch.matmul(self.mel_basis, torch.sqrt(power_spec + 1e-9))
        return self.spectral_normalize(mel_spec)

    @classmethod
    def spectral_normalize(cls, spec, C=1, clip_val=1e-5):
        output = cls.dynamic_range_compression(spec, C, clip_val)
        return output

    @classmethod
    def spectral_de_normalize_torch(cls, spec, C=1, clip_val=1e-5):
        output = cls.dynamic_range_decompression(spec, C, clip_val)
        return output

    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def dynamic_range_decompression(x, C=1):
        return torch.exp(x) / C

class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

@tables.register("model_classes", "LLMDiffusionInterleave")
class LLMDiffusionInterleave(nn.Module):
    """ """

    def __init__(
        self,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        del_lm_head: bool = True,
        dit_conf: Dict = {},
        mel_feat_conf: Dict = {},
        downsample_encoder_conf: Dict = {},
        upsample_encoder_conf: Dict = {},
        lm_use_spk_emb: bool = False,
        pad_left: bool = True,
        init_cv2: bool = True,
        bce_weight: list = [1, 1],
        use_l1_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        self.sample_rate = kwargs.get("sample_rate", 24000)
        self.token_rate = kwargs.get("token_rate", 25)
        self.subsampling_factor = kwargs.get("subsampling_factor", 8)
        self.left_size = kwargs.get("left_size", 8)
        self.sos = kwargs.get("sos", 0)
        self.eos = kwargs.get("eos", 1)
        self.turn_of_speech = kwargs.get("turn_of_speech", 2)
        self.fill_token = kwargs.get("fill_token", 3)
        self.pad_left = pad_left
        self.t_scheduler = kwargs.get("t_scheduler", "cosine")
        self.downsample_type = kwargs.get("downsample_type", "linear")
        self.upsample_type = kwargs.get("upsample_type", "linear")
        self.init_cv2 = init_cv2
        self.v_prediction = kwargs.get("v_prediction", False)
        self.use_l1_loss = use_l1_loss

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            **llm_load_kwargs,
        )

        if self.init_cv2:
            llm_state_dict = torch.load('/nfs/ankeyu.aky/CosyVoice_github/pretrained_models/CosyVoice2-0.5B/llm.pt')
            model.load_state_dict(llm_state_dict, strict=False)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}")
        if llm_conf.get("use_lora", False):
            from omegaconf import OmegaConf, DictConfig

            lora_conf = llm_conf.get("lora_conf", {})
            if isinstance(lora_conf, (OmegaConf, DictConfig)):
                lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
            from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

            lora_init_param_path = lora_conf.get("init_param_path", None)
            if lora_init_param_path is not None:
                logging.info(f"lora_init_param_path: {lora_init_param_path}")
                model = PeftModel.from_pretrained(model, lora_init_param_path)
                for name, param in model.named_parameters():
                    if not lora_conf.get("freeze_lora", False):
                        if "lora_" in name:
                            param.requires_grad = True
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        if del_lm_head:
            del self.llm.lm_head
        self.error_calculator = None

        specaug = kwargs.get("specaug", None)
        specaug_conf = kwargs.get("specaug_conf", {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug
        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")
        xvec_size = kwargs.get("xvec_size", 192)

        self.dit_conf = dit_conf
        if kwargs.get("dit", "dit") == "dit":
            self.dit_class = "dit"
            from funasr.models.cosyvoice.modules.dit_flow_matching.dit_model import DiT
            self.dit_model = DiT(**dit_conf)
            self.xvec_proj = torch.nn.Linear(xvec_size, self.dit_model.input_embed.spk_dim)
        else:
            self.dit_class = "matcha"
            from funasr.models.cosyvoice.modules.matcha_flow_matching.matcha_patch_decoder import ConditionalPatchDecoder
            self.dit_model = ConditionalPatchDecoder(**dit_conf)           
            self.xvec_proj = torch.nn.Linear(xvec_size, self.dit_model.out_channels)

        self.mel_extractor = Audio2Mel(**mel_feat_conf) 
        self.mel_norm_type = kwargs.get("mel_norm_type", None)

        if self.downsample_type == "linear":
            from .downsample_encoder import LinearEncoder
            self.downsample_encoder = LinearEncoder(output_size=llm_dim, **downsample_encoder_conf)
        else:
            from .downsample_encoder import TransformerDownsampleBlock
            self.downsample_encoder = TransformerDownsampleBlock(output_dim=llm_dim, **downsample_encoder_conf)


        if self.upsample_type == "linear":
            from .upsample_encoder import LinearUpsample1D_V2
            self.upsample_encoder = LinearUpsample1D_V2(input_size=llm_dim, **upsample_encoder_conf)
        else:
            from .upsample_encoder import TransformerUpsampleBlock
            self.upsample_encoder = TransformerUpsampleBlock(input_dim=llm_dim, **upsample_encoder_conf)

        self.llm_embedding = torch.nn.Embedding(4, llm_dim)

        self.training_cfg_rate = kwargs.get("training_cfg_rate", 0)

        self.ignore_id = kwargs.get("ignore_id", -100)
        self.binary_decoder = nn.Linear(llm_dim, 2)
        self.criterion_ce = LabelSmoothingLoss(
            size=2,
            padding_idx=self.ignore_id,
            smoothing=0.0,
            normalize_length=True,
        )
        #self.criterion_ce = WeightedBCELoss(size=2, padding_idx=self.ignore_id, weight=bce_weight)
        self.length_normalized_loss = length_normalized_loss

        self.lm_use_spk_emb = lm_use_spk_emb
        if self.lm_use_spk_emb:
            self.xvec_proj_llm = torch.nn.Linear(xvec_size, llm_dim)

        self.diff_use_global_emb = kwargs.get("diff_use_global_emb", False)
        if self.diff_use_global_emb:
           from .seg_transformer import SegmentTransformer
           self.seg_transformer = SegmentTransformer(input_dim=80, output_dim=80, hidden_dim=256, num_heads=4, num_layers=2)

        self.llm_use_global_emb = kwargs.get("llm_use_global_emb", False)
        self.spk_llm_use_global_emb = kwargs.get("spk_llm_use_global_emb", False)
        if self.llm_use_global_emb or self.spk_llm_use_global_emb:
           from .seg_transformer import SegmentTransformer
           self.seg_transformer_llm = SegmentTransformer(input_dim=80, output_dim=llm_dim, hidden_dim=256, num_heads=4, num_layers=2)


    @torch.no_grad()
    def extract_feat(self, y: torch.Tensor, y_lengths: torch.Tensor):
        mel_extractor = self.mel_extractor.float()
        feat = mel_extractor(y)
        feat = feat.transpose(1, 2)
        feat_lengths = (y_lengths / self.mel_extractor.hop_length).to(y_lengths)
        if self.mel_norm_type is not None:
            feat = self.normalize_mel_feat(feat, feat_lengths)
        return feat, feat_lengths

    def add_cond(self, x1, y, chunk_size, left_context_size):
        N_chunks = y.size(1) // chunk_size
        y = y.reshape(y.size(0) * N_chunks, chunk_size, y.size(2))

        if left_context_size == 0:
            return y

        x1 = x1.reshape(x1.size(0) * N_chunks, chunk_size, x1.size(2))
        left_context = torch.zeros(y.size()[0], left_context_size, y.size()[2], device=y.device)
        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, :
                ] = x1[: -N + idx, :, :]
            for idx in range(N):
                left_context[idx::N_chunks, : (N - idx) * chunk_size, :] = 0
        else:
            left_context[1:, :, :] = x1[:-1, -left_context_size:, :]
            left_context[0::N_chunks, :, :] = 0

        cond_y = torch.cat((left_context, y), dim=1)
        return cond_y

    def add_ctx(self, mu, chunk_size, left_context_size):
        N_chunks = mu.size(1) // chunk_size
        mu = mu.reshape(mu.size(0) * N_chunks, chunk_size, mu.size(2))

        if left_context_size == 0:
            return mu

        left_context = torch.zeros(mu.size()[0], left_context_size, mu.size()[2], device=mu.device)
        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, :
                ] = mu[: -N + idx, :, :]
            for idx in range(N):
                left_context[idx::N_chunks, : (N - idx) * chunk_size, :] = 0
        else:
            left_context[1:, :, :] = mu[:-1, -left_context_size:, :]
            left_context[0::N_chunks, :, :] = 0

        ctx_mu = torch.cat((left_context, mu), dim=1)
        return ctx_mu

    def repeat(self, mu, inputs, chunk_size):
        N_chunks = mu.size(1) // chunk_size
        outputs = inputs.repeat_interleave(N_chunks, dim=0)
        return outputs

    def diffusion_forward(self, x1, mu, spks, x1_len, mask, global_emb=None, z=None, t=None, cond=None, cfg_mask=None):
        """
        x1: B,T,n_mel
        mu: B,T,n_mel
        spks: B,1,n_mel
        cond: B,T,n_mel
        xs_len: B
        """
        b, T, _ = mu.shape
        if t is None:
            t = torch.rand([b], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        if z is None:
            z = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * z + t[:, None, None] * x1

        if self.v_prediction:
            u = x1 - z
        else:
            u = x1

        if self.training_cfg_rate > 0:
            if cfg_mask is None:
                cfg_mask = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) > self.training_cfg_rate
            mu = mu * cfg_mask
            spks = spks * cfg_mask
            if global_emb is not None:
                global_emb = global_emb * cfg_mask.squeeze(-1)

        if cond is None:
            cond = x1
        cond_x_t = self.add_cond(cond, x_t, self.subsampling_factor, self.left_size)
        ctx_mu = self.add_ctx(mu, self.subsampling_factor, self.left_size)

        if spks is not None:
            spks = self.repeat(mu, spks, self.subsampling_factor)

        if global_emb is not None:
            global_emb = self.repeat(mu, global_emb, self.subsampling_factor)
            global_emb = repeat(global_emb, "b c -> b t c", t=cond_x_t.shape[1])

        t = self.repeat(mu, t, self.subsampling_factor)

        #print_memory_usage("Before dit:")
        if self.dit_class == "dit":
            u_hat = self.dit_model(
                cond_x_t, global_emb, ctx_mu, spks, t,
                mask=None, mask_rand=None
            )
        else:
            cond_x_t = cond_x_t.transpose(1,2)
            ctx_mu = ctx_mu.transpose(1,2)
            u_hat = self.dit_model(cond_x_t, ctx_mu, t, spks.squeeze(dim=1), cond=global_emb, return_hidden=False)
            u_hat = u_hat.transpose(1,2)
        #print_memory_usage("After dit:")
        u_hat = u_hat[:,-self.subsampling_factor:,:].reshape(b, T, -1)

        with torch.cuda.amp.autocast(False):
            if self.use_l1_loss:
                l1_loss = (F.l1_loss(u, u_hat, reduction="none") * mask.unsqueeze(2)).sum() / (torch.sum(mask) * u.shape[2])
            loss = (F.mse_loss(u, u_hat, reduction="none") * mask.unsqueeze(2)).sum() / (torch.sum(mask) * u.shape[2])

        if self.use_l1_loss:
            return loss, u_hat, l1_loss
        else:
            return loss, u_hat

    @torch.inference_mode()
    def diffusion_infer(self, mu, n_timesteps, temperature=1.0, spks=None, mel_history=None, global_emb=None, **kwargs):
        z = torch.randn(mu.shape[0], self.subsampling_factor, self.dit_model.mel_dim).to(mu.device).to(mu.dtype) * temperature
        if mel_history is not None:
            z = torch.cat((mel_history, z), dim=1)
        if global_emb is not None:
            global_emb = repeat(global_emb, "b c -> b t c", t=mu.shape[1])

        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        infer_t_scheduler = kwargs.get("infer_t_scheduler", "cosine")
        if infer_t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        mel = self.solve_euler(z, t_span=t_span, mu=mu, spks=spks, mel_history=mel_history, global_emb=global_emb)
        return mel

    def solve_euler(self, x, t_span, mu, spks=None, mel_history=None, global_emb=None, **kwargs):
        inference_cfg_rate = kwargs.get("inference_cfg_rate", 0.7)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        steps = 1
        z, bz = x, x.shape[0]
        while steps <= len(t_span) - 1:
            if mel_history is not None:
                x[:,:-self.subsampling_factor,:] = mel_history
            if inference_cfg_rate > 0:
                x_in = torch.concat([x, x], dim=0)
                spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
                mu_in = torch.concat([mu, torch.zeros_like(mu)], dim=0)
                t_in = torch.concat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                if global_emb is not None:
                    global_emb_in = torch.concat([global_emb, torch.zeros_like(global_emb)], dim=0)
                else:
                    global_emb_in = None
            else:
                x_in, mu_in, spks_in, t_in = x, mu, spks, t

            if self.dit_class == "dit":
                dphi_dt = self.dit_model(
                    x_in, global_emb_in, mu_in, spks_in, t_in,
                    mask=None,
                    mask_rand=torch.ones_like(t_in).reshape(-1, 1, 1)
                )
            else:
                dphi_dt = self.dit_model(
                    x_in.transpose(1, 2), mu_in.transpose(1, 2), t_in.squeeze(), spks_in.squeeze(), cond=global_emb_in
                ).transpose(1, 2)

            if inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [bz, bz], dim=0)
                dphi_dt = ((1.0 + inference_cfg_rate) * dphi_dt -
                           inference_cfg_rate * cfg_dphi_dt)

            if self.v_prediction:
                x = x + dt * dphi_dt
            else:
                x = x + dt * (dphi_dt - x) / (1-t)
            t = t + dt
            # sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return x

    # def pad_unpad_sequence(self, text_token, text_token_len, speech, speech_len, spk_emb_llm=None, global_emb_llm=None):
    #     text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
    #     speech = unpad_sequence(speech, speech_len.cpu(), batch_first=True)

    #     sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
    #     turn_of_speech_emb = self.llm_embedding.weight[self.turn_of_speech].reshape(1, 1, -1)

    #     if global_emb_llm is not None:
    #         global_emb_llm = global_emb_llm.unsqueeze(dim=1)

    #     if self.lm_use_spk_emb:
    #         if self.llm_use_global_emb:
    #             lm_input = [torch.concat([sos_emb.squeeze(dim=0), spk_emb_llm[i], global_emb_llm[i], text_token[i], turn_of_speech_emb.squeeze(dim=0), speech[i]], dim=0)
    #                 for i in range(len(text_token))]
    #         else:
    #             lm_input = [torch.concat([sos_emb.squeeze(dim=0), spk_emb_llm[i], text_token[i], turn_of_speech_emb.squeeze(dim=0), speech[i]], dim=0)
    #                 for i in range(len(text_token))]
    #     else:
    #         if self.llm_use_global_emb:
    #             lm_input = [torch.concat([sos_emb.squeeze(dim=0), global_emb_llm[i], text_token[i], turn_of_speech_emb.squeeze(dim=0), speech[i]], dim=0)
    #                 for i in range(len(text_token))]
    #         else:
    #             lm_input = [torch.concat([sos_emb.squeeze(dim=0), text_token[i], turn_of_speech_emb.squeeze(dim=0), speech[i]], dim=0)
    #                 for i in range(len(text_token))]
    #     attention_mask = [torch.concat([torch.tensor([1] * len(lm_input[i]), dtype=torch.int32)], dim=0) for i in range(len(text_token))]
    #     lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
    #     lm_input = pad_sequence(lm_input, batch_first=True, padding_value=0.0)
    #     attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0).to(lm_input.device)
    #     return lm_input, lm_input_len, attention_mask


    def get_llm_input(self, text_token, text_token_len, speech, speech_len, spk_emb_llm=None, global_emb_llm=None):
        # 这个函数可以实现将输入的文本和语音进行交替拼接，并返回text_token和speech_token所在位置的mask
        text_first = random.random() < 0.5
        B, N, D = text_token.shape
        B, T, D = speech.shape
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech = unpad_sequence(speech, speech_len.cpu(), batch_first=True)
        results = []
        text_mask = []
        speech_mask = []
        if text_first:
            text_token, speech = text_token[::2], speech[1::2]
            text_token, speech = text_token[:min(len(text_token), len(speech))], speech[:min(len(text_token), len(speech))]
            for text, sph in zip(text_token, speech):
                results.append(torch.concat([text, speech], dim=1))
                text_mask.append(torch.concat([torch.ones(text.size(0)), torch.zeros(speech.size(0))], dim=0))
                speech_mask.append(torch.concat([torch.zeros(text.size(0)), torch.ones(speech.size(0))], dim=0))
        else:
            text_token, speech = text_token[1::2], speech[::2]
            text_token, speech = text_token[:min(len(text_token), len(speech))], speech[:min(len(text_token), len(speech))]
            for text, sph in zip(text_token, speech):
                results.append(torch.concat([speech, text], dim=1))
                text_mask.append(torch.concat([torch.zeros(text.size(0)), torch.ones(speech.size(0))], dim=0))
                speech_mask.append(torch.concat([torch.ones(text.size(0)), torch.zeros(speech.size(0))], dim=0))
        llm_input = torch.concat(results, dim=0)
        text_mask = torch.concat(text_mask, dim=0)
        speech_mask = torch.concat(speech_mask, dim=0)
        attention_mask = torch.ones(llm_input.shape[0], llm_input.shape[1]).to(llm_input.device).long()
        return llm_input, text_mask, speech_mask, attention_mask


    def get_llm_output(self, llm_output, text_len, feat_len, feat):
        feat_pred = torch.zeros(feat.shape).to(feat.device)
        for i, j in enumerate(feat_len):
            if self.lm_use_spk_emb:
                if self.llm_use_global_emb:
                    feat_pred[i, :j] = llm_output[i, 3 + text_len[i]: 3 + text_len[i] + j]
                else:
                    feat_pred[i, :j] = llm_output[i, 2 + text_len[i]: 2 + text_len[i] + j]
            else:
                if self.llm_use_global_emb:
                    feat_pred[i, :j] = llm_output[i, 2 + text_len[i]: 2 + text_len[i] + j]
                else:
                    feat_pred[i, :j] = llm_output[i, 1 + text_len[i]: 1 + text_len[i] + j]
        return feat_pred

    @staticmethod
    def norm_spk_emb(xvec):
        xvec_mask = (~xvec.norm(dim=-1).isnan()) * (~xvec.norm(dim=-1).isinf())
        xvec = xvec * xvec_mask.unsqueeze(-1)
        xvec = xvec.mean(dim=1)
        xvec = F.normalize(xvec, dim=1)

        return xvec

    def extract_segment_features(self, input_tensor, input_lengths, feature_extractor=None):
        B, T, D = input_tensor.shape
        device = input_tensor.device
    
        # 1. 计算随机截取长度 (原长度的10%~50%)
        min_ratio, max_ratio = 0.1, 0.5
        rand_ratios = torch.rand(B, device=device) * (max_ratio - min_ratio) + min_ratio
        segment_lengths = (input_lengths.float() * rand_ratios).long().clamp(min=1)
        segment_lengths = torch.min(segment_lengths, input_lengths)
    
        # 2. 随机选择起始位置
        max_starts = input_lengths - segment_lengths
        starts = torch.rand(B, device=device) * (max_starts.float() + 1e-8)
        starts = starts.long().clamp(min=0)
    
        # 3. 创建片段张量和掩码
        max_seg_len = segment_lengths.max().item()
        segments = torch.zeros(B, max_seg_len, D, device=device)
        mask = torch.ones(B, max_seg_len, dtype=torch.bool, device=device)
    
        # 4. 填充每个序列的片段
        for i in range(B):
            seg_len = segment_lengths[i].item()
            end = starts[i] + seg_len
        
            # 提取片段
            seg = input_tensor[i, starts[i]:end]
        
            # 填充到最大长度
            segments[i, :seg_len] = seg
        
            # 创建掩码（有效位置为False）
            mask[i, :seg_len] = False
        return segments, mask
        #return self.seg_transformer(segments, mask)

    def forward(
        self,
        text: torch.Tensor, # B, R, N
        text_len: torch.Tensor, 
        audio: torch.Tensor,  # B, R, T
        audio_len: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
        spk_emb_len: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if self.pad_left:
            stride_lens = self.subsampling_factor * self.mel_extractor.hop_length
            audio_len = (audio_len + stride_lens - 1) // stride_lens * stride_lens
            pad_lens = audio_len.max()-audio.size(1)

            if pad_lens > 0:
                left_pad = torch.zeros(audio.size(0), pad_lens).to(audio.device)
                audio = torch.cat([left_pad, audio], dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            feat, feat_lengths = self.extract_feat(audio.float(), audio_len)
            if not self.pad_left:
                if feat_lengths.max() < self.subsampling_factor:
                    feat_pad = torch.zeros(feat.size(0), self.subsampling_factor-feat_lengths.max(), feat.size(2)).to(audio.device)
                    feat = torch.cat([feat_pad, feat], dim=1)
                    feat_lengths = (feat_lengths + self.subsampling_factor) // self.subsampling_factor * self.subsampling_factor
                else:
                    feat_lengths = feat_lengths // self.subsampling_factor * self.subsampling_factor
                    feat = feat[:,:feat_lengths.max(),:]       
        feat_lengths = feat_lengths.squeeze(-1)
        text_len = text_len.squeeze(-1)

        if self.diff_use_global_emb or self.llm_use_global_emb:
            segments, seg_mask = self.extract_segment_features(feat, feat_lengths)

        if self.diff_use_global_emb:
            global_emb = self.seg_transformer(segments, seg_mask)
        else:
            global_emb = None

        if self.llm_use_global_emb:
            global_emb_llm = self.seg_transformer_llm(segments, seg_mask)
        else:
            global_emb_llm = None

        encoder_in = feat.clone()
        if self.specaug is not None:
            encoder_in, _ = self.specaug(encoder_in)

        encoder_out, encoder_mask = self.downsample_encoder(encoder_in, feat_lengths)
        encoder_out_lengths = encoder_mask.sum(dim=2).squeeze(dim=1) 
        
        text_mask = text > 0
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            text_embeds = self.llm.model.get_input_embeddings()(text * text_mask) * text_mask.unsqueeze(-1)

        if self.lm_use_spk_emb and spk_emb is not None:
            spk_emb_llm = spk_emb[:, :spk_emb_len.max()]
            spk_emb_llm = self.norm_spk_emb(spk_emb_llm.reshape(1, 1, -1)).float()
            spk_emb_llm = self.xvec_proj_llm(spk_emb_llm).unsqueeze(dim=1)
        else:
            spk_emb_llm = None

        # inputs_embeds, inputs_embeds_len, attention_mask = self.pad_unpad_sequence(text_embeds, text_len, encoder_out, encoder_out_lengths, spk_emb_llm, global_emb_llm)
        llm_input, text_mask, speech_mask, attention_mask = self.get_llm_input(text_embeds, text_len, encoder_out, encoder_out_lengths, spk_emb_llm, global_emb_llm)
        inputs_embeds, text_mask, speech_mask, attention_mask = llm_input.unsqueeze(dim=0), text_mask.unsqueeze(dim=0), speech_mask.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0)


        stats = {}
        batch_size, feat_num, dims = inputs_embeds.shape
        #print_memory_usage("Before llm:")
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            attention_mask[attention_mask < 0] = 0
            outputs = self.llm.model(
                # inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            lm_outputs = outputs[0]
        #print_memory_usage("After llm:")
        if self.lm_use_spk_emb:
            if self.llm_use_global_emb:
                lm_binary_target = [torch.tensor([self.ignore_id] * (3 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
            else:
                lm_binary_target = [torch.tensor([self.ignore_id] * (2 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
        else:
            if self.llm_use_global_emb:
                lm_binary_target = [torch.tensor([self.ignore_id] * (2 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
            else:
                lm_binary_target = [torch.tensor([self.ignore_id] * (1 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
        lm_binary_target = pad_sequence(lm_binary_target, batch_first=True, padding_value=self.ignore_id).to(lm_outputs.device)

        binary_logits = self.binary_decoder(lm_outputs)

        # binary loss
        loss_binary = self.criterion_ce(binary_logits, lm_binary_target)
        acc_binary = th_accuracy(binary_logits.view(-1, 2), lm_binary_target, ignore_label=self.ignore_id)

        hidden = self.get_llm_output(lm_outputs, text_len, encoder_out_lengths, encoder_out)
        hidden, h_lengths = self.upsample_encoder(hidden, encoder_out_lengths)
        if spk_emb is not None:
            spk_emb_diff = spk_emb[:, :spk_emb_len.max()]
            spk_emb_diff = self.norm_spk_emb(spk_emb_diff.reshape(1, 1, -1)).float()
            spk_emb_diff = self.xvec_proj(spk_emb_diff).unsqueeze(dim=1)

        #print_memory_usage("Before diffusion:")
        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(hidden)

        if self.use_l1_loss:
            diff_loss, feat_pred, diff_loss_l1 = self.diffusion_forward(feat, hidden, spk_emb_diff, feat_lengths, mask, global_emb)
            #print_memory_usage("After diffusion:")
            loss = diff_loss + diff_loss_l1 + loss_binary
        else:
            diff_loss, feat_pred = self.diffusion_forward(feat, hidden, spk_emb_diff, feat_lengths, mask, global_emb)
            #print_memory_usage("After diffusion:")
            loss = diff_loss + loss_binary


        #print("loss", loss)
        # Shift so that tokens < n predict n

        stats["loss"] = torch.clone(loss.detach())
        stats["diff_loss"] = torch.clone(diff_loss.detach())
        if self.use_l1_loss:
            stats["diff_loss_l1"] = torch.clone(diff_loss_l1.detach())
        stats["binary_loss"] = torch.clone(loss_binary.detach())
        stats["acc_binary"] = acc_binary

        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = feat_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def data_template(self, data):
        system, user, assistant, meta_dict = [], [], [], {}
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "user_prompt_text" in item:
                    meta_dict["user_prompt_text"] = item["user_prompt_text"]
                user.append(content)
            elif role == "assistant":
                if "prompt_codec" in item:
                    meta_dict["prompt_codec"] = item["prompt_codec"]
                if "prompt_wav" in item:
                    meta_dict["prompt_wav"] = item["prompt_wav"]
                if "spk_emb" in item:
                    meta_dict["spk_emb"] = item["spk_emb"]
                assistant.append(content)
        system = system * len(user)
        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
            "meta_dict": meta_dict,
        }
        return contents

    def load_data(self, contents: dict, **kwargs):
        uttid = kwargs.get("uttid", None)
        streaming_mode = kwargs.get("lm_streaming", False)
        if isinstance(streaming_mode, bool):
            streaming_mode = str(streaming_mode).lower()
        lm_use_prompt = kwargs.get("lm_use_prompt", True)
        tokenizer = kwargs.get("tokenizer")
        prompt_text = contents["meta_dict"].get("user_prompt_text", None)
        llm_generator_conf = kwargs.get("llm_generator_conf", {})

        prompt_wav = contents["meta_dict"].get("prompt_wav", None)
        prompt_wav_lengths = None
        if prompt_wav is not None and lm_use_prompt:
            if isinstance(prompt_wav, str) and os.path.exists(prompt_wav):
                prompt_wav = load_audio_text_image_video(prompt_wav, fs=self.sample_rate)
                prompt_wav = prompt_wav[None, :]
            if isinstance(prompt_wav, np.ndarray):
                prompt_wav = torch.from_numpy(prompt_wav)
            prompt_wav = prompt_wav.to(kwargs["device"])
            prompt_wav_lengths = torch.tensor([prompt_wav.shape[1]], dtype=torch.int64)

            if self.pad_left:
                stride_lens = self.subsampling_factor * self.mel_extractor.hop_length
                prompt_wav_lengths = (prompt_wav_lengths + stride_lens - 1) // stride_lens * stride_lens
                zero_pad = torch.zeros(prompt_wav.size(0), prompt_wav_lengths.max()-prompt_wav.size(1)).to(prompt_wav.device)
                #prompt_wav = torch.cat([prompt_wav, zero_pad], dim=1)
                prompt_wav = torch.cat([zero_pad, prompt_wav], dim=1)
                prompt_mel, prompt_mel_lengths = self.extract_feat(prompt_wav.float(), prompt_wav_lengths)
            else:
                prompt_mel, prompt_mel_lengths = self.extract_feat(prompt_wav.float(), prompt_wav_lengths)
                prompt_mel_lengths = prompt_mel_lengths//self.subsampling_factor*self.subsampling_factor
                prompt_mel = prompt_mel[:,:prompt_mel_lengths.max(),:]
        else:
            prompt_wav = None
            prompt_mel = None
            prompt_mel_lengths = 0

        # single turn
        raw_text = ""
        sys_instruct = contents["system"][0] + "<|endofprompt|>"
        text = contents["user"][0]
        if text.startswith("语音合成："):
            text = text.replace("语音合成：", "")
        if text.startswith("Speech Synthesis:"):
            text = text.replace("Speech Synthesis:", "")

        if streaming_mode == "false":
            input_source_ids = tokenizer.encode(text)
            raw_text = raw_text + text
            if prompt_text is not None and lm_use_prompt:
                input_source_ids = tokenizer.encode(prompt_text) + input_source_ids
                raw_text = prompt_text + raw_text
            if kwargs.get("use_sys_instruct", False):
                hint_once("use_sys_instruct: True", "use_sys_instruct")
                input_source_ids = tokenizer.encode(sys_instruct) + input_source_ids
                raw_text = sys_instruct + raw_text
            all_text_len = len(input_source_ids)
            input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)[None, :]

        if prompt_text is not None and lm_use_prompt:
            prompt_text = tokenizer.encode(prompt_text)
            prompt_text = torch.tensor(prompt_text, dtype=torch.int64)[None, :]
            prompt_text_lengths = torch.tensor([prompt_text.shape[1]], dtype=torch.int64)
        else:
            prompt_text = None
            prompt_text_lengths = 0

        text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
        text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.int64)
        instruct_ids = torch.tensor(tokenizer.encode(sys_instruct), dtype=torch.int64)[None, :]
        instruct_lengths = torch.tensor([instruct_ids.shape[1]], dtype=torch.int64)

        spk_emb = kwargs.get("spk_emb", None)
        spk_emb_lengths = None
        if spk_emb is not None:
            if isinstance(spk_emb, str) and os.path.exists(spk_emb):
                spk_emb = np.load(spk_emb)
            if isinstance(spk_emb, np.ndarray):
                spk_emb = torch.from_numpy(spk_emb)[None, :]
            spk_emb_lengths = torch.tensor([spk_emb.shape[1]], dtype=torch.int64)

        logging.info(f"generating for {uttid}: input text: {raw_text}, input ids: {input_source_ids}")
        output = {
            "source_ids": input_source_ids,
            "text": text_ids,
            "text_lengths": text_lengths,
            "prompt_text": prompt_text,
            "prompt_text_lengths": prompt_text_lengths,
            "prompt_mel": prompt_mel,
            "prompt_mel_lengths": prompt_mel_lengths,
            "prompt_wav": prompt_wav,
            "prompt_wav_lengths": prompt_wav_lengths, 
            "sys_instruct": instruct_ids,
            "sys_instruct_lengths": instruct_lengths,
            "xvec": spk_emb,
            "xvec_lengths": spk_emb_lengths,
        }

        return output

    def inference_prepare(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            **kwargs,
    ):
        uttid = key[0]
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        kwargs.update(**contents["meta_dict"])
        kwargs["uttid"] = uttid
        output = self.load_data(
            contents, **kwargs
        )
        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]
        prompt_mel = batch["prompt_mel"]
        prompt_mel_lengths = batch["prompt_mel_lengths"]
        text_embeder = self.llm.model.get_input_embeddings()
        inputs_text_embeds = text_embeder(source_ids)

        if prompt_mel is not None:
            encoder_out, encoder_mask = self.downsample_encoder(prompt_mel, prompt_mel_lengths)
            encoder_out_lengths = encoder_mask.sum(dim=2).squeeze(dim=1)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        turn_of_speech_emb = self.llm_embedding.weight[self.turn_of_speech].reshape(1, 1, -1)

        if self.llm_use_global_emb:
            #segments, seg_mask = self.extract_segment_features(prompt_mel, prompt_mel_lens)
            segments = prompt_mel
            max_seg_len = prompt_mel_lengths.max().item()
            seg_mask = torch.ones(segments.shape[0], max_seg_len, dtype=torch.bool, device=prompt_mel.device)
            for i in range(segments.shape[0]):
                seg_len = prompt_mel_lengths[i].item()
                seg_mask[i, :seg_len] = False

        if self.llm_use_global_emb:
            global_emb_llm = self.seg_transformer_llm(segments, seg_mask).unsqueeze(dim=1)
        else:
            global_emb_llm = None

        if self.lm_use_spk_emb:
            spk_emb = batch["xvec"]
            spk_emb_len = batch["xvec_lengths"]

            spk_emb_llm = spk_emb[:, :spk_emb_len.max()]
            spk_emb_llm = self.norm_spk_emb(spk_emb_llm.reshape(1, 1, -1)).float()
            spk_emb_llm = self.xvec_proj_llm(spk_emb_llm).unsqueeze(dim=1)

            if prompt_mel is not None:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, global_emb_llm, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
            else:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, global_emb_llm, inputs_text_embeds, turn_of_speech_emb], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, inputs_text_embeds, turn_of_speech_emb], dim=1)
        else:
            if prompt_mel is not None:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, global_emb_llm, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
            else:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, global_emb_llm, inputs_text_embeds, turn_of_speech_emb], dim=1)
                else: 
                    inputs_embeds = torch.concat([sos_emb, inputs_text_embeds, turn_of_speech_emb], dim=1)
        return inputs_embeds, batch

    @staticmethod
    def get_avoid_token(avoid_tokens):
        if avoid_tokens is None:
            pass
        elif isinstance(avoid_tokens, list):
            pass
        elif isinstance(avoid_tokens, str):
            if "-" in avoid_tokens:
                st, ed = avoid_tokens.split('-')
                avoid_tokens = list(range(int(st), int(ed)))
                return avoid_tokens
            elif "," in avoid_tokens:
                avoid_tokens = [int(x) for x in avoid_tokens.split(",")]
            else:
                avoid_tokens = [int(avoid_tokens)]
        elif isinstance(avoid_tokens, int):
            avoid_tokens = [avoid_tokens]
        else:
            raise ValueError(f"avoid_tokens must be a list or a string or a int, given {avoid_tokens}")

        return avoid_tokens

    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        uttid = key[0]
        streaming_mode = kwargs.get("lm_streaming", False)
        if isinstance(streaming_mode, bool):
            streaming_mode = str(streaming_mode).lower()
        inputs_emb, batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        text, text_lengths = batch["text"], batch["text_lengths"]
        prompt_text, prompt_text_lens = batch["prompt_text"], batch["prompt_text_lengths"]
        prompt_mel, prompt_mel_lens = batch["prompt_mel"], batch["prompt_mel_lengths"]

        if self.diff_use_global_emb or self.diff_use_global_emb:
            #segments, seg_mask = self.extract_segment_features(prompt_mel, prompt_mel_lens)
            segments = prompt_mel
            max_seg_len = prompt_mel_lens.max().item()
            seg_mask = torch.ones(segments.shape[0], max_seg_len, dtype=torch.bool, device=prompt_mel.device)
            for i in range(segments.shape[0]):
                seg_len = prompt_mel_lens[i].item()
                seg_mask[i, :seg_len] = False

        if self.diff_use_global_emb:
            global_emb = self.seg_transformer(segments, seg_mask)
        else:
            global_emb = None

        sys_instruct, instruct_lengths = batch["sys_instruct"], batch["sys_instruct_lengths"]
        
        max_length = kwargs.get("max_length", 60 * 25)
        if "max_length" in kwargs:
            kwargs.pop("max_length")
        min_length = kwargs.get("min_length", None)
        if "min_length" in kwargs:
            kwargs.pop("min_length")
        min_length_ratio = kwargs.get("min_length_ratio", None)
        max_length_ratio = kwargs.get("max_length_ratio", None)
        if min_length_ratio is not None and prompt_text_lens is not None and prompt_mel_lens is not None:
            min_length = int(prompt_mel_lens / prompt_text_lens * text_lengths * min_length_ratio)
        if max_length_ratio is not None and prompt_text_lens is not None and prompt_mel_lens is not None:
            max_length = int(prompt_mel_lens / prompt_text_lens * text_lengths * max_length_ratio)
        logging.info(f"{uttid}: min length: {min_length}, max length: {max_length}")

        spk_emb = batch["xvec"]
        spk_emb_len = batch["xvec_lengths"]
        if spk_emb is not None:
            spk_emb = spk_emb[:, :spk_emb_len.max()]
            spk_emb = self.norm_spk_emb(spk_emb.reshape(1, 1, -1)).float()
            spk_emb = self.xvec_proj(spk_emb).unsqueeze(dim=1)

        use_llm_cache = kwargs.get("use_llm_cache", True)
        states = kwargs.get("states", {})
        llm_cache = states.get("llm_cache", None)

        diff_input_h = None

        diff_input_mel = torch.zeros(1, self.left_size, 80).to(inputs_emb.device)

        if prompt_mel is not None:
            diff_input_mel = torch.cat((diff_input_mel, prompt_mel), dim=1)[:,-self.left_size:, :]

        output = None

        for t in range(max_length):
        #for t in range(50):
            with torch.cuda.amp.autocast(
                enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype] 
            ):
                outputs = self.llm.model(
                    input_ids=None,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=llm_cache,
                    inputs_embeds=inputs_emb,
                    use_cache=use_llm_cache,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=None,
                )
                lm_hidden_states = outputs.hidden_states[-1]
                logp = self.binary_decoder(lm_hidden_states[:, -1]).log_softmax(dim=-1)
                # logp = h.log_softmax(dim=-1).squeeze(0)
                eos = (logp.argmax().item() == 1)

                if t>0 and eos is True:
                    break

                if diff_input_h is None:
                    diff_input_h = torch.zeros(1, self.left_size, self.upsample_encoder.out_channels).to(lm_hidden_states.device)
                    hidden, _ = self.upsample_encoder(lm_hidden_states[:, -(1+prompt_mel_lens//self.subsampling_factor):], None)
                else:
                    hidden, _ = self.upsample_encoder(lm_hidden_states[:, -1:], None)
          
                diff_input_h = torch.cat((diff_input_h, hidden), dim=1)[:,-(self.left_size+self.subsampling_factor):, :]
                pred_mel = self.diffusion_infer(
                    mu=diff_input_h.contiguous(),
                    spks=spk_emb,
                    n_timesteps=10,
                    mel_history=diff_input_mel.contiguous() if self.left_size>0 else None,
                    global_emb=global_emb,
                )

                pred_mel = pred_mel[:,-self.subsampling_factor:,:]

                if output is None:
                    output = pred_mel
                else:
                    output = torch.concat([output, pred_mel], dim=1)
                
                diff_input_mel = torch.cat((diff_input_mel, pred_mel), dim=1)[:,-self.left_size:, :]
                this_input_embed, _ = self.downsample_encoder(pred_mel, torch.IntTensor([pred_mel.shape[1]]).to(lm_hidden_states.device))

                if use_llm_cache:
                    llm_cache = outputs.past_key_values

                if use_llm_cache:
                    inputs_emb = this_input_embed
                else:
                    inputs_emb = torch.cat([
                        inputs_emb,
                        this_input_embed,
                    ], dim=1)

        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, "mel")
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"{key[0]}.npy"),
                output[0].cpu().numpy()
            )
        return output
