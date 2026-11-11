import os.path

import torch
import random
import torch.nn as nn
from funasr.register import tables
from typing import Dict, Optional
import logging
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from torch.cuda.amp import autocast
from funasr.train_utils.device_funcs import force_gatherable
from funasr.train_utils.device_funcs import to_device
import numpy as np
from funasr.utils.load_utils import load_audio_text_image_video
import time
from funasr.models.cosyvoice.utils import dtype_map


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


class LookaheadBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels, channels,
            kernel_size=pre_lookahead_len+1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, in_channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs, ilens):
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(ilens).unsqueeze(-1).to(inputs.device))
        # residual connection
        outputs = (outputs + inputs) * mask

        return outputs, ilens


@tables.register("model_classes", "CosyVoiceFlowMatchingOld")
class CosyVoiceFlowMatching(nn.Module):
    def __init__(
            self,
            codebook_size: int,
            model_size: int,
            xvec_size: int = 198,
            dit_conf: Dict = {},
            mel_feat_conf: Dict = {},
            prompt_conf: Dict = None,
            **kwargs):
        super().__init__()

        # feat related
        self.feat_token_ratio = kwargs.get("feat_token_ratio", None)
        self.mel_extractor = Audio2Mel(**mel_feat_conf)
        self.mel_norm_type = kwargs.get("mel_norm_type", None)
        self.num_mels = num_mels = mel_feat_conf["n_mel_channels"]
        self.token_rate = kwargs.get("token_rate", 25)
        self.sample_rate = self.mel_extractor.sampling_rate

        # condition related
        self.prompt_conf = prompt_conf
        if self.prompt_conf is not None:
            self.prompt_masker = self.build_prompt_masker()

        # codec related
        self.codec_embedder = nn.Embedding(codebook_size, num_mels)
        lookahead_length = kwargs.get("lookahead_length", 4)
        self.lookahead_conv1d = LookaheadBlock(num_mels, model_size, lookahead_length)

        # spk embed related
        if xvec_size is not None:
            self.xvec_proj = torch.nn.Linear(xvec_size, num_mels)

        # dit model related
        self.dit_conf = dit_conf
        from funasr.models.cosyvoice.modules.dit_model import DiT
        self.dit_model = DiT(**dit_conf)

        self.training_cfg_rate = kwargs.get("training_cfg_rate", 0)
        self.only_mask_loss = kwargs.get("only_mask_loss", True)

    def build_prompt_masker(self):
        prompt_type = self.prompt_conf.get("prompt_type", "free")
        if prompt_type == "prefix":
            from funasr.models.cosyvoice.utils.mask_along_axis import MaskTailVariableMaxWidth
            masker = MaskTailVariableMaxWidth(
                mask_width_ratio_range=self.prompt_conf["prompt_width_ratio_range"],
            )
        else:
            raise NotImplementedError

        return masker

    @staticmethod
    def norm_spk_emb(xvec):
        xvec_mask = (~xvec.norm(dim=-1).isnan()) * (~xvec.norm(dim=-1).isinf())
        xvec = xvec * xvec_mask.unsqueeze(-1)
        xvec = xvec.mean(dim=1)
        xvec = F.normalize(xvec, dim=1)

        return xvec

    def select_target_prompt(self, y: torch.Tensor, y_lengths: torch.Tensor):
        # cond_mask: 1, 1, 1, ..., 0, 0, 0
        cond_mask = self.prompt_masker(y, y_lengths, return_mask=True)

        return cond_mask

    @torch.no_grad()
    def normalize_mel_feat(self, feat, feat_lengths):
        # feat in B,T,D
        if self.mel_norm_type == "mean_std":
            max_length = feat.shape[1]
            mask = (~make_pad_mask(feat_lengths, maxlen=max_length))
            mask = mask.unsqueeze(-1).to(feat)
            mean = ((feat * mask).sum(dim=(1, 2), keepdim=True) /
                    (mask.sum(dim=(1, 2), keepdim=True) * feat.shape[-1]))
            var = (((feat - mean)**2 * mask).sum(dim=(1, 2), keepdim=True) /
                   (mask.sum(dim=(1, 2), keepdim=True) * feat.shape[-1] - 1))  # -1 for unbiased estimation
            std = torch.sqrt(var)
            feat = (feat - mean) / std
            feat = feat * mask
            return feat
        if self.mel_norm_type == "min_max":
            bb, tt, dd = feat.shape
            mask = (~make_pad_mask(feat_lengths, maxlen=tt))
            mask = mask.unsqueeze(-1).to(feat)
            feat_min = (feat * mask).reshape([bb, tt * dd]).min(dim=1, keepdim=True).values.unsqueeze(-1)
            feat_max = (feat * mask).reshape([bb, tt * dd]).max(dim=1, keepdim=True).values.unsqueeze(-1)
            feat = (feat - feat_min) / (feat_max - feat_min)
            # noise ~ N(0, I), P(x >= 3sigma) = 0.001, 3 is enough.
            feat = (feat * 3) * mask  # feat in [-3, 3]
            return feat
        else:
            raise NotImplementedError

    @torch.no_grad()
    def extract_feat(self, y: torch.Tensor, y_lengths: torch.Tensor):
        feat = self.mel_extractor(y)
        feat = feat.transpose(1, 2)
        feat_lengths = (y_lengths / self.mel_extractor.hop_length).to(y_lengths)
        if self.mel_norm_type is not None:
            feat = self.normalize_mel_feat(feat, feat_lengths)
        return feat, feat_lengths

    def flow_matching_forward(self, x1, mu, spks, cond, x1_len):
        """
        x1: B,T,n_mel
        mu: B,T,n_mel
        spks: B,1,n_mel
        cond: B,T,n_mel
        xs_len: B
        """
        b, t, _ = mu.shape
        t = torch.rand([b], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * z + t[:, None, None] * x1
        u = x1 - z

        if spks is not None:
            cond = cond + spks
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) > self.training_cfg_rate
            mu = mu * cfg_mask
            cond = cond * cfg_mask

        mask = (~make_pad_mask(x1_len)).to(mu.device)
        u_hat = self.dit_model(x_t, cond, mu, t, mask)
        loss = F.l1_loss(u, u_hat, reduction="none") * mask.unsqueeze(-1)

        return loss, u_hat

    def forward(
            self,
            codec: torch.Tensor,
            codec_len: torch.Tensor,
            audio: torch.Tensor,
            audio_len: torch.Tensor,
            spk_emb: Optional[torch.Tensor] = None,
            spk_emb_len: Optional[torch.Tensor] = None,
            **kwargs
    ):
        batch_size = audio.shape[0]
        codec_len = codec_len.squeeze(1)
        audio_len = audio_len.squeeze(1)
        spk_emb_len = spk_emb_len.squeeze(1)
        # for data parallel
        codec = codec[:, :codec_len.max()]
        audio = audio[:, :audio_len.max()]

        with autocast(False):
            mask = (codec != -1).float().unsqueeze(-1)
            codec_emb = self.codec_embedder(torch.clamp(codec, min=0)) * mask

            if spk_emb is not None:
                spk_emb = spk_emb[:, :spk_emb_len.max()]
                spk_emb = self.norm_spk_emb(spk_emb)

            feat, feat_lengths = self.extract_feat(audio, audio_len)

        if spk_emb is not None:
            spk_emb = self.xvec_proj(spk_emb).unsqueeze(dim=1)

        if self.prompt_conf is not None:
            cond_mask = self.prompt_masker(feat, feat_lengths)
            conditions = feat * cond_mask
        else:
            cond_mask, conditions = None, None

        stats = dict(
            batch_size=batch_size,
            in_lengths=codec_len.max(),
            out_lengths=feat_lengths.max(),
        )

        mask = (~make_pad_mask(feat_lengths)).to(feat)
        # y, h in (B, T, D)
        mu, _ = self.lookahead_conv1d(codec_emb, codec_len)
        mu = mu.repeat_interleave(self.feat_token_ratio, dim=1)
        loss, u_hat = self.flow_matching_forward(feat, mu, spk_emb, conditions, feat_lengths)

        all_loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * loss.shape[-1])
        if cond_mask is not None:
            loss_mask = mask.unsqueeze(-1) * (~cond_mask)
        else:
            loss_mask = mask.unsqueeze(-1)
        masked_loss = (loss * loss_mask).sum() / (loss_mask.sum() * loss.shape[-1])
        stats["all_loss"] = all_loss.item()
        stats["masked_loss"] = masked_loss.item()

        loss = masked_loss if self.only_mask_loss else all_loss

        stats["loss"] = loss.item()

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
        tokenizer = kwargs.get("tokenizer")
        # single turn
        text = contents["user"][0]
        text = text.replace("语音合成：", "")
        text = text.replace("Speech Synthesis:", "")
        text = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
        text_lengths = torch.tensor([text.shape[1]], dtype=torch.int64)

        codec = contents["assistant"][0]
        if isinstance(codec, str) and codec.startswith("<|startofspeech|>!"):
            codec = codec.replace("<|startofspeech|>!", "")
            codec = codec.replace("<|endofspeech|>", "")
            codec = np.load(codec)[None, :]
        if isinstance(codec, np.ndarray):
            codec = torch.from_numpy(codec)
        codec_lengths = torch.tensor([codec.shape[1]], dtype=torch.int64)

        prompt_text = kwargs.get("user_prompt_text", None)
        prompt_text_lengths = None
        if prompt_text is not None:
            prompt_text = tokenizer.encode(prompt_text)
            prompt_text = torch.tensor(prompt_text, dtype=torch.int64)[None, :]
            prompt_text_lengths = torch.tensor([prompt_text.shape[1]], dtype=torch.int64)

        prompt_codec = kwargs.get("prompt_codec", None)
        prompt_codec_lengths = None
        if prompt_codec is not None:
            if isinstance(prompt_codec, str) and os.path.exists(prompt_codec):
                prompt_codec = np.load(prompt_codec)
            if isinstance(prompt_codec, np.ndarray):
                prompt_codec = torch.from_numpy(prompt_codec)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)

        prompt_wav = kwargs.get("prompt_wav", None)
        prompt_wav_lengths = None
        if prompt_wav is not None:
            if isinstance(prompt_wav, str) and os.path.exists(prompt_wav):
                prompt_wav = load_audio_text_image_video(prompt_wav, fs=self.sample_rate)
                prompt_wav = prompt_wav[None, :]
            if isinstance(prompt_wav, np.ndarray):
                prompt_wav = torch.from_numpy(prompt_wav)
            prompt_wav_lengths = torch.tensor([prompt_wav.shape[1]], dtype=torch.int64)

        if prompt_wav is not None and prompt_codec is not None:
            exp_audio_len = int(prompt_codec.shape[1] / self.token_rate * self.sample_rate)
            if prompt_wav.shape[1] > exp_audio_len:
                prompt_wav = prompt_wav[:, :exp_audio_len]
            elif prompt_wav.shape[1] < exp_audio_len:
                exp_token_len = int(prompt_wav.shape[1] / self.sample_rate * self.token_rate)
                prompt_codec = prompt_codec[:, :exp_token_len]
                exp_audio_len = int(prompt_codec.shape[1] / self.token_rate * self.sample_rate)
                prompt_wav = prompt_wav[:, :exp_audio_len]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
            prompt_wav_lengths = torch.tensor([prompt_wav.shape[1]], dtype=torch.int64)

        spk_emb = kwargs.get("spk_emb", None)
        spk_emb_lengths = None
        if spk_emb is not None:
            if isinstance(spk_emb, str) and os.path.exists(spk_emb):
                spk_emb = np.load(spk_emb)
            if isinstance(spk_emb, np.ndarray):
                spk_emb = torch.from_numpy(spk_emb)[None, :]
            spk_emb_lengths = torch.tensor([spk_emb.shape[1]], dtype=torch.int64)

        output = {
            "text": text,
            "text_lengths": text_lengths,
            "codec": codec,
            "codec_lengths": codec_lengths,
            "prompt_text": prompt_text,
            "prompt_text_lengths": prompt_text_lengths,
            "prompt_codec": prompt_codec,
            "prompt_codec_lengths": prompt_codec_lengths,
            "prompt_wav": prompt_wav,
            "prompt_wav_lengths": prompt_wav_lengths,
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
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        kwargs.update(**contents["meta_dict"])
        batch = self.load_data(
            contents, **kwargs
        )
        batch = to_device(batch, kwargs["device"])

        return batch

    @torch.no_grad()
    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            **kwargs,
    ):
        uttid = key[0]
        batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        feat = self._inference(**batch, **kwargs)
        feat = feat.float()
        logging.info(f"{uttid}: feat lengths {feat.shape[1]}")

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            feat_out_dir = os.path.join(output_dir, "feat")
            os.makedirs(feat_out_dir, exist_ok=True)
            np.save(os.path.join(feat_out_dir, f"{key[0]}.npy"), feat[0].cpu().numpy())

        return feat

    @torch.no_grad()
    def _inference(
            self,
            codec, codec_lengths,
            prompt_codec=None, prompt_codec_lengths=None,
            prompt_wav=None, prompt_wav_lengths=None,
            xvec=None, xvec_lengths=None,
            **kwargs
    ):
        fm_dtype = dtype_map[kwargs.get("fm_dtype", "fp32")]
        rand_xvec = None
        if xvec is not None:
            if xvec.dim() == 2:
                xvec = xvec.unsqueeze(1).to(fm_dtype)
                xvec_lens = torch.ones_like(xvec_lengths)
            rand_xvec = self.norm_spk_emb(xvec)
            rand_xvec = self.xvec_proj(rand_xvec)

        if prompt_codec is not None:
            codec, codec_lengths = self.concat_prompt(prompt_codec, prompt_codec_lengths, codec, codec_lengths)
        mask = (codec != -1).float().unsqueeze(-1)
        codec_emb = self.codec_embedder(torch.clamp(codec, min=0)) * mask

        mu, _ = self.lookahead_conv1d(codec_emb, codec_lengths)
        mu = mu.repeat_interleave(self.feat_token_ratio, dim=1)

        conditions = torch.zeros([1, mu.shape[1], self.num_mels]).to(mu)
        # get conditions
        if prompt_wav is not None:
            if prompt_wav.ndim == 2:
                prompt_wav, prompt_wav_lengths = self.extract_feat(prompt_wav, prompt_wav_lengths)
            prompt_wav = prompt_wav.to(fm_dtype)
            for i, _len in enumerate(prompt_wav_lengths):
                conditions[i, :_len] = prompt_wav[i]

        feat_lengths = codec_lengths * self.feat_token_ratio
        mask = (~make_pad_mask(feat_lengths)).to(mu.device)
        feat = self.solve_ode(mu, rand_xvec, conditions, mask, **kwargs)

        if prompt_codec is not None and prompt_wav is not None:
            feat, feat_lens = self.remove_prompt(None, prompt_wav_lengths, feat, feat_lengths)

        return feat

    @staticmethod
    def concat_prompt(prompt, prompt_lengths, text, text_lengths):
        xs_list, x_len_list = [], []
        for idx, (_prompt_len, _text_len) in enumerate(zip(prompt_lengths, text_lengths)):
            xs_list.append(torch.concat([prompt[idx, :_prompt_len], text[idx, :_text_len]], dim=0))
            x_len_list.append(_prompt_len + _text_len)

        xs = torch.nn.utils.rnn.pad_sequence(xs_list, batch_first=True, padding_value=0.0)
        x_lens = torch.tensor(x_len_list, dtype=torch.int64).to(xs.device)

        return xs, x_lens

    @staticmethod
    def remove_prompt(prompt, prompt_lengths, padded, padded_lengths):
        xs_list = []
        for idx, (_prompt_len, _x_len) in enumerate(zip(prompt_lengths, padded_lengths)):
            xs_list.append(padded[idx, _prompt_len: _x_len])

        xs = torch.nn.utils.rnn.pad_sequence(xs_list, batch_first=True, padding_value=0.0)

        return xs, padded_lengths - prompt_lengths

    def get_rand_noise(self, mu: torch.Tensor, **kwargs):
        use_fixed_noise_infer = kwargs.get("use_fixed_noise_infer", True)
        max_len = kwargs.get("max_len", 50*300)
        if use_fixed_noise_infer:
            if not hasattr(self, "rand_noise") or self.rand_noise is None or self.rand_noise.shape[2] < mu.shape[2]:
                self.rand_noise = torch.randn([1, max_len, mu.shape[2]]).to(mu)
                logging.info("init random noise for Flow")
            return self.rand_noise[:, :mu.shape[1], :]
        else:
            return torch.randn_like(mu)

    def solve_ode(self, mu, rand_xvec, conditions, mask, **kwargs):
        temperature = kwargs.get("temperature", 1.0)
        n_timesteps = kwargs.get("n_timesteps", 10)
        infer_t_scheduler = kwargs.get("infer_t_scheduler", "cosine")

        z = self.get_rand_noise(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(mu)
        if infer_t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        fm_time = time.time()
        feat = self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask,
            spks=rand_xvec, cond=conditions, **kwargs
        )
        escape_time = (time.time() - fm_time) * 1000.0
        logging.info(f"fm dec {n_timesteps} step time: {escape_time:.2f}, avg {escape_time/n_timesteps:.2f} ms")
        return feat

    def solve_euler(self, x, t_span, mu, mask, spks=None, cond=None, **kwargs):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        inference_cfg_rate = kwargs.get("inference_cfg_rate", 0.7)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        steps = 1
        z, bz = x, x.shape[0]
        while steps <= len(t_span) - 1:
            if inference_cfg_rate > 0:
                x_in = torch.concat([x, x], dim=0)
                mask_in = torch.concat([mask, mask], dim=0)
                mu_in = torch.concat([mu, torch.zeros_like(mu)], dim=0)
                t_in = torch.concat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                if isinstance(cond, torch.Tensor):
                    cond_in = torch.concat([cond, torch.zeros_like(cond)], dim=0)
                else:
                    cond_in = dict(
                        prompt=[
                            torch.concat([cond["prompt"][0], torch.zeros_like(cond["prompt"][0])], dim=0),
                            torch.concat([cond["prompt"][1], cond["prompt"][1]], dim=0),
                        ]
                    )
            else:
                x_in, mask_in, mu_in, t_in, cond_in = x, mask, mu, t, cond

            if spks is not None:
                cond_in = cond_in + spks

            dphi_dt = self.dit_model(x_in, cond_in, mu_in, t_in, mask_in)
            if inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [bz, bz], dim=0)
                dphi_dt = ((1.0 + inference_cfg_rate) * dphi_dt -
                           inference_cfg_rate * cfg_dphi_dt)

            x = x + dt * dphi_dt
            t = t + dt
            # sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return x
