import os.path
import torch
import torch.nn as nn
from funasr.register import tables
from typing import Dict, Optional
import logging
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.train_utils.device_funcs import force_gatherable
from funasr.train_utils.device_funcs import to_device
import numpy as np
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.train_utils.set_all_random_seed import set_all_random_seed
import time, random
from funasr.models.cosyvoice.utils import dtype_map
from funasr.utils.hinter import hint_once
from funasr.models.cosyvoice.utils.masks import add_optional_chunk_mask
from funasr.models.cosyvoice.utils.common import causal_linear_interpolation, CausalUpsample
from funasr.utils.oss_io import oss_remote_numpy_load


# NOTE 这一段是从matcha-tts copy来的，要确保vocoder特征和这个一致
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
        self.mel_fmax = mel_fmax
        self.center = center
        self.feat_type = feat_type

    def forward(self, audioin):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        # NOTE: torch.stft on CUDA may trigger CUFFT_INTERNAL_ERROR on certain
        # GPU/driver combinations; move computation to CPU to avoid this.
        orig_device = audio.device
        audio_cpu = audio.cpu()
        window_cpu = self.window.cpu()
        fft = torch.stft(
            audio_cpu,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_cpu,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).to(orig_device)
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

    def forward(self, inputs, ilens, context: torch.Tensor = torch.zeros(0, 0, 0)):
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        # look ahead
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0)
        else:
            assert context.size(2) == self.pre_lookahead_len
            outputs = torch.concat([outputs, context], dim=2)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(ilens).unsqueeze(-1).to(inputs.device))
        # residual connection
        outputs = (outputs + inputs) * mask

        return outputs, ilens


@tables.register("model_classes", "CosyVoiceFlowMatching")
class CosyVoiceFlowMatching(nn.Module):
    def __init__(
            self,
            codebook_size: int,
            model_size: int,
            xvec_size: int = 198,
            dit_conf: Dict = {},
            mel_feat_conf: Dict = {},
            **kwargs):
        super().__init__()

        # feat related
        self.feat_token_ratio = kwargs.get("feat_token_ratio", None)
        self.mel_extractor = Audio2Mel(**mel_feat_conf)
        self.sample_rate = self.mel_extractor.sampling_rate
        self.num_mels = num_mels = mel_feat_conf["n_mel_channels"]
        self.token_rate = kwargs.get("token_rate", 25)
        self.codebook_size = codebook_size

        # codec related
        self.embed_dim = kwargs.get('embed_dim', 80)
        self.codec_embedder = nn.Embedding(codebook_size, self.embed_dim)
        lookahead_length = kwargs.get("lookahead_length", 3)
        self.lookahead_conv1d = LookaheadBlock(self.embed_dim, model_size, lookahead_length)

        # spk embed related
        self.xvec_proj = torch.nn.Linear(xvec_size, num_mels)

        # dit model related
        self.dit_conf = dit_conf
        self.flash_attn = kwargs.get("flash_attn", False)
        if self.flash_attn is False:
            from .modules.dit_flow_matching.dit_model import DiT as DiT
        else:
            from .modules.dit_flow_matching.dit_model import FlashAttnDiT as DiT
        self.dit_model = DiT(**dit_conf)

        self.training_cfg_rate = kwargs.get("training_cfg_rate", 0)
        self.only_mask_loss = kwargs.get("only_mask_loss", True)

        # NOTE(lyuxiang.lx) fm需要右看的下文
        self.context_size = self.lookahead_conv1d.pre_lookahead_len
        self.unified = kwargs.get('unified', False)
        self.attn_type = kwargs.get('attn_type', 'v1')
        self.upsample_method = kwargs.get('upsample_method', 'repeat')
        self.max_vol = kwargs.get('max_vol', 1.0)
        self.no_prompt_ratio = kwargs.get('no_prompt_ratio', 0.5)
        self.no_spk_emb_prompt_ratio = kwargs.get('no_spk_emb_prompt_ratio', 0.0)
        self.max_prompt_ratio = kwargs.get('max_prompt_ratio', 0.3)
        self.max_prompt_num_token = kwargs.get('max_prompt_num_token', -1)
        self.speech_token_extractor_conf = kwargs.get('speech_token_extractor_conf', {}) or {}
        self.speech_token_extractor_enabled = self.speech_token_extractor_conf.get("enabled", False)
        if self.speech_token_extractor_enabled:
            self._init_speech_token_extractor()
        if self.upsample_method == 'causal_upsample':
            self.upsample_layer = CausalUpsample(self.embed_dim, ratio=self.feat_token_ratio)

    @staticmethod
    def norm_spk_emb(xvec):
        xvec_mask = (~xvec.norm(dim=-1).isnan()) * (~xvec.norm(dim=-1).isinf())
        xvec = xvec * xvec_mask.unsqueeze(-1)
        xvec = xvec.mean(dim=1)
        xvec = F.normalize(xvec, dim=1)
        return xvec

    @torch.no_grad()
    def extract_feat(self, y: torch.Tensor, y_lengths: torch.Tensor):
        mel_extractor = self.mel_extractor.float()
        max_val = y.abs().max(dim=1)[0]
        ratio = (self.max_vol / max_val).clip(max=1)
        y = y * ratio.unsqueeze(dim=1)
        feat = mel_extractor(y)
        feat = feat.transpose(1, 2)
        feat_lengths = (y_lengths / self.mel_extractor.hop_length).to(y_lengths)
        return feat, feat_lengths

    def _init_speech_token_extractor(self):
        self.speech_token_extractor_type = self.speech_token_extractor_conf.get("type", "automodel")
        if self.speech_token_extractor_type == "automodel":
            print (f'SpeechTokenExtraction use automodel')
            from funasr.auto.auto_model import AutoModel
            automodel_conf = self.speech_token_extractor_conf.get("automodel_conf", {})
            if "model" not in automodel_conf:
                raise ValueError("speech_token_extractor_conf.automodel_conf.model is required when speech token extraction is enabled")
            self.speech_token_extractor = AutoModel(**automodel_conf)
            device = "cuda"
            self.speech_token_extractor.model = self.speech_token_extractor.model.to(device)
            self.speech_token_extractor.model.eval()
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")

    def _extract_speech_token(self, audio_feat=None, audio_feat_lengths=None, audio=None):
        if audio is not None:
            audio_feat, audio_feat_lengths = extract_fbank(
                audio,
                data_type="sound",
                frontend=self.frontend,
                is_final=True
            )
        else:
            audio_feat = audio_feat.permute(0, 2, 1)
            audio_feat_lengths = audio_feat_lengths[:, 0]

        if self.speech_token_extractor_type == "automodel":
            with torch.no_grad():
                token_result, _ = self.speech_token_extractor.model.extract_token(audio_feat,
                                        data_lengths=audio_feat_lengths,
                                        data_type="fbank",
                                        key=["any"],
                                        device="cuda")
                codec = token_result[0]["token"]
                codec_len = token_result[0]["token_lengths"]
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")
        return codec, codec_len

    def flow_matching_forward(self, x1, mu, spks, cond, x1_len, cond_mask, **kwargs):
        """
        x1: B,T,n_mel
        mu: B,T,n_mel
        spks: B,1,n_mel
        cond: B,T,n_mel
        xs_len: B
        """
        # Ensure all inputs have the same dtype as the model
        dtype = next(self.dit_model.parameters()).dtype
        x1 = x1.to(dtype)
        mu = mu.to(dtype)
        spks = spks.to(dtype)
        cond = cond.to(dtype)

        b, t, _ = mu.shape
        t = torch.rand([b], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * z + t[:, None, None] * x1
        u = x1 - z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) > self.training_cfg_rate
            mu = mu * cfg_mask
            cond = cond * cfg_mask
            spks = spks * cfg_mask
            # NOTE cond_mask也要相应的调整
            for i in range(cfg_mask.shape[0]):
                if cfg_mask[i].item() is False:
                    cond_mask[i] = True

        mask = (~make_pad_mask(x1_len)).to(mu.device).unsqueeze(1)
        mask_rand = torch.rand((b, 1, 1), device=mu.device, dtype=torch.float32).to(mu.dtype)
        if self.attn_type == 'v1':
            causal_mask = self.dit_model.build_mix_causal_mask(mask.unsqueeze(1), rand=mask_rand.unsqueeze(-1))
            assert ((causal_mask[:, 0, -1].sum(dim=-1) - x1_len) == 0).all()
        elif self.attn_type == 'v2':
            # NOTE 支持num_left_chunk
            causal_mask = self.dit_model.build_mix_causal_mask_v2(mask.unsqueeze(1), rand=mask_rand.unsqueeze(-1))
        else:
            raise NotImplementedError

        # NOTE 验证因果性
        # self.eval()
        # a = self.dit_model(x_t, cond, mu, spks, t, mask=causal_mask.squeeze(1), attn_type=self.attn_type)[0].detach().cpu()
        # for i in range(x1_len.shape[0]):
        #     length=int(x1_len[i].item())
        #     b = self.dit_model(x_t[:,:length], cond[:,:length], mu[:,:length], spks, t, mask=causal_mask[:,:,:length,:length].squeeze(1), attn_type=self.attn_type)[0].detach().cpu()
        #     assert (a[i,:length]-b[i]).abs().sum() == 0
        u_hat, hidden_list = self.dit_model(x_t, cond, mu, spks, t, mask=causal_mask.squeeze(1), attn_type=self.attn_type)

        loss = F.l1_loss(u, u_hat, reduction="none") * mask.transpose(1, 2)
        return loss, u_hat, hidden_list, cond_mask

    def forward(
            self,
            audio: torch.Tensor,
            audio_len: torch.Tensor,
            spk_emb: Optional[torch.Tensor] = None,
            spk_emb_len: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch_size = audio.shape[0]
        if self.unified is True and 'codec_hidden' in kwargs:
            if 'codec_len' not in kwargs:
                raise RuntimeError(
                    "CosyVoiceFlowMatching requires codec_len with codec_hidden"
                )
            codec = None
            codec_len = kwargs['codec_len'].squeeze(1)
        elif 'codec' in kwargs:
            codec = kwargs['codec'].squeeze(1)
            codec_len = kwargs['codec_len'].squeeze(1)
        else:
            if not self.speech_token_extractor_enabled:
                raise RuntimeError(
                    "CosyVoiceFlowMatching requires codec/codec_len, or "
                    "codec_hidden/codec_len in unified mode, when "
                    "speech_token_extractor_conf is omitted or disabled"
                )
            whisper_feat = torch.nn.utils.rnn.pad_sequence([i[-1] for i in kwargs['audio_list']], batch_first=True, padding_value=0)
            whisper_feat_len = torch.tensor([i[-1] for i in kwargs['audio_len_list']])
            codec, codec_len = self._extract_speech_token(audio_feat=whisper_feat, audio_feat_lengths=whisper_feat_len.unsqueeze(1))
            samples_per_token = self.mel_extractor.hop_length * self.feat_token_ratio
            assert ((audio_len.flatten() / samples_per_token).int() == codec_len).all()

        audio_len = audio_len.squeeze(1)
        codec_seq_len = (
            kwargs['codec_hidden'].shape[1] if self.unified is True else codec.shape[1]
        )
        assert codec_len.max() == codec_seq_len and audio_len.max() == audio.shape[1] and (spk_emb_len==1).all()

        # token condition
        mask = (~make_pad_mask(codec_len)).unsqueeze(-1).to(codec_len)
        if self.unified is True:
            codec_emb = kwargs['codec_hidden'] * mask
            # Keep the discrete-codec embedding in the autograd graph with a
            # zero gradient.  Megatron overlap grad reduction requires every
            # trainable parameter in a bucket to make its gradient available,
            # while unified mode intentionally bypasses this embedding.
            codec_emb = codec_emb + self.codec_embedder.weight.sum() * 0.0
        else:
            codec_emb = self.codec_embedder(torch.clamp(codec, min=0)) * mask
        mu, _ = self.lookahead_conv1d(codec_emb, codec_len)
        if self.upsample_method == 'repeat':
            mu = mu.repeat_interleave(self.feat_token_ratio, dim=1)
        elif self.upsample_method == 'causal_linear':
            mu = causal_linear_interpolation(mu, self.feat_token_ratio)
        elif self.upsample_method == 'causal_upsample':
            mu = self.upsample_layer(mu)
        else:
            raise NotImplementedError

        # spk condition
        spk_emb = F.normalize(spk_emb, dim=-1)
        spk_emb = spk_emb.to(self.xvec_proj.weight.dtype)
        spk_emb = self.xvec_proj(spk_emb)
        for i in range(spk_emb.shape[0]):
            if random.random() < self.no_spk_emb_prompt_ratio:
                spk_emb[i] = 0

        # feat condition
        feat, feat_lengths = self.extract_feat(audio.float(), audio_len)
        conditions = torch.zeros(feat.shape, device=feat.device)
        cond_mask = (~make_pad_mask(feat_lengths)).unsqueeze(-1).to(feat.device)
        for i, j in enumerate(feat_lengths):
            # NOTE 一定概率没有mel cond，依赖hidden进行生成，这样基模型也可以进行sft推理，否则会有问题
            if random.random() < self.no_prompt_ratio:
                continue
            index = random.randint(0, int(self.max_prompt_ratio * j))
            if self.max_prompt_num_token != -1:
                index = min(index, self.max_prompt_num_token)
            conditions[i, :index] = feat[i, :index]
            cond_mask[i, :index] = False

        # loss summary
        stats = dict(
            batch_size=batch_size,
            in_lengths=codec_len.max(),
            out_lengths=feat_lengths.max(),
        )
        mask = (~make_pad_mask(feat_lengths)).to(feat)
        loss, u_hat, hidden_list, cond_mask = self.flow_matching_forward(feat, mu, spk_emb, conditions, feat_lengths, cond_mask=cond_mask)
        all_loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * loss.shape[-1])
        # NOTE 注意这里去掉了取反符，mask/connd_mask都是需要计算loss部分为true
        loss_mask = mask.unsqueeze(-1) * cond_mask
        masked_loss = (loss * loss_mask).sum() / (loss_mask.sum() * loss.shape[-1])
        loss = masked_loss if self.only_mask_loss else all_loss
        stats["loss"] = loss.item()
        stats["all_loss"] = all_loss.item()
        stats["masked_loss"] = masked_loss.item()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # NOTE(lyuxiang.lx) hidden部分惩罚过大值
        for i in range(len(hidden_list)):
            index = (hidden_list[i] * mask.unsqueeze(dim=-1)).abs() >= (i + 1) * 10 + 100
            if index.sum() == 0:
                continue
            loss = loss + ((hidden_list[i] * index).norm(dim=-1).sum() / index.sum()) * 1e-08 * (len(hidden_list) - 1)
        return loss, stats, weight

    def data_template(self, data, **kwargs):
        system, user, assistant, meta_dict = [], [], [], {}
        # fix fyu force gender, need to fix later
        if len(data) <=3 and 'gender' not in data[1] and data[0]['role'] == 'system' and "speaker" not in data[-1]:
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
        elif data[-1]['role'] == 'assistant':
            assistant_text, assistant_token, assistant_speaker = [], [], []
            user_text, user_token, user_speaker = [], [], []
            for i, item in enumerate(data):
                role = item["role"]
                if role != "system":
                    content = item["wav_path"]
                else:
                    content = item['content']
                if role == "system":
                    system.append(content)
                elif role == "user":
                    if "text_content" in item:
                        user.append(item['text_content'])
                        user_text.append(item['text_content'])
                    if "token" in item:
                        user_token.append(item['token'])
                    if "spk_emb" in item:
                        user_speaker.append(item['spk_emb'])
                    if kwargs.get("use_infer_conversation_use_firstP_2spk",False):
                        assistant.append(content)
                    if kwargs.get("use_infer_single_turn",False):
                        assistant.append(content)
                elif role == "assistant":
                    if "token" in item:
                        assistant_token.append(item['token'])
                    if "spk_emb" in item:
                        assistant_speaker.append(item['spk_emb'])
                    assistant.append(content)
                    if "text_content" in item:
                        assistant_text.append(item['text_content'])
            #assis prompt:倒数第二句
            user = assistant_text[-1:]


            if kwargs.get("use_infer_conversation_use_firstP_2spk",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict['prompt_codec'] = [user_token[0],assistant_token[0]]
                meta_dict["prompt_wav"] = assistant[:2]
                meta_dict["spk_emb"] = [user_speaker[0],assistant_speaker[0]]
            elif kwargs.get("use_infer_only_user",False):
                meta_dict["user_prompt_text"] = None
                meta_dict["prompt_codec"] = None
                meta_dict["prompt_wav"] = None
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_first_prompt",False) and kwargs.get("use_infer_single_turn",False):
                meta_dict["user_prompt_text"] = user_text[0]
                meta_dict["prompt_codec"] = user_token[0]
                meta_dict["prompt_wav"] = assistant[0]
                meta_dict["spk_emb"] = user_speaker[0]
            elif kwargs.get("use_infer_conversation_use_first_prompt",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict["prompt_codec"] = assistant_token[0]
                meta_dict["prompt_wav"] = assistant[0]
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_fmall_firstP",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict['prompt_codec'] = assistant_token[:-1]
                meta_dict["prompt_wav"] = assistant[0]
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_fmall",False):
                meta_dict["user_prompt_text"] = ''.join(assistant_text[:-1])
                meta_dict["prompt_codec"] = assistant_token[:-1]
                meta_dict["prompt_wav"] = assistant[:-1]
                meta_dict["spk_emb"] = assistant_speaker[0]
            else:
                meta_dict["user_prompt_text"] = assistant_text[-2]
                meta_dict["prompt_codec"] = assistant_token[-2]
                meta_dict["prompt_wav"] = assistant[-2]
                # meta_dict["spk_emb"] = assistant_speaker[-1] #bug 250819 这里spk emb应该用上一句
                meta_dict["spk_emb"] = assistant_speaker[-2]
            assistant = assistant[-1:]
        else:
            #for user sysn
            #user assistant 对换
            assistant_text, assistant_token, assistant_speaker = [], [], []
            user_text, user_token, user_speaker = [], [], []
            for i, item in enumerate(data):
                role = item["role"]
                if role != "system":
                    content = item["wav_path"]
                else:
                    content = item['content']
                if role == "system":
                    system.append(content)
                elif role == "assistant":
                    if "text_content" in item:
                        user.append(item['text_content'])
                        user_text.append(item['text_content'])
                    if "token" in item:
                        user_token.append(item['token'])
                    if "spk_emb" in item:
                        user_speaker.append(item['spk_emb'])
                    if kwargs.get("use_infer_conversation_use_firstP_2spk",False):
                        assistant.append(content)
                    if kwargs.get("use_infer_single_turn",False):
                        assistant.append(content)
                elif role == "user":
                    if "token" in item:
                        assistant_token.append(item['token'])
                    if "spk_emb" in item:
                        assistant_speaker.append(item['spk_emb'])
                    assistant.append(content)
                    if "text_content" in item:
                        assistant_text.append(item['text_content'])
            #assis prompt:倒数第二句
            user = assistant_text[-1:]
            if kwargs.get("use_infer_conversation_use_firstP_2spk",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict['prompt_codec'] = [user_token[0],assistant_token[0]]
                meta_dict["prompt_wav"] = assistant[:2]
                meta_dict["spk_emb"] = [user_speaker[0],assistant_speaker[0]]
            elif kwargs.get("use_infer_only_user",False):
                meta_dict["user_prompt_text"] = None
                meta_dict["prompt_codec"] = None
                meta_dict["prompt_wav"] = None
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_first_prompt",False) and kwargs.get("use_infer_single_turn",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict["prompt_codec"] = assistant_token[0]
                meta_dict["prompt_wav"] = assistant[0]
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_first_prompt",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict["prompt_codec"] = assistant_token[0]
                meta_dict["prompt_wav"] = assistant[0]
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_fmall_firstP",False):
                meta_dict["user_prompt_text"] = assistant_text[0]
                meta_dict['prompt_codec'] = assistant_token[:-1]
                meta_dict["prompt_wav"] = assistant[0]
                meta_dict["spk_emb"] = assistant_speaker[0]
            elif kwargs.get("use_infer_conversation_use_fmall",False):
                meta_dict["user_prompt_text"] = ''.join(assistant_text[:-1])
                meta_dict["prompt_codec"] = assistant_token[:-1]
                meta_dict["prompt_wav"] = assistant[:-1]
                meta_dict["spk_emb"] = assistant_speaker[0]
            else:
                meta_dict["user_prompt_text"] = assistant_text[-2]
                meta_dict["prompt_codec"] = assistant_token[-2]
                meta_dict["prompt_wav"] = assistant[-2]
                # meta_dict["spk_emb"] = assistant_speaker[-1] #bug 250819 这里spk emb应该用上一句
                meta_dict["spk_emb"] = assistant_speaker[-2]
            assistant = assistant[-1:]

        if len(system) == 0:
            system = ["You are a helpful assistant."]

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
            "meta_dict": meta_dict,
        }
        # import pdb;pdb.set_trace()
        return contents

    def load_data(self, contents: dict, **kwargs):
        # import pdb;pdb.set_trace()
        fm_use_prompt = kwargs.get("fm_use_prompt", True)
        tokenizer = kwargs.get("tokenizer")
        # single turn
        text = contents["user"][0]
        text_lengths = None
        if text and tokenizer is not None:
            text = text.replace("语音合成：", "")
            text = text.replace("Speech Synthesis:", "")
            text = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
            text_lengths = torch.tensor([text.shape[1]], dtype=torch.int64)

        codec = contents["assistant"][0]
        if isinstance(codec, str) and codec.startswith("<|startofspeech|>!"):
            codec = codec.replace("<|startofspeech|>!", "")
            codec = codec.replace("<|endofspeech|>", "")
            codec = np.load(codec)
            if len(codec.shape) == 1:
                codec = codec[None, :]
        if isinstance(codec, str) and not codec.startswith("<|startofspeech|>!"):
            # import pdb;pdb.set_trace()
            codec = np.load(codec)
            if len(codec.shape) == 1:
                codec = codec[None, :]
        if isinstance(codec, np.ndarray):
            codec = torch.from_numpy(codec)
        codec_lengths = torch.tensor([codec.shape[1]], dtype=torch.int64)

        prompt_text = kwargs.get("user_prompt_text", None)
        prompt_text_lengths = None
        if prompt_text is not None and tokenizer is not None:
            prompt_text = tokenizer.encode(prompt_text)
            prompt_text = torch.tensor(prompt_text, dtype=torch.int64)[None, :]
            prompt_text_lengths = torch.tensor([prompt_text.shape[1]], dtype=torch.int64)

        prompt_codec = kwargs.get("prompt_codec", None)
        prompt_codec_lengths = None
        if prompt_codec is not None and fm_use_prompt:
            if isinstance(prompt_codec, str) and os.path.exists(prompt_codec):
                prompt_codec = np.load(prompt_codec)
            elif isinstance(prompt_codec, list):
                features = []
                for p in prompt_codec:
                    if isinstance(p, str):
                        if os.path.exists(p):
                            feat = np.load(p)  # 每个 (T_i,)
                        else:
                            raise FileNotFoundError(f"Prompt codec file not found: {p}")
                    elif isinstance(p, np.ndarray):
                        feat = p
                    else:
                        raise TypeError(f"Unsupported type in prompt_codec list: {type(p)}")
                    features.append(feat)
                # 沿时间轴拼接: (T1,) + (T2,) + ... -> (T1+T2+...,)
                prompt_codec = np.concatenate(features, axis=0)
            if isinstance(prompt_codec, np.ndarray):
                prompt_codec = torch.from_numpy(prompt_codec)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
        else:
            prompt_codec = None

        prompt_wav = kwargs.get("prompt_wav", None)
        prompt_wav_lengths = None
        # import pdb;pdb.set_trace()
        if prompt_wav is not None and fm_use_prompt:
            logging.info(f"fm using prompt")
            if isinstance(prompt_wav, str) and os.path.exists(prompt_wav):
                prompt_wav = load_audio_text_image_video(prompt_wav, fs=self.sample_rate)
                prompt_wav = prompt_wav[None, :]
            elif isinstance(prompt_wav, list):
                wavs = []
                for w in prompt_wav:
                    if isinstance(w, str):
                        if os.path.exists(w):
                            wav = load_audio_text_image_video(w, fs=self.sample_rate)
                        else:
                            raise FileNotFoundError(f"Prompt wav file not found: {w}")
                    elif isinstance(w, np.ndarray):
                        wav = w
                    else:
                        raise TypeError(f"Unsupported type in prompt_wav list: {type(w)}")
                    wavs.append(wav)
                # 拼接音频数据
                prompt_wav = np.concatenate(wavs, axis=0)  # (T_total,)
                if isinstance(prompt_wav, np.ndarray):
                    prompt_wav = torch.from_numpy(prompt_wav)[None, :]  # (1, T_total)
            if isinstance(prompt_wav, np.ndarray):
                prompt_wav = torch.from_numpy(prompt_wav)
            prompt_wav_lengths = torch.tensor([prompt_wav.shape[1]], dtype=torch.int64)
        else:
            prompt_wav = None

        # if prompt_wav is not None and prompt_codec is not None and fm_use_prompt:
        # NOTE(lyuxiang.lx) 不要对音频做裁剪，提取的mel做interploate即可
        if False:
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
            if isinstance(spk_emb, str) and spk_emb.startswith("oss://"):
                spk_emb = oss_remote_numpy_load(spk_emb)
            if isinstance(spk_emb, str) and os.path.exists(spk_emb):
                spk_emb = np.load(spk_emb)
            if isinstance(spk_emb, np.ndarray):
                spk_emb = torch.from_numpy(spk_emb)[None, :]
            if isinstance(spk_emb, list):
                # import pdb;pdb.set_trace()
                spk_emb = np.concatenate([np.load(s) for s in spk_emb],axis=0)
                # import pdb;pdb.set_trace()
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

        contents = self.data_template(data_in[0],**kwargs)
        # print(f"data_template: {contents}")
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
            chunk_size: int = -1,
            finalize: bool = True,
            **kwargs,
    ):
        uttid = key[0]
        # print(f"data in before infer_prepaer {data_in}")
        # import pdb; pdb.set_trace()
        batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        batch.update({'finalize': finalize, 'chunk_size': chunk_size})
        with torch.amp.autocast(device_type="cuda", enabled=True if kwargs['fm_dtype'] != "fp32" else False, dtype=dtype_map[kwargs['fm_dtype']]):
            feat = self._inference(**batch, **kwargs)
        feat = feat.float()
        logging.info(f"{uttid}: feat lengths {feat.shape[1]}")

        return feat

    @torch.no_grad()
    def _inference(
            self,
            codec, codec_lengths,
            prompt_codec=None, prompt_codec_lengths=None,
            prompt_wav=None, prompt_wav_lengths=None,
            xvec=None, xvec_lengths=None, chunk_size=-1, num_left_chunks=-1, finalize=False,
            **kwargs
    ):
        if xvec is not None:
            if xvec.dim() == 2:
                #origin 单个spk emb的情况
                # xvec = xvec.unsqueeze(1)
                #多个spk emb的情况
                xvec = xvec.unsqueeze(0)
                xvec_lens = torch.ones_like(xvec_lengths)
            xvec = self.norm_spk_emb(xvec)
            xvec = xvec.to(device=self.xvec_proj.weight.device, dtype=self.xvec_proj.weight.dtype)
            xvec = self.xvec_proj(xvec).unsqueeze(1)
        else:
            logging.info("using zeros as speaker embedding")
            xvec = torch.zeros((codec.shape[0], 1, self.xvec_proj.weight.shape[0]), dtype=torch.float32, device=codec.device)

        if (codec >= self.codebook_size).any():
            new_codec = codec[codec < self.codebook_size].unsqueeze(0)
            logging.info(f"remove out-of-range token for FM: from {codec.shape[1]} to {new_codec.shape[1]}.")
            codec_lengths = codec_lengths - (codec.shape[1] - new_codec.shape[1])
            codec = new_codec
        if prompt_codec is not None:
            codec, codec_lengths = self.concat_prompt(prompt_codec, prompt_codec_lengths, codec, codec_lengths)
        mask = (codec != -1).unsqueeze(-1)

        if self.unified is True:
            llm_hidden = torch.concat([kwargs['prompt_llm_hidden'], kwargs['llm_hidden']], dim=1) if 'prompt_llm_hidden' in kwargs else kwargs['llm_hidden']
            logging.info(f"llm_hidden shape {llm_hidden.shape}, codec shape {codec.shape}")
            assert llm_hidden.shape[1] >= codec.shape[1]
            codec_emb = llm_hidden[:, :codec.shape[1]]
        else:
            codec_emb = self.codec_embedder(torch.clamp(codec, min=0))
        codec_emb = codec_emb * mask.to(device=codec_emb.device, dtype=codec_emb.dtype)

        if finalize is True:
            context = torch.zeros(
                1,
                0,
                self.codec_embedder.embedding_dim,
                device=codec_emb.device,
                dtype=codec_emb.dtype,
            )
        else:
            codec_emb, context = codec_emb[:, :-self.context_size], codec_emb[:, -self.context_size:]
            codec_lengths = codec_lengths - self.context_size
        mu, _ = self.lookahead_conv1d(codec_emb, codec_lengths, context)
        if self.upsample_method == 'repeat':
            mu = mu.repeat_interleave(self.feat_token_ratio, dim=1)
        elif self.upsample_method == 'causal_linear':
            mu = causal_linear_interpolation(mu, self.feat_token_ratio)
        elif self.upsample_method == 'causal_upsample':
            mu = self.upsample_layer(mu)
        else:
            raise NotImplementedError

        conditions = torch.zeros([mu.size(0), mu.shape[1], self.num_mels]).to(mu)
        # get conditions
        if prompt_wav is not None:
            if prompt_wav.ndim == 2:
                # NOTE(lyuxiang.lx) 推理的时候不要对音量进行规整，避免复刻音频音量与原始音频不一致
                self.max_vol = 1.0
                prompt_wav, prompt_wav_lengths = self.extract_feat(prompt_wav, prompt_wav_lengths)
            # NOTE(lyuxiang.lx) 在fmax12k fm中，尝试mel interploate成token 2倍shape，而不是强制截断
            # prompt_wav = F.interpolate(prompt_wav.transpose(1, 2), size=prompt_codec.shape[1] * self.feat_token_ratio, mode='linear').transpose(1, 2)
            # conditions[:, :prompt_wav.shape[1]] = prompt_wav
            for i, _len in enumerate(prompt_wav_lengths):
                conditions[i, :_len] = prompt_wav[i]

        # NOTE(lyuxiang.lx) add_optional_chunk_mask支持生成-1/1/15/30不同chunk_size的mask
        mask = add_optional_chunk_mask(mu, torch.ones([1, 1, mu.shape[1]]).to(mu).bool(), False, False, 0, chunk_size * self.feat_token_ratio, num_left_chunks)
        feat = self.solve_ode(mu, xvec, conditions, mask, **kwargs)

        if prompt_codec is not None and prompt_wav is not None:
            feat = feat[:, prompt_codec.shape[1] * self.feat_token_ratio:]
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

    def get_rand_noise(self, mu: torch.Tensor, **kwargs):
        use_fixed_noise_infer = kwargs.get("use_fixed_noise_infer", True)
        max_len = kwargs.get("max_len", 50*300)
        if use_fixed_noise_infer:
            if not hasattr(self, "rand_noise"):
                set_all_random_seed(kwargs.get("random_seed", 0))
                self.rand_noise = torch.randn([1, max_len, self.dit_conf['mel_dim']]).to(mu)
                logging.info("init random noise for Flow")
            return torch.concat([self.rand_noise[:, :mu.shape[1], :] for _ in range(mu.size(0))], dim = 0)
        else:
            return torch.randn_like(mu)

    def solve_ode(self, mu, rand_xvec, conditions, mask, **kwargs):
        fm_dtype = dtype_map[kwargs.get("fm_dtype", "fp32")]
        temperature = kwargs.get("temperature", 1.0)
        n_timesteps = kwargs.get("n_timesteps", 10)
        infer_t_scheduler = kwargs.get("infer_t_scheduler", "cosine")

        z = self.get_rand_noise(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(mu)
        if infer_t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        fm_time = time.time()
        self.dit_model.to(fm_dtype)
        feat = self.solve_euler(
            z.to(fm_dtype), t_span=t_span.to(fm_dtype), mu=mu.to(fm_dtype), mask=mask,
            spks=rand_xvec.to(fm_dtype), cond=conditions.to(fm_dtype), **kwargs
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
        # print("solve_euler cond", cond.size())
        steps = 1
        z, bz = x, x.shape[0]
        while steps <= len(t_span) - 1:
            if inference_cfg_rate > 0:
                x_in = torch.concat([x, x], dim=0)
                spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
                mask_in = torch.concat([mask, mask], dim=0)
                mu_in = torch.concat([mu, torch.zeros_like(mu)], dim=0)
                t_in = torch.concat([t.unsqueeze(0) for _ in range(mu_in.size(0))], dim=0)
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
                x_in, mask_in, mu_in, spks_in, t_in, cond_in = x, mask, mu, spks, t, cond

            dphi_dt, _ = self.dit_model(x_in, cond_in, mu_in, spks_in, t_in, mask=mask_in, attn_type=self.attn_type)
            if inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [bz, bz], dim=0)
                dphi_dt = ((1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt)

            x = x + dt * dphi_dt
            t = t + dt
            # sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return x
