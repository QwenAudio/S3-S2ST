import torch
import torch.nn as nn
from funasr.register import tables
import logging
from funasr.utils.load_utils import load_audio_text_image_video
import numpy as np
from funasr.train_utils.device_funcs import to_device
from funasr.auto.auto_model import AutoModel
from funasr.train_utils.set_all_random_seed import set_all_random_seed
import os
import torchaudio


@tables.register("model_classes", "LLMDiffusionTTSInferModel")
class LLMDiffusionTTSInferModel(nn.Module):
    def __init__(
        self,
        lm_model: AutoModel,
        voc_model: AutoModel,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = lm_model.kwargs["tokenizer"]
        self.lm_model = lm_model.model
        self.voc_model = voc_model.model
        mel_extractor = self.lm_model.mel_extractor
        self.mel_frame_rate = mel_extractor.sampling_rate // mel_extractor.hop_length
        self.sample_rate = mel_extractor.sampling_rate
        self.vocoder_hop = kwargs.pop("vocoder_hop", 1920)
        self.vocoder_hop_in_feat = self.vocoder_hop // mel_extractor.hop_length

    @staticmethod
    def init_fm_voc_states():
        return dict(
            cached_token=None,
            cached_feat=None,
            cached_wav=None,
        )

    @staticmethod
    def cross_fade(cached, current, hop_len):
        # cached and current in B, T, ...

        hop_len = min(hop_len, current.shape[1])
        old_overlap = cached[:, -hop_len:]
        # sin_wind = torch.tensor(
        #     np.sin((np.arange(hop_len * 2) + 1) / (hop_len * 2 + 1) * np.pi)[None, :]
        # ).to(old_overlap)
        # cf_overlap = ((old_overlap * sin_wind[:, -hop_len:] +
        #                current[:, :hop_len] * sin_wind[:, :hop_len]) /
        #               (sin_wind[:, :hop_len] + sin_wind[:, -hop_len:]))
        # sin_wind = torch.hann_window(2*hop_len)[None, :].to(old_overlap)
        sin_wind = torch.blackman_window(2 * hop_len)[None, :].to(old_overlap)
        cf_overlap = (old_overlap * sin_wind[:, -hop_len:] +
                      current[:, :hop_len] * sin_wind[:, :hop_len])
        current[:, :hop_len] = cf_overlap

        return current

    def streaming_flow_voc(
        self, data_in, data_lengths, key,
        fm_voc_states, is_last,
        **kwargs
    ):
        cached_token = fm_voc_states.get("cached_token", None)
        cached_feat = fm_voc_states.get("cached_feat", None)
        cached_wav = fm_voc_states.get("cached_wav", None)
        if cached_token is not None:
            data_in[0][-1]["content"] = torch.cat([cached_token, data_in[0][-1]["content"]], dim=1)

        diff_out = self.fm_model.inference(data_in, data_lengths, key, **kwargs)
        diff_out = diff_out.transpose(1, 2)  # convert to B,C,T
        # feat in B,C,T. remove lookahead feat and history, which is inaccurate
        feat = diff_out
        padded_feat_len = 0
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            feat = diff_out[:, :, cached_feat_len:]
            # pad history feat for vocoder synthesis consistency
            padded_feat_len = min(self.mel_frame_rate, cached_feat_len)
            feat = torch.cat([cached_feat[:, :, -padded_feat_len:], feat], dim=2)
            # involving vocoder_hop for cross fade
            padded_feat_len = padded_feat_len - self.vocoder_hop_in_feat

        # if not last, reback feat
        if not is_last:
            feat = feat[:, :, :feat.shape[-1]-self.feat_reback]

        wav = self.voc_model.inference([feat[0].transpose(0, 1)], data_lengths, key, **kwargs).unsqueeze(1)
        # wav = self.vocoder_model(feat)[0].unsqueeze(1)
        # remove padded history wav
        pad_wav_len = int(padded_feat_len * (self.sample_rate / self.mel_frame_rate))
        wav = wav[:, :, pad_wav_len:]
        if cached_wav is not None:
            wav = self.cross_fade(cached_wav.squeeze(1), wav.squeeze(1), self.vocoder_hop).unsqueeze(1)

        # play wav without tail hop length
        cur_wav = wav[:, :, :-self.vocoder_hop] if not is_last else wav

        # update caches
        cached_token = data_in[0][-1]["content"]
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            if is_last:
                cached_feat = torch.cat([cached_feat, diff_out[:, :, cached_feat_len:]], dim=2)
            else:
                cached_feat = torch.cat([cached_feat, diff_out[:, :, cached_feat_len:diff_out.shape[-1]-self.feat_reback]], dim=2)
        else:
            if is_last:
                cached_feat = diff_out[:, :, :]
            else:
                cached_feat = diff_out[:, :, :diff_out.shape[-1]-self.feat_reback]
        if cached_wav is not None:
            cached_wav = torch.cat([cached_wav[:, :, :-self.vocoder_hop], wav], dim=2)
        else:
            cached_wav = wav

        return (cur_wav.squeeze(1), diff_out[:, :, :diff_out.shape[-1]-self.feat_reback]), dict(
            cached_token=cached_token,
            cached_feat=cached_feat,
            cached_wav=cached_wav,
        )

    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        uttid = key[0]
        logging.info(f"generating {uttid}")
        # text -> codec in [1, T]
        kwargs["tokenizer"] = self.tokenizer
        set_all_random_seed(kwargs.get("random_seed", 0))
        feat = self.lm_model.inference(data_in, data_lengths, key, **kwargs)

        wav, batch_data_time = None, 1.0
        if feat.shape[1] > 0:
            # feat -> wav
            set_all_random_seed(kwargs.get("random_seed", 0))
            wav = self.voc_model.inference([feat[0]], data_lengths, key, **kwargs)

            output_dir = kwargs.get("output_dir", None)
            if output_dir is not None:
                wav_out_dir = os.path.join(output_dir, "wav")
                os.makedirs(wav_out_dir, exist_ok=True)
                torchaudio.save(
                    os.path.join(wav_out_dir, f"{key[0]}.wav"), wav.cpu(),
                    sample_rate=self.sample_rate, encoding='PCM_S', bits_per_sample=16
                )

            batch_data_time = wav.shape[1] / self.voc_model.sample_rate
        return [[wav]], {"batch_data_time": batch_data_time}

