import torch
import torch.nn as nn
from funasr.register import tables
import logging
from funasr.utils.load_utils import load_audio_text_image_video
import numpy as np
from funasr.train_utils.device_funcs import to_device
from funasr.auto.auto_model import AutoModel


@tables.register("model_classes", "CosyVoiceInferModel")
class CosyVoiceInferModel(nn.Module):
    def __init__(
        self,
        lm_model: AutoModel,
        fm_model: AutoModel,
        voc_model: AutoModel,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = lm_model.kwargs["tokenizer"]
        self.frontend = fm_model.kwargs["frontend"]
        self.lm_model = lm_model.model
        self.fm_model = fm_model.model
        self.voc_model = voc_model.model

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
        # text -> codec
        kwargs["tokenizer"] = self.tokenizer
        codec, hit_eos, states = self.lm_model.inference(data_in, data_lengths, key, **kwargs)

        # codec -> feat
        data_in[0][-1]["content"] = codec
        feat = self.fm_model.inference(data_in, data_lengths, key, **kwargs)

        # feat -> wav
        wav = self.voc_model.inference([feat[0]], data_lengths, key, **kwargs)

        return wav, {"batch_data_time": wav.shape[1]/self.voc_model.sample_rate}
