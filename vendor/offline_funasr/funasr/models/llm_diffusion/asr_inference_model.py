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


@tables.register("model_classes", "LLMDiffusionASRInferModel")
class LLMDiffusionASRInferModel(nn.Module):
    def __init__(
        self,
        lm_model: AutoModel,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = lm_model.kwargs["tokenizer"]
        self.lm_model = lm_model.model
        mel_extractor = self.lm_model.mel_extractor
        self.mel_frame_rate = mel_extractor.sampling_rate // mel_extractor.hop_length
        self.sample_rate = mel_extractor.sampling_rate

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
        res = self.lm_model.asr_inference(data_in, data_lengths, key, **kwargs)

        return [[res]]

