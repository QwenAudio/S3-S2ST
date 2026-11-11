import logging
import os.path
import torchaudio

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math

from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import compute_accuracy
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.train_utils.set_all_random_seed import set_all_random_seed
import traceback
import numpy as np
import random
import json
import types
# from funasr.models.ctc.ctc import CTC
from funasr.models.ctc.ctc_zk import CTC
from funasr.frontends.whisper_frontend import WhisperFrontend

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# register speech_encoder via import it
from .sensevoice_encoder import *


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


@tables.register("model_classes", "MinMoSense_CTC")
class MinMoSense_CTC(nn.Module):
    """ """
    def __init__(
            self,
            audio_encoder: str = None,
            audio_encoder_conf: dict = None,
            audio_adaptor: str = None,
            audio_adaptor_conf: dict = None,
            llm: str = None,
            llm_conf: dict = None,
            input_size: int = 80,
            length_normalized_loss: bool = False,
            **kwargs,
    ):

        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
        if hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            if hasattr(model, "model"):
                model = model.model

            logging.info(model)
            audio_encoder_output_size = (
                model.encoder_output_size
                if hasattr(model, "encoder_output_size")
                else -1
            )
            vocab_size = model.vocab_size
            audio_encoder = model.model.encoder if hasattr(model, "model") else model.encoder

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        ctc_vocab_size = kwargs.get("ctc_vocab_size", 60515)

        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))

        # encoder_freeze_exlude_key = audio_encoder_conf.get("freeze_exclude_key", None)
        # if not isinstance(encoder_freeze_exlude_key, (list, tuple)):
        # encoder_freeze_exlude_key = (encoder_freeze_exlude_key,)
        if freeze:
            for name, param in audio_encoder.named_parameters():
                if freeze_layer_num > 0:
                    idx = re.search(r"\.\d+\.", name)
                    if idx is not None:
                        beg, end = idx.regs[0]
                        layer_id = int(name[beg + 1: end - 1])
                        if layer_id < freeze_layer_num:
                            param.requires_grad = False
                    elif "ln_post." not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
                # for exlcude_key in encoder_freeze_exlude_key:
                #     if name.startswith(exlcude_key):
                #         param.requires_grad = True
            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None
        init_param_path = llm_conf.get("init_param_path", None)
        llm_dim = None
        if init_param_path is not None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


            llm_load_kwargs = llm_conf.get("load_kwargs", {})

            if not llm_conf.get("low_cpu", False):
                model = AutoModelForCausalLM.from_pretrained(
                    init_param_path,
                    attn_implementation="flash_attention_2",
                    load_in_8bit=None,
                    device_map=None,
                    use_cache=None,
                    **llm_load_kwargs,
                )
            else:
                import os

                if int(os.environ.get("RANK", 0)) == 0:
                    model = AutoModelForCausalLM.from_pretrained(
                        init_param_path,
                        attn_implementation="flash_attention_2",
                        load_in_8bit=None,
                        device_map="cpu",
                        use_cache=None,
                        **llm_load_kwargs,
                    )
                else:
                    llm_config = AutoConfig.from_pretrained(init_param_path)
                    model = AutoModelForCausalLM.from_config(llm_config)

            if llm_conf.get("reform_vocab_token", False):
                llm_dim = model.get_input_embeddings().weight.shape[-1]
                avoid_meta_f = open(f"{init_param_path}/avoid_meta.json", "r")
                avoid_meta = json.load(avoid_meta_f)["avoid_token"]

                llm_vocab_size = model.lm_head.weight.shape[0]
                lm_head = nn.Linear(llm_dim, llm_vocab_size, bias=True)
                lm_head.weight.data = model.lm_head.weight.data
                bais = torch.zeros(llm_vocab_size, dtype=model.lm_head.weight.data.dtype)
                for i in avoid_meta:
                    bais[i] = -1000000
                lm_head.bias.data = nn.Parameter(bais)
                model.set_output_embeddings(lm_head)

            freeze = llm_conf.get("freeze", True)
            if freeze:
                for name, param in model.named_parameters():
                    param.requires_grad = False
                model.eval()
            # from .utils.stream_utils import _sample
            #
            # model._sample = types.MethodType(_sample, model)
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

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim if llm_dim is not None else audio_adaptor_conf["llm_dim"]
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        init_param_path = audio_adaptor_conf.get("init_param_path", None)
        if init_param_path is not None:
            src_state = torch.load(init_param_path, map_location="cpu")
            flag = audio_adaptor.load_state_dict(src_state, strict=False)
            logging.info(f"Loading audio_adaptor ckpt: {init_param_path}, status: {flag}")
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for name, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()
        self.audio_adaptor = audio_adaptor

        # ctc decoder
        ctc_decoder = kwargs.get("ctc_decoder", "")
        ctc_decoder_class = tables.adaptor_classes.get(ctc_decoder)
        ctc_decoder_conf = kwargs.get("ctc_decoder_conf", {})
        if ctc_decoder_class is None:
            ctc_decoder = None
        else:
            if audio_encoder_output_size > 0:
                ctc_decoder_conf["encoder_dim"] = audio_encoder_output_size
            # ctc_decoder_conf["llm_dim"] = llm_dim
            ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
            init_param_path = ctc_decoder_conf.get("init_param_path", None)
            if init_param_path is not None:
                src_state = torch.load(init_param_path, map_location="cpu")
                flag = ctc_decoder.load_state_dict(src_state, strict=False)
                logging.info(f"Loading ctc_decoder ckpt: {init_param_path}, status: {flag}")
            freeze = ctc_decoder_conf.get("freeze", False)
            if freeze:
                for name, param in ctc_decoder.named_parameters():
                    param.requires_grad = False
                ctc_decoder.eval()
        self.ctc_decoder = ctc_decoder

        # ctc projector and loss, llm_n_vocab = 151665, whisper_n_vocab=60515, blank_id = 60514
        ctc_conf = kwargs.get("ctc_conf", None)
        if ctc_conf is None:
            ctc_conf = {}
        blank_id = ctc_conf.get("blank_id", -1) if ctc_conf.get("blank_id", -1) > -1 else ctc_vocab_size - 1
        assert blank_id > -1, f"blank_id must be an positive int number, now it is {blank_id}"
        ctc = CTC(odim=ctc_vocab_size, encoder_output_size=audio_encoder_output_size, blank_id=blank_id, **ctc_conf)
        # ctc = model.ctc
        # blank_id = ctc.blank_id
        ctc_weight = kwargs.get("ctc_weight", 0.3)
        self.ctc_weight = ctc_weight
        self.blank_id = blank_id
        self.ctc = ctc
        self.detach_ctc_decoder = kwargs.get("detach_ctc_decoder", True)
        print(f"=====ctc blank id is {ctc.blank_id}====")

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

    def forward(
            self,
            speech: torch.Tensor = None,
            speech_lengths: torch.Tensor = None,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            labels_ids: torch.Tensor = None,
            fbank_beg: torch.Tensor = None,
            fbank_mask: torch.Tensor = None,
            **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb
        #
        # pdb.set_trace()
        batch_size, token_num = input_ids.shape
        stats = {}
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        if speech is not None:
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out_ctc, encoder_out_lens_ctc = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out_ctc, encoder_out_lens_ctc = self.encode(speech, speech_lengths)

            # audio_adaptor, detach the grad from encoder_out to audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out_ctc, encoder_out_lens_ctc)
            # encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out_ctc.detach(), encoder_out_lens_ctc)

            batch_size, token_num, dims = inputs_embeds.shape
            fake_token_len = kwargs.get("fake_token_len")
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            speech_idx = 0
            for batch_idx in range(batch_size):

                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    if fbank_beg_idx > 0:
                        speech_token_len = fake_token_len[batch_idx, turn_id]
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]

                        try:
                            inputs_embeds[
                            batch_idx, fbank_beg_idx: fbank_beg_idx + speech_token_len, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            # import pdb;
                            # pdb.set_trace()
                            speech_token_len = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :speech_token_len, :]
                            inputs_embeds[
                            batch_idx, fbank_beg_idx: fbank_beg_idx + speech_token_len, :
                            ] = speech_token

                        speech_idx += 1

            stats["batch_size_speech"] = batch_size_speech
            stats["batch_size_x_frames"] = frames * batch_size_speech
            stats["batch_size_real_frames"] = speech_lengths.sum().item()
            stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        if self.ctc_weight == 1.0:
            loss_att = None
        else:
            with torch.cuda.amp.autocast(
                    enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
            ):
                labels_ids[labels_ids == -1] = -100
                attention_mask[attention_mask < 0] = 0
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                    attention_mask=attention_mask,
                    labels=labels_ids,
                )
                loss_att = model_outputs.loss

            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
                stats["acc"] = acc_att

        ctc_labels = kwargs.get("ctc_labels", None)
        ctc_labels_len = kwargs.get("ctc_labels_len", None)
        if self.ctc_weight > 0.0 and ctc_labels is not None and ctc_labels_len is not None:
            try:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out_ctc, encoder_out_lens_ctc, ctc_labels, ctc_labels_len
                )
                if loss_att is not None:
                    loss = (1 - self.ctc_weight) * loss_att + self.ctc_weight * loss_ctc
                else:
                    loss = loss_ctc

            except Exception as e:
                logging.error(f"\n=====\nctc_loss compute error:\n{str(e)}, {traceback.format_exc()}\n")
                loss_ctc = None
                cer_ctc = None
                logging.error(
                    f"\n=====\n(encoder_out_ctc,encoder_out_lens_ctc, ctc_labels, ctc_labels_len, speech).shape is{encoder_out_ctc.shape} {encoder_out_lens_ctc.shape} {ctc_labels.shape} {ctc_labels_len.shape} {speech.shape}\n=====\n")
                loss = loss_att
        else:
            loss_ctc, cer_ctc = None, None
            loss = loss_att

        stats["cer_ctc"] = cer_ctc
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_att"] = torch.clone(loss_att.detach()) if loss_att is not None else None

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (fbank_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward_export(self, speech, speech_lengths, **kwargs):

        x, olens = self.audio_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)

        encoder_out_ctc, encoder_out_lens_ctc = None, None
        if hasattr(self, "ctc_decoder") and self.ctc_decoder is not None:
            encoder_out_ctc, encoder_out_lens_ctc = self.ctc_decoder(x, olens)
            encoder_out_ctc = self.ctc.log_softmax(encoder_out_ctc)

        return encoder_out, encoder_out_lens, encoder_out_ctc, encoder_out_lens_ctc




    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        if self.detach_ctc_decoder:
            encoder_out = encoder_out.detach()
            encoder_out_lens = encoder_out_lens.detach()
            ys_pad = ys_pad.detach()
            ys_pad_lens = ys_pad_lens.detach()
        if self.ctc_decoder is not None:
            encoder_out, encoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)

        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        do_think = True
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", False)
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"

            splits = pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech, speech_lengths = [], []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(
                                sub_str, fs=frontend.fs, **kwargs
                            )
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                                speech_lengths.sum().item()
                                * frontend.frame_shift
                                * frontend.lfr_n
                                / 1000
                        )

                        if kwargs.get("permute", True):
                            speech = speech.permute(0, 2, 1)
                        if speech_lengths > kwargs.get("max_source_length", 5500):
                            # logging.info(
                            #     f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                            # )
                            badcase_flag = True

                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        # fbank = speech[0, :, :]
        # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference_prepare(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        meta_data = {}
        prompt = kwargs.get("prompt", None)

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = to_device(output, kwargs["device"])

        # audio encoder
        speech = batch["speech"]

        if len(speech) > 0:
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
                speech_lengths = batch["speech_lengths"][:, 0]
                # fp16
                if kwargs.get("fp16", False):
                    speech = speech.to(torch.float16)
                elif kwargs.get("bf16", False):
                    speech = speech.to(torch.bfloat16)
                # audio encoder
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                # audio_adaptor
                encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
                # with open('/nfs/zoukun.zk/projects/FunASR/examples/industrial_data_pretraining/minmo/embedding_audio.pt', 'wb') as f:
                #     torch.save(encoder_out, f)
                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):

            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = encoder_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                        batch_idx, fbank_beg_idx: fbank_beg_idx + speech_token_len, :
                        ] = speech_token
                    except Exception as e:
                        #
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        # import pdb;
                        # pdb.set_trace()
                        speech_token_len = encoder_out_lens[speech_idx].item()
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                        batch_idx, fbank_beg_idx: fbank_beg_idx + speech_token_len, :
                        ] = speech_token

                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data


    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models

    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):
        use_ctc_decoding = kwargs.get("use_ctc_decoding", False)
        use_ctc_batch_decoding = kwargs.get("use_ctc_batch_decoding", False)
        print(f"{__file__} use_ctc_decoding: {use_ctc_decoding}")
        print(f"{__file__} use_ctc_batch_decoding: {use_ctc_batch_decoding}")
        if use_ctc_batch_decoding:
            return self.inference_ctc_batch(
                data_in,
                data_lengths=data_lengths,
                key=key,
                tokenizer=tokenizer,
                frontend=frontend,
                **kwargs,
            )
        elif use_ctc_decoding:
            return self.inference_ctc(
                data_in,
                data_lengths=data_lengths,
                key=key,
                tokenizer=tokenizer,
                frontend=frontend,
                **kwargs,
            )
        else:
            return self.inference_llm(
                data_in,
                data_lengths=data_lengths,
                key=key,
                tokenizer=tokenizer,
                frontend=frontend,
                **kwargs,
            )

    def inference_ctc_batch(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        ibest_writer = kwargs.get("ibest_writer", None)

        if ibest_writer is not None:
            self.writer = ibest_writer

        if kwargs.get("output_dir") is not None and ibest_writer is None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"1best_recog"]
        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.encoder.input_size, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        ctc_tokenizer = kwargs.get("ctc_tokenizer", None)
        ctc_tokenizer_conf = kwargs.get("ctc_tokenizer_conf", None)
        if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
            ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
            # print(f"ctc_tokenizer_conf: {ctc_tokenizer_conf}")
            ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
            self.ctc_tokenizer = ctc_tokenizer

        meta_data = {}
        if (
                isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=ctc_tokenizer,
            )

            if (
                    isinstance(kwargs.get("data_type", None), (list, tuple))
                    and len(kwargs.get("data_type", [])) > 1
            ):
                audio_sample_list, text_token_int_list = audio_sample_list
                text_token_int = text_token_int_list[0]
            else:
                text_token_int = None

            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        if isinstance(frontend, WhisperFrontend):
            speech = speech.permute(0, 2, 1)

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if self.ctc_decoder is not None:
            encoder_out = self.ctc_decoder(encoder_out, encoder_out_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)


            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            # Change integer-ids to tokens
            text = self.ctc_tokenizer.decode(token_int)

            result_i = {"key": key[i], "text": text, "ctc_logits": x}
            results.append(result_i)

            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text

        return results, meta_data

    def inference_llm(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )
        # with open('/nfs/zoukun.zk/projects/FunASR/examples/industrial_data_pretraining/minmo/embedding_inputs_embeds.pt', 'wb') as f:
        #     torch.save(inputs_embeds, f)
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
                enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    **llm_kwargs,
                )

                response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=kwargs.get("skip_special_tokens", True)
                )[0]

                loss = None
            else:

                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                # attention_mask = attention_mask.to(dtype_map[llm_dtype])
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
                )

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1]:]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {"key": key[0], "text": response, "text_tn": response_clean, "label": label}
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data

    def inference_ctc(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.encoder.input_size, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )
        ctc_tokenizer = kwargs.get("ctc_tokenizer", None) if "ctc_tokenizer" in kwargs else kwargs["dataset_conf"]["ctc_tokenizer"]
        ctc_tokenizer_conf = kwargs.get("ctc_tokenizer_conf", None) if "ctc_tokenizer_conf" in kwargs else kwargs["dataset_conf"]["ctc_tokenizer_conf"]
        if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
            ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
            # print(f"ctc_tokenizer_conf: {ctc_tokenizer_conf}")
            ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
            self.ctc_tokenizer = ctc_tokenizer
        assert ctc_tokenizer is not None, f"ctc_tokenizer must be set"
        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"][0]
        meta_data = {}
        # if (
        #         isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        # ):  # fbank
        #     speech, speech_lengths = data_in, data_lengths
        #     if len(speech.shape) < 3:
        #         speech = speech[None, :, :]
        #     if speech_lengths is None:
        # #         speech_lengths = speech.shape[1]
        # # else:
        #     # extract fbank feats
        #     time1 = time.perf_counter()
        #     audio_sample_list = load_audio_text_image_video(
        #         data_in,
        #         fs=frontend.fs if hasattr(frontend, "fs") else 16000,
        #         audio_fs=kwargs.get("fs", 16000),
        #         data_type=kwargs.get("data_type", "sound"),
        #         tokenizer=ctc_tokenizer,
        #     )

        #     if (
        #             isinstance(kwargs.get("data_type", None), (list, tuple))
        #             and len(kwargs.get("data_type", [])) > 1
        #     ):
        #         audio_sample_list, text_token_int_list = audio_sample_list
        #         text_token_int = text_token_int_list[0]
        #     else:
        #         text_token_int = None

        #     time2 = time.perf_counter()
        #     meta_data["load_data"] = f"{time2 - time1:0.3f}"
        #     speech, speech_lengths = extract_fbank(
        #         audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
        #     )
        #     time3 = time.perf_counter()
        #     meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        #     frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
        #     lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
        #     meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # if isinstance(frontend, WhisperFrontend):
        #     speech = speech.permute(0, 2, 1)

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if self.ctc_decoder is not None:
            encoder_out, encoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b

        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            # Change integer-ids to tokens
            text = self.ctc_tokenizer.decode(token_int)

            result_i = {"key": key[i], "text": text, "ctc_logits": x}
            results.append(result_i)

            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text

        return results, meta_data

