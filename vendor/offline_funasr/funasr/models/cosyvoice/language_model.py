import logging
import os.path
import torchaudio
from typing import Union, Dict, List, Tuple, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import re
import math
from funasr.models.scama.utils import sequence_mask
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.train_utils.set_all_random_seed import set_all_random_seed
import numpy as np
from funasr.models.cosyvoice.utils import dtype_map

try:
    from scipy.io import savemat
except:
    pass



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


@tables.register("model_classes", "CosyVoiceLM")
class CosyVoiceLM(nn.Module):
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

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            **llm_load_kwargs,
        )

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

        del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim, 0)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.codec_unit,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
            # reduction=False,
        )

        specaug = kwargs.get("specaug", None)
        specaug_conf = kwargs.get("specaug_conf", {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug
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

        codec = kwargs.get("codec")
        codec[codec < 0] = 0
        codec_len = kwargs.get("codec_len")
        codec_len[codec_len < 0] = 0
        codec_beg = kwargs.get("codec_beg")
        codec_beg[codec_beg < 0] = 0
        fake_codec_len = kwargs.get("fake_codec_len")
        fake_codec_len[fake_codec_len < 0] = 0

        audio = kwargs.get("audio")
        audio_len = kwargs.get("audio_len")
        spk_emb = kwargs.get("spk_emb")
        spk_emb_len = kwargs.get("spk_emb_len")

        batch_size, token_num, dims = inputs_embeds.shape

        codec_emb = self.codec_embed(codec)
        idx = 0
        for i, batch_idx in enumerate(range(batch_size)):

            for j, turn_id in enumerate(range(codec_beg.shape[1])):
                codec_beg_idx = codec_beg[batch_idx, turn_id].item()
                if codec_beg_idx > 0:
                    codec_len_i = codec_len[batch_idx, turn_id]
                    codec_i = codec_emb[idx : idx + 1, :codec_len_i, :]
                    codec_i, _ = self.specaug(codec_i, codec_len_i)
                    inputs_embeds[
                        batch_idx : batch_idx + 1, codec_beg_idx : codec_beg_idx + codec_len_i, :
                    ] = codec_i
                    idx += 1

        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids < 0] = self.ignore_id
            attention_mask[attention_mask < 0] = 0
            outputs = self.llm.model(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
            )
            labels = labels_ids
            # loss = model_outputs.loss
            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
            logits = logits.float()

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.criterion_ce(shift_logits, shift_labels)

        with torch.no_grad():
            preds = torch.argmax(logits, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], labels_ids[:, 1:], ignore_label=self.ignore_id
            )
            stats["acc"] = acc_att

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
        prompt_text = kwargs.get("user_prompt_text", None)

        prompt_codec = kwargs.get("prompt_codec", None)
        prompt_codec_lengths = None
        if prompt_codec is not None:
            prompt_codec = np.load(prompt_codec)
            prompt_codec = prompt_codec.tolist()
            prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)

        # single turn
        text = contents["user"][0]
        if text.startswith("语音合成："):
            text = text.replace("语音合成：", "")
        if text.startswith("Speech Synthesis:"):
            text = text.replace("Speech Synthesis:", "")
        cat_text = text
        if prompt_text is not None:
            cat_text = prompt_text + text
        source_input = f"<|im_start|>{cat_text}<|im_end|>"
        input_source_ids = tokenizer.encode(source_input)
        input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)[None, :]

        prompt_text_lengths = None
        if prompt_text is not None:
            prompt_text = tokenizer.encode(prompt_text)
            prompt_text = torch.tensor(prompt_text, dtype=torch.int64)[None, :]
            prompt_text_lengths = torch.tensor([prompt_text.shape[1]], dtype=torch.int64)

        text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
        text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.int64)
        output = {
            "source_ids": input_source_ids,
            "text": text_ids,
            "text_lengths": text_lengths,
            "prompt_text": prompt_text,
            "prompt_text_lengths": prompt_text_lengths,
            "prompt_codec": prompt_codec,
            "prompt_codec_lengths": prompt_codec_lengths,
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
        output = self.load_data(
            contents, **kwargs
        )
        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]
        inputs_embeds = self.llm.model.get_input_embeddings()(source_ids)

        prompt_codec = batch.get("prompt_codec", None)
        if prompt_codec is not None:
            codec_emb = self.codec_embed(prompt_codec)
            inputs_embeds = torch.cat((inputs_embeds, codec_emb), dim=1)

        return inputs_embeds, batch

    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        uttid = key[0]
        inputs_emb, batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        text, text_lengths = batch["text"], batch["text_lengths"]
        prompt_text, prompt_text_lens = batch["prompt_text"], batch["prompt_text_lengths"]
        prompt_codec, prompt_codec_lens = batch["prompt_codec"], batch["prompt_codec_lengths"]

        max_length = kwargs.get("max_length", 30 * 25)
        min_length = kwargs.get("min_length", None)
        min_length_ratio = kwargs.get("min_length_ratio", None)
        max_length_ratio = kwargs.get("max_length_ratio", None)
        if min_length_ratio is not None and prompt_text_lens is not None and prompt_codec_lens is not None:
            min_length = int(prompt_codec_lens / prompt_text_lens * text_lengths * min_length_ratio)
        if max_length_ratio is not None and prompt_text_lens is not None and prompt_codec_lens is not None:
            max_length = int(prompt_codec_lens / prompt_text_lens * text_lengths * max_length_ratio)
        logging.info(f"{uttid}: min length: {min_length}, max length: {max_length}")

        dtype = dtype_map[kwargs.get("llm_dtype", "fp32")]
        if not hasattr(self, "llm_generator"):
            from funasr.models.cosyvoice.utils.llm_decoding import LLMDecoder
            llm_generator_conf = kwargs.get("llm_generator_conf", {})
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)
        self.llm.lm_head = self.codec_head.to(dtype)
        gen_codec, hit_eos, states = self.llm_generator(
            inputs_emb.to(dtype), self.llm,
            states=kwargs.get("states", {}),
            max_length=max_length,
            min_length=min_length,
            **kwargs
        )
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, "codec")
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"{key[0]}.npy"),
                gen_codec[0].cpu().numpy()
            )

        return gen_codec, hit_eos, states


@tables.register("model_classes", "CosyVoiceMixedLM")
class CosyVoiceMixedLM(nn.Module):
    """ """

    def __init__(
        self,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            **llm_load_kwargs,
        )

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

        del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim, 0)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.codec_unit,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
            # reduction=False,
        )

        specaug = kwargs.get("specaug", None)
        specaug_conf = kwargs.get("specaug_conf", {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug
        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        text_flag: torch.Tensor = None,
        codec_flag: torch.Tensor = None,
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        stats = {}
        input_mask = input_ids > 0
        text_flag = (text_flag > 0) * input_mask
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        codec_flag = (codec_flag > 0) * input_mask
        codec_embeds = self.codec_embed(input_ids * codec_flag) * codec_flag.unsqueeze(-1)
        inputs_embeds = inputs_embeds + codec_embeds

        batch_size, token_num, dims = inputs_embeds.shape
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids < 0] = self.ignore_id
            attention_mask[attention_mask < 0] = 0
            outputs = self.llm.model(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
            )
            labels = labels_ids
            # loss = model_outputs.loss
            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
            logits = logits.float()

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.criterion_ce(shift_logits, shift_labels)

        with torch.no_grad():
            preds = torch.argmax(logits, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], labels_ids[:, 1:], ignore_label=self.ignore_id
            )
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        stats["dialog_turns_max"] = 1
        stats["dialog_turns_avg"] = 1

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
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
        prompt_text = kwargs.get("user_prompt_text", None)

        prompt_codec = kwargs.get("prompt_codec", None)
        prompt_codec_lengths = None
        if prompt_codec is not None:
            prompt_codec = np.load(prompt_codec)
            prompt_codec = prompt_codec.tolist()
            prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)

        # single turn
        text = contents["user"][0]
        if text.startswith("语音合成："):
            text = text.replace("语音合成：", "")
        if text.startswith("Speech Synthesis:"):
            text = text.replace("Speech Synthesis:", "")
        cat_text = text
        if prompt_text is not None:
            cat_text = prompt_text + text
        source_input = f"<|im_start|>{cat_text}<|im_end|>"
        input_source_ids = tokenizer.encode(source_input)
        input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)[None, :]

        prompt_text_lengths = None
        if prompt_text is not None:
            prompt_text = tokenizer.encode(prompt_text)
            prompt_text = torch.tensor(prompt_text, dtype=torch.int64)[None, :]
            prompt_text_lengths = torch.tensor([prompt_text.shape[1]], dtype=torch.int64)

        text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
        text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.int64)
        output = {
            "source_ids": input_source_ids,
            "text": text_ids,
            "text_lengths": text_lengths,
            "prompt_text": prompt_text,
            "prompt_text_lengths": prompt_text_lengths,
            "prompt_codec": prompt_codec,
            "prompt_codec_lengths": prompt_codec_lengths,
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
        output = self.load_data(
            contents, **kwargs
        )
        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]
        inputs_embeds = self.llm.model.get_input_embeddings()(source_ids)

        prompt_codec = batch.get("prompt_codec", None)
        if prompt_codec is not None:
            codec_emb = self.codec_embed(prompt_codec)
            inputs_embeds = torch.cat((inputs_embeds, codec_emb), dim=1)

        return inputs_embeds, batch

    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        uttid = key[0]
        inputs_emb, batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        text, text_lengths = batch["text"], batch["text_lengths"]
        prompt_text, prompt_text_lens = batch["prompt_text"], batch["prompt_text_lengths"]
        prompt_codec, prompt_codec_lens = batch["prompt_codec"], batch["prompt_codec_lengths"]

        max_length = kwargs.get("max_length", 30 * 25)
        min_length = kwargs.get("min_length", None)
        min_length_ratio = kwargs.get("min_length_ratio", None)
        max_length_ratio = kwargs.get("max_length_ratio", None)
        if min_length_ratio is not None and prompt_text_lens is not None and prompt_codec_lens is not None:
            min_length = int(prompt_codec_lens / prompt_text_lens * text_lengths * min_length_ratio)
        if max_length_ratio is not None and prompt_text_lens is not None and prompt_codec_lens is not None:
            max_length = int(prompt_codec_lens / prompt_text_lens * text_lengths * max_length_ratio)
        logging.info(f"{uttid}: min length: {min_length}, max length: {max_length}")

        dtype = dtype_map[kwargs.get("llm_dtype", "fp32")]
        if not hasattr(self, "llm_generator"):
            from funasr.models.cosyvoice.utils.llm_decoding import LLMDecoder
            llm_generator_conf = kwargs.get("llm_generator_conf", {})
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)
        self.llm.lm_head = self.codec_head.to(dtype)
        gen_codec, hit_eos, states = self.llm_generator(
            inputs_emb.to(dtype), self.llm,
            states=kwargs.get("states", {}),
            max_length=max_length,
            min_length=min_length,
            **kwargs
        )
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, "codec")
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"{key[0]}.npy"),
                gen_codec[0].cpu().numpy()
            )

        return gen_codec, hit_eos, states

@tables.register("model_classes", "CosyVoiceMixedLMSimpleSys")
class CosyVoiceMixedLMSimpleSys(nn.Module):
    """ """

    def __init__(
        self,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
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

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            **llm_load_kwargs,
        )

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

        del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim, 0)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.codec_unit,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
            # reduction=False,
        )

        specaug = kwargs.get("specaug", None)
        specaug_conf = kwargs.get("specaug_conf", {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug
        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        text_flag: torch.Tensor = None,
        codec_flag: torch.Tensor = None,
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        stats = {}
        input_mask = input_ids > 0
        text_flag = (text_flag > 0) * input_mask
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        codec_flag = (codec_flag > 0) * input_mask
        codec_embeds = self.codec_embed(input_ids * codec_flag) * codec_flag.unsqueeze(-1)
        inputs_embeds = inputs_embeds + codec_embeds

        batch_size, token_num, dims = inputs_embeds.shape
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids < 0] = self.ignore_id
            attention_mask[attention_mask < 0] = 0
            outputs = self.llm.model(
                # inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            labels = labels_ids
            # loss = model_outputs.loss
            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
        logits = logits.float()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.criterion_ce(shift_logits, shift_labels)

        with torch.no_grad():
            preds = torch.argmax(logits, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], labels_ids[:, 1:], ignore_label=self.ignore_id
            )
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        stats["dialog_turns_max"] = 1
        stats["dialog_turns_avg"] = 1

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
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
        text_codec_mix_ratio = kwargs.get("text_codec_mix", [5, 15])
        lm_use_prompt = kwargs.get("lm_use_prompt", True)
        tokenizer = kwargs.get("tokenizer")
        prompt_text = contents["meta_dict"].get("user_prompt_text", None)
        llm_generator_conf = kwargs.get("llm_generator_conf", {})

        prompt_codec = contents["meta_dict"].get("prompt_codec", None)
        prompt_codec_lengths = None
        if prompt_codec is not None and lm_use_prompt:
            prompt_codec = np.load(prompt_codec)
            prompt_codec = prompt_codec.tolist()
            prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
        else:
            prompt_codec = None

        prompt_wav = contents["meta_dict"].get("prompt_wav", None)
        prompt_wav_lengths = None
        if prompt_wav is not None and lm_use_prompt:
            if isinstance(prompt_wav, str) and os.path.exists(prompt_wav):
                prompt_wav = load_audio_text_image_video(prompt_wav, fs=self.sample_rate)
                prompt_wav = prompt_wav[None, :]
            if isinstance(prompt_wav, np.ndarray):
                prompt_wav = torch.from_numpy(prompt_wav)
            prompt_wav_lengths = torch.tensor([prompt_wav.shape[1]], dtype=torch.int64)
        else:
            prompt_wav = None

        if prompt_wav is not None and prompt_codec is not None and lm_use_prompt:
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

        # single turn
        raw_text = ""
        sys_instruct = contents["system"][0] + "<|endofprompt|>"
        text = contents["user"][0]
        if text.startswith("语音合成："):
            text = text.replace("语音合成：", "")
        if text.startswith("Speech Synthesis:"):
            text = text.replace("Speech Synthesis:", "")

        # offline mode
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
            input_source_ids = [
                llm_generator_conf.get("sos", 6561),
                *input_source_ids,
                llm_generator_conf.get("turn_of_speech", 6563)
            ]
            if prompt_codec is not None and lm_use_prompt:
                input_source_ids.extend(prompt_codec[0].tolist())
            input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)[None, :]
            text_emb_flag = torch.zeros_like(input_source_ids)
            text_emb_flag[:, 1:1+all_text_len] = 1
            res_text_ids = [[]]
        # online mode mixed
        elif streaming_mode in ["true", "mixed"]:
            sos_token = llm_generator_conf.get("sos", 6561)
            tos_token = llm_generator_conf.get("turn_of_speech", 6563)
            instruct_ids = torch.tensor(tokenizer.encode(sys_instruct), dtype=torch.int64)[None, :]
            if prompt_codec is None:
                raw_text = sys_instruct + text
                text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
                if text_ids.shape[1] < text_codec_mix_ratio[0]:
                    input_source_ids = torch.cat([
                        torch.tensor([[sos_token]], dtype=torch.int64),
                        instruct_ids,
                        text_ids,
                        torch.tensor([[tos_token]], dtype=torch.int64),
                    ], dim=1)
                    text_emb_flag = torch.ones_like(input_source_ids)
                    text_emb_flag[:, 0] = 0
                    text_emb_flag[:, -1] = 0
                    res_text_ids = text_ids[:, text_codec_mix_ratio[0]:]
                else:
                    input_source_ids = torch.cat([
                        torch.tensor([[sos_token]], dtype=torch.int64),
                        instruct_ids,
                        text_ids[:, :text_codec_mix_ratio[0]],
                    ], dim=1)
                    res_text_ids = text_ids[:, text_codec_mix_ratio[0]:]
                    text_emb_flag = torch.ones_like(input_source_ids)
                    text_emb_flag[:, 0] = 0
            else:
                raw_text = sys_instruct + prompt_text + text
                text_ids = torch.tensor(
                    tokenizer.encode(prompt_text) + tokenizer.encode(text),
                    dtype=torch.int64
                )[None, :]
                text_len = text_ids.shape[1]
                codec_len = prompt_codec.shape[1]
                text_interval, codec_interval = text_codec_mix_ratio
                text_exp_intervals = int(math.floor(text_len / text_interval))
                codec_exp_intervals = int(math.floor(codec_len / codec_interval))
                exp_intervals = min(text_exp_intervals, codec_exp_intervals)

                exp_codec_len = exp_intervals * codec_interval
                exp_text_len = exp_intervals * text_interval
                # Calculate the length of the mixed sequence
                mixed_length = exp_text_len + exp_codec_len
                mixed_seq = torch.zeros([1, mixed_length], dtype=torch.int64)
                text_emb_flag = torch.ones([1, mixed_length], dtype=torch.int64)
                # Fill the mixed sequence
                for i in range(text_interval):
                    mixed_seq[:, i::text_interval + codec_interval] = text_ids[:, i:exp_text_len:text_interval]
                for i in range(codec_interval):
                    mixed_seq[:, i + text_interval::text_interval + codec_interval] = prompt_codec[:,
                                                                                   i:exp_codec_len:codec_interval]
                    text_emb_flag[:, i + text_interval::text_interval + codec_interval] = 0

                inputs_list = [
                    torch.tensor([[sos_token]], dtype=torch.int64),
                    instruct_ids,
                    mixed_seq
                ]
                text_flag_list = [
                    torch.tensor([[0]], dtype=torch.int64),
                    torch.ones_like(instruct_ids),
                    text_emb_flag,
                ]
                res_text = text_ids[:, exp_text_len:]
                res_codec = prompt_codec[:, exp_codec_len:]
                if res_text.shape[1] < text_interval:
                    inputs_list.extend([res_text, torch.tensor([[tos_token]], dtype=torch.int64)])
                    text_flag_list.extend([torch.ones_like(res_text), torch.tensor([[0]], dtype=torch.int64)])
                    res_text = res_text[:, res_text.shape[1]:]
                inputs_list.extend([res_text[:, :text_interval], res_codec])
                text_flag_list.extend([torch.ones_like(res_text[:, :text_interval]), torch.zeros_like(res_codec)])
                res_text_ids = res_text[:, text_interval:]
                input_source_ids = torch.cat(inputs_list, dim=1)
                text_emb_flag = torch.cat(text_flag_list, dim=1)
        elif streaming_mode in ["fixed_ratio", "syn_to_end"]:
            streaming_ratio = kwargs.get("streaming_ratio", [5, 15])
            text_ids = tokenizer.encode(text)
            input_source_ids = text_ids[:streaming_ratio[0]]
            res_text_ids = [text_ids[streaming_ratio[0]:]]
            raw_text = text
            if prompt_text is not None and lm_use_prompt:
                input_source_ids = tokenizer.encode(prompt_text) + input_source_ids
                raw_text = prompt_text + raw_text
            if kwargs.get("use_sys_instruct", False):
                hint_once("use_sys_instruct: True", "use_sys_instruct")
                input_source_ids = tokenizer.encode(sys_instruct) + input_source_ids
                raw_text = sys_instruct + raw_text
            all_text_len = len(input_source_ids)
            input_source_ids = [
                llm_generator_conf.get("sos", 6561),
                *input_source_ids,
                llm_generator_conf.get("turn_of_speech", 6563)
            ]
            if prompt_codec is not None and lm_use_prompt:
                input_source_ids.extend(prompt_codec[0].tolist())
            input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)[None, :]
            text_emb_flag = torch.zeros_like(input_source_ids)
            text_emb_flag[:, 1:1 + all_text_len] = 1
        else:
            raise NotImplementedError("streaming_mode must be in true, false, mixed, fixed_ratio")

        prompt_text_lengths = None
        if prompt_text is not None and lm_use_prompt:
            prompt_text = tokenizer.encode(prompt_text)
            prompt_text = torch.tensor(prompt_text, dtype=torch.int64)[None, :]
            prompt_text_lengths = torch.tensor([prompt_text.shape[1]], dtype=torch.int64)
        else:
            prompt_text = None

        text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)[None, :]
        text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.int64)
        instruct_ids = torch.tensor(tokenizer.encode(sys_instruct), dtype=torch.int64)[None, :]
        instruct_lengths = torch.tensor([instruct_ids.shape[1]], dtype=torch.int64)
        res_text_ids = torch.tensor(res_text_ids, dtype=torch.int64)
        logging.info(f"generating for {uttid}: input text: {raw_text}, input ids: {input_source_ids}")
        output = {
            "source_ids": input_source_ids,
            "text": text_ids,
            "text_lengths": text_lengths,
            "prompt_text": prompt_text,
            "prompt_text_lengths": prompt_text_lengths,
            "prompt_codec": prompt_codec,
            "prompt_codec_lengths": prompt_codec_lengths,
            "text_emb_flag": text_emb_flag,
            "sys_instruct": instruct_ids,
            "sys_instruct_lengths": instruct_lengths,
            "res_text_ids": res_text_ids,
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
        text_emb_flag = batch["text_emb_flag"]
        text_embeder = self.llm.model.get_input_embeddings()
        inputs_text_embeds = text_embeder(source_ids * text_emb_flag) * text_emb_flag.unsqueeze(-1)
        input_codec_embs = self.codec_embed(source_ids * (1-text_emb_flag)) * (1-text_emb_flag).unsqueeze(-1)
        inputs_embeds = inputs_text_embeds + input_codec_embs
        batch["res_text_emb"] = text_embeder(batch["res_text_ids"])

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
        prompt_codec, prompt_codec_lens = batch["prompt_codec"], batch["prompt_codec_lengths"]

        max_length = kwargs.get("max_length", 60 * 25)
        if "max_length" in kwargs:
            kwargs.pop("max_length")
        min_length = kwargs.get("min_length", None)
        if "min_length" in kwargs:
            kwargs.pop("min_length")
        min_length_ratio = kwargs.get("min_length_ratio", None)
        max_length_ratio = kwargs.get("max_length_ratio", None)
        if min_length_ratio is not None and prompt_text_lens is not None and prompt_codec_lens is not None:
            min_length = int(prompt_codec_lens / prompt_text_lens * text_lengths * min_length_ratio)
        if max_length_ratio is not None and prompt_text_lens is not None and prompt_codec_lens is not None:
            max_length = int(prompt_codec_lens / prompt_text_lens * text_lengths * max_length_ratio)
        logging.info(f"{uttid}: min length: {min_length}, max length: {max_length}")

        dtype = dtype_map[kwargs.get("llm_dtype", "fp32")]
        llm_generator_conf = kwargs.get("llm_generator_conf", {})
        if not hasattr(self, "llm_generator"):
            from funasr.models.cosyvoice.utils.llm_decoding import LLMDecoder
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)
        self.llm.lm_head = self.codec_head.to(dtype)
        avoid_token = self.get_avoid_token(llm_generator_conf.get("avoid_token", None))
        if streaming_mode == "false":
            gen_codec, hit_eos, states = self.llm_generator(
                inputs_emb.to(dtype), self.llm,
                states=kwargs.get("states", {}),
                max_length=max_length,
                min_length=min_length,
                avoid_token=avoid_token,
                **kwargs
            )
        elif streaming_mode in ["true", "mixed"]:
            kwargs["use_llm_cache"] = True
            gen_raw_list, gen_codec_list = [], []
            res_text_emb = batch["res_text_emb"]
            text_interval, codec_interval = kwargs.get("text_codec_mix", [5, 15])
            eos_tokens = [
                llm_generator_conf.get("eos", 6562),
                llm_generator_conf.get("fill_token", 6564),
            ]
            states = kwargs.get("states", {})
            while True:
                gen_codec, hit_eos, states = self.llm_generator(
                    inputs_emb.to(dtype), self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=None,
                    include_eos=True,
                    custom_eos_token=eos_tokens,
                    avoid_token=avoid_token,
                    **kwargs
                )
                if gen_codec[0, -1] == llm_generator_conf.get("fill_token", 6564):
                    if res_text_emb.shape[1] < text_interval:
                        tos_idx = torch.tensor(
                            [[llm_generator_conf.get("turn_of_speech", 6563)]],
                            dtype=torch.int64, device=res_text_emb.device
                        )
                        tos_emb = self.codec_embed(tos_idx)
                        inputs_emb = torch.cat([res_text_emb, tos_emb], dim=1)
                    else:
                        inputs_emb = res_text_emb[:, :text_interval]

                    res_text_emb = res_text_emb[:, text_interval:]

                gen_raw_list.append(gen_codec)
                gen_codec_list.append(gen_codec[:, :-1])
                if gen_codec[0, -1] == llm_generator_conf.get("eos", 6562):
                    break

            gen_raw_codes = torch.cat(gen_raw_list, dim=1)
            logging.info(f"{uttid}: gen {gen_raw_codes.shape[1]} raw tokens: {gen_raw_codes[0].cpu().tolist()}")
            gen_codec = torch.cat(gen_codec_list, dim=1)
        elif streaming_mode in ["fixed_ratio"]:
            streaming_ratio = kwargs.get("streaming_ratio", [5, 15])
            tos_pos = torch.where(batch["text_emb_flag"][0] == 1)[0][-1].item() + 1
            res_text_emb = batch["res_text_emb"]
            states = kwargs.get("states", {})
            chunk_len = streaming_ratio[1]
            gen_codec_list = []
            while True:
                gen_codec, hit_eos, _ = self.llm_generator(
                    inputs_emb.to(dtype), self.llm,
                    states=states,
                    max_length=chunk_len,
                    min_length=None,
                    avoid_token=avoid_token,
                    **kwargs
                )
                gen_codec_list.append(gen_codec)
                if hit_eos:
                    break
                if res_text_emb.shape[1] <= streaming_ratio[0]:
                    chunk_len = max_length
                gen_codec_emb = self.codec_embed(gen_codec)
                inputs_emb = torch.cat([
                    inputs_emb[:, :tos_pos],
                    res_text_emb[:, :streaming_ratio[0]],
                    inputs_emb[:, tos_pos:],
                    gen_codec_emb,
                ], dim=1)
                tos_pos = tos_pos + res_text_emb[:, :streaming_ratio[0]].shape[1]
                res_text_emb = res_text_emb[:, streaming_ratio[0]:]
            gen_codec = torch.cat(gen_codec_list, dim=1)
        elif streaming_mode in ["syn_to_end"]:
            streaming_ratio = kwargs.get("streaming_ratio", [5, 15])
            break_num = kwargs.get("break_num", 5)
            tos_pos = torch.where(batch["text_emb_flag"][0] == 1)[0][-1].item() + 1
            res_text_emb = batch["res_text_emb"]
            states = kwargs.get("states", {})
            gen_codec_list = []
            while True:
                gen_codec, hit_eos, _ = self.llm_generator(
                    inputs_emb.to(dtype), self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=None,
                    include_eos=False,
                    **kwargs
                )
                if res_text_emb.shape[1] > 0:
                    _break_num = break_num
                    if isinstance(break_num, float):
                        _break_num = int(break_num * gen_codec.shape[1])
                    gen_codec = gen_codec[:, :-_break_num]
                gen_codec_list.append(gen_codec)
                if res_text_emb.shape[1] == 0:
                    break
                gen_codec_emb = self.codec_embed(gen_codec)
                inputs_emb = torch.cat([
                    inputs_emb[:, :tos_pos],
                    res_text_emb[:, :streaming_ratio[0]],
                    inputs_emb[:, tos_pos:],
                    gen_codec_emb,
                ], dim=1)
                tos_pos = tos_pos + res_text_emb[:, :streaming_ratio[0]].shape[1]
                res_text_emb = res_text_emb[:, streaming_ratio[0]:]
            gen_codec = torch.cat(gen_codec_list, dim=1)
        else:
            raise NotImplementedError(f"{streaming_mode} is not supported.")

        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, "codec")
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"{key[0]}.npy"),
                gen_codec[0].cpu().numpy()
            )

        return gen_codec, hit_eos, states
