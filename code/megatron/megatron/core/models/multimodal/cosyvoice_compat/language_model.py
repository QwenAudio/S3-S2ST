import logging
import os.path
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
import numpy as np
from .llm_decoding import dtype_map
from funasr.utils.hinter import hint_once
import math
import random
from copy import deepcopy
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
try:
    from funasr.utils.oss_io import oss_remote_numpy_load
except ImportError:
    def oss_remote_numpy_load(path):
        raise ImportError(
            "funasr.utils.oss_io is required to load OSS speaker embeddings: "
            f"{path}"
        )

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
        self.llm_conf = llm_conf
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig , BitsAndBytesConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        init_hf_model_path = llm_conf.get("init_hf_model_path", None)
        lm_init_hf_model_path = kwargs.get("lm_init_hf_model_path","")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        self.sample_rate = kwargs.get("sample_rate", 24000)
        self.token_rate = kwargs.get("token_rate", 25)

        if kwargs.get("infer_lora_merged", False):
            llm_conf["use_qlora"] = False
            llm_conf["use_lora"] = False
            kwargs["infer_use_lora"] = False

        if kwargs.get("infer_use_lora",False):
            llm_conf['use_qlora'] = False
            llm_conf['use_lora'] = False
            logging.info(f"load lora pretrained model from {lm_init_hf_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                lm_init_hf_model_path,
                device_map=None,
                **llm_load_kwargs,
            )
        elif llm_conf.get("use_qlora", False) and not llm_conf.get("use_lora_int8", False):
            llm_conf['use_lora'] = True
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logging.info(f"load qlora pretrained from {init_hf_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                init_hf_model_path,
                quantization_config=bnb_config,
                device_map=None,
                **llm_load_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                init_param_path,
                device_map=None,
                **llm_load_kwargs,
            )

        tokenizer_vocab_size = llm_conf.get("vocab_size", kwargs.get("vocab_size", None))
        if tokenizer_vocab_size == -1:
            tokenizer_vocab_size = None
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_vocab_size is not None and tokenizer_vocab_size > model_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        freeze_18 = llm_conf.get("freeze_18", False)
        if freeze_18:
            for name, param in model.named_parameters():
                if name.startswith("model.layers.") and int(name.split(".")[2]) < 18:
                    param.requires_grad = False

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}, use_qlora: {llm_conf.get('use_qlora', False)}, infer_use_lora: {kwargs.get('infer_use_lora',False)}, infer_lora_merged: {kwargs.get('infer_lora_merged',False)}")


        if kwargs.get("infer_use_lora",False):

            del model.lm_head

            from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel
            init_param_lora = kwargs.get("init_param_lora", None)
            if init_param_lora is not None:
                logging.info(f"init_param_lora: {init_param_lora}")
                model = PeftModel.from_pretrained(model, init_param_lora)


        if llm_conf.get("use_lora", False):
            del model.lm_head
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
            elif kwargs.get("init_param_lora", None):
                init_param_lora = kwargs.get("init_param_lora", None)
                logging.info(f"init_param_lora: {init_param_lora}")
                model = PeftModel.from_pretrained(model, init_param_lora)
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        if not llm_conf.get("use_qlora", False):
            self.llm = model.to(dtype_map[self.llm_dtype])
        else:
            self.llm = model
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        if (not llm_conf.get("use_lora", False)) and (not kwargs.get("infer_use_lora",False)):
            del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim)
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
        if self.llm_conf.get("use_lora",False) or kwargs.get("infer_use_lora",False):
            inputs_embeds = self.llm.base_model.model.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        else:
            inputs_embeds = self.llm.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        codec_flag = (codec_flag > 0) * input_mask
        codec_embeds = self.codec_embed(input_ids * codec_flag) * codec_flag.unsqueeze(-1)
        inputs_embeds = inputs_embeds + codec_embeds

        batch_size, token_num, dims = inputs_embeds.shape
        with torch.amp.autocast(
            device_type="cuda", enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids < 0] = self.ignore_id
            attention_mask[attention_mask < 0] = 0
            if self.llm_conf.get("use_lora", False) or kwargs.get("infer_use_lora",False):
                outputs = self.llm.base_model.model.model(
                    # inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
            else:
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
            if isinstance(prompt_codec, str):
                prompt_codec = np.load(prompt_codec)
                prompt_codec = prompt_codec.tolist()
                prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
        else:
            prompt_codec = None

        #prompt_wav = contents["meta_dict"].get("prompt_wav", None)
        # fix fyu
        prompt_wav = None
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

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("use_qlora_int8",False)) and (not kwargs.get("infer_lora_merged",False)):
            text_embeder = self.llm.base_model.model.model.get_input_embeddings()
        elif hasattr(self, 'llm_quant'):
            text_embeder = self.llm_quant.model.get_input_embeddings()
        else:
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
            from .llm_decoding import LLMDecoder
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
            self.llm.base_model.model.lm_head = self.codec_head.to(dtype)
        else:
            self.llm.lm_head = self.codec_head.to(dtype)
        avoid_token = self.get_avoid_token(llm_generator_conf.get("avoid_token", None))
        if streaming_mode == "false":
            gen_codec, hit_eos, states = self.llm_generator(
                inputs_emb,
                self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                states=kwargs.get("states", {}),
                max_length=max_length,
                min_length=min_length,
                avoid_token=avoid_token,
                quantize=True if hasattr(self, 'llm_quant') else False,
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
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=None,
                    include_eos=True,
                    custom_eos_token=eos_tokens,
                    avoid_token=avoid_token,
                    quantize=True if hasattr(self, 'llm_quant') else False,
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
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=chunk_len,
                    min_length=None,
                    avoid_token=avoid_token,
                    quantize=True if hasattr(self, 'llm_quant') else False,
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
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=None,
                    include_eos=False,
                    quantize=True if hasattr(self, 'llm_quant') else False,
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

        return gen_codec, hit_eos, states


@tables.register("model_classes", "CosyVoiceGroupLMSimpleSys")
class CosyVoiceGroupLMSimpleSys(nn.Module):
    """
        Support group and delay-pattern
    """

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
        self.llm_conf = llm_conf
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        init_hf_model_path = llm_conf.get("init_hf_model_path", None)
        lm_init_hf_model_path = kwargs.get("lm_init_hf_model_path","")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        self.sample_rate = kwargs.get("sample_rate", 24000)
        self.token_rate = kwargs.get("token_rate", 25)

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            device_map=None,
            **llm_load_kwargs,
        )

        tokenizer_vocab_size = llm_conf.get("vocab_size", kwargs.get("vocab_size", None))
        if tokenizer_vocab_size == -1:
            tokenizer_vocab_size = None
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_vocab_size is not None and tokenizer_vocab_size > model_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size)

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        self.token_embed = nn.Embedding(2, llm_dim, 0)

        # del self.llm.lm_head
        self.group_size = llm_conf.get("group_size", 1)
        if self.group_size > 1:
            codec_dim = 1024
        else:
            codec_dim = llm_dim
        self.codec_unit = kwargs.get("codec_unit", [4097])
        self.codec_embed = nn.ModuleList([
            nn.Embedding(self.codec_unit[i], codec_dim, 0) for i in range(len(self.codec_unit))
        ])
        # self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)

        # Token grouping related
        self.group_decoder = None
        self.group_encoder = None
        self.group_decoder_lm_head = None
        self.group_encoder_downcast = None
        self.hidden_states_downcast = None
        self.hidden_states_upcast = None
        if self.group_size > 1:
            self.group_decoder = Qwen2Model(
                Qwen2Config(
                    hidden_size=1024,
                    num_hidden_layers=16,
                    num_attention_heads=64,
                    num_key_value_heads=64,
                    head_dim=16,
                    intermediate_size=4096,
                    attention_dropout=0.1
                    )
                )
            self.group_decoder.embed_tokens = None
            self.group_encoder = Qwen2Model(
                Qwen2Config(
                    hidden_size=1024,
                    num_hidden_layers=6,
                    num_attention_heads=64,
                    num_key_value_heads=64,
                    head_dim=16,
                    intermediate_size=4096,
                    attention_dropout=0.1
                    )
                )
            self.group_encoder.embed_tokens = None

            self.group_decoder_lm_head = nn.ModuleList([
                nn.Linear(
                    1024,
                    self.codec_unit[i],
                    bias=False,
                ) for i in range(len(self.codec_unit))
            ])

            self.group_encoder_downcast = nn.Linear(
                1024 * self.group_size,
                self.llm.config.hidden_size,
                bias=False,
            )
            self.hidden_states_downcast = nn.Linear(
                self.llm.config.hidden_size,
                1024,
                bias=False,
            )
            self.hidden_states_upcast = nn.Linear(
                1024,
                self.llm.config.hidden_size,
                bias=False,
            )

        self.length_normalized_loss = length_normalized_loss

        dataset_conf = kwargs.get("dataset_conf", {})
        self.sos = torch.tensor([dataset_conf.get("sos", 0)], dtype=torch.int32)
        self.eos = torch.tensor([dataset_conf.get("eos", 1024)], dtype=torch.int32)
        self.turn_of_speech = torch.tensor([dataset_conf.get("turn_of_speech", 1)], dtype=torch.int32)
        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=None,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

    def get_div5_mask_padded(self, lengths, group_size, max_output_len=None):
        """
        返回一个 padded tensor，shape: [B, max_mask_len]
        """
        device = lengths.device
        ceil_lens = (lengths + group_size - 1) // group_size  # ceil(L/group_size)
        num_ones = lengths // group_size

        if max_output_len is None:
            max_output_len = ceil_lens.max().item()

        # 生成位置索引
        positions = torch.arange(0, max_output_len, device=device).unsqueeze(0)  # [1, T]

        # mask_len_mask: [B, T]，表示每个位置是否在有效长度内
        mask_len_mask = positions < ceil_lens.unsqueeze(1)  # [B, T]

        # value_mask: 前 num_ones 个为 1
        value_mask = positions < num_ones.unsqueeze(1)  # [B, T]

        # 结合：只在有效长度内才是 1，否则 0
        mask = value_mask.float() * mask_len_mask.float()

        return mask.squeeze(1)

    def apply_delay_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (step, channel) 张量转换为 (step + channel - 1, channel) 的 delay pattern。

        Args:
            x: Tensor of shape (step, channel)

        Returns:
            y: Tensor of shape (step + channel - 1, channel)
        """
        B, T, step, channel, H = x.shape
        total_steps = step + channel - 1

        # 初始化输出张量（全零）
        y = torch.zeros(B, T, total_steps, channel, H, dtype=x.dtype, device=x.device)

        # 对每个 channel 维度，将其内容放入偏移后的位置
        for c in range(channel):
            start = c
            end = start + step
            y[:, :, start:end, c, :] = x[:, :, :, c, :]

        return y

    def _prepare_input_embeds(
        self,
        text_ids: torch.Tensor,
        text_ids_len: torch.Tensor,
        codec_embeds: torch.Tensor,
        codec_padded_len: torch.Tensor,
        **kwargs,
    ):
        sos = self.sos.to(text_ids.device)
        turn_of_speech = self.turn_of_speech.to(text_ids.device)
        sos_embed = self.token_embed(sos)
        turn_of_speech_embed = self.token_embed(turn_of_speech)

        codec_embeds = torch.stack(codec_embeds).sum(dim=0)
        B, T, H = codec_embeds.shape
        T_group = T // self.group_size

        codec_embeds_mask = self.get_div5_mask_padded(codec_padded_len, self.group_size) # B * T_group

        # Process each group independently
        codec_embeds = codec_embeds.reshape(B * T_group, self.group_size, H)
        output = self.group_encoder(
            inputs_embeds=codec_embeds,
            return_dict=True,
            is_causal=False)
        codec_embeds = output.last_hidden_state

        codec_embeds = codec_embeds.reshape(B, T_group, self.group_size * H)
        codec_embeds = codec_embeds * codec_embeds_mask.unsqueeze(-1)
        codec_embeds = self.group_encoder_downcast(codec_embeds)

        text_ids_mask = text_ids > 0
        text_embeds = self.llm.model.get_input_embeddings()(text_ids * text_ids_mask) * text_ids_mask.unsqueeze(-1)

        input_embeds = []
        attention_mask = []
        for i in range(B):
            input_embed = torch.cat(
                (
                    sos_embed,
                    text_embeds[i, :text_ids_len[i].item()],
                    turn_of_speech_embed,
                    codec_embeds[i, :(codec_padded_len[i].item() + self.group_size - 1) // self.group_size],
                ),
                dim=0
            )
            input_embeds.append(input_embed)
            attention_mask.append(torch.ones(input_embed.shape[0], dtype=torch.int32, device=input_embed.device))
        input_embeds = torch.nn.utils.rnn.pad_sequence(input_embeds, batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return input_embeds, attention_mask

    def _prepare_output_embeds(
        self,
        hidden_states: torch.Tensor,
        text_ids_len: torch.Tensor,
        codec_embeds: torch.Tensor,
        codec_padded_len: torch.Tensor,
        **kwargs,
    ):
        if type(codec_embeds) == list:
            codec_embeds = torch.stack(codec_embeds, dim=2)  # B, T, 8, H_local
            B, T, C, H_local = codec_embeds.shape
            T_group = T // self.group_size
            codec_embeds_group = codec_embeds.reshape(B, T_group, self.group_size, C, H_local)
            codec_embeds_group = self.apply_delay_pattern(codec_embeds_group)
            codec_embeds_group = codec_embeds_group.sum(dim=3)[:,:,:-1]
        else:
            B, T, H_local = codec_embeds.shape
            T_group = T // self.group_size
            codec_embeds_group = codec_embeds.reshape(B, T_group, self.group_size, H_local)
            codec_embeds_group = codec_embeds_group[:,:,:-1,:]

        output_embeds = []
        for i in range(B):
            start = text_ids_len[i]+1
            end = start + codec_padded_len[i].item() // self.group_size
            output_embed = hidden_states[i, start:end, :]
            output_embeds.append(output_embed)
        output_embeds = torch.nn.utils.rnn.pad_sequence(output_embeds, batch_first=True, padding_value=0.0)

        output_embeds = self.hidden_states_downcast(output_embeds)
        output_embeds = output_embeds.unsqueeze(2)
        output_embeds = torch.cat((output_embeds, codec_embeds_group), dim=2) # B, T_group, delayed_iters, H_local
        delayed_iters = output_embeds.shape[2]
        output_embeds = output_embeds.reshape(B * T_group, delayed_iters, H_local)
        output = self.group_decoder(
            inputs_embeds=output_embeds,
            return_dict=True,
            is_causal=True)
        output_embeds = output.last_hidden_state
        output_embeds = output_embeds.reshape(B, T_group, delayed_iters, H_local)

        return output_embeds


    def forward(
        self,
        text_ids: torch.Tensor = None,
        text_ids_len: torch.Tensor = None,
        codec_padded: torch.Tensor = None,
        codec_padded_len: torch.Tensor = None,
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

        codec_padded_mask = codec_padded > 0
        codec_embeds = []
        for idx in range(len(self.codec_embed)):
            codec_embed = self.codec_embed[idx](codec_padded[...,idx] * codec_padded_mask[...,idx])
            codec_embed *= codec_padded_mask[...,idx].unsqueeze(-1)
            codec_embeds.append(codec_embed)

        inputs_embeds, attention_mask = self._prepare_input_embeds(
            text_ids,
            text_ids_len,
            codec_embeds,
            codec_padded_len,
            **kwargs,
        )

        batch_size, token_num, dims = inputs_embeds.shape
        with torch.amp.autocast(
            device_type="cuda", enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            outputs = self.llm.model(
                # inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            labels = codec_padded.to(torch.int64)
            labels[labels < 0] = self.ignore_id
            # loss = model_outputs.loss
            hidden_states = outputs[0]
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            output_embeds = self._prepare_output_embeds(
                hidden_states,
                text_ids_len,
                codec_embeds,
                codec_padded_len,
            )
            logits = []
            for i in range(len(self.group_decoder_lm_head)):
                logits_i = self.group_decoder_lm_head[i](output_embeds)
                logits_i = logits_i[:, :, i:i+self.group_size, :]
                B, T_group, _, classes = logits_i.shape
                logits_i = logits_i.reshape(B, -1, classes)
                logits.append(logits_i)

        loss = 0.0
        for i in range(len(logits)):
            shift_logits = logits[i].float().contiguous()
            shift_labels = labels[:,:,i].contiguous()
            loss += self.criterion_ce(shift_logits, shift_labels, size=logits[i].shape[-1])
        loss /= len(logits)

        with torch.no_grad():
            for i in range(len(logits)):
                preds_i = torch.argmax(logits[i], -1)
                acc_i = compute_accuracy(preds_i, labels[:,:,i], ignore_label=self.ignore_id)
                stats[f"acc_{i}"] = acc_i

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        # stats["batch_size_real_tokens"] = attention_mask.sum().item()
        # stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        stats["dialog_turns_max"] = 1
        stats["dialog_turns_avg"] = 1

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels > 0 + 1).sum())
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

        # single turn
        raw_text = ""
        sys_instruct = contents["system"][0] + "<|endofprompt|>"
        text = contents["user"][0]
        if text.startswith("语音合成："):
            text = text.replace("语音合成：", "")
        if text.startswith("Speech Synthesis:"):
            text = text.replace("Speech Synthesis:", "")

        # group mode
        all_text_ids = tokenizer.encode(text)
        text_lengths = torch.tensor([len(all_text_ids)], dtype=torch.int64)
        raw_text = raw_text + text
        prompt_text_lengths = None
        if prompt_text is not None and lm_use_prompt:
            prompt_text_ids = tokenizer.encode(prompt_text)
            prompt_text_lengths = torch.tensor([len(prompt_text_ids)], dtype=torch.int64)
            all_text_ids = prompt_text_ids + all_text_ids
            raw_text = prompt_text + raw_text
        if kwargs.get("use_sys_instruct", False):
            hint_once("use_sys_instruct: True", "use_sys_instruct")
            all_text_ids = tokenizer.encode(sys_instruct) + all_text_ids
            raw_text = sys_instruct + raw_text
        all_text_ids = torch.tensor(all_text_ids, dtype=torch.int64)[None, :]

        logging.info(f"generating for {uttid}: input text: {raw_text}, input ids: {all_text_ids}")
        output = {
            "all_text_ids": all_text_ids,
            "text_lengths": text_lengths,
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
        kwargs["uttid"] = uttid
        output = self.load_data(
            contents, **kwargs
        )
        batch = to_device(output, kwargs["device"])
        return batch

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
        batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        all_text_ids = batch["all_text_ids"]               # 包括sys_instruct/prompt_text/text
        text_lengths = batch["text_lengths"]               # 只包括text
        prompt_text_lens = batch["prompt_text_lengths"]
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
            from funasr.models.cosyvoice.utils.llm_decoding import GroupLLMDecoder
            self.llm_generator = GroupLLMDecoder(
                token_embeder=self.codec_embed,
                group_size=self.group_size,
                **llm_generator_conf
            ).to(dtype)

        # self.llm.lm_head = self.codec_head.to(dtype)
        avoid_token = self.get_avoid_token(llm_generator_conf.get("avoid_token", None))

        gen_codec, hit_eos, states = self.llm_generator(
            all_text_ids,
            prompt_codec,
            self.llm,
            self.codec_embed,
            self.group_encoder,
            self.group_decoder,
            self.group_decoder_lm_head,
            self.group_encoder_downcast,
            self.hidden_states_downcast,
            self.hidden_states_upcast,
            states=kwargs.get("states", {}),
            max_length=max_length,
            min_length=min_length,
            avoid_token=avoid_token,
            **kwargs
        )

        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        return gen_codec, hit_eos, states


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


@tables.register("model_classes", "CosyVoiceMixedLMSimpleSysV2")
class CosyVoiceMixedLMSimpleSysV2(nn.Module):
    """
        Support online token extraction
    """

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
        self.llm_conf = llm_conf
        self.llm = None

        from transformers import AutoModelForCausalLM

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        init_hf_model_path = llm_conf.get("init_hf_model_path", None)
        lm_init_hf_model_path = kwargs.get("lm_init_hf_model_path","")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        self.moe_aux_loss = llm_conf.get("moe_aux_loss", False)

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            # attn_implementation="flash_attention_2",
            device_map=None,
            **llm_load_kwargs,
        )

        tokenizer_vocab_size = llm_conf.get("vocab_size", kwargs.get("vocab_size", None))
        if tokenizer_vocab_size == -1:
            tokenizer_vocab_size = None
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_vocab_size is not None and tokenizer_vocab_size > model_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size)

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        self.length_normalized_loss = kwargs.get("length_normalized_loss", False)
        self.instruction_ce_loss = kwargs.get("instruction_ce_loss", False)
        if not self.instruction_ce_loss:
            del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)

        self.length_normalized_loss = length_normalized_loss

        dataset_conf = kwargs.get("dataset_conf", {})
        self.codebook_size = dataset_conf.get("codebook_size", 4096)
        self.sos = torch.tensor([dataset_conf.get("sos", self.codebook_size)], dtype=torch.int32)
        self.eos = torch.tensor([dataset_conf.get("eos", self.codebook_size + 1)], dtype=torch.int32)
        self.turn_of_speech = torch.tensor([dataset_conf.get("turn_of_speech", self.codebook_size + 2)], dtype=torch.int32)
        self.fill_token = torch.tensor([dataset_conf.get("fill_token", self.codebook_size + 3)], dtype=torch.int32)
        self.ignore_id = dataset_conf.get("ignore_id", -100)

        self.text_codec_mix_ratio = dataset_conf.get("text_codec_mix_ratio", [1, 2])
        self.mixup_prob = dataset_conf.get("mixup_prob", 0.5)
        self.text_codec_mix_tag = dataset_conf.get("text_codec_mix_tag", False)
        if self.text_codec_mix_tag:
            logging.info(
                f"text_codec_mix_tag enabled in V2 model, ratio={self.text_codec_mix_ratio}"
            )

        self.criterion_ce = LabelSmoothingLoss(
            size=self.codec_unit,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
        if self.instruction_ce_loss:
            self.criterion_ce_instruct = LabelSmoothingLoss(
                size=model.get_output_embeddings().weight.shape[0],
                padding_idx=self.ignore_id,
                smoothing=kwargs.get("lsm_weight", 0.0),
                normalize_length=self.length_normalized_loss,
            )

        self.speech_token_extractor_conf = llm_conf.get("speech_token_extractor_conf", {})
        self.speech_token_extractor_enabled = self.speech_token_extractor_conf.get("enabled", True)
        if self.speech_token_extractor_enabled:
            print (f'SpeechTokenExtraction enabled, initializing speech token extractor')
            self._init_speech_token_extractor()
            self.frontend = kwargs.get("frontend", None)
        self.unified = kwargs.get('unified', False)
        self.duration_embedding = kwargs.get('duration_embedding', False)
        self.duration_embedding_type = kwargs.get('duration_embedding_type', 'pe')
        if self.duration_embedding is True:
            if self.duration_embedding_type == 'pe':
                from funasr.models.transformer.embedding import PositionalEncoding
                self.duration_embedding_layer = nn.Embedding(5000, llm_dim)
                self.duration_embedding_layer.weight.data = PositionalEncoding(llm_dim, 0, max_len=5000).pe.squeeze(0)
            elif self.duration_embedding_type == 'rope':
                from transformers.models.qwen2.modeling_qwen2 import rotate_half
                self.rope_sin, self.rope_cos = self.llm.model.rotary_emb(torch.zeros(1), torch.tensor([i for i in range(5000)]).view(1, -1))
                self.rope_sin_rotate, self.rope_cos_rotate = rotate_half(self.rope_sin), rotate_half(self.rope_cos)
                self.duration_embedding_layer = nn.Embedding(5000, 256)
                self.duration_embedding_layer.weight.data = torch.concat([self.rope_sin, self.rope_cos, self.rope_sin_rotate, self.rope_cos_rotate], dim=-1).squeeze(dim=0)
                self.duration_embedding_proj = nn.Linear(256, llm_dim)
            else:
                raise NotImplementedError
            for m, n in self.duration_embedding_layer.named_parameters():
                n.detach_()
        self.speaker_embedding = kwargs.get('speaker_embedding', False)
        if self.speaker_embedding is True:
            self.speaker_embedding_proj = nn.Linear(192, llm_dim)

        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

    def _init_speech_token_extractor(self):
        self.speech_token_extractor_type = self.speech_token_extractor_conf.get("type", "automodel")
        if self.speech_token_extractor_type == "automodel":
            print (f'SpeechTokenExtraction use automodel')
            from funasr.auto.auto_model import AutoModel
            self.speech_token_extractor = AutoModel(**self.speech_token_extractor_conf.get("automodel_conf", {}))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.speech_token_extractor_device = (
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
            self.speech_token_extractor.model = self.speech_token_extractor.model.to(
                self.speech_token_extractor_device
            )
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
                                        device=self.speech_token_extractor_device)
                codec = token_result[0]["token"]
                codec_len = token_result[0]["token_lengths"]
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")
        return codec, codec_len

    def mixup_text_codec(self, text: torch.Tensor, codec: torch.Tensor,
                         text_is_tag: torch.Tensor = None, **kwargs):
        sos = self.sos.to(text.device)
        eos = self.eos.to(text.device)
        turn_of_speech = self.turn_of_speech.to(text.device)
        fill_token = self.fill_token.to(text.device)
        ignore_id = torch.tensor([self.ignore_id], dtype=torch.int32, device=text.device)
        # split instruct and text
        instruct_end_token = kwargs.get("instruct_end_token", torch.tensor([151646], dtype=torch.int32))
        pos = torch.where(text == instruct_end_token)[0]
        if len(pos) > 0:
            pos = pos[0].item()
        else:
            pos = -1
        instruct = text[:pos+1]
        text_only = text[pos+1:]
        if text_is_tag is not None:
            is_tag_only = text_is_tag[pos+1:].to(text.device)
        else:
            is_tag_only = torch.zeros_like(text_only)

        text_len = text_only.shape[0]
        codec_len = codec.shape[0]
        text_interval, codec_interval = self.text_codec_mix_ratio
        use_tag_aware = (
            getattr(self, "text_codec_mix_tag", False)
            and text_is_tag is not None
        )

        if use_tag_aware:
            # 按「数到 text_interval 个非 tag token 关闭一个 chunk」的不定长切分
            chunks_tok = []          # 每个 chunk 的 text token 张量
            chunks_first_normal = [] # 每个 chunk 内第一个 normal token 的局部 idx (用于打 fill 标签)
            cur_tokens = []
            cur_first_normal = None
            n_count = 0
            for i in range(text_len):
                tok = text_only[i]
                tag_flag = bool(is_tag_only[i].item())
                if (not tag_flag) and cur_first_normal is None:
                    cur_first_normal = len(cur_tokens)
                cur_tokens.append(tok)
                if not tag_flag:
                    n_count += 1
                    if n_count == text_interval:
                        chunks_tok.append(torch.stack(cur_tokens))
                        chunks_first_normal.append(cur_first_normal)
                        cur_tokens, cur_first_normal, n_count = [], None, 0
            residual_text = (
                torch.stack(cur_tokens)
                if len(cur_tokens) > 0
                else torch.zeros(0, dtype=text.dtype, device=text.device)
            )

            num_chunks = len(chunks_tok)
            exp_codec_len = num_chunks * codec_interval

            if (
                self.mixup_prob is not None
                and random.random() < self.mixup_prob
                and exp_codec_len < codec_len
                and num_chunks > 0
            ):
                in_parts, lbl_parts, tflag_parts = [], [], []
                for k in range(num_chunks):
                    t_chunk = chunks_tok[k]
                    c_chunk = codec[k * codec_interval:(k + 1) * codec_interval]
                    in_parts += [t_chunk, c_chunk]

                    t_lbl = torch.full_like(t_chunk, self.ignore_id)
                    fn = chunks_first_normal[k]
                    if fn is not None:
                        t_lbl[fn] = fill_token.item()
                    lbl_parts += [t_lbl, c_chunk]

                    tflag_parts += [
                        torch.ones_like(t_chunk),
                        torch.zeros_like(c_chunk),
                    ]

                mixed_input = torch.cat(in_parts, dim=0)
                mixed_label = torch.cat(lbl_parts, dim=0)
                mixed_tflag = torch.cat(tflag_parts, dim=0)
                res_codec = codec[exp_codec_len:]

                input_ids = torch.cat(
                    [sos, instruct, mixed_input,
                     residual_text, turn_of_speech, res_codec, eos],
                    dim=0,
                )

                res_text_label = torch.full_like(residual_text, self.ignore_id)
                if res_text_label.shape[0] >= 1:
                    res_text_label[0] = fill_token.item()
                labels = torch.cat(
                    [ignore_id, torch.full_like(instruct, self.ignore_id),
                     mixed_label,
                     res_text_label, ignore_id,
                     res_codec, eos],
                    dim=0,
                )

                text_flag = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=text.device),
                        torch.ones_like(instruct),
                        mixed_tflag,
                        torch.ones_like(residual_text),
                        torch.zeros(1, dtype=torch.int32, device=text.device),
                        torch.zeros_like(res_codec),
                        torch.zeros(1, dtype=torch.int32, device=text.device),
                    ],
                    dim=0,
                )
                codec_flag = 1 - text_flag
                
            else:
                text_len_total = instruct.shape[0] + text_only.shape[0]
                sequence = [sos, instruct, text_only, turn_of_speech, codec, eos]
                input_ids = torch.cat(sequence, dim=0)
                text_flag = torch.zeros(input_ids.shape[0], dtype=torch.int32, device=text.device)
                text_flag[1:text_len_total + 1] = 1
                codec_flag = 1 - text_flag
                labels = torch.cat(sequence, dim=0)
                labels[:text_len_total + 2] = self.ignore_id

        else:
            exp_intervals = text_len // text_interval
            exp_codec_len = exp_intervals * codec_interval
            if (
                    self.mixup_prob is not None
                    and random.random() < self.mixup_prob
                    and exp_codec_len < codec_len
            ):
                # streaming mode
                exp_text_len = exp_intervals * text_interval
                # Calculate the length of the mixed sequence
                mixed_length = exp_text_len + exp_codec_len

                # build mixed inputs
                mixed_seq = torch.zeros(mixed_length, dtype=torch.int32, device=text.device)
                for i in range(text_interval):
                    mixed_seq[i::text_interval+codec_interval] = text_only[i:exp_text_len:text_interval]
                for i in range(codec_interval):
                    mixed_seq[i+text_interval::text_interval+codec_interval] = codec[i:exp_codec_len:codec_interval]
                sequence = [
                    sos, instruct, mixed_seq,
                    text_only[exp_text_len:], turn_of_speech,
                    codec[exp_codec_len:], eos
                ]
                input_ids = torch.cat(sequence, dim=0)

                # build labels
                mixed_seq = torch.zeros(mixed_length, dtype=torch.int32, device=text.device)
                for i in range(text_interval):
                    if i == 0:
                        mixed_seq[i::text_interval + codec_interval] = fill_token
                    else:
                        mixed_seq[i::text_interval + codec_interval] = ignore_id
                for i in range(codec_interval):
                    mixed_seq[i + text_interval::text_interval + codec_interval] = codec[i:exp_codec_len:codec_interval]
                res_text_label = torch.ones_like(text_only[exp_text_len:]) * ignore_id
                if res_text_label.shape[0] >= 1:
                    res_text_label[0] = fill_token
                label_sequence = [
                    ignore_id, torch.ones_like(instruct) * ignore_id,
                    mixed_seq,
                    res_text_label, ignore_id,
                    codec[exp_codec_len:], eos
                ]
                labels = torch.cat(label_sequence, dim=0)

                # build input flags
                flag_seq = torch.zeros(mixed_length, dtype=torch.int32, device=text.device)
                for i in range(text_interval):
                    flag_seq[i::text_interval + codec_interval] = 1
                text_flag = [
                    torch.zeros(1, dtype=torch.int32, device=text.device), torch.ones_like(instruct), flag_seq,
                    torch.ones_like(text_only[exp_text_len:]), torch.zeros(1, dtype=torch.int32, device=text.device),
                    torch.zeros_like(codec[exp_codec_len:]), torch.zeros(1, dtype=torch.int32, device=text.device)
                ]
                text_flag = torch.cat(text_flag, dim=0)
                codec_flag = 1 - text_flag

            else:
                text_len_total = instruct.shape[0] + text_only.shape[0]
                sequence = [sos, instruct, text_only, turn_of_speech,
                            codec, eos]
                input_ids = torch.cat(sequence, dim=0)
                text_flag = torch.zeros(input_ids.shape[0], dtype=torch.int32, device=text.device)
                text_flag[1:text_len_total+1] = 1
                codec_flag = 1 - text_flag
                sequence = [sos, instruct, text_only, turn_of_speech,
                            codec, eos]
                labels = torch.cat(sequence, dim=0)
                labels[:text_len_total+2] = self.ignore_id

        labels_instruct = torch.ones_like(labels) * ignore_id
        labels_instruct[1:len(instruct)] = instruct[:-1]

        return input_ids, labels, text_flag, codec_flag, labels_instruct

    def _prepare_input_embeds(
        self,
        text_ids: torch.Tensor,
        text_ids_len: torch.Tensor,
        audio_feat: torch.Tensor,
        audio_feat_len: torch.Tensor,
        text_is_tag: torch.Tensor = None,
        **kwargs,
    ):
        codec, codec_len = self._extract_speech_token(audio_feat=audio_feat, audio_feat_lengths=audio_feat_len)

        # process one by one
        input_ids, labels, text_flag, codec_flag, labels_instruct = [], [], [], [], []
        attention_mask = []
        for i in range(text_ids.shape[0]):
            cur_is_tag = (
                text_is_tag[i, :text_ids_len[i]]
                if text_is_tag is not None
                else None
            )
            _input_ids, _labels, _text_flag, _codec_flag, _labels_instruct = self.mixup_text_codec(
                text_ids[i, :text_ids_len[i]],
                codec[i, :codec_len[i]],
                text_is_tag=cur_is_tag,
                **kwargs,
            )
            input_ids.append(_input_ids)
            labels.append(_labels)
            text_flag.append(_text_flag)
            codec_flag.append(_codec_flag)
            labels_instruct.append(_labels_instruct)
            attention_mask.append(torch.ones(_input_ids.shape[0], dtype=torch.int32, device=_input_ids.device))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=-1)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        text_flag = torch.nn.utils.rnn.pad_sequence(text_flag, batch_first=True, padding_value=-1)
        codec_flag = torch.nn.utils.rnn.pad_sequence(codec_flag, batch_first=True, padding_value=-1)
        labels_instruct = torch.nn.utils.rnn.pad_sequence(labels_instruct, batch_first=True, padding_value=-1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        input_mask = input_ids >= 0
        text_flag = (text_flag > 0) * input_mask
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        codec_flag = (codec_flag > 0) * input_mask
        codec_embeds = self.codec_embed(input_ids * codec_flag) * codec_flag.unsqueeze(-1)
        inputs_embeds = inputs_embeds + codec_embeds

        # NOTE 加dur_emb/spk_emb/emo_emb
        embs_to_add = []
        if self.speaker_embedding is True and kwargs.get("spk_emb", None) is not None:
            spk_emb = kwargs["spk_emb"]
            spk_emb = self.speaker_embedding_proj(spk_emb)
            if random.random() < 0.3:
                spk_emb.zero_()
            embs_to_add.append(spk_emb)
        if self.duration_embedding is True:
            if self.duration_embedding_type == 'pe':
                dur_emb = self.duration_embedding_layer(codec_len).unsqueeze(1)
            elif self.duration_embedding_type == 'rope':
                dur_emb = self.duration_embedding_proj(self.duration_embedding_layer(attention_mask.sum(dim=-1)).unsqueeze(1))
            else:
                raise NotImplementedError
            if random.random() < 0.7:
                dur_emb.zero_()
            embs_to_add.append(dur_emb)
        if len(embs_to_add) > 0:
            B = input_ids.size(0)
            embeddings_to_add = torch.cat(embs_to_add, dim=1)
            inputs_embeds = torch.cat([inputs_embeds[:, :1], embeddings_to_add, inputs_embeds[:, 1:]], dim=1)
            labels = torch.cat([labels[:, :1], torch.full((B, len(embs_to_add)), self.ignore_id, dtype=torch.int32, device=labels.device), labels[:, 1:]], dim=1)
            labels_instruct = torch.cat([labels_instruct[:, :1], torch.full((B, len(embs_to_add)), self.ignore_id, dtype=torch.int32, device=labels_instruct.device), labels_instruct[:, 1:]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :1], torch.ones((B, len(embs_to_add)), dtype=torch.int32, device=attention_mask.device), attention_mask[:, 1:]], dim=1)

        return inputs_embeds, labels, labels_instruct, attention_mask

    def forward(
        self,
        text_ids: torch.Tensor = None,
        text_ids_len: torch.Tensor = None,
        audio_feat: torch.Tensor = None,
        audio_feat_len: torch.Tensor = None,
        text_is_tag: torch.Tensor = None,
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

        inputs_embeds, labels, labels_instruct, attention_mask = self._prepare_input_embeds(
            text_ids,
            text_ids_len,
            audio_feat,
            audio_feat_len,
            text_is_tag=text_is_tag,
            **kwargs,
        )

        batch_size, token_num, dims = inputs_embeds.shape
        with torch.amp.autocast(
            device_type="cuda", enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            outputs = self.llm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_router_logits=self.moe_aux_loss,
            )

            labels = labels.to(torch.int64)
            labels[labels < 0] = self.ignore_id
            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)

            if self.instruction_ce_loss:
                labels_instruct = labels_instruct.to(torch.int64)
                labels_instruct[labels_instruct < 0] = self.ignore_id
                logits_instruct = self.llm.lm_head(hidden_states)

        logits = logits.float()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss, denom = self.criterion_ce(shift_logits, shift_labels, return_denom=True)

        if self.instruction_ce_loss:
            logits_instruct = logits_instruct.float()
            shift_logits_instruct = logits_instruct[..., :-1, :].contiguous()
            shift_labels_instruct = labels_instruct[..., 1:].contiguous()
            loss_instruct, denom_instruct = self.criterion_ce_instruct(shift_logits_instruct, shift_labels_instruct, return_denom=True)
            loss = (loss + loss_instruct) / (denom + denom_instruct)
        else:
            loss = loss / denom

        if self.moe_aux_loss:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.llm.num_experts,
                self.llm.num_experts_per_tok,
                attention_mask,
            )
            loss += self.llm.router_aux_loss_coef * aux_loss.to(loss.device)

        with torch.no_grad():
            preds = torch.argmax(logits, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], labels[:, 1:], ignore_label=self.ignore_id
            )
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        # stats["batch_size_real_tokens"] = attention_mask.sum().item()
        # stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        stats["dialog_turns_max"] = 1
        stats["dialog_turns_avg"] = 1

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels > 0 + 1).sum())

        # NOTE(lyuxiang.lx) 从llm计算结果中索引codec/hidden
        if self.unified is True:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            codec, codec_hidden = [], []
            for i in range(shift_labels.shape[0]):
                index = ((shift_labels[i] >= 0).float() * (shift_labels[i] < self.sos.item()).float()).bool()
                codec.append(shift_labels[i, index])
                codec_hidden.append(shift_hidden_states[i, index])
            stats['codec'] = pad_sequence(codec, batch_first=True, padding_value=0)
            stats['codec_len'] = torch.tensor([i.size(0) for i in codec], dtype=torch.int32).view(-1, 1)
            stats['codec_hidden'] = pad_sequence(codec_hidden, batch_first=True, padding_value=0)
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
            if isinstance(prompt_codec, str) and os.path.exists(prompt_codec):
                prompt_codec = np.load(prompt_codec)
            if isinstance(prompt_codec, np.ndarray):
                prompt_codec = prompt_codec.tolist()
                prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
        else:
            prompt_codec = None

        spk_emb = contents["meta_dict"].get("spk_emb", None)
        if self.speaker_embedding is True:
            if spk_emb is not None:
                if isinstance(spk_emb, str) and spk_emb.startswith("oss://"):
                    spk_emb = oss_remote_numpy_load(spk_emb)
                if isinstance(spk_emb, str) and os.path.exists(spk_emb):
                    spk_emb = np.load(spk_emb)
                if isinstance(spk_emb, np.ndarray):
                    spk_emb = torch.from_numpy(spk_emb)[None, :]
            else:
                spk_emb = torch.zeros([1, 1, 192])

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
            input_source_ids = tokenizer.encode(text) if text else []
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
        # fixed_ratio shares this initial-input construction with true/mixed; it
        # only differs at generation time (inject text by count, not on fill).
        elif streaming_mode in ["true", "mixed", "fixed_ratio"]:
            sos_token = llm_generator_conf.get("sos", 6561)
            tos_token = llm_generator_conf.get("turn_of_speech", 6563)
            instruct_ids = torch.tensor(tokenizer.encode(sys_instruct), dtype=torch.int64)[None, :]
            if prompt_codec is None:
                raw_text = sys_instruct + text
                text_ids = tokenizer.encode(text) if text else []
                text_ids = torch.tensor(text_ids, dtype=torch.int64)[None, :]
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
                text_ids = tokenizer.encode(text) if text else []
                text_ids = torch.tensor(
                    tokenizer.encode(prompt_text) + text_ids,
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
            "spk_emb": spk_emb,
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

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("use_qlora_int8",False)) and (not kwargs.get("infer_lora_merged",False)):
            text_embeder = self.llm.base_model.model.model.get_input_embeddings()
        elif hasattr(self, 'llm_quant'):
            text_embeder = self.llm_quant.model.get_input_embeddings()
        else:
            text_embeder = self.llm.model.get_input_embeddings()
        inputs_text_embeds = text_embeder(source_ids * text_emb_flag) * text_emb_flag.unsqueeze(-1)
        input_codec_embs = self.codec_embed(source_ids * (1-text_emb_flag)) * (1-text_emb_flag).unsqueeze(-1)
        inputs_embeds = inputs_text_embeds + input_codec_embs
        batch["res_text_emb"] = text_embeder(batch["res_text_ids"])

        # NOTE 加dur_emb/spk_emb/emo_emb
        embs_to_add = []
        if self.speaker_embedding is True:
            spk_emb = batch["spk_emb"]
            spk_emb_proj = self.speaker_embedding_proj(spk_emb)
            if spk_emb_proj.dim() == 2:
                spk_emb_proj = spk_emb_proj.unsqueeze(1)
            embs_to_add.append(spk_emb_proj)
        if self.duration_embedding is True:
            target_codec_len = kwargs.get('target_codec_len', -1)
            if target_codec_len == -1:
                dur_emb = torch.zeros(1, 1, inputs_embeds.shape[2]).to(inputs_embeds.device)
            else:
                target_codec_len = torch.tensor([target_codec_len]).to(inputs_embeds.device, torch.int32)
                if self.duration_embedding_type == 'pe':
                    prompt_codec_len = batch["prompt_codec_lengths"][0].item()
                    dur_emb = self.duration_embedding_layer(target_codec_len + prompt_codec_len).unsqueeze(1)
                elif self.duration_embedding_type == 'rope':
                    dur_emb = self.duration_embedding_proj(self.duration_embedding_layer(target_codec_len + inputs_embeds.shape[1]).unsqueeze(1))
                else:
                    raise NotImplementedError
            embs_to_add.append(dur_emb)
        if len(embs_to_add) > 0:
            embeddings_to_add = torch.cat(embs_to_add, dim=1)
            inputs_embeds = torch.cat([inputs_embeds[:, :1], embeddings_to_add, inputs_embeds[:, 1:]], dim=1)
            text_emb_flag = batch['text_emb_flag']
            text_emb_flag = torch.concat([text_emb_flag[:, :1], torch.zeros(1, len(embs_to_add)).to(text_emb_flag), text_emb_flag[:, 1:]], dim=1)
            batch['text_emb_flag'] = text_emb_flag
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
            from .llm_decoding import LLMDecoder
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
            self.llm.base_model.model.lm_head = self.codec_head.to(dtype)
        else:
            self.llm.lm_head = self.codec_head.to(dtype)
        avoid_token = self.get_avoid_token(llm_generator_conf.get("avoid_token", None))
        hiddens = []
        text_emb_flag = batch["text_emb_flag"]
        initial_audio_in_input = int((1 - text_emb_flag).sum().item())
        if streaming_mode == "false":
            outputs = self.llm_generator(
                inputs_emb,
                self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                states=kwargs.get("states", {}),
                max_length=max_length,
                min_length=min_length,
                avoid_token=avoid_token,
                quantize=True if hasattr(self, 'llm_quant') else False,
                return_hidden=self.unified,
                **kwargs
            )
            gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
            if self.unified:
                hiddens += outputs[3]
                text_emb_flag = torch.cat([text_emb_flag, torch.zeros([1, gen_codec.shape[1]], device=text_emb_flag.device)], dim=1)
        elif streaming_mode in ["true", "mixed"]:
            kwargs["use_llm_cache"] = True
            gen_raw_list, gen_codec_list = [], []
            res_text_emb = batch["res_text_emb"]
            text_interval, codec_interval = kwargs.get("text_codec_mix", [5, 15])
            real_eos = llm_generator_conf.get("eos", 6562)
            eos_tokens = [
                real_eos,
                llm_generator_conf.get("fill_token", 6564),
            ]
            states = kwargs.get("states", {})
            # cumulative codec tokens emitted so far (markers excluded); enforces the
            # global min_length across chunks while blocking only the real eos so the
            # fill chunk-switch stays available during the min-length window.
            gen_codec_total = 0
            while True:
                chunk_min_length = (
                    max(0, min_length - gen_codec_total)
                    if min_length is not None else None
                )
                outputs = self.llm_generator(
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=chunk_min_length,
                    min_length_token=[real_eos],
                    include_eos=True,
                    custom_eos_token=eos_tokens,
                    avoid_token=avoid_token,
                    quantize=True if hasattr(self, 'llm_quant') else False,
                    return_hidden=self.unified,
                    **kwargs
                )
                gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
                # last token is always a marker (fill switch or eos) here
                gen_codec_total += gen_codec.shape[1] - 1
                if self.unified:
                    hiddens += outputs[3]
                    text_emb_flag = torch.cat([text_emb_flag, torch.zeros([1, gen_codec.shape[1]-1], device=text_emb_flag.device)], dim=1)

                if gen_codec[0, -1] == llm_generator_conf.get("fill_token", 6564):
                    if res_text_emb.shape[1] < text_interval:
                        tos_idx = torch.tensor(
                            [[llm_generator_conf.get("turn_of_speech", 6563)]],
                            dtype=torch.int64, device=res_text_emb.device
                        )
                        tos_emb = self.codec_embed(tos_idx)
                        inputs_emb = torch.cat([res_text_emb, tos_emb], dim=1)
                        if self.unified:
                            text_emb_flag = torch.cat([text_emb_flag,
                                                torch.ones([1, res_text_emb.shape[1]], device=text_emb_flag.device)], dim=1)


                    else:
                        inputs_emb = res_text_emb[:, :text_interval]
                        if self.unified:
                            text_emb_flag = torch.cat([text_emb_flag,
                                                torch.ones([1, inputs_emb.shape[1]-1], device=text_emb_flag.device)], dim=1)

                    res_text_emb = res_text_emb[:, text_interval:]

                gen_raw_list.append(gen_codec)
                gen_codec_list.append(gen_codec[:, :-1])
                if gen_codec[0, -1] == llm_generator_conf.get("eos", 6562):
                    break

            gen_raw_codes = torch.cat(gen_raw_list, dim=1)
            logging.info(f"{uttid}: gen {gen_raw_codes.shape[1]} raw tokens: {gen_raw_codes[0].cpu().tolist()}")
            gen_codec = torch.cat(gen_codec_list, dim=1)
        elif streaming_mode == "fixed_ratio":
            # Count-based streaming: inject the next text chunk every codec_interval
            # generated codec tokens, WITHOUT relying on the model emitting
            # fill_token (robust when the LM can't reliably predict fill, e.g. GDN).
            # Mirrors the train-time 5:8 interleave and keeps text leading codec.
            # text_emb_flag bookkeeping mirrors the true/mixed branch so the unified
            # hidden/mask alignment assert below still holds: a chunk that ends on
            # the eos marker contributes gen_codec.shape[1]-1 codec hiddens (the
            # marker has no hidden); a count-stopped chunk has no marker, so it
            # contributes the full gen_codec.shape[1].
            kwargs["use_llm_cache"] = True
            gen_raw_list, gen_codec_list = [], []
            res_text_emb = batch["res_text_emb"]
            text_interval, codec_interval = kwargs.get("text_codec_mix", [5, 8])
            eos_token = llm_generator_conf.get("eos", 6562)
            fill_token = llm_generator_conf.get("fill_token", 6564)
            tos_token = llm_generator_conf.get("turn_of_speech", 6563)
            # fill-or-count: let the model emit fill to end a chunk EARLY (natural,
            # variable-length output; avoids the stutter from force-filling
            # codec_interval codec when the model wants to switch sooner). If it
            # does NOT emit fill within codec_interval, the count cap forces the
            # switch (robust to GDN occasionally failing to emit fill).
            states = kwargs.get("states", {})
            text_done = res_text_emb.shape[1] == 0
            # cumulative codec tokens emitted so far (markers excluded); enforces the
            # global min_length across chunks while blocking only the real eos so the
            # fill / count chunk-switch stays available during the min-length window.
            gen_codec_total = 0
            while True:
                chunk_min_length = (
                    max(0, min_length - gen_codec_total)
                    if min_length is not None else None
                )
                outputs = self.llm_generator(
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=(max_length if text_done else codec_interval),
                    min_length=chunk_min_length,
                    min_length_token=[eos_token],
                    include_eos=True,
                    custom_eos_token=[eos_token, fill_token],
                    avoid_token=avoid_token,
                    quantize=True if hasattr(self, 'llm_quant') else False,
                    return_hidden=self.unified,
                    **kwargs
                )
                gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
                last_tok = gen_codec[0, -1].item()
                ended = bool(last_tok == eos_token)
                # fill or eos is a trailing marker with no collected hidden.
                has_marker = ended or bool(last_tok == fill_token)
                gen_codec_total += gen_codec.shape[1] - (1 if has_marker else 0)
                if self.unified:
                    hiddens += outputs[3]
                    n_codec = gen_codec.shape[1] - (1 if has_marker else 0)
                    text_emb_flag = torch.cat(
                        [text_emb_flag,
                         torch.zeros([1, n_codec], device=text_emb_flag.device)], dim=1)
                gen_raw_list.append(gen_codec)
                gen_codec_list.append(gen_codec[:, :-1] if has_marker else gen_codec)
                if ended or text_done:
                    break
                # force-inject the next text chunk by count (no fill needed)
                if res_text_emb.shape[1] < text_interval:
                    tos_idx = torch.tensor(
                        [[tos_token]], dtype=torch.int64, device=res_text_emb.device)
                    tos_emb = self.codec_embed(tos_idx)
                    inputs_emb = torch.cat([res_text_emb, tos_emb], dim=1)
                    if self.unified:
                        text_emb_flag = torch.cat(
                            [text_emb_flag,
                             torch.ones([1, res_text_emb.shape[1]], device=text_emb_flag.device)], dim=1)
                    res_text_emb = res_text_emb[:, res_text_emb.shape[1]:]
                    text_done = True
                else:
                    inputs_emb = res_text_emb[:, :text_interval]
                    if self.unified:
                        text_emb_flag = torch.cat(
                            [text_emb_flag,
                             torch.ones([1, inputs_emb.shape[1] - 1], device=text_emb_flag.device)], dim=1)
                    res_text_emb = res_text_emb[:, text_interval:]
            gen_raw_codes = torch.cat(gen_raw_list, dim=1)
            logging.info(f"{uttid}: [fixed_ratio] gen {gen_raw_codes.shape[1]} raw tokens: {gen_raw_codes[0].cpu().tolist()}")
            gen_codec = torch.cat(gen_codec_list, dim=1)
        else:
            raise NotImplementedError(f"{streaming_mode} is not supported.")

        # get codec tokens only
        if self.unified:
            hiddens = torch.cat(hiddens, dim=1)
            mask_audio = text_emb_flag[0,1:]                  # 输出为text_emb_flag的移位
            assert hiddens.shape[1] == mask_audio.shape[0], \
                f"{uttid}: hiddens {hiddens.shape[1]} != mask_audio {mask_audio.shape[0]}, text_emb_flag {text_emb_flag.shape}"
            mask_audio = mask_audio.cpu().numpy() == 0
            hiddens = hiddens[:,mask_audio,:]                 # 获取音频部分，但包含 dur/spk/emo, <|turnofspeech|>
            # NOTE 仅丢弃 dur/spk/emo 等特殊 token 的 hidden，保留 prompt_codec 段，
            # 使返回的 hidden 覆盖 [prompt_codec + generated]，与 inference_model 中
            # llm_hidden[:, prompt_codec_len:] / [:, :prompt_codec_len] 的切分对齐
            prompt_codec_len = prompt_codec_lens[0].item() if prompt_codec_lens is not None else 0
            hiddens = hiddens[:, initial_audio_in_input - 1 - prompt_codec_len:, :]

        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        if self.unified:
            return gen_codec, hit_eos, states, hiddens
        return gen_codec, hit_eos, states


@tables.register("model_classes", "CosyVoiceMixedLMSimpleSysV2LongContext")
class CosyVoiceMixedLMSimpleSysV2LongContext(nn.Module):
    """
        Support online token extraction
    """

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
        self.llm_conf = llm_conf
        self.llm = None

        from transformers import AutoModelForCausalLM

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        init_hf_model_path = llm_conf.get("init_hf_model_path", None)
        lm_init_hf_model_path = kwargs.get("lm_init_hf_model_path","")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        self.moe_aux_loss = llm_conf.get("moe_aux_loss", False)

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            # attn_implementation="flash_attention_2",
            **llm_load_kwargs,
        )

        tokenizer_vocab_size = llm_conf.get("vocab_size", kwargs.get("vocab_size", None))
        if tokenizer_vocab_size == -1:
            tokenizer_vocab_size = None
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_vocab_size is not None and tokenizer_vocab_size > model_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size)

        if llm_conf.get("use_lora", False):
            del model.lm_head
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
            elif kwargs.get("init_param_lora", None):
                init_param_lora = kwargs.get("init_param_lora", None)
                logging.info(f"init_param_lora: {init_param_lora}")
                model = PeftModel.from_pretrained(model, init_param_lora)
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        self.length_normalized_loss = kwargs.get("length_normalized_loss", False)
        self.instruction_ce_loss = kwargs.get("instruction_ce_loss", False)
        if not self.instruction_ce_loss and not llm_conf.get("use_lora", False):
            del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)

        self.length_normalized_loss = length_normalized_loss

        dataset_conf = kwargs.get("dataset_conf", {})
        self.codebook_size = dataset_conf.get("codebook_size", 4096)
        self.sos = torch.tensor([dataset_conf.get("sos", self.codebook_size)], dtype=torch.int32)
        self.eos = torch.tensor([dataset_conf.get("eos", self.codebook_size + 1)], dtype=torch.int32)
        self.turn_of_speech = torch.tensor([dataset_conf.get("turn_of_speech", self.codebook_size + 2)], dtype=torch.int32)
        self.fill_token = torch.tensor([dataset_conf.get("fill_token", self.codebook_size + 3)], dtype=torch.int32)
        self.ignore_id = dataset_conf.get("ignore_id", -100)
        self.begin_token = torch.tensor([dataset_conf.get("begin_token", 151644)], dtype=torch.int32)
        self.end_token = torch.tensor([dataset_conf.get("end_token", 151645)], dtype=torch.int32)
        self.line_token = torch.tensor([dataset_conf.get("line_token", 198)], dtype=torch.int32)

        self.text_codec_mix_ratio = dataset_conf.get("text_codec_mix_ratio", [1, 2])
        self.mixup_prob = dataset_conf.get("mixup_prob", 0.5)

        self.use_drop_prompt_token = dataset_conf.get("use_drop_prompt_token", False)
        self.data_template_train = dataset_conf.get("data_template", "default")
        self.data_mode = "default"
        
        if self.data_template_train == "qwen3d5-byturn-mode2":
            logging.info(f"data_template: {self.data_template_train}")
            self.data_template_train = "qwen3d5-byturn"
            self.data_mode = "mode2"
        if self.data_template_train == "qwen3d5-byturn-mode3":
            logging.info(f"data_template: {self.data_template_train}")
            self.data_template_train = "qwen3d5-byturn"
            self.data_mode = "mode3"
        if self.data_template_train == "qwen3d5-byturn-mode4":
            logging.info(f"data_template: {self.data_template_train}")
            self.data_template_train = "qwen3d5-byturn"
            self.data_mode = "mode4"

        self.criterion_ce = LabelSmoothingLoss(
            size=self.codec_unit,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
        if self.instruction_ce_loss:
            self.criterion_ce_instruct = LabelSmoothingLoss(
                size=model.get_output_embeddings().weight.shape[0],
                padding_idx=self.ignore_id,
                smoothing=kwargs.get("lsm_weight", 0.0),
                normalize_length=self.length_normalized_loss,
            )

        self.speech_token_extractor_conf = llm_conf.get("speech_token_extractor_conf", {})
        self.speech_token_extractor_enabled = self.speech_token_extractor_conf.get("enabled", True)
        if self.speech_token_extractor_enabled:
            print (f'SpeechTokenExtraction enabled, initializing speech token extractor')
            self._init_speech_token_extractor()
            self.frontend = kwargs.get("frontend", None)
        self.unified = kwargs.get('unified', False)
        self.unified_target_turn = kwargs.get('unified_target_turn', 'LAST_ASSISTANT')
        self.duration_embedding = kwargs.get('duration_embedding', False)
        self.duration_embedding_type = kwargs.get('duration_embedding_type', 'pe')
        if self.duration_embedding is True:
            if self.duration_embedding_type == 'pe':
                from funasr.models.transformer.embedding import PositionalEncoding
                self.duration_embedding_layer = nn.Embedding(5000, llm_dim)
                self.duration_embedding_layer.weight.data = PositionalEncoding(llm_dim, 0, max_len=5000).pe.squeeze(0)
            elif self.duration_embedding_type == 'rope':
                from transformers.models.qwen2.modeling_qwen2 import rotate_half
                self.rope_sin, self.rope_cos = self.llm.model.rotary_emb(torch.zeros(1), torch.tensor([i for i in range(5000)]).view(1, -1))
                self.rope_sin_rotate, self.rope_cos_rotate = rotate_half(self.rope_sin), rotate_half(self.rope_cos)
                self.duration_embedding_layer = nn.Embedding(5000, 256)
                self.duration_embedding_layer.weight.data = torch.concat([self.rope_sin, self.rope_cos, self.rope_sin_rotate, self.rope_cos_rotate], dim=-1).squeeze(dim=0)
                self.duration_embedding_proj = nn.Linear(256, llm_dim)
            else:
                raise NotImplementedError
            for m, n in self.duration_embedding_layer.named_parameters():
                n.detach_()
        self.speaker_embedding = kwargs.get('speaker_embedding', False)
        if self.speaker_embedding is True:
            self.speaker_embedding_proj = nn.Linear(192, llm_dim)

        import os

        rank = int(os.environ.get("RANK", 0))
        self.tokenizer = deepcopy(kwargs['tokenizer'])
        logging.info(f"rank: {rank}, model is builded.")

    def _init_speech_token_extractor(self):
        self.speech_token_extractor_type = self.speech_token_extractor_conf.get("type", "automodel")
        if self.speech_token_extractor_type == "automodel":
            print (f'SpeechTokenExtraction use automodel')
            from funasr.auto.auto_model import AutoModel
            self.speech_token_extractor = AutoModel(**self.speech_token_extractor_conf.get("automodel_conf", {}))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.speech_token_extractor_device = (
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
            self.speech_token_extractor.model = self.speech_token_extractor.model.to(
                self.speech_token_extractor_device
            )
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
                                        device=self.speech_token_extractor_device)
                codec = token_result[0]["token"]
                codec_len = token_result[0]["token_lengths"]
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")
        return codec, codec_len

    def mixup_text_codec(self, text: torch.Tensor, codec: torch.Tensor, mix_random: float = 0.5, ignore_codec: bool = False, **kwargs):
        sos = self.sos.to(text.device)
        eos = self.eos.to(text.device)
        turn_of_speech = self.turn_of_speech.to(text.device)
        fill_token = self.fill_token.to(text.device)
        ignore_id = torch.tensor([self.ignore_id], dtype=torch.int32, device=text.device)
        # split instruct and text
        instruct_end_token = kwargs.get("instruct_end_token", torch.tensor([151646], dtype=torch.int32))
        pos = torch.where(text == instruct_end_token)[0]
        if len(pos) > 0:
            pos = pos[0].item()
        else:
            pos = -1
        instruct = text[:pos+1]
        text = text[pos+1:]

        text_len = text.shape[0]
        codec_len = codec.shape[0]
        text_interval, codec_interval = self.text_codec_mix_ratio
        exp_intervals = text_len // text_interval
        exp_codec_len = exp_intervals * codec_interval

        if (
                self.mixup_prob is not None
                and mix_random < self.mixup_prob
                and exp_codec_len < codec_len
        ):
            # streaming mode
            exp_text_len = exp_intervals * text_interval
            # Calculate the length of the mixed sequence
            mixed_length = exp_text_len + exp_codec_len

            # build mixed inputs
            mixed_seq = torch.zeros(mixed_length, dtype=torch.int32, device=text.device)
            for i in range(text_interval):
                mixed_seq[i::text_interval+codec_interval] = text[i:exp_text_len:text_interval]
            for i in range(codec_interval):
                mixed_seq[i+text_interval::text_interval+codec_interval] = codec[i:exp_codec_len:codec_interval]
            sequence = [
                sos, instruct, mixed_seq,
                text[exp_text_len:], turn_of_speech,
                codec[exp_codec_len:], eos
            ]
            input_ids = torch.cat(sequence, dim=0)

            # build labels
            mixed_seq = torch.zeros(mixed_length, dtype=torch.int32, device=text.device)
            for i in range(text_interval):
                if i == 0:
                    mixed_seq[i::text_interval + codec_interval] = fill_token
                else:
                    mixed_seq[i::text_interval + codec_interval] = ignore_id
            for i in range(codec_interval):
                if ignore_codec:
                    mixed_seq[i + text_interval::text_interval + codec_interval] = ignore_id
                else:
                    mixed_seq[i + text_interval::text_interval + codec_interval] = codec[i:exp_codec_len:codec_interval]
            res_text_label = torch.ones_like(text[exp_text_len:]) * ignore_id
            if res_text_label.shape[0] >= 1:
                res_text_label[0] = fill_token
            if ignore_codec:
                label_sequence = [
                    ignore_id, torch.ones_like(instruct) * ignore_id,
                    mixed_seq,
                    res_text_label, ignore_id,
                    torch.ones_like(codec[exp_codec_len:]) * ignore_id, eos
                ]
            else:
                label_sequence = [
                    ignore_id, torch.ones_like(instruct) * ignore_id,
                    mixed_seq,
                    res_text_label, ignore_id,
                    codec[exp_codec_len:], eos
                ]
            labels = torch.cat(label_sequence, dim=0)

            # build input flags
            flag_seq = torch.zeros(mixed_length, dtype=torch.int32, device=text.device)
            for i in range(text_interval):
                flag_seq[i::text_interval + codec_interval] = 1
            text_flag = [
                torch.zeros(1, dtype=torch.int32, device=text.device), torch.ones_like(instruct), flag_seq,
                torch.ones_like(text[exp_text_len:]), torch.zeros(1, dtype=torch.int32, device=text.device),
                torch.zeros_like(codec[exp_codec_len:]), torch.zeros(1, dtype=torch.int32, device=text.device)
            ]
            text_flag = torch.cat(text_flag, dim=0)
            codec_flag = 1 - text_flag

        else:
            text_len = instruct.shape[0] + text.shape[0]
            sequence = [sos, instruct, text, turn_of_speech,
                        codec, eos]
            input_ids = torch.cat(sequence, dim=0)
            text_flag = torch.zeros(input_ids.shape[0], dtype=torch.int32, device=text.device)
            text_flag[1:text_len+1] = 1
            codec_flag = 1 - text_flag
            if ignore_codec:
                 sequence = [sos, instruct, text, turn_of_speech,
                            torch.ones_like(codec) * ignore_id, eos]
            else:
                sequence = [sos, instruct, text, turn_of_speech,
                            codec, eos]
            labels = torch.cat(sequence, dim=0)
            labels[:text_len+2] = self.ignore_id

        labels_instruct = torch.ones_like(labels) * ignore_id
        labels_instruct[1:len(instruct)] = instruct[:-1]

        return input_ids, labels, text_flag, codec_flag, labels_instruct


    def mixup_text_codec_no_stream(self, text: torch.Tensor, codec: torch.Tensor, sos_number, turn_of_speech_number, eos_number):

        text_len = text.shape[0]
        sequence = [sos_number, *text.tolist(), turn_of_speech_number,
                        *codec.tolist(), eos_number]
        input_ids = torch.tensor(sequence, dtype=torch.int64)
        text_flag = torch.zeros(len(sequence), dtype=torch.int64)
        text_flag[1:text_len+1] = 1
        codec_flag = 1 - text_flag
        sequence = [sos_number, *text.tolist(), turn_of_speech_number,
                        *codec.tolist(), eos_number]


        return input_ids, text_flag, codec_flag


    def _prepare_input_embeds(
        self,
        text_id_list: list,
        text_ids_len_list: list,
        audio_list: list,
        audio_len_list: list,
        **kwargs,
    ):
        codec_list = []
        codec_len_list = []
        # device = "cuda"
        # audio_len_list = torch.tensor(audio_len_list, dtype=torch.int32)
        for i, audio in enumerate(audio_list):
            if audio is not None:
                pad_value = 0.0
                audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=pad_value)
                codec, codec_len = self._extract_speech_token(audio_feat=audio, audio_feat_lengths=torch.tensor(audio_len_list[i], dtype=torch.int32, device=audio.device).unsqueeze(1))
                codec_list.append(codec)
                codec_len_list.append(codec_len)
        if kwargs.get('only_extract_speech_token', False) is True:
            return codec_list, codec_len_list

        # record target turns codec length per batch element for unified mode
        # collect codec lengths for every-other turn from the end (turns -1, -3, -5, ...)
        target_codec_len_list = []
        for i in range(len(codec_len_list)):
            lens = codec_len_list[i]  # tensor of shape [num_turns]
            num_turns = lens.shape[0]
            # indices from the end at step 2: -1, -3, -5, ...
            if self.unified_target_turn == "LAST_ASSISTANT":
                selected_lens = [int(lens[-1].item())]
            elif self.unified_target_turn == "ALL_ASSISTANT":
                selected_lens = [int(lens[-(k * 2 + 1)].item()) for k in range(num_turns) if (k * 2 + 1) <= num_turns]
            elif self.unified_target_turn == "ALL":
                selected_lens = [int(lens[k].item()) for k in range(num_turns-1, -1, -1)]
            else:
                raise ValueError(f"Invalid target turns: {TARGET_TURNS}")
            target_codec_len_list.append(selected_lens)
        instruct_end_token = kwargs.get("instruct_end_token", torch.tensor([151646], dtype=torch.int32))
        # process one by ones
        input_ids, labels, text_flag, codec_flag, labels_instruct = [], [], [], [], []
        attention_mask = []
        for i in range(len(text_id_list)): #B
            text_i = text_id_list[i]
            text_ids_len_i = text_ids_len_list[i]
            codec_i = codec_list[i]
            codec_len_i = codec_len_list[i]
            _input_ids = text_i[0]
            _labels = [-100] * int(_input_ids.shape[0])
            _labels = torch.tensor(_labels,dtype=torch.int64, device=_input_ids.device)
            _text_flag = [1.0] * int(_input_ids.shape[0])
            _text_flag = torch.tensor(_text_flag,dtype=torch.float32, device=_input_ids.device)
            _codec_flag = [0.0] * int(_input_ids.shape[0])
            _codec_flag = torch.tensor(_codec_flag,dtype=torch.float32, device=_input_ids.device)
            _labels_instruct = [1.0] * int(_input_ids.shape[0])
            _labels_instruct = torch.tensor(_labels_instruct,dtype=torch.float32, device=_input_ids.device)

            assert len(text_i) == len(codec_i)+1
            for j in range(len(text_i)-1):
                if self.data_template_train == "qwen3d5-byturn" and self.data_mode in ["mode2","mode3","mode4"] and j==len(text_i)-2:
                    _input_ids = torch.cat([_input_ids, instruct_end_token.to(_input_ids.device)])
                    _labels = torch.cat([_labels, torch.tensor([self.ignore_id]).to(_labels.device)])
                    _text_flag = torch.cat([_text_flag, torch.tensor([1.0]).to(_text_flag.device)])
                    _codec_flag = torch.cat([_codec_flag, torch.tensor([0.0]).to(_codec_flag.device)])
                    _labels_instruct = torch.cat([_labels_instruct, torch.tensor([self.ignore_id]).to(_labels_instruct.device)])
                mix_random = random.random()
                if text_ids_len_i[j+1] > 0 and codec_len_i[j] > 0:
                    ignore_codec = True if self.unified is True and j < len(codec_i)-1 else False
                    _input_ids_j, _labels_j, _text_flag_j, _codec_flag_j, _labels_instruct_j = self.mixup_text_codec(text_i[j+1], codec_i[j,:codec_len_i[j]], mix_random=mix_random, ignore_codec=ignore_codec, **kwargs)
                    if self.data_template_train == "qwen3d5-byturn" and self.data_mode in ["mode3","mode4"] and j==len(text_i)-2:
                        #mode3 最后一轮 用sos 不用imstart 和qwen2原模版一样
                        pass
                    elif self.data_template_train == "qwen3d5-byturn":
                        _input_ids_j = torch.cat([self.begin_token.to(_input_ids_j.device), _input_ids_j[1:], self.end_token.to(_input_ids_j.device), self.line_token.to(_input_ids_j.device)],dim=0)
                        _labels_j = torch.cat([torch.tensor([self.ignore_id]).to(_labels_j.device), _labels_j[1:], torch.tensor([self.ignore_id]).to(_labels_j.device), torch.tensor([self.ignore_id]).to(_labels_j.device)],dim=0)
                        _text_flag_j = torch.cat([torch.tensor([1.0]).to(_text_flag_j.device), _text_flag_j[1:], torch.tensor([1.0]).to(_text_flag_j.device), torch.tensor([1.0]).to(_text_flag_j.device)],dim=0)
                        _codec_flag_j = torch.cat([torch.tensor([0.0]).to(_codec_flag_j.device), _codec_flag_j[1:], torch.tensor([0.0]).to(_codec_flag_j.device), torch.tensor([0.0]).to(_codec_flag_j.device)],dim=0)
                        _labels_instruct_j = torch.cat([torch.tensor([self.ignore_id]).to(_labels_instruct_j.device), _labels_instruct_j[1:], torch.tensor([self.ignore_id]).to(_labels_instruct_j.device), torch.tensor([self.ignore_id]).to(_labels_instruct_j.device)],dim=0)
                    _input_ids = torch.cat([_input_ids, _input_ids_j])
                    _labels = torch.cat([_labels, _labels_j])
                    _text_flag = torch.cat([_text_flag, _text_flag_j])
                    _codec_flag = torch.cat([_codec_flag, _codec_flag_j])
                    _labels_instruct = torch.cat([_labels_instruct, _labels_instruct_j])
                elif text_ids_len_i[j+1] > 0 and codec_len_i[j] == 0:
                    _input_ids_j = text_i[j+1]
                    if self.data_template_train == "qwen3d5-byturn":
                        if self.data_mode == "mode4":
                            if j == 0:
                                _input_ids_j = torch.cat([self.begin_token.to(_input_ids_j.device), _input_ids_j],dim=0)
                            if len(text_i) >= 3 and j == len(text_i)-3:
                                _input_ids_j = torch.cat([_input_ids_j, self.end_token.to(_input_ids_j.device), self.line_token.to(_input_ids_j.device)],dim=0)
                        else:
                            _input_ids_j = torch.cat([self.begin_token.to(_input_ids_j.device), _input_ids_j, self.end_token.to(_input_ids_j.device), self.line_token.to(_input_ids_j.device)],dim=0)
                    _labels_j = torch.full_like(_input_ids_j, self.ignore_id)
                    _text_flag_j = torch.full_like(_input_ids_j, 1)
                    _codec_flag_j = torch.full_like(_input_ids_j, 0)
                    _labels_instruct_j = torch.full_like(_input_ids_j, self.ignore_id)

                    _input_ids = torch.cat([_input_ids, _input_ids_j])
                    _labels = torch.cat([_labels, _labels_j])
                    _text_flag = torch.cat([_text_flag, _text_flag_j])
                    _codec_flag = torch.cat([_codec_flag, _codec_flag_j])
                    _labels_instruct = torch.cat([_labels_instruct, _labels_instruct_j])
                else:
                    log_rank = int(os.environ.get("RANK", 0))
                    logging.info(f"error in 5814 {log_rank} text_ids_len_i {text_ids_len_i[j+1]} codec_len_i {codec_len_i[j]}")
                    import pdb;pdb.set_trace()
            hint_once(self.tokenizer.decode(_input_ids, skip_special_tokens=False), 'tokenizer_decode_check')
            input_ids.append(_input_ids)
            labels.append(_labels)
            text_flag.append(_text_flag)
            codec_flag.append(_codec_flag)
            labels_instruct.append(_labels_instruct)
            attention_mask.append(torch.ones(_input_ids.shape[0], dtype=torch.int32, device=_input_ids.device))
            hint_once(f"_input_ids: {_input_ids}", "_input_ids")
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=-1)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        text_flag = torch.nn.utils.rnn.pad_sequence(text_flag, batch_first=True, padding_value=-1)
        codec_flag = torch.nn.utils.rnn.pad_sequence(codec_flag, batch_first=True, padding_value=-1)
        labels_instruct = torch.nn.utils.rnn.pad_sequence(labels_instruct, batch_first=True, padding_value=-1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        input_mask = input_ids >= 0
        text_flag = (text_flag > 0) * input_mask
        if self.llm_conf.get("use_lora", False):
            inputs_embeds = self.llm.base_model.model.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        else:
            inputs_embeds = self.llm.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        codec_flag = (codec_flag > 0) * input_mask
        codec_embeds = self.codec_embed(input_ids * codec_flag) * codec_flag.unsqueeze(-1)
        inputs_embeds = inputs_embeds + codec_embeds

        return inputs_embeds, labels, labels_instruct, attention_mask, target_codec_len_list

    def forward(
        self,
        text_id_list: list = None,
        text_ids_len_list: list = None,
        audio_list: list = None,
        audio_len_list: list = None,
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
        if kwargs.get('only_extract_speech_token', False) is True:
            return self._prepare_input_embeds(
                text_id_list,
                text_ids_len_list,
                audio_list,
                audio_len_list,
                **kwargs,
            )
        inputs_embeds, labels, labels_instruct, attention_mask, target_codec_len_list = self._prepare_input_embeds(
            text_id_list,
            text_ids_len_list,
            audio_list,
            audio_len_list,
            **kwargs,
        )


        batch_size, token_num, dims = inputs_embeds.shape
        with torch.amp.autocast(
            device_type="cuda", enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            if self.llm_conf.get("use_lora", False):
                outputs = self.llm.base_model.model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_router_logits=self.moe_aux_loss,
                )
            else:
                outputs = self.llm.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_router_logits=self.moe_aux_loss,
                )

            labels = labels.to(torch.int64)
            labels[labels < 0] = self.ignore_id
            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)

            if self.instruction_ce_loss:
                labels_instruct = labels_instruct.to(torch.int64)
                labels_instruct[labels_instruct < 0] = self.ignore_id
                logits_instruct = self.llm.lm_head(hidden_states)

        logits = logits.float()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss, denom = self.criterion_ce(shift_logits, shift_labels, return_denom=True)

        if self.instruction_ce_loss:
            logits_instruct = logits_instruct.float()
            shift_logits_instruct = logits_instruct[..., :-1, :].contiguous()
            shift_labels_instruct = labels_instruct[..., 1:].contiguous()
            loss_instruct, denom_instruct = self.criterion_ce_instruct(shift_logits_instruct, shift_labels_instruct, return_denom=True)
            loss = (loss + loss_instruct) / (denom + denom_instruct)
        else:
            loss = loss / denom

        if self.moe_aux_loss:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.llm.num_experts,
                self.llm.num_experts_per_tok,
                attention_mask,
            )
            loss += self.llm.router_aux_loss_coef * aux_loss.to(loss.device)

        with torch.no_grad():
            preds = torch.argmax(logits, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], labels[:, 1:], ignore_label=self.ignore_id
            )
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size



        stats["dialog_turns_max"] = 1
        stats["dialog_turns_avg"] = 1

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels > 0 + 1).sum())

        # NOTE unified mode: extract codec/hidden from every-other turn from the end (-1, -3, -5, ...)
        # per-sequence: all selected turns are concat in chronological order; codec_len records each turn's length
        if self.unified is True:
            from torch.nn.utils.rnn import pad_sequence
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            codec_list, codec_hidden_list, codec_len_out = [], [], []
            for i in range(shift_labels.shape[0]):
                # all valid codec positions (value in [0, sos))
                all_codec_index = ((shift_labels[i] >= 0).float() * (shift_labels[i] < self.sos.item()).float()).bool()
                all_codec_positions = all_codec_index.nonzero(as_tuple=False).squeeze(-1)
                # tgt_lens: [len_turn_-1, len_turn_-3, ...] (from last turn backwards)
                tgt_lens = target_codec_len_list[i]
                # collect per-turn positions from the back, then reverse for chronological order
                turn_positions_list = []
                end_idx = all_codec_positions.shape[0]
                for tgt_len in tgt_lens:
                    if tgt_len > 0 and end_idx >= tgt_len:
                        turn_pos = all_codec_positions[end_idx - tgt_len: end_idx]
                        turn_positions_list.append(turn_pos)
                    # skip turns with tgt_len == 0 entirely (not written to output)
                    end_idx = max(end_idx - tgt_len, 0)
                # reverse to chronological order (earliest selected turn first)
                turn_positions_list = turn_positions_list[::-1]
                # concat all selected turns along the token dimension
                if turn_positions_list:
                    all_positions = torch.cat(turn_positions_list, dim=0)
                else:
                    all_positions = all_codec_positions[:0]
                codec_list.append(shift_labels[i, all_positions])
                codec_hidden_list.append(shift_hidden_states[i, all_positions])
                # codec_len: [num_selected_turns] lengths in chronological order
                codec_len_out.append(torch.tensor([p.shape[0] for p in turn_positions_list], dtype=torch.int32))
            stats['codec'] = pad_sequence(codec_list, batch_first=True, padding_value=0)
            stats['codec_len'] = pad_sequence(codec_len_out, batch_first=True, padding_value=0)  # [B, num_turns]
            stats['codec_hidden'] = pad_sequence(codec_hidden_list, batch_first=True, padding_value=0)

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def data_template(self, data):
        system, user, assistant, meta_dict = [], [], [], {}
        user_text, user_codec, assistant_text, assistant_codec = [], [], [], []
        target_role = data[-1]['role']
        if target_role == "assistant":
            for i, item in enumerate(data):
                role = item["role"]
                if "wav_path" in item:
                    content = item["wav_path"]
                else:
                    content = item["content"]
                if role == "system":
                    system.append(content)
                elif role == "user":
                    if "user_prompt_text" in item:
                        meta_dict["user_prompt_text"] = item["user_prompt_text"]
                    user.append(content)
                    if "text_content" in item:
                        user_text.append(item["text_content"])
                    if "token" in item:
                        user_codec.append(item["token"])

                elif role == "assistant":
                    if "prompt_codec" in item:
                        meta_dict["prompt_codec"] = item["prompt_codec"]
                    if "prompt_wav" in item:
                        meta_dict["prompt_wav"] = item["prompt_wav"]
                    if "spk_emb" in item:
                        meta_dict["spk_emb"] = item["spk_emb"]
                    assistant.append(content)
                    if "text_content" in item:
                        assistant_text.append(item["text_content"])
                    if "token" in item:
                        assistant_codec.append(item["token"])
        else:
            for i, item in enumerate(data):
                role = item["role"]
                content = item["wav_path"]
                if role == "system":
                    system.append(content)
                elif role == "assistant":
                    if "user_prompt_text" in item:
                        meta_dict["user_prompt_text"] = item["user_prompt_text"]
                    user.append(content)
                    if "text_content" in item:
                        user_text.append(item["text_content"])
                    if "token" in item:
                        user_codec.append(item["token"])

                elif role == "user":
                    if "prompt_codec" in item:
                        meta_dict["prompt_codec"] = item["prompt_codec"]
                    if "prompt_wav" in item:
                        meta_dict["prompt_wav"] = item["prompt_wav"]
                    if "spk_emb" in item:
                        meta_dict["spk_emb"] = item["spk_emb"]
                    assistant.append(content)
                    if "text_content" in item:
                        assistant_text.append(item["text_content"])
                    if "token" in item:
                        assistant_codec.append(item["token"])
        if len(system) == 0:
            system = ["You are a helpful assistant."]
        system = system * len(user)
        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
            "meta_dict": meta_dict,
        }
        if len(user_text) > 0:
            contents["user_text"] = user_text
        if len(user_codec) > 0:
            contents["user_codec"] = user_codec
        if len(assistant_text) > 0:
            contents["assistant_text"] = assistant_text
        if len(assistant_codec) > 0:
            contents["assistant_codec"] = assistant_codec
        return contents


    def load_data(self, contents: dict, **kwargs):
        uttid = kwargs.get("uttid", None)
        streaming_mode = kwargs.get("lm_streaming", False)
        data_template = self.data_template_train
        data_mode = self.data_mode
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
            if isinstance(prompt_codec, str) and os.path.exists(prompt_codec):
                prompt_codec = np.load(prompt_codec)
            if isinstance(prompt_codec, np.ndarray):
                prompt_codec = prompt_codec.tolist()
                prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
        else:
            prompt_codec = None

        spk_emb = contents["meta_dict"].get("spk_emb", None)
        if self.speaker_embedding is True:
            if spk_emb is not None:
                if isinstance(spk_emb, str) and spk_emb.startswith("oss://"):
                    spk_emb = oss_remote_numpy_load(spk_emb)
                if isinstance(spk_emb, str) and os.path.exists(spk_emb):
                    spk_emb = np.load(spk_emb)
                if isinstance(spk_emb, np.ndarray):
                    spk_emb = torch.from_numpy(spk_emb)[None, :]
            else:
                spk_emb = torch.zeros([1, 1, 192])

        # single turn
        raw_text = ""
        if data_template == 'qwen3d5-byturn':
            sys_text = contents["system"][0]
            if 'You are a helpful assistant.' in sys_text:
                system_sys = 'You are a helpful assistant.'
                user_sys = sys_text.replace(system_sys, '').strip()
            sys_instruct = '<|im_start|>'+'system\n'+system_sys+'<|im_end|>\n'
            
            sys_instruct += '<|im_start|>'+'user\n'+user_sys+'<|im_end|>\n'
        else:
            sys_instruct = contents["system"][0] + "<|endofprompt|>"
        text = contents["user"][0]
        if text.startswith("语音合成："):
            text = text.replace("语音合成：", "")
        if text.startswith("Speech Synthesis:"):
            text = text.replace("Speech Synthesis:", "")
        # offline mode
        if streaming_mode == "false":
            input_source_ids = tokenizer.encode(text) if text else []
            raw_text = raw_text + text
            if prompt_text is not None and lm_use_prompt:
                input_source_ids = tokenizer.encode(prompt_text) + input_source_ids
                raw_text = prompt_text + raw_text
            if data_template == 'qwen3d5-byturn':
                assert kwargs.get("use_sys_instruct", False)
                assis_prefix = 'assistant\n'
                assert not text.startswith(assis_prefix)
                if data_mode == 'mode3':
                    pass
                else:
                    input_source_ids = tokenizer.encode(assis_prefix) + input_source_ids
                    raw_text = assis_prefix + raw_text

                sys_ids = tokenizer.encode(sys_instruct)
                raw_text = sys_instruct + raw_text
                sys_len = len(sys_ids)
                all_text_len = len(input_source_ids)
                if data_mode == 'mode2':
                    input_source_ids = [
                        *sys_ids,
                        llm_generator_conf.get("end_prompt_token", 151646),
                        llm_generator_conf.get("begin_token", 151644),
                        *input_source_ids,
                        llm_generator_conf.get("turn_of_speech", 6563),
                    ]
                elif data_mode == 'mode3':
                    input_source_ids = [
                        *sys_ids,
                        llm_generator_conf.get("end_prompt_token", 151646),
                        llm_generator_conf.get("sos", 59049),
                        *input_source_ids,
                        llm_generator_conf.get("turn_of_speech", 6563),
                    ]
                else:
                    input_source_ids = [
                        *sys_ids,
                        llm_generator_conf.get("begin_token", 151644),
                        *input_source_ids,
                        llm_generator_conf.get("turn_of_speech", 6563),
                    ]
            else:
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
            if data_template == 'qwen3d5-byturn':
                if data_mode == 'mode2':
                    text_emb_flag[:,0:sys_len+2+all_text_len] = 1
                elif data_mode == 'mode3':
                    text_emb_flag[:,0:sys_len+2+all_text_len] = 1
                    text_emb_flag[:,sys_len+1] = 0
                else:
                    text_emb_flag[:,0:sys_len+1+all_text_len] = 1
            else:
                text_emb_flag[:, 1:1+all_text_len] = 1
            res_text_ids = [[]]
        # online mode mixed
        elif streaming_mode in ["true", "mixed"]:
            sos_token = llm_generator_conf.get("sos", 6561)
            tos_token = llm_generator_conf.get("turn_of_speech", 6563)
            begin_token = llm_generator_conf.get("begin_token", 151644)
            end_prompt_token = llm_generator_conf.get("end_prompt_token", 151646)
            instruct_ids = torch.tensor(tokenizer.encode(sys_instruct), dtype=torch.int64)[None, :]
            text_interval, codec_interval = text_codec_mix_ratio

            # 构造前缀 ids 和对应 flag（不含目标文本）
            if data_template == 'qwen3d5-byturn':
                last_turn_text = 'assistant\n' + text
                if data_mode == 'mode2':
                    prefix_ids = torch.cat([
                        instruct_ids,
                        torch.tensor([[end_prompt_token]], dtype=torch.int64),
                        torch.tensor([[begin_token]], dtype=torch.int64),
                    ], dim=1)
                else:
                    prefix_ids = torch.cat([
                        instruct_ids,
                        torch.tensor([[begin_token]], dtype=torch.int64),
                    ], dim=1)
                prefix_flag = torch.ones_like(prefix_ids)
            else:
                last_turn_text = text
                prefix_ids = torch.cat([
                    torch.tensor([[sos_token]], dtype=torch.int64),
                    instruct_ids,
                ], dim=1)
                prefix_flag = torch.ones_like(prefix_ids)
                prefix_flag[:, 0] = 0

            raw_text = sys_instruct + last_turn_text
            target_text_ids = torch.tensor(tokenizer.encode(last_turn_text), dtype=torch.int64)[None, :]

            if prompt_codec is None:
                # 无 prompt_codec：纯文本 interleave
                if target_text_ids.shape[1] < text_interval:
                    input_source_ids = torch.cat([
                        prefix_ids,
                        target_text_ids,
                        torch.tensor([[tos_token]], dtype=torch.int64),
                    ], dim=1)
                    text_emb_flag = torch.cat([
                        prefix_flag,
                        torch.ones_like(target_text_ids),
                        torch.tensor([[0]], dtype=torch.int64),
                    ], dim=1)
                else:
                    input_source_ids = torch.cat([
                        prefix_ids,
                        target_text_ids[:, :text_interval],
                    ], dim=1)
                    text_emb_flag = torch.cat([
                        prefix_flag,
                        torch.ones(1, text_interval, dtype=torch.int64),
                    ], dim=1)
                res_text_ids = target_text_ids[:, text_interval:]
            else:
                # 有 prompt_codec：文本 + codec 交织（prompt_text 拼到目标文本前面做 interleave）
                # qwen3d5-byturn 时 assistant\n 加在 prompt_text 前面
                assis_prefix = 'assistant\n' if data_template == 'qwen3d5-byturn' else ''
                raw_text = sys_instruct + assis_prefix + (prompt_text or '') + text
                full_text_ids = torch.tensor(
                    tokenizer.encode(assis_prefix + (prompt_text or '')) + tokenizer.encode(text),
                    dtype=torch.int64
                )[None, :]
                text_len = full_text_ids.shape[1]
                codec_len = prompt_codec.shape[1]
                text_exp_intervals = int(math.floor(text_len / text_interval))
                codec_exp_intervals = int(math.floor(codec_len / codec_interval))
                exp_intervals = min(text_exp_intervals, codec_exp_intervals)
                exp_codec_len = exp_intervals * codec_interval
                exp_text_len = exp_intervals * text_interval
                # 构造交织序列
                mixed_length = exp_text_len + exp_codec_len
                mixed_seq = torch.zeros([1, mixed_length], dtype=torch.int64)
                mixed_flag = torch.ones([1, mixed_length], dtype=torch.int64)
                for i in range(text_interval):
                    mixed_seq[:, i::text_interval + codec_interval] = full_text_ids[:, i:exp_text_len:text_interval]
                for i in range(codec_interval):
                    mixed_seq[:, i + text_interval::text_interval + codec_interval] = prompt_codec[:, i:exp_codec_len:codec_interval]
                    mixed_flag[:, i + text_interval::text_interval + codec_interval] = 0
                inputs_list = [prefix_ids, mixed_seq]
                text_flag_list = [prefix_flag, mixed_flag]
                res_text = full_text_ids[:, exp_text_len:]
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
        else:
            raise NotImplementedError("streaming_mode must be in true, false, mixed")

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
        if not isinstance(res_text_ids, torch.Tensor):
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
            "spk_emb": spk_emb,
        }

        return output

    def load_data_long(self, contents: dict, **kwargs):
        lm_use_prompt = kwargs.get("lm_use_prompt", True)
        tokenizer = kwargs.get("tokenizer")
        prompt_text = kwargs.get("user_prompt_text", None)
        data_template = self.data_template_train
        data_mode = self.data_mode
        drop_user_token = kwargs.get("drop_user_token", False)
        drop_assis_token = kwargs.get("drop_assis_token", False)
        streaming_mode = kwargs.get("lm_streaming", "false")
        if isinstance(streaming_mode, bool):
            streaming_mode = str(streaming_mode).lower()
        llm_generator_conf = kwargs['llm_generator_conf']
        text_codec_mix_ratio = kwargs.get("text_codec_mix", [5, 15])

        if data_template == 'qwen3d5-byturn':
            sys_instruct = '<|im_start|>'+'system\n'+contents["system"][0]+'<|im_end|>\n'
        else:
            sys_instruct = contents["system"][0] + "<|endofprompt|>"

        user_text = contents['user_text']
        user_codec = contents['user_codec']
        assistant_text = contents['assistant_text']
        assistant_codec = contents['assistant_codec']

        long_input_ids = torch.tensor(tokenizer.encode(sys_instruct), dtype=torch.int32)
        sequence = [1.0] * int(long_input_ids.shape[0])
        long_text_flag = torch.tensor(sequence, dtype=torch.int64)

        turn_number = len(user_text)

        #构造context
        if len(user_text) < len(assistant_text):
            #target turn role == user
            for index in range(turn_number):
                text = assistant_text[index]
                token = assistant_codec[index]
                text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
                codec = torch.from_numpy(np.load(token))
                input_ids_tmp, text_flag_tmp, codec_flag_tmp = self.mixup_text_codec_no_stream(
                    text_ids, codec, kwargs['llm_generator_conf'].get("sos", 59049),kwargs['llm_generator_conf'].get("turn_of_speech", 59051),kwargs['llm_generator_conf'].get("eos", 59050)
                )
                long_input_ids = torch.cat((long_input_ids, input_ids_tmp))
                long_text_flag = torch.cat((long_text_flag, text_flag_tmp))
                text = user_text[index]
                token = user_codec[index]
                text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
                codec = torch.from_numpy(np.load(token))
                input_ids_tmp, text_flag_tmp, codec_flag_tmp = self.mixup_text_codec_no_stream(
                    text_ids, codec, kwargs['llm_generator_conf'].get("sos", 59049),kwargs['llm_generator_conf'].get("turn_of_speech", 59051),kwargs['llm_generator_conf'].get("eos", 59050)
                )
                long_input_ids = torch.cat((long_input_ids, input_ids_tmp))
                long_text_flag = torch.cat((long_text_flag, text_flag_tmp))
        else:
            #target turn role == assistant
            for index in range(turn_number):
                text = user_text[index]
                if data_template == 'qwen3d5-byturn':
                    if index == 0 and data_mode == 'mode4':
                        text = 'user\n' + '用户说:'+ text
                    elif data_mode == 'mode4':
                        text = '用户说:'+ text
                    else:
                        text = 'user\n'+text
                token = user_codec[index]
                text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
                if drop_user_token:
                    input_ids_tmp = torch.tensor([kwargs['llm_generator_conf'].get("begin_token", 151644)]+tokenizer.encode(text), dtype=torch.int32)
                    text_flag_tmp = torch.tensor([1.0] * int(input_ids_tmp.shape[0]), dtype=torch.int64)
                    codec_flag_tmp = torch.tensor([0.0] * int(input_ids_tmp.shape[0]), dtype=torch.int64)
                else:
                    codec = torch.from_numpy(np.load(token))
                    input_ids_tmp, text_flag_tmp, codec_flag_tmp = self.mixup_text_codec_no_stream(
                        text_ids, codec, kwargs.get("sos", 59049),kwargs.get("turn_of_speech", 59051),kwargs.get("eos", 59050)
                    )
                if data_template == 'qwen3d5-byturn':
                    if index == 0 and data_mode == "mode4":
                        pass
                    elif data_mode == "mode4":
                        input_ids_tmp = input_ids_tmp[1:]
                        text_flag_tmp = text_flag_tmp[1:]
                        codec_flag_tmp = codec_flag_tmp[1:]
                    else:
                        input_ids_tmp[0] = kwargs['llm_generator_conf'].get("begin_token", 151644)
                        input_ids_tmp = torch.cat([input_ids_tmp, torch.tensor([kwargs['llm_generator_conf'].get("end_token", 151645)]),torch.tensor([kwargs['llm_generator_conf'].get("line_token", 198)])],dim=0)
                        text_flag_tmp[0] = 1
                        text_flag_tmp = torch.cat([text_flag_tmp, torch.tensor([1]), torch.tensor([1])],dim=0)
                        codec_flag_tmp[0] = 0
                        codec_flag_tmp = torch.cat([codec_flag_tmp, torch.tensor([0]), torch.tensor([0])],dim=0)
                # import pdb;pdb.set_trace()
                long_input_ids = torch.cat((long_input_ids, input_ids_tmp))
                long_text_flag = torch.cat((long_text_flag, text_flag_tmp))
                if index < turn_number - 1:
                    text = assistant_text[index]
                    if data_template == 'qwen3d5-byturn':
                        if data_mode == 'mode4':
                            text = '助手说:'+ text
                        else:
                            text = 'assistant\n'+text
                    token = assistant_codec[index]
                    text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
                    if drop_assis_token:
                        input_ids_tmp = torch.tensor([kwargs['llm_generator_conf'].get("begin_token", 151644)]+tokenizer.encode(text), dtype=torch.int32)
                        text_flag_tmp = torch.tensor([1.0] * int(input_ids_tmp.shape[0]), dtype=torch.int64)
                        codec_flag_tmp = torch.tensor([0.0] * int(input_ids_tmp.shape[0]), dtype=torch.int64)
                    else:
                        codec = torch.from_numpy(np.load(token))
                        input_ids_tmp, text_flag_tmp, codec_flag_tmp = self.mixup_text_codec_no_stream(
                            text_ids, codec, kwargs['llm_generator_conf'].get("sos", 59049),kwargs['llm_generator_conf'].get("turn_of_speech", 59051),kwargs['llm_generator_conf'].get("eos", 59050)
                        )
                    if data_template == 'qwen3d5-byturn':
                        if data_mode == 'mode4':
                            input_ids_tmp = input_ids_tmp[1:]
                            text_flag_tmp = text_flag_tmp[1:]
                            codec_flag_tmp = codec_flag_tmp[1:]
                        else:
                            input_ids_tmp[0] = kwargs['llm_generator_conf'].get("begin_token", 151644)
                            input_ids_tmp = torch.cat([input_ids_tmp, torch.tensor([kwargs['llm_generator_conf'].get("end_token", 151645)]),torch.tensor([kwargs['llm_generator_conf'].get("line_token", 198)])],dim=0)
                            text_flag_tmp[0] = 1
                            text_flag_tmp = torch.cat([text_flag_tmp, torch.tensor([1]), torch.tensor([1])],dim=0)
                            codec_flag_tmp[0] = 0
                            codec_flag_tmp = torch.cat([codec_flag_tmp, torch.tensor([0]), torch.tensor([0])],dim=0)
                    # import pdb;pdb.set_trace()
                    long_input_ids = torch.cat((long_input_ids, input_ids_tmp))
                    long_text_flag = torch.cat((long_text_flag, text_flag_tmp))
            if data_template == 'qwen3d5-byturn' and data_mode == 'mode4':
                #如果是mode4 需要在context最后收尾
                input_ids_tmp = torch.cat([torch.tensor([kwargs['llm_generator_conf'].get("end_token", 151645)]),torch.tensor([kwargs['llm_generator_conf'].get("line_token", 198)])],dim=0)
                text_flag_tmp = torch.cat([torch.tensor([1]), torch.tensor([1])],dim=0)
                codec_flag_tmp = torch.cat([torch.tensor([0]), torch.tensor([0])],dim=0)
                long_input_ids = torch.cat((long_input_ids, input_ids_tmp))
                long_text_flag = torch.cat((long_text_flag, text_flag_tmp))


        text = assistant_text[-1]
        if streaming_mode == "false":
            if data_template == 'qwen3d5-byturn':
                if data_mode == 'mode2':
                    text_ids = torch.tensor([kwargs['llm_generator_conf'].get("end_prompt_token", 151646)]+[kwargs['llm_generator_conf'].get("begin_token", 151644)]+tokenizer.encode('assistant\n'+text)+[kwargs['llm_generator_conf'].get("turn_of_speech", 6563)], dtype=torch.int32)
                elif data_mode in ['mode3','mode4']:
                    text_ids = torch.tensor([kwargs['llm_generator_conf'].get("end_prompt_token", 151646)]+[kwargs['llm_generator_conf'].get("sos", 59049)]+tokenizer.encode(text)+[kwargs['llm_generator_conf'].get("turn_of_speech", 6563)], dtype=torch.int32)
                else:
                    text_ids = torch.tensor([kwargs['llm_generator_conf'].get("begin_token", 151644)]+tokenizer.encode('assistant\n'+text)+[kwargs['llm_generator_conf'].get("turn_of_speech", 6563)], dtype=torch.int32)
            else:
                text_ids = torch.tensor([kwargs['llm_generator_conf'].get("sos", 6561)]+tokenizer.encode(text)+[kwargs['llm_generator_conf'].get("turn_of_speech", 6563)], dtype=torch.int32)
            long_input_ids = torch.cat((long_input_ids, text_ids))
            sequence = [1.0] * int(text_ids.shape[0])
            if data_template == 'qwen3d5-byturn':
                if data_mode in ['mode3','mode4']:
                    sequence[1] = 0
                    sequence[-1] = 0
                else:
                    sequence[0] = 1.0
                    sequence[-1] = 0
            else:
                sequence[0] = 0
                sequence[-1] = 0
            text_flag_tmp = torch.tensor(sequence, dtype=torch.int64)
            long_text_flag = torch.cat((long_text_flag, text_flag_tmp))
            res_text_ids = torch.tensor([], dtype=torch.int32).unsqueeze(0)
        elif streaming_mode in ["true", "mixed"]:
            sos_token = llm_generator_conf.get("sos", 6561)
            tos_token = llm_generator_conf.get("turn_of_speech", 6563)
            begin_token = llm_generator_conf.get("begin_token", 151644)
            end_prompt_token = llm_generator_conf.get("end_prompt_token", 151646)

            #多轮暂不支持-最后一轮给 speech prompt
            text_interval = text_codec_mix_ratio[0]
            if data_template == 'qwen3d5-byturn':
                if data_mode in ['mode3','mode4']:
                    last_turn_text = text
                else: 
                    last_turn_text = 'assistant\n'+text
            else:
                last_turn_text = text
            text_ids = torch.tensor(tokenizer.encode(last_turn_text), dtype=torch.int32)
            if text_ids.shape[0] < text_interval:
                if data_mode == 'mode2' and data_template == 'qwen3d5-byturn':
                    new_ids = torch.cat([
                        torch.tensor([end_prompt_token], dtype=torch.int32),
                        torch.tensor([begin_token], dtype=torch.int32),
                        text_ids,
                        torch.tensor([tos_token], dtype=torch.int32),
                    ], dim=0)
                    new_flag = torch.cat([
                        torch.tensor([1], dtype=torch.int64),
                        torch.tensor([1], dtype=torch.int64),
                        torch.ones_like(text_ids, dtype=torch.int64),
                        torch.tensor([0], dtype=torch.int64),
                    ], dim=0)
                elif data_mode in ['mode3','mode4'] and data_template == 'qwen3d5-byturn':
                    new_ids = torch.cat([
                        torch.tensor([end_prompt_token], dtype=torch.int32),
                        torch.tensor([sos_token], dtype=torch.int32),
                        text_ids,
                        torch.tensor([tos_token], dtype=torch.int32),
                    ], dim=0)
                    new_flag = torch.cat([
                        torch.tensor([1], dtype=torch.int64),
                        torch.tensor([0], dtype=torch.int64),
                        torch.ones_like(text_ids, dtype=torch.int64),
                        torch.tensor([0], dtype=torch.int64),
                    ], dim=0)
                else:
                    new_ids = torch.cat([
                        torch.tensor([begin_token if data_template == 'qwen3d5-byturn' else sos_token], dtype=torch.int32),
                        text_ids,
                        torch.tensor([tos_token], dtype=torch.int32),
                    ], dim=0)
                    new_flag = torch.cat([
                        torch.tensor([1 if data_template == 'qwen3d5-byturn' else 0], dtype=torch.int64),
                        torch.ones_like(text_ids, dtype=torch.int64),
                        torch.tensor([0], dtype=torch.int64),
                    ], dim=0)
                res_text_ids = torch.tensor([], dtype=torch.int32).unsqueeze(0)  # [1, 0]
            else:
                if data_mode == 'mode2' and data_template == 'qwen3d5-byturn':
                    new_ids = torch.cat([
                        torch.tensor([end_prompt_token], dtype=torch.int32),
                        torch.tensor([begin_token], dtype=torch.int32),
                        text_ids[:text_interval],
                    ], dim=0)
                    new_flag = torch.cat([
                        torch.tensor([1], dtype=torch.int64),
                        torch.tensor([1], dtype=torch.int64),
                        torch.ones(text_interval, dtype=torch.int64),
                    ], dim=0)
                elif data_mode in ['mode3','mode4'] and data_template == 'qwen3d5-byturn':
                    new_ids = torch.cat([
                        torch.tensor([end_prompt_token], dtype=torch.int32),
                        torch.tensor([sos_token], dtype=torch.int32),
                        text_ids[:text_interval],
                    ], dim=0)
                    new_flag = torch.cat([
                        torch.tensor([1], dtype=torch.int64),
                        torch.tensor([0], dtype=torch.int64),
                        torch.ones(text_interval, dtype=torch.int64),
                    ], dim=0)
                else:
                    new_ids = torch.cat([
                        torch.tensor([begin_token if data_template == 'qwen3d5-byturn' else sos_token], dtype=torch.int32),
                        text_ids[:text_interval],
                    ], dim=0)
                    new_flag = torch.cat([
                        torch.tensor([1 if data_template == 'qwen3d5-byturn' else 0], dtype=torch.int64),
                        torch.ones(text_interval, dtype=torch.int64),
                    ], dim=0)
                res_text_ids = text_ids[text_interval:].unsqueeze(0)  # [1, L]

            long_input_ids = torch.cat((long_input_ids, new_ids))
            long_text_flag = torch.cat((long_text_flag, new_flag))

        logging.info(f"lm inference : long_input_text: {tokenizer.decode(long_input_ids[long_text_flag == 1])}, \n {long_input_ids}")
        logging.info(f"text emb flag: {long_text_flag}")
     
        output = {
            "source_ids": long_input_ids.unsqueeze(0),
            "text": None,
            "text_lengths": None,
            "prompt_text": None,
            "prompt_text_lengths": None,
            "prompt_codec": None,
            "prompt_codec_lengths": None,
            "res_text_ids": res_text_ids,
            "text_emb_flag": long_text_flag.unsqueeze(0),
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
        if 'assistant_codec' in contents:
            output = self.load_data_long(
                contents, **kwargs
            )
        else:
            output = self.load_data(
                contents, **kwargs
            )
        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]
        text_emb_flag = batch["text_emb_flag"]

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("use_qlora_int8",False)) and (not kwargs.get("infer_lora_merged",False)):
            text_embeder = self.llm.base_model.model.model.get_input_embeddings()
        elif hasattr(self, 'llm_quant'):
            text_embeder = self.llm_quant.model.get_input_embeddings()
        else:
            text_embeder = self.llm.model.get_input_embeddings()
        inputs_text_embeds = text_embeder(source_ids * text_emb_flag) * text_emb_flag.unsqueeze(-1)
        input_codec_embs = self.codec_embed(source_ids * (1-text_emb_flag)) * (1-text_emb_flag).unsqueeze(-1)
        inputs_embeds = inputs_text_embeds + input_codec_embs
        if "res_text_ids" in batch:
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
            from .llm_decoding import LLMDecoder
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
            self.llm.base_model.model.lm_head = self.codec_head.to(dtype)
        else:
            self.llm.lm_head = self.codec_head.to(dtype)
        avoid_token = self.get_avoid_token(llm_generator_conf.get("avoid_token", None))
        hiddens = []
        text_emb_flag = batch["text_emb_flag"]
        if streaming_mode == "false":
            outputs = self.llm_generator(
                inputs_emb,
                self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                states=kwargs.get("states", {}),
                max_length=max_length,
                min_length=min_length,
                avoid_token=avoid_token,
                quantize=True if hasattr(self, 'llm_quant') else False,
                return_hidden=self.unified,
                **kwargs
            )
            gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
            if self.unified:
                hiddens += outputs[3]
                text_emb_flag = torch.cat([text_emb_flag, torch.zeros([1, gen_codec.shape[1]], device=text_emb_flag.device)], dim=1)
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
                outputs = self.llm_generator(
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=None,
                    include_eos=True,
                    custom_eos_token=eos_tokens,
                    avoid_token=avoid_token,
                    quantize=True if hasattr(self, 'llm_quant') else False,
                    return_hidden=self.unified,
                    **kwargs
                )
                gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
                if self.unified:
                    hiddens += outputs[3]
                    text_emb_flag = torch.cat([text_emb_flag, torch.zeros([1, gen_codec.shape[1]-1], device=text_emb_flag.device)], dim=1)

                if gen_codec[0, -1] == llm_generator_conf.get("fill_token", 6564):
                    if res_text_emb.shape[1] < text_interval:
                        tos_idx = torch.tensor(
                            [[llm_generator_conf.get("turn_of_speech", 6563)]],
                            dtype=torch.int64, device=res_text_emb.device
                        )
                        tos_emb = self.codec_embed(tos_idx)
                        inputs_emb = torch.cat([res_text_emb, tos_emb], dim=1)
                        if self.unified:
                            text_emb_flag = torch.cat([text_emb_flag,
                                                torch.ones([1, res_text_emb.shape[1]], device=text_emb_flag.device)], dim=1)
                    else:
                        inputs_emb = res_text_emb[:, :text_interval]
                        if self.unified:
                            text_emb_flag = torch.cat([text_emb_flag,
                                                torch.ones([1, inputs_emb.shape[1]-1], device=text_emb_flag.device)], dim=1)

                    res_text_emb = res_text_emb[:, text_interval:]

                gen_raw_list.append(gen_codec)
                gen_codec_list.append(gen_codec[:, :-1])
                if gen_codec[0, -1] == llm_generator_conf.get("eos", 6562):
                    break

            gen_raw_codes = torch.cat(gen_raw_list, dim=1)
            logging.info(f"{uttid}: gen {gen_raw_codes.shape[1]} raw tokens: {gen_raw_codes[0].cpu().tolist()}")
            gen_codec = torch.cat(gen_codec_list, dim=1)
        else:
            raise NotImplementedError(f"{streaming_mode} is not supported.")

        # get codec tokens only
        if self.unified:
            hiddens_cat = torch.cat(hiddens, dim=1)
            mask_audio = text_emb_flag[0,1:]                  # 输出为text_emb_flag的移位
            logging.info(f"{uttid}: hiddens_cat.shape[1]={hiddens_cat.shape[1]}, mask_audio.shape={mask_audio.shape}, text_emb_flag.shape={text_emb_flag.shape}")
            assert hiddens_cat.shape[1] == mask_audio.shape[0], \
                f"{uttid}: hiddens {hiddens_cat.shape[1]} != mask_audio {mask_audio.shape[0]}, text_emb_flag {text_emb_flag.shape}"
            mask_audio = mask_audio.cpu().numpy() == 0
            hiddens = hiddens_cat[:,mask_audio,:]             # 获取音频部分，但包含 dur/spk/emo, <|turnofspeech|>
            if prompt_codec is not None:
                assert hiddens.shape[1] >= gen_codec.shape[1] + prompt_codec.shape[1]

        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        if self.unified:
            return gen_codec, hit_eos, states, hiddens
        return gen_codec, hit_eos, states

    @torch.no_grad()
    def inference_conversation(
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
            from .llm_decoding import LLMDecoder
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)

        if kwargs.get("infer_use_lora",False):
            self.llm.base_model.model.lm_head = self.codec_head.to(dtype)
        else:
            self.llm.lm_head = self.codec_head.to(dtype)
        avoid_token = self.get_avoid_token(llm_generator_conf.get("avoid_token", None))
        hiddens = []
        text_emb_flag = batch["text_emb_flag"]
        if streaming_mode == "false":
            outputs = self.llm_generator(
                inputs_emb,
                self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                states=kwargs.get("states", {}),
                max_length=max_length,
                min_length=min_length,
                avoid_token=avoid_token,
                quantize=True if hasattr(self, 'llm_quant') else False,
                return_hidden=self.unified,
                **kwargs
            )
            gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
            if self.unified:
                hiddens += outputs[3]
                text_emb_flag = torch.cat([text_emb_flag, torch.zeros([1, gen_codec.shape[1]], device=text_emb_flag.device)], dim=1)
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
                outputs = self.llm_generator(
                    inputs_emb,
                    self.llm_quant if hasattr(self, 'llm_quant') else self.llm,
                    states=states,
                    max_length=max_length,
                    min_length=None,
                    include_eos=True,
                    custom_eos_token=eos_tokens,
                    avoid_token=avoid_token,
                    quantize=True if hasattr(self, 'llm_quant') else False,
                    return_hidden=self.unified,
                    **kwargs
                )
                gen_codec, hit_eos, states = outputs[0], outputs[1], outputs[2]
                if self.unified:
                    hiddens += outputs[3]
                    n_codec = gen_codec.shape[1]-1 if hit_eos else gen_codec.shape[1]
                    text_emb_flag = torch.cat([text_emb_flag, torch.zeros([1, n_codec], device=text_emb_flag.device)], dim=1)

                if gen_codec[0, -1] == llm_generator_conf.get("fill_token", 6564):
                    if res_text_emb.shape[1] < text_interval:
                        tos_idx = torch.tensor(
                            [[llm_generator_conf.get("turn_of_speech", 6563)]],
                            dtype=torch.int64, device=res_text_emb.device
                        )
                        tos_emb = self.codec_embed(tos_idx)
                        inputs_emb = torch.cat([res_text_emb, tos_emb], dim=1)
                        if self.unified and hit_eos:
                            text_emb_flag = torch.cat([text_emb_flag,
                                                torch.ones([1, res_text_emb.shape[1]], device=text_emb_flag.device)], dim=1)
                    else:
                        inputs_emb = res_text_emb[:, :text_interval]
                        if self.unified and hit_eos:
                            text_emb_flag = torch.cat([text_emb_flag,
                                                torch.ones([1, inputs_emb.shape[1]-1], device=text_emb_flag.device)], dim=1)

                    res_text_emb = res_text_emb[:, text_interval:]

                gen_raw_list.append(gen_codec)
                gen_codec_list.append(gen_codec[:, :-1])
                if gen_codec[0, -1] == llm_generator_conf.get("eos", 6562) or torch.cat(gen_codec_list, dim=1).shape[1] > max_length:
                    break

            gen_raw_codes = torch.cat(gen_raw_list, dim=1)
            logging.info(f"{uttid}: gen {gen_raw_codes.shape[1]} raw tokens: {gen_raw_codes[0].cpu().tolist()}")
            gen_codec = torch.cat(gen_codec_list, dim=1)
        else:
            raise NotImplementedError(f"{streaming_mode} is not supported.")

        # get codec tokens only
        if self.unified:
            hiddens_cat = torch.cat(hiddens, dim=1)
            mask_audio = text_emb_flag[0,1:]                  # 输出为text_emb_flag的移位
            logging.info(f"{uttid}: hiddens_cat.shape[1]={hiddens_cat.shape[1]}, mask_audio.shape={mask_audio.shape}, text_emb_flag.shape={text_emb_flag.shape}")
            assert hiddens_cat.shape[1] == mask_audio.shape[0], \
                f"{uttid}: hiddens {hiddens_cat.shape[1]} != mask_audio {mask_audio.shape[0]}, text_emb_flag {text_emb_flag.shape}"
            mask_audio = mask_audio.cpu().numpy() == 0
            hiddens = hiddens_cat[:,mask_audio,:]             # 获取音频部分，但包含 dur/spk/emo, <|turnofspeech|>

        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length
        logging.info(f"{uttid}: gen {gen_codec.shape[1]} tokens: {gen_codec[0].cpu().tolist()}")

        if self.unified:
            return gen_codec, hit_eos, states, hiddens
        return gen_codec, hit_eos, states
