import logging
import os.path
import torchaudio

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
import math
import whisper

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
from funasr.models.minmo.dual_head_model import FunAudioForDualChannelModel, FunAudioMoeFusedForDualChannelModel


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


class SpeechTokenExtractionONNX(nn.Module):
    def __init__(self,
                 model_dir='/path/to/CosyVoice3',
                 onnx_intra_op_num_threads=20,
                 onnx_use_cuda=False,
                 num_threads=0, # 0: sequential, >=1: threadpool
                 **kwargs):
        try:
            import onnxruntime
        except Exception as e:
            print("ERROR: onnxruntime is not installed. it is required for SpeechTokenExtractionONNX")
            raise e
        super().__init__(**kwargs)
        print(f'SpeechTokenExtractionONNX init with model_dir: {model_dir}, onnx_intra_op_num_threads: {onnx_intra_op_num_threads}, onnx_use_cuda: {onnx_use_cuda}, kwargs: {kwargs}')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        intra_op_num_threads = onnx_intra_op_num_threads
        print (f'SpeechTokenExtractionONNX onnx intra_op_num_threads: {intra_op_num_threads}')
        option.intra_op_num_threads = intra_op_num_threads
        # from modelscope import snapshot_download, HubApi
        # api = HubApi()
        # _, cookies = api.login(access_token=os.environ['token'])
        # model_dir = snapshot_download('FunAudioLLM/CosyVoice3', cookies=cookies)
        self.model_dir = model_dir
        print (f'SpeechTokenExtractionONNX model_dir: {model_dir}')
        if onnx_use_cuda:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            print (f'SpeechTokenExtractionONNX local_rank: {local_rank}')
            providers = [("CUDAExecutionProvider", {"device_id": local_rank})]
        else:
            providers = ["CPUExecutionProvider"]
        print (f'SpeechTokenExtractionONNX onnx providers: {providers}')
        self.speech_tokenizer_session = onnxruntime.InferenceSession('{}/speech_tokenizer_v3.onnx'.format(model_dir), sess_options=option, providers=providers)
        # self.warmup(n_warmup=1000)
        self.num_threads = num_threads
        if self.num_threads > 0:
            print (f'SpeechTokenExtractionONNX use threadpool with num_threads: {self.num_threads}')
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        else:
            print (f'SpeechTokenExtractionONNX use sequential')

    def warmup(self, n_warmup):
        print ('start SpeechTokenExtractionONNX warmup')
        start_time = time.time()
        from tqdm import tqdm
        device = 'cuda:0'
        for _ in tqdm(range(n_warmup)):
            speech_lengths = torch.randint(size=(1,), low=16000, high=16000 * 60).to(device)
            speech = torch.rand(1, speech_lengths.item()).to(device)
            self(speech, speech_lengths)
        end_time = time.time()
        print ('SpeechTokenExtractionONNX warmup time: {}'.format(end_time - start_time))

    def forward(self, speech, speech_lengths):
        if self.num_threads <= 0:
            speech_token_list = []
            for i in range(len(speech)):
                speech_token = self._extract_speech_token(speech[i: i + 1][:, :speech_lengths[i].item()])
                speech_token_list.append(torch.tensor(speech_token, dtype=torch.int32))
            speech_token_len = torch.tensor([i.size(0) for i in speech_token_list], dtype=torch.int32).to(speech.device)
            speech_token = pad_sequence(speech_token_list,
                                        batch_first=True,
                                        padding_value=-1).to(speech_lengths.device)
            return speech_token, speech_token_len
        else:
            """
            Parallelized forward pass with padding applied directly (-1).
            Speech: [batch_size, time_samples]
            Speech_lengths: [batch_size] valid length in samples.
            """
            batch_size = speech.size(0)

            def process_one(i: int) -> torch.Tensor:
                # Slice valid portion for sample i
                valid_len = speech_lengths[i].item()
                speech_piece = speech[i:i+1, :valid_len]
                # Extract token sequence (list[int])
                token_ids = self._extract_speech_token(speech_piece)
                # Convert to tensor
                return torch.tensor(token_ids, dtype=torch.int32)

            # Parallelized calls — keeps order
            max_workers = min(batch_size, 8)  # cap workers
            speech_token_list = list(self.executor.map(process_one, range(batch_size)))

            # Lengths in tokens
            speech_token_len = torch.tensor(
                [t.size(0) for t in speech_token_list],
                dtype=torch.int32,
                device=speech.device
            )

            # Pad *with -1 directly*
            speech_token = pad_sequence(
                speech_token_list,
                batch_first=True,
                padding_value=-1  # any shorter sequence gets padded here
            ).to(speech_lengths.device)

            return speech_token, speech_token_len

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 60, 'do not support extract speech token for audio longer than 60s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        return speech_token



@tables.register("model_classes", "LLMASR4")
@tables.register("model_classes", "MinMo_S2T")
class MinMo_S2T(nn.Module):
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
        qwen_init_encoder = audio_encoder_conf.get("qwen_init_encoder", False)
        if qwen_init_encoder:
            from funasr.models.minmo.qwen_audio_encoder import QwenAudioEncoder
            audio_encoder = QwenAudioEncoder(
                audio_encoder,
                llm_dtype = audio_encoder_conf.get("llm_dtype", "bf16"),
                output_size=audio_encoder_conf.get("output_size", 2048),
                fix_layer = -1
            )
            audio_encoder_output_size = audio_encoder_conf.get("output_size", 2048)
        elif hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder
        # llm
        self.llm = None
        init_param_path = llm_conf.get("init_param_path", None)
        llm_dim = None
        if init_param_path is not None:

            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            llm_load_kwargs = llm_conf.get("load_kwargs", {})
            print(f"llm_conf-qwen3moefused: {llm_conf.get('qwen3moefused', False)}")
            if llm_conf.get('qwen3moefused', False):
                print(f"use qwen3moefused")
                # from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
                # from qwen3_moe_fused.lora import patch_lora_config
                from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM

                import os
                os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

                print(f"patchs")
                # patch_bnb_quantizer()
                # patch_lora_config()
                # patch_Qwen3MoeFusedSparseMoeBlock_forward()

                print (f"load Qwen3MoeFused model from {init_param_path}")
                model = Qwen3MoeFusedForCausalLM.from_pretrained(init_param_path)
            else:
                print(f"not use qwen3moefused")
                model = AutoModelForCausalLM.from_pretrained(
                    init_param_path,
                    attn_implementation="flash_attention_2",
                    load_in_8bit=None,
                    device_map=None,
                    use_cache=None,
                    **llm_load_kwargs,
                )

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

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.feat_permute = audio_encoder_conf.get("feat_permute", True)
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

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

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
            loss = model_outputs.loss

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
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

    def forward_export(self, speech, speech_lengths, **kwargs):

        x, olens = self.audio_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)

        encoder_out_ctc, encoder_out_lens_ctc = None, None
        if hasattr(self, "ctc_decoder") and self.ctc_decoder is not None:
            encoder_out_ctc, encoder_out_lens_ctc = self.ctc_decoder(x, olens)
            encoder_out_ctc = self.ctc.log_softmax(encoder_out_ctc)

        return encoder_out, encoder_out_lens, encoder_out_ctc, encoder_out_lens_ctc

    def encode(self, speech, speech_lengths):
        # audio encoder
        if self.feat_permute:
            encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)
        else:
            encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

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
        do_think = True
        sys_prompt = True
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", True)
            sys_prompt = kwargs["dataset_conf"].get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
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
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
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

                        if self.feat_permute:
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

        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

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


@tables.register("model_classes", "MinMo_S2S_Interleave")
class MinMo_S2S_Interleave(nn.Module):
    """ """

    def __init__(
            self,
            audio_encoder: str = None,
            audio_encoder_conf: dict = None,
            audio_adaptor: str = None,
            audio_adaptor_conf: dict = None,
            vision_encoder: str = None,
            vision_encoder_conf: dict = None,
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
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
        audio_adaptor_conf["llm_dim"] = llm_dim
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
            audio_tokens: torch.Tensor = None,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            labels_ids: torch.Tensor = None,
            feature_beg: torch.Tensor = None,
            feature_type: torch.Tensor = None,
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

        # pdb.set_trace()
        stats = {}
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size_speech = 0
        frames = 0
        acoustic_audio_emb = None
        speech_encoder_loss = None
        if speech is not None:
            # speech
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
        else:
            # Resolve the issue of asymmetric gradients across multiple GPUs
            speech = torch.randn(1, 64, 128).to(inputs_embeds.device)
            speech_lengths = torch.tensor([64], dtype=torch.int32).to(inputs_embeds.device)
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            speech_encoder_loss = encoder_out.mean() * 0.0

        if audio_tokens is not None:
            # audio_tokens
            audio_tokens[audio_tokens == -1] = 0
            audio_emb = self.llm.model.get_input_embeddings()(audio_tokens)
            if acoustic_audio_emb is not None:
                acoustic_audio_emb = (acoustic_audio_emb + audio_emb) * math.sqrt(2)
            else:
                acoustic_audio_emb = audio_emb

        batch_size, token_num, _ = inputs_embeds.shape
        fake_token_len = kwargs.get("fake_token_len")

        speech_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1

        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item() if speech_lengths is not None else 0
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

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
            loss = model_outputs.loss
            if speech_encoder_loss is not None:
                loss += speech_encoder_loss

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (feature_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
        pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        image_processor = kwargs.get("image_processor", None)

        input_ids, labels, fbank, fbank_lens, \
            pixel_values_lst, image_grid_thws, pixel_values_videos_lst, \
            video_grid_thws, video_sencond_per_grids, feature_attention_masks, input_features_lst, \
            feature_beg, feature_type, fake_token_len = (
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            ## add vision_start/vision_end
            user_prompt = user_prompt.replace("<|startofimage|>", "<|vision_start|><|startofimage|>").replace(
                "<|endofimage|>", "<|endofimage|><|vision_end|>")
            user_prompt = user_prompt.replace("<|startofvideo|>", "<|vision_start|><|startofvideo|>").replace(
                "<|endofvideo|>", "<|endofvideo|><|vision_end|>")

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

            splits = pattern.split(source_input)
            source_ids = []
            # audio
            speech = []
            speech_lengths = []
            # image
            pixel_values_turn = []
            image_grid_thw_turn = []
            # video
            pixel_values_videos_turn = []
            video_grid_thw_turn = []
            video_sencond_per_grid_turn = []
            feature_attention_mask_turn = []
            input_features_turn = []
            # audio:1, image:2, video:3
            feature_type_turn = 0
            output_kwargs = {'text_kwargs': {'padding': True, 'padding_side': 'left', 'return_tensors': 'pt'},
                             'images_kwargs': {'return_tensors': 'pt'},
                             'audio_kwargs': {'sampling_rate': 16000, 'padding': 'max_length',
                                              'return_attention_mask': True, 'return_tensors': 'pt'},
                             'videos_kwargs': {'return_tensors': 'pt'},
                             'common_kwargs': {'return_tensors': 'pt'}}
            for k, sub_str in enumerate(splits):
                if sub_str is None or sub_str == "":
                    continue
                fake_token_len_i = 0
                featrue_beg_i = -1
                if not any(sub_str.startswith(prefix) for prefix in
                           ["<|startofspeech|>", "<|startofimage|>", "<|startofvideo|>"]):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                elif sub_str.startswith("<|startofspeech|>"):
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
                        featrue_beg_i = len(source_ids)
                        feature_type_turn = 1
                        source_ids += fake_token
                elif sub_str.startswith("<|startofimage|>") or sub_str.startswith("<|startofvideo|>"):
                    from transformers.image_utils import make_batched_videos
                    import qwen_omni_utils
                    image_inputs = []
                    video_inputs = []
                    video_sample_fps_list = []

                    if sub_str.startswith("<|startofimage|>"):
                        try:
                            sub_str = sub_str.replace("<|startofimage|>", "").replace(
                                "<|endofimage|>", "")
                            image_inputs.append(qwen_omni_utils.fetch_image({"image": sub_str[1:]}))
                            if len(image_inputs) == 0:
                                image_inputs = None

                            images_inputs = image_processor(images=image_inputs, videos=None,
                                                            **output_kwargs["images_kwargs"])
                            image_grid_thw = images_inputs["image_grid_thw"]
                            # image_index = 0
                            fake_token_len_i = image_grid_thw[0].prod() // (image_processor.merge_size ** 2)
                            fake_token = [0] * fake_token_len_i
                            featrue_beg_i = len(source_ids)
                            feature_type_turn = 2
                            source_ids += fake_token
                            pixel_values_turn.extend(images_inputs["pixel_values"])
                            image_grid_thw_turn.extend(image_grid_thw)

                        except Exception as e:
                            logging.warning(
                                f"Loading Image failed! {str(e)}, {traceback.format_exc()}"
                            )
                            badcase_flag = True
                            continue
                    elif sub_str.startswith("<|startofvideo|>"):
                        use_audio_in_video = False
                        try:
                            sub_str = sub_str.replace("<|startofvideo|>", "").replace(
                                "<|endofvideo|>", "")
                            video_input, video_sample_fps = qwen_omni_utils.fetch_video({"video": sub_str[1:]},
                                                                                        return_video_sample_fps=True)
                            video_sample_fps_list.append(video_sample_fps)
                            video_inputs.append(video_input)
                            if len(video_inputs) == 0:
                                video_inputs = None

                            videos = make_batched_videos(video_inputs)
                            videos_inputs = image_processor(images=None, videos=videos,
                                                            **output_kwargs["videos_kwargs"])
                            fps = output_kwargs["videos_kwargs"].pop("fps", None)
                            if fps is None:
                                fps = [2.0] * len(videos)
                            videos_inputs["video_second_per_grid"] = [
                                image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
                            ]
                            video_grid_thw = videos_inputs["video_grid_thw"]

                            if not use_audio_in_video:
                                # video_index = 0
                                fake_token_len_i = video_grid_thw[0].prod() // (image_processor.merge_size ** 2)
                                fake_token = [0] * fake_token_len_i
                                featrue_beg_i = len(source_ids)
                                feature_type_turn = 3
                                source_ids += fake_token
                                pixel_values_videos_turn.extend(videos_inputs["pixel_values_videos"])
                                video_grid_thw_turn.extend(videos_inputs["video_grid_thw"])
                                # video_sencond_per_grid_turn.append(videos_inputs["video_second_per_grid"])
                                # feature_attention_mask_turn.append(videos_inputs.get("feature_attention_mask", None))
                                # input_features_turn.append(videos_inputs.get("input_features", None))
                            else:
                                # TODO: add audio in video
                                pass
                        except Exception as e:
                            logging.warning(
                                f"Loading Image failed! {str(e)}, {traceback.format_exc()}"
                            )
                            badcase_flag = True
                            continue
                if featrue_beg_i > 0:
                    feature_beg += [featrue_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                    feature_type += [feature_type_turn]

            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)
            if len(pixel_values_turn) > 0:
                pixel_values_lst.extend(pixel_values_turn)
                image_grid_thws.extend(image_grid_thw_turn)
            if len(pixel_values_videos_turn) > 0:
                pixel_values_videos_lst.extend(pixel_values_videos_turn)
                video_grid_thws.extend(video_grid_thw_turn)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        # fbank = speech[0, :, :]
        # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
        feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
        feature_type = torch.tensor(feature_type, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        output = {
            "feature_beg": feature_beg[None, :],
            "feature_type": feature_type[None, :],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None, :],
            "attention_mask": attention_mask[None, :],
            "labels_ids": labels[None, :],
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        if len(fbank) > 0:
            output["speech"] = fbank
            output["speech_lengths"] = fbank_lens
        if len(pixel_values_lst) > 0:
            output["pixel_values"] = pixel_values_lst
            output["image_grid_thw"] = image_grid_thws
        if len(pixel_values_videos_turn) > 0:
            output["pixel_values_videos"] = pixel_values_videos_lst
            output["video_grid_thw"] = video_grid_thws

        for key, data_list in output.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                    pad_value = -1
                else:
                    pad_value = 0.0

                output[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

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

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        feature_beg = batch["feature_beg"]
        feature_type = batch["feature_type"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, _ = inputs_embeds.shape

        # audio encoder
        speech = batch.get("speech", None)
        if speech is not None:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.dtype)
            image_embeds = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        pixel_values_videos = batch.get("pixel_values_videos", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.vision_encoder.dtype)
            video_embeds = self.vision_encoder(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        speech_idx = 0
        image_idx = 0
        video_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = encoder_out[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1
                    # image
                    elif feature_type[batch_idx, feature_idx] == 2:
                        image_token = image_embeds[image_idx:image_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = image_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        image_idx += feature_length
                    # video
                    elif feature_type[batch_idx, feature_idx] == 3:
                        video_token = video_embeds[video_idx:video_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = video_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        video_idx += feature_length

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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

@tables.register("model_classes", "MinMo_S2S_Interleave_Pretrain")
class MinMo_S2S_Interleave_Pretrain(nn.Module):
    """ """

    def __init__(
            self,
            audio_encoder: str = None,
            audio_encoder_conf: dict = None,
            audio_adaptor: str = None,
            audio_adaptor_conf: dict = None,
            vision_encoder: str = None,
            vision_encoder_conf: dict = None,
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size() if hasattr(audio_encoder, "output_size") else -1
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

        self.input_size = input_size

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder
        self.feat_permute = audio_encoder_conf.get("feat_permute", True)

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        print(f"llm_conf-qwen3moefused: {llm_conf.get('qwen3moefused', False)}")
        if llm_conf.get('qwen3moefused', False):
            print(f"use qwen3moefused")
            import os
            os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
            print (f"load Qwen3MoeFused model from {init_param_path}")
            model = FunAudioMoeFusedForDualChannelModel.from_pretrained(
                init_param_path,
                attn_implementation="flash_attention_2",
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )
        else:
            print(f"not use qwen3moefused")
            model = FunAudioForDualChannelModel.from_pretrained(
                init_param_path,
                attn_implementation="flash_attention_2",
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )


        # if not llm_conf.get("low_cpu", False):
        #     model = AutoModelForCausalLM.from_pretrained(
        #         init_param_path,
        #         attn_implementation="flash_attention_2",
        #         load_in_8bit=None,
        #         device_map=None,
        #         use_cache=None,
        #         **llm_load_kwargs,
        #     )
        # else:
        #     import os

        #     if int(os.environ.get("RANK", 0)) == 0:
        #         model = AutoModelForCausalLM.from_pretrained(
        #             init_param_path,
        #             attn_implementation="flash_attention_2",
        #             load_in_8bit=None,
        #             device_map="cpu",
        #             use_cache=None,
        #             **llm_load_kwargs,
        #         )
        #     else:
        #         llm_config = AutoConfig.from_pretrained(init_param_path)
        #         model = AutoModelForCausalLM.from_config(llm_config)

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
        audio_adaptor_conf["llm_dim"] = llm_dim
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
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.speech_token_extractor_conf = llm_conf.get("speech_token_extractor_conf", {})
        self.speech_token_extractor_enabled = self.speech_token_extractor_conf.get("enabled", True)
        if self.speech_token_extractor_enabled:
            print (f'SpeechTokenExtraction enabled, initializing speech token extractor')
            self._init_speech_token_extractor()

        self.audio_code_book_size = 59049  # 6761 59049
        self.mask_token_id = -60

        # audio embed and head
        # TODO: take care the param initialization
        self.special_tokens_audio_count = 463
        self.audio_embed = nn.Embedding(self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
        self.audio_head = nn.Linear(llm_dim, self.audio_code_book_size + self.special_tokens_audio_count, bias=False)

        audio_embed_ckpt_path = llm_conf.get("audio_embed_ckpt_path", None)
        if audio_embed_ckpt_path is not None:
            audio_embed_weight = torch.load(audio_embed_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
            if audio_embed_weight.shape == expected_shape:
                self.audio_embed.weight.data = audio_embed_weight
                logging.info(f"Loading audio_embed from {audio_embed_ckpt_path}, shape: {audio_embed_weight.shape}")
            else:
                logging.warning(f"audio_embed shape mismatch: expected {expected_shape}, got {audio_embed_weight.shape}")
        else:
            logging.info(f"No audio_embed_ckpt_path provided, using default initialization")
        
        audio_head_ckpt_path = llm_conf.get("audio_head_ckpt_path", None)
        if audio_head_ckpt_path is not None:
            audio_head_weight = torch.load(audio_head_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
            if audio_head_weight.shape == expected_shape:
                self.audio_head.weight.data = audio_head_weight
                logging.info(f"Loading audio_head weight from {audio_head_ckpt_path}, shape: {audio_head_weight.shape}")
            else:
                logging.warning(f"audio_head weight shape mismatch: expected {expected_shape}, got {audio_head_weight.shape}")
        else:
            logging.info(f"No audio_head_ckpt_path provided, using default initialization")
        
        if llm_conf.get("use_lora", False):
            # self.llm.model.embed_tokens_audio = self.audio_embed
            # self.llm.model.lm_head_audio = self.audio_head

            self.llm.base_model.model.embed_tokens_audio = self.audio_embed
            self.llm.base_model.model.lm_head_audio = self.audio_head

            # self.llm.base_model.model.model.embed_tokens_audio = self.audio_embed
            # self.llm.base_model.model.model.lm_head_audio = self.audio_head
        else:
            self.llm.embed_tokens_audio = self.audio_embed
            self.llm.lm_head_audio = self.audio_head

        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

        # self.speech_token_id_offset = llm_conf.get("speech_token_id_offset", 151694) # 151692, 151694

    def _init_speech_token_extractor(self):
        self.speech_token_extractor_type = self.speech_token_extractor_conf.get("type", "automodel")
        if self.speech_token_extractor_type == "automodel":
            print (f'SpeechTokenExtraction use automodel')
            from funasr.auto.auto_model import AutoModel
            self.speech_token_extractor = AutoModel(**self.speech_token_extractor_conf.get("automodel_conf", {}))
            device = "cuda"
            self.speech_token_extractor.model = self.speech_token_extractor.model.to(device)
            self.speech_token_extractor.model.eval()

            self.speech_token_extractor_sliding_window_size = self.speech_token_extractor_conf.get("sliding_window_size", 4000)
            print (f'SpeechTokenExtractor sliding window size: {self.speech_token_extractor_sliding_window_size}')
            self.speech_token_extractor_sliding_window_left_context = self.speech_token_extractor_conf.get("sliding_window_left_context", 200)
            print (f'SpeechTokenExtractor sliding window left context: {self.speech_token_extractor_sliding_window_left_context}')
            self.speech_token_extractor_sliding_window_bin_threshold = self.speech_token_extractor_conf.get("sliding_window_bin_threshold", 1000)
            print (f'SpeechTokenExtractor sliding window bin threshold: {self.speech_token_extractor_sliding_window_bin_threshold}')
            self.speech_token_extractor_downsample_factor = self.speech_token_extractor_conf.get("downsample_factor", 4)
            print (f'SpeechTokenExtractor downsample factor: {self.speech_token_extractor_downsample_factor}')
        elif self.speech_token_extractor_type == "onnx":
            print (f'SpeechTokenExtraction use onnx')
            self.speech_token_extractor = SpeechTokenExtractionONNX(**self.speech_token_extractor_conf.get("onnx_conf", {}))
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")

    @staticmethod
    def process_with_sliding_window(segment_fbank, segment_fbank_lengths, 
                                    model, window_size=4000, left_context=2000,
                                    bin_threshold=256, downsample_factor=4, device="cuda"):

        B, Lmax, D = segment_fbank.shape
        true_lengths = segment_fbank_lengths[:, 0]

        # How many tokens correspond to the overlapped part
        overlap_token_len = int(left_context / downsample_factor)

        # Step 1: create sliding window chunks
        chunks = []
        for idx in range(B):
            length = true_lengths[idx].item()
            feat = segment_fbank[idx, :length]  # Trim padded part

            start = 0
            chunk_idx_in_original = 0
            while start < length:
                end = min(start + window_size, length)
                if chunk_idx_in_original > 0:
                    context_start = start - left_context
                else:
                    context_start = start
                chunk = feat[context_start:end]  # (chunk_len, D)
                chunks.append({
                    "tensor": chunk,
                    "length": chunk.shape[0],
                    "orig_idx": idx,
                    "chunk_idx": chunk_idx_in_original
                })
                chunk_idx_in_original += 1
                if end == length:
                    break
                start += window_size

        # Step 2: sort by length to minimize padding
        chunks_sorted = sorted(chunks, key=lambda x: x["length"])

        # Step 3: bin chunks by length threshold
        bins = []
        current_bin = []
        current_bin_maxlen = 0
        for ch in chunks_sorted:
            if not current_bin:
                current_bin = [ch]
                current_bin_maxlen = ch["length"]
            else:
                if ch["length"] <= current_bin_maxlen + bin_threshold:
                    current_bin.append(ch)
                    current_bin_maxlen = max(current_bin_maxlen, ch["length"])
                else:
                    bins.append(current_bin)
                    current_bin = [ch]
                    current_bin_maxlen = ch["length"]
        if current_bin:
            bins.append(current_bin)

        # Step 4: run model per bin
        results = {}  # (orig_idx, chunk_idx) -> trimmed_output_tensor

        for bin_chunks in bins:
            maxlen = max(ch["length"] for ch in bin_chunks)
            batch_tensors = torch.stack([
                torch.cat([
                    ch["tensor"],
                    torch.zeros(maxlen - ch["length"], D, dtype=ch["tensor"].dtype, device=device)
                ], dim=0)
                for ch in bin_chunks
            ], dim=0)
            batch_lengths = torch.tensor([ch["length"] for ch in bin_chunks], device=device)

            # Pass through model
            batch_tensors = batch_tensors.permute(0, 2, 1)
            token_result, _ = model.extract_token(
                batch_tensors,
                data_lengths=batch_lengths,
                data_type="fbank",
                key=["any"],
                device=device
            )
            tokens = token_result[0]["token"]

            # Now assign each output explicitly
            for ch, out_tokens in zip(bin_chunks, tokens):
                # if ch["orig_idx"] == 0:
                #     print(f"ch['chunk_idx']: {ch['chunk_idx']}, out_tokens: {out_tokens.shape}, ch['length']: {ch['length']}")
                max_token_len = math.ceil(ch["length"] / downsample_factor)
                if ch["chunk_idx"] == 0:
                    trimmed = out_tokens[:max_token_len]
                else:
                    drop = min(math.ceil(left_context / downsample_factor), max_token_len)
                    trimmed = out_tokens[drop:max_token_len]
                results[(ch["orig_idx"], ch["chunk_idx"])] = trimmed

        # Step 5: reassemble to original order
        final_outputs = []
        final_output_lengths = []
        for idx in range(B):
            chunk_idx = 0
            sample_tokens = []
            while (idx, chunk_idx) in results:
                sample_tokens.append(results[(idx, chunk_idx)])
                # print(f"len(results[(idx, chunk_idx)]): {len(results[(idx, chunk_idx)])}")
                chunk_idx += 1
            if sample_tokens:
                cat_tokens = torch.cat(sample_tokens, dim=0)
                final_outputs.append(cat_tokens)
                final_output_lengths.append(cat_tokens.shape[0])
            else:
                final_outputs.append(torch.empty(0))  # no data
                final_output_lengths.append(0)

        # Optional sanity check: lengths match expected downsample length
        for i in range(B):
            expected_len = math.ceil(true_lengths[i].item() / downsample_factor)
            assert final_outputs[i].shape[0] == expected_len, \
                f"Mismatch for sample {i}: got {final_outputs[i].shape[0]}, expected {expected_len}"
        
        # Pad final_outputs to the same length
        final_outputs = torch.nn.utils.rnn.pad_sequence(final_outputs, batch_first=True, padding_value=0).to(device)
        final_output_lengths = torch.tensor(final_output_lengths, dtype=torch.int32)

        return final_outputs, final_output_lengths

    def _extract_speech_token(self, segment_speech: torch.Tensor, segment_speech_lengths: torch.Tensor, segment_fbank: torch.Tensor, segment_fbank_lengths: torch.Tensor):
        if self.speech_token_extractor_type == "automodel":
            # fbank input
            assert segment_fbank is not None and segment_fbank_lengths is not None, "segment_fbank and segment_fbank_lengths are required"
            with torch.no_grad():
                # with huge padding
                # token_result, _ = self.speech_token_extractor.model.extract_token(segment_fbank.permute(0, 2, 1),
                #                         data_lengths=segment_fbank_lengths[:, 0],
                #                         data_type="fbank",
                #                         key=["any"],
                #                         device="cuda")
                # segment_speech_token = token_result[0]["token"]
                # segment_speech_token_lengths = token_result[0]["token_lengths"].cpu()

                # with sliding window and bin threshold
                segment_speech_token, segment_speech_token_lengths = MinMo_S2S_Interleave_Pretrain.process_with_sliding_window(
                    segment_fbank,
                    segment_fbank_lengths,
                    self.speech_token_extractor.model,
                    window_size=self.speech_token_extractor_sliding_window_size,
                    left_context=self.speech_token_extractor_sliding_window_left_context,
                    bin_threshold=self.speech_token_extractor_sliding_window_bin_threshold,
                    downsample_factor=self.speech_token_extractor_downsample_factor,
                    device="cuda"
                )
        elif self.speech_token_extractor_type == "onnx":
            # audio input
            segment_speech_token, segment_speech_token_lengths = self.speech_token_extractor(segment_speech.to(torch.bfloat16), segment_speech_lengths)
        else:
            raise ValueError(f"Invalid speech token extractor type: {self.speech_token_extractor_type}")
        return segment_speech_token, segment_speech_token_lengths

    def forward(
            self,
            segment_speech: torch.Tensor = None,
            segment_speech_lengths: torch.Tensor = None,
            segment_token: torch.Tensor = None,
            segment_token_lengths: torch.Tensor = None,
            speech: torch.Tensor = None,
            speech_lengths: torch.Tensor = None,
            audio_tokens: torch.Tensor = None,
            input_ids: torch.Tensor = None,
            labels_ids: torch.Tensor = None,
            input_ids_audio: torch.Tensor = None,
            labels_ids_audio: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            feature_beg: torch.Tensor = None,
            feature_type: torch.Tensor = None,
            return_logits: bool = False,
            return_inputs_embeds_only: bool = False,
            **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # deal discrete speech token
        if kwargs.get("segment_fbank_lengths", None) is not None:
            # segment_speech_lengths = torch.tensor(segment_speech_lengths)
            # segment_token_lengths = torch.tensor(segment_token_lengths)
            # NOTE(lyuxiang.lx) speech->speech_token
            stime = time.time()
            segment_speech_token, segment_speech_token_lengths = self._extract_speech_token(
                None,
                None,
                kwargs.get("segment_fbank", None),
                kwargs.get("segment_fbank_lengths", None))

            # mock data
            # segment_speech_token = torch.zeros(segment_speech.shape[0], math.ceil(segment_speech.shape[1] / 16000 * 25), dtype=torch.int32, device='cpu')
            # segment_speech_token_lengths = torch.ceil(segment_speech_lengths / 16000 * 25).to(torch.int32).to(device='cuda')
            # lengths_cpu = segment_speech_token_lengths.cpu()
            # batch_size, max_len = segment_speech_token.shape
            # # Make a mask for padding positions
            # idxs = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)  # shape: [batch_size, max_len]
            # mask = idxs >= lengths_cpu.unsqueeze(1)  # True where padding
            # segment_speech_token[mask] = -1

            etime = time.time()
            # print (f"speech_token_extractor time(batch): {etime - stime}")
            idx = 0
            if self.speech_token_extractor_downsample_factor == 8:
                frame_rate = 12.5
            elif self.speech_token_extractor_downsample_factor == 4:
                frame_rate = 25
            else:
                raise ValueError(f"Invalid speech token extractor downsample factor: {self.speech_token_extractor_downsample_factor}")
            slice_speech_tokens = []
            for i, item in enumerate(kwargs["item"]):
                segments = {}
                for segment_id, segment in item["segments_dict"].items():
                    if not segment["onlie_discrete_speech_token"]:
                        continue
                    segment_speech_token_length = segment_speech_token_lengths[idx].item()
                    segment["tokens"] = segment_speech_token[idx][:segment_speech_token_length]
                    idx += 1
                    segments[segment["segment_id"]] = segment

                slice_speech_tokens.append([])
                user = item["user"]
                assistant = item["assistant"]
                for user_item, assistant_item in zip(user, assistant):
                    for _slice in [user_item, assistant_item]:
                        if _slice is None:
                            continue
                        if not _slice["onlie_discrete_speech_token"]:
                            continue
                        segment = segments[_slice["segment_id"]]
                        token_offset = int((_slice["start_time"] - segment["start_time"]) * frame_rate)
                        token_duration = int((_slice["end_time"] - _slice["start_time"]) * frame_rate)
                        speech_tokens = segment["tokens"][token_offset:token_offset + token_duration]
                        # assert len(speech_tokens) == token_duration, f'[speech_tokens length not match], {len(speech_tokens)} vs {token_duration}'
                        if len(speech_tokens) != token_duration:
                            print (f'[MODEL FORWARD] speech_tokens length not match], {len(speech_tokens)} vs {token_duration}')
                            print (f'[MODEL FORWARD] segment tokens length: {len(segment["tokens"])}')
                            print (f'[MODEL FORWARD] token_offset: {token_offset}, token_duration: {token_duration}')
                            print (f'[MODEL FORWARD] segment time start: {segment["start_time"]}, end: {segment["end_time"]}')
                            print (f'[MODEL FORWARD] _slice time start: {_slice["start_time"]}, end: {_slice["end_time"]}')
                            print (f'[MODEL FORWARD] item: {item}')

                            # tmp fix
                            if len(segment["tokens"]) < token_duration:
                                padding = torch.zeros(token_duration - len(segment["tokens"]), dtype=segment["tokens"].dtype, device=segment["tokens"].device)
                                speech_tokens = torch.cat([segment["tokens"], padding])
                                print (f'[MODEL FORWARD] segment tokens less than token_duration: {len(segment["tokens"])} vs {token_duration}')
                            else:
                                speech_tokens = segment["tokens"][-token_duration:]
                                print (f'[MODEL FORWARD] pick last token duration: {len(segment["tokens"])} vs {token_duration}')

                        slice_speech_tokens[-1].append(speech_tokens)

                if len(slice_speech_tokens[-1]) == 0:
                    # onlie_discrete_speech_token_flag_list = []
                    # for user_item, assistant_item in zip(user, assistant):
                    #     for _slice in [user_item, assistant_item]:
                    #         if _slice is None:
                    #             onlie_discrete_speech_token_flag_list.append(None)
                    #         else:
                    #             onlie_discrete_speech_token_flag_list.append(_slice["onlie_discrete_speech_token"])
                    # print (f'[MODEL FORWARD] onlie_discrete_speech_token_flag_list: {onlie_discrete_speech_token_flag_list}')

                    # 纯文本，或者audio to text的形式
                    assert len(input_ids_audio[i][input_ids_audio[i] == -50]) == 0, f'[MODEL FORWARD] input_ids[i][input_ids[i] == -50] is not empty, {len(input_ids_audio[i][input_ids_audio[i] == -50])}'
                    slice_speech_tokens[-1] = torch.tensor([], dtype=torch.long)
                else:
                    slice_speech_tokens[-1] = torch.cat(slice_speech_tokens[-1])

                # print (f'[MODEL FORWARD] slice_speech_tokens[i]: {slice_speech_tokens[i]}')
                
                # not need offset anymore
                # slice_speech_tokens[i] += kwargs["speech_token_id_offset"][0]

                # print (f'[MODEL FORWARD] slice_speech_tokens[i] after offset: {slice_speech_tokens[i]}')

            for i in range(input_ids_audio.size(0)):
                # audio_placeholder
                # assert len(input_ids[i][input_ids[i] == -50]) == len(slice_speech_tokens[i]), f'[length not match], {len(input_ids[i][input_ids[i] == -50])} vs {len(slice_speech_tokens[i])}, total input length: {len(input_ids[i])}'
                if len(input_ids_audio[i][input_ids_audio[i] == -50]) != len(slice_speech_tokens[i]):
                    print (f'[MODEL FORWARD] input_ids_audio[i][input_ids_audio[i] == -50] length not match, {len(input_ids_audio[i][input_ids_audio[i] == -50])} vs {len(slice_speech_tokens[i])}, total input length: {len(input_ids_audio[i])}')
                    print (f'[MODEL FORWARD] item: {kwargs["item"][i]}')
                    # -50的赋值成0
                    input_ids_audio[i][input_ids_audio[i] == -50] = 0

                input_ids_audio[i][input_ids_audio[i] == -50] = slice_speech_tokens[i].long().to(input_ids_audio.device)
            labels_ids_audio = torch.where(labels_ids_audio == -50, input_ids_audio, labels_ids_audio)
        # else:
        #     print (f'[MODEL FORWARD] no speech data or speech data is offline tokenized')
        # print (f'[MODEL FORWARD] input_ids: {input_ids.shape}')

        # assert input_ids.min() >= 0
        assert input_ids.max() < self.llm.model.config.vocab_size, f'input_ids.max() {input_ids.max()} >= self.llm.model.config.vocab_size {self.llm.model.config.vocab_size}'
        # assert input_ids_audio.min() >= 0
        assert input_ids_audio.max() < self.audio_embed.num_embeddings, f'input_ids_audio.max() {input_ids_audio.max()} >= self.audio_embed.num_embeddings {self.audio_embed.num_embeddings}'

        # set position according to self.mask_token_id to 0
        input_ids_audio_mask = input_ids_audio == self.mask_token_id
        input_ids_audio[input_ids_audio < 0] = 0
        inputs_embeds_audio = self.audio_embed(input_ids_audio)
        inputs_embeds_audio[input_ids_audio_mask] = 0

        stats = {}
        input_ids_mask = input_ids == self.mask_token_id
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        inputs_embeds[input_ids_mask] = 0

        batch_size_speech = 0
        frames = 0
        acoustic_audio_emb = None
        if speech is not None:
            # speech
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
            # print (f"acoustic_audio_emb shape: {acoustic_audio_emb.shape}")
            loss_fake = None
            # print (f'[MODEL FORWARD] with speech data, shape: {speech.shape}')
        else:
            batch_size_speech = 1
            frames = 64
            speech = torch.zeros(1, 64, self.input_size).to(inputs_embeds.device)
            speech_lengths = torch.tensor([64], dtype=torch.int32).to(inputs_embeds.device)
            
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
            loss_fake = encoder_out.mean() * 0.0

        batch_size, token_num, _ = inputs_embeds.shape
        fake_token_len = kwargs.get("fake_token_len")

        speech_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"assign speech token failed. batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1

        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item() if speech_lengths is not None else 0
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        inputs_embeds = inputs_embeds + inputs_embeds_audio

        if return_inputs_embeds_only:
            return inputs_embeds

        inputs_embeds = inputs_embeds[:, :-1, :]
        with torch.amp.autocast('cuda',
                enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids == -1] = -100
            attention_mask[attention_mask < 0] = 0
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = model_outputs.hidden_states[-1]
            logits_text = self.llm.lm_head(hidden_states)
            logits_audio = self.audio_head(hidden_states)

            labels_ids = labels_ids[:, 1:]
            labels_ids_audio = labels_ids_audio[:, 1:]

            is_all_minus_100 = (labels_ids == -100).all() 
            if not is_all_minus_100:
                text_loss = F.cross_entropy(
                            logits_text.view(-1, logits_text.size(-1)),
                            labels_ids.view(-1),
                            ignore_index=-100,
                        )
            else:
                text_loss = 0.0
            is_all_minus_100_audio = (labels_ids_audio == -100).all()
            if not is_all_minus_100_audio:
                audio_loss = F.cross_entropy(
                                logits_audio.view(-1, logits_audio.size(-1)),
                                labels_ids_audio.view(-1),
                                ignore_index=-100,
                            )
            else:
                audio_loss = 0.0
            # print (f'[MODEL FORWARD] is_all_minus_100: {is_all_minus_100}, is_all_minus_100_audio: {is_all_minus_100_audio}')
            # print (f'[MODEL FORWARD] text_loss: {text_loss}, audio_loss: {audio_loss}')
            loss = text_loss + audio_loss
            if loss_fake is not None:
                loss += loss_fake

        if is_all_minus_100 and is_all_minus_100_audio:
            # print (f'[MODEL FORWARD] all minus 100, text and audio')
            # print (f'[MODEL FORWARD] input_ids: {input_ids}')
            # print (f'[MODEL FORWARD] input_ids_audio: {input_ids_audio}')
            stats["loss"] = 0.0
        else:
            stats["loss"] = torch.clone(loss.detach())

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        
        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (feature_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        if return_logits:
            return loss, stats, weight, (logits_text, logits_audio), (labels_ids, labels_ids_audio)
        else:
            return loss, stats, weight

    def encode(self, speech, speech_lengths):
        # audio encoder
        if self.feat_permute:
            speech = speech.permute(0, 2, 1)
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

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
        pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        image_processor = kwargs.get("image_processor", None)

        input_ids, labels, fbank, fbank_lens, \
            pixel_values_lst, image_grid_thws, pixel_values_videos_lst, \
            video_grid_thws, video_sencond_per_grids, feature_attention_masks, input_features_lst, \
            feature_beg, feature_type, fake_token_len = (
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            ## add vision_start/vision_end
            user_prompt = user_prompt.replace("<|startofimage|>", "<|vision_start|><|startofimage|>").replace(
                "<|endofimage|>", "<|endofimage|><|vision_end|>")
            user_prompt = user_prompt.replace("<|startofvideo|>", "<|vision_start|><|startofvideo|>").replace(
                "<|endofvideo|>", "<|endofvideo|><|vision_end|>")

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

            splits = pattern.split(source_input)
            source_ids = []
            # audio
            speech = []
            speech_lengths = []
            # image
            pixel_values_turn = []
            image_grid_thw_turn = []
            # video
            pixel_values_videos_turn = []
            video_grid_thw_turn = []
            video_sencond_per_grid_turn = []
            feature_attention_mask_turn = []
            input_features_turn = []
            # audio:1, image:2, video:3
            feature_type_turn = 0
            output_kwargs = {'text_kwargs': {'padding': True, 'padding_side': 'left', 'return_tensors': 'pt'},
                             'images_kwargs': {'return_tensors': 'pt'},
                             'audio_kwargs': {'sampling_rate': 16000, 'padding': 'max_length',
                                              'return_attention_mask': True, 'return_tensors': 'pt'},
                             'videos_kwargs': {'return_tensors': 'pt'},
                             'common_kwargs': {'return_tensors': 'pt'}}
            for k, sub_str in enumerate(splits):
                if sub_str is None or sub_str == "":
                    continue
                fake_token_len_i = 0
                featrue_beg_i = -1
                if not any(sub_str.startswith(prefix) for prefix in
                           ["<|startofspeech|>", "<|startofimage|>", "<|startofvideo|>"]):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                elif sub_str.startswith("<|startofspeech|>"):
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
                        featrue_beg_i = len(source_ids)
                        feature_type_turn = 1
                        source_ids += fake_token
                elif sub_str.startswith("<|startofimage|>") or sub_str.startswith("<|startofvideo|>"):
                    from transformers.image_utils import make_batched_videos
                    import qwen_omni_utils
                    image_inputs = []
                    video_inputs = []
                    video_sample_fps_list = []

                    if sub_str.startswith("<|startofimage|>"):
                        try:
                            sub_str = sub_str.replace("<|startofimage|>", "").replace(
                                "<|endofimage|>", "")
                            image_inputs.append(qwen_omni_utils.fetch_image({"image": sub_str[1:]}))
                            if len(image_inputs) == 0:
                                image_inputs = None

                            images_inputs = image_processor(images=image_inputs, videos=None,
                                                            **output_kwargs["images_kwargs"])
                            image_grid_thw = images_inputs["image_grid_thw"]
                            # image_index = 0
                            fake_token_len_i = image_grid_thw[0].prod() // (image_processor.merge_size ** 2)
                            fake_token = [0] * fake_token_len_i
                            featrue_beg_i = len(source_ids)
                            feature_type_turn = 2
                            source_ids += fake_token
                            pixel_values_turn.extend(images_inputs["pixel_values"])
                            image_grid_thw_turn.extend(image_grid_thw)

                        except Exception as e:
                            logging.warning(
                                f"Loading Image failed! {str(e)}, {traceback.format_exc()}"
                            )
                            badcase_flag = True
                            continue
                    elif sub_str.startswith("<|startofvideo|>"):
                        use_audio_in_video = False
                        try:
                            sub_str = sub_str.replace("<|startofvideo|>", "").replace(
                                "<|endofvideo|>", "")
                            video_input, video_sample_fps = qwen_omni_utils.fetch_video({"video": sub_str[1:]},
                                                                                        return_video_sample_fps=True)
                            video_sample_fps_list.append(video_sample_fps)
                            video_inputs.append(video_input)
                            if len(video_inputs) == 0:
                                video_inputs = None

                            videos = make_batched_videos(video_inputs)
                            videos_inputs = image_processor(images=None, videos=videos,
                                                            **output_kwargs["videos_kwargs"])
                            fps = output_kwargs["videos_kwargs"].pop("fps", None)
                            if fps is None:
                                fps = [2.0] * len(videos)
                            videos_inputs["video_second_per_grid"] = [
                                image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
                            ]
                            video_grid_thw = videos_inputs["video_grid_thw"]

                            if not use_audio_in_video:
                                # video_index = 0
                                fake_token_len_i = video_grid_thw[0].prod() // (image_processor.merge_size ** 2)
                                fake_token = [0] * fake_token_len_i
                                featrue_beg_i = len(source_ids)
                                feature_type_turn = 3
                                source_ids += fake_token
                                pixel_values_videos_turn.extend(videos_inputs["pixel_values_videos"])
                                video_grid_thw_turn.extend(videos_inputs["video_grid_thw"])
                                # video_sencond_per_grid_turn.append(videos_inputs["video_second_per_grid"])
                                # feature_attention_mask_turn.append(videos_inputs.get("feature_attention_mask", None))
                                # input_features_turn.append(videos_inputs.get("input_features", None))
                            else:
                                # TODO: add audio in video
                                pass
                        except Exception as e:
                            logging.warning(
                                f"Loading Image failed! {str(e)}, {traceback.format_exc()}"
                            )
                            badcase_flag = True
                            continue
                if featrue_beg_i > 0:
                    feature_beg += [featrue_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                    feature_type += [feature_type_turn]

            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)
            if len(pixel_values_turn) > 0:
                pixel_values_lst.extend(pixel_values_turn)
                image_grid_thws.extend(image_grid_thw_turn)
            if len(pixel_values_videos_turn) > 0:
                pixel_values_videos_lst.extend(pixel_values_videos_turn)
                video_grid_thws.extend(video_grid_thw_turn)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        # fbank = speech[0, :, :]
        # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
        feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
        feature_type = torch.tensor(feature_type, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        output = {
            "feature_beg": feature_beg[None, :],
            "feature_type": feature_type[None, :],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None, :],
            "attention_mask": attention_mask[None, :],
            "labels_ids": labels[None, :],
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        if len(fbank) > 0:
            output["speech"] = fbank
            output["speech_lengths"] = fbank_lens
        if len(pixel_values_lst) > 0:
            output["pixel_values"] = pixel_values_lst
            output["image_grid_thw"] = image_grid_thws
        if len(pixel_values_videos_turn) > 0:
            output["pixel_values_videos"] = pixel_values_videos_lst
            output["video_grid_thw"] = video_grid_thws

        for key, data_list in output.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                    pad_value = -1
                else:
                    pad_value = 0.0

                output[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

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

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        feature_beg = batch["feature_beg"]
        feature_type = batch["feature_type"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, _ = inputs_embeds.shape

        # audio encoder
        speech = batch.get("speech", None)
        if speech is not None:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.dtype)
            image_embeds = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        pixel_values_videos = batch.get("pixel_values_videos", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.vision_encoder.dtype)
            video_embeds = self.vision_encoder(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        speech_idx = 0
        image_idx = 0
        video_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = encoder_out[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1
                    # image
                    elif feature_type[batch_idx, feature_idx] == 2:
                        image_token = image_embeds[image_idx:image_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = image_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        image_idx += feature_length
                    # video
                    elif feature_type[batch_idx, feature_idx] == 3:
                        video_token = video_embeds[video_idx:video_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = video_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        video_idx += feature_length

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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

@tables.register("model_classes", "MinMo_S2S_Channels_SFT")
class MinMo_S2S_Channels_SFT(nn.Module):
    """ """

    def __init__(
            self,
            audio_encoder: str = None,
            audio_encoder_conf: dict = None,
            audio_adaptor: str = None,
            audio_adaptor_conf: dict = None,
            vision_encoder: str = None,
            vision_encoder_conf: dict = None,
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
            print (f'The audio encoder is model: {audio_encoder}')
            model = AutoModel(model=audio_encoder, model_revision="master")
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size() if hasattr(audio_encoder, "output_size") else -1
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        print(f"llm_conf-qwen3moefused: {llm_conf.get('qwen3moefused', False)}")
        if llm_conf.get('qwen3moefused', False):
            print(f"use qwen3moefused")
            import os
            os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
            print (f"load Qwen3MoeFused model from {init_param_path}")
            model = FunAudioMoeFusedForDualChannelModel.from_pretrained(
                init_param_path,
                attn_implementation="flash_attention_2",
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )
        else:
            print(f"not use qwen3moefused")
            model = FunAudioForDualChannelModel.from_pretrained(
                init_param_path,
                attn_implementation="flash_attention_2",
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )


        # if not llm_conf.get("low_cpu", False):
        #     model = FunAudioForDualChannelModel.from_pretrained(
        #         init_param_path,
        #         attn_implementation="flash_attention_2",
        #         load_in_8bit=None,
        #         device_map=None,
        #         use_cache=None,
        #         **llm_load_kwargs,
        #     )
        # else:
        #     import os

        #     if int(os.environ.get("RANK", 0)) == 0:
        #         model = FunAudioForDualChannelModel.from_pretrained(
        #             init_param_path,
        #             attn_implementation="flash_attention_2",
        #             load_in_8bit=None,
        #             device_map="cpu",
        #             use_cache=None,
        #             **llm_load_kwargs,
        #         )
        #     else:
        #         llm_config = AutoConfig.from_pretrained(init_param_path)
        #         model = FunAudioForDualChannelModel.from_config(llm_config)

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
        audio_adaptor_conf["llm_dim"] = llm_dim
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
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.audio_code_book_size = 59049  # 6761 59049
        self.mask_token_id = -60

        # audio embed and head
        # TODO: take care the param initialization
        self.special_tokens_audio_count = 463
        self.audio_embed = nn.Embedding(self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
        self.audio_head = nn.Linear(llm_dim, self.audio_code_book_size + self.special_tokens_audio_count, bias=False)

        audio_embed_ckpt_path = llm_conf.get("audio_embed_ckpt_path", None)
        if audio_embed_ckpt_path is not None:
            audio_embed_weight = torch.load(audio_embed_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
            if audio_embed_weight.shape == expected_shape:
                self.audio_embed.weight.data = audio_embed_weight
                logging.info(f"Loading audio_embed from {audio_embed_ckpt_path}, shape: {audio_embed_weight.shape}")
            else:
                logging.warning(f"audio_embed shape mismatch: expected {expected_shape}, got {audio_embed_weight.shape}")
        else:
            logging.info(f"No audio_embed_ckpt_path provided, using default initialization")
        
        audio_head_ckpt_path = llm_conf.get("audio_head_ckpt_path", None)
        if audio_head_ckpt_path is not None:
            audio_head_weight = torch.load(audio_head_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
            if audio_head_weight.shape == expected_shape:
                self.audio_head.weight.data = audio_head_weight
                logging.info(f"Loading audio_head weight from {audio_head_ckpt_path}, shape: {audio_head_weight.shape}")
            else:
                logging.warning(f"audio_head weight shape mismatch: expected {expected_shape}, got {audio_head_weight.shape}")
        else:
            logging.info(f"No audio_head_ckpt_path provided, using default initialization")
        
        if llm_conf.get("use_lora", False):
            # self.llm.model.embed_tokens_audio = self.audio_embed
            # self.llm.model.lm_head_audio = self.audio_head

            self.llm.base_model.model.embed_tokens_audio = self.audio_embed
            self.llm.base_model.model.lm_head_audio = self.audio_head

            # self.llm.base_model.model.model.embed_tokens_audio = self.audio_embed
            # self.llm.base_model.model.model.lm_head_audio = self.audio_head
        else:
            self.llm.embed_tokens_audio = self.audio_embed
            self.llm.lm_head_audio = self.audio_head

        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

        self.input_size = input_size


    def forward(
            self,
            speech: torch.Tensor = None,
            speech_lengths: torch.Tensor = None,
            input_ids: torch.Tensor = None,
            labels_ids: torch.Tensor = None,
            input_ids_audio: torch.Tensor = None,
            labels_ids_audio: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            feature_beg: torch.Tensor = None,
            feature_type: torch.Tensor = None,
            output_embeds: bool = False, 
            **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # print (f'[Model forward] input length: {input_ids.shape}')
        
        assert input_ids.max() < self.llm.model.config.vocab_size, f'input_ids.max() {input_ids.max()} >= self.llm.model.config.vocab_size {self.llm.model.config.vocab_size}'
        assert input_ids_audio.max() < self.audio_embed.num_embeddings, f'input_ids_audio.max() {input_ids_audio.max()} >= self.audio_embed.num_embeddings {self.audio_embed.num_embeddings}'

        assert labels_ids.max() < self.llm.model.config.vocab_size, f'labels_ids.max() {labels_ids.max()} >= self.llm.model.config.vocab_size {self.llm.model.config.vocab_size}'
        assert labels_ids_audio.max() < self.audio_embed.num_embeddings, f'labels_ids_audio.max() {labels_ids_audio.max()} >= self.audio_embed.num_embeddings {self.audio_embed.num_embeddings}'

        # set position according to self.mask_token_id to 0
        input_ids_audio_mask = input_ids_audio == self.mask_token_id
        input_ids_audio[input_ids_audio < 0] = 0
        inputs_embeds_audio = self.audio_embed(input_ids_audio)
        inputs_embeds_audio[input_ids_audio_mask] = 0

        stats = {}
        input_ids_mask = input_ids == self.mask_token_id
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        inputs_embeds[input_ids_mask] = 0

        batch_size_speech = 0
        frames = 0
        acoustic_audio_emb = None
        if speech is not None:
            # speech
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
            # print (f"acoustic_audio_emb shape: {acoustic_audio_emb.shape}")
            loss_fake = None
            # print (f'[MODEL FORWARD] with speech data, shape: {speech.shape}')
        else:
            batch_size_speech = 1
            frames = 64
            speech = torch.zeros(1, 64, 128).to(inputs_embeds.device)
            speech_lengths = torch.tensor([64], dtype=torch.int32).to(inputs_embeds.device)
            
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
            loss_fake = encoder_out.mean() * 0.0

        batch_size, token_num, _ = inputs_embeds.shape
        fake_token_len = kwargs.get("fake_token_len")

        speech_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"assign speech token failed. batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1

        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item() if speech_lengths is not None else 0
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        inputs_embeds = inputs_embeds + inputs_embeds_audio
        if output_embeds:
            return inputs_embeds
        inputs_embeds = inputs_embeds[:, :-1, :]
        with torch.amp.autocast('cuda',
                enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids == -1] = -100
            labels_ids_audio[labels_ids_audio == -1] = -100
            attention_mask[attention_mask < 0] = 0
            attention_mask = attention_mask[:, :-1]
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = model_outputs.hidden_states[-1]
            logits_text = self.llm.lm_head(hidden_states)
            logits_audio = self.audio_head(hidden_states)

            labels_ids = labels_ids[:, 1:]
            labels_ids_audio = labels_ids_audio[:, 1:]

            is_all_minus_100 = (labels_ids == -100).all() 
            if not is_all_minus_100:
                text_loss = F.cross_entropy(
                            logits_text.view(-1, logits_text.size(-1)),
                            labels_ids.contiguous().view(-1),
                            ignore_index=-100,
                            reduction='none'
                        )
            else:
                # text_loss = 0.0
                logits_text_mean = torch.nan_to_num(
                logits_text.mean(dim=-1), 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
                )
                text_loss = logits_text_mean * 0.0
            is_all_minus_100_audio = (labels_ids_audio == -100).all()
            if not is_all_minus_100_audio:
                audio_loss = F.cross_entropy(
                                logits_audio.view(-1, logits_audio.size(-1)),
                                labels_ids_audio.contiguous().view(-1),
                                ignore_index=-100,
                                reduction='none'
                            )
            else:
                # audio_loss = 0.0
                logits_audio_mean = torch.nan_to_num(
                logits_audio.mean(dim=-1), 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
                )
                audio_loss = logits_audio_mean * 0.0

            # loss = text_loss + audio_loss

            # valid-token level average loss
            loss_mask_text = (labels_ids != -100).float().view(-1)
            loss_mask_audio = (labels_ids_audio != -100).float().view(-1)
            text_loss = torch.sum(text_loss.view(-1) * loss_mask_text) / (loss_mask_text.sum() + loss_mask_audio.sum())
            audio_loss = torch.sum(audio_loss.view(-1) * loss_mask_audio) / (loss_mask_text.sum() + loss_mask_audio.sum())
            loss = text_loss + audio_loss

            # print (f'[MODEL FORWARD] is_all_minus_100: {is_all_minus_100}, is_all_minus_100_audio: {is_all_minus_100_audio}, text_loss: {text_loss}, audio_loss: {audio_loss}')

            if loss_fake is not None:
                loss += loss_fake

        if is_all_minus_100 and is_all_minus_100_audio:
            print (f'[MODEL FORWARD] all minus 100, text and audio')
            print (f'[MODEL FORWARD] input_ids: {input_ids}')
            print (f'[MODEL FORWARD] input_ids_audio: {input_ids_audio}')
            stats["loss"] = 0.0
            stats["text_loss"] = 0.0
            stats["text_loss"] = 0.0
        else:
            stats["loss"] = torch.clone(loss.detach())
            stats["text_loss"] = torch.clone(text_loss.detach())
            stats["audio_loss"] = torch.clone(audio_loss.detach())

        with torch.no_grad():
            if not is_all_minus_100:
                preds = torch.argmax(logits_text, -1)
                acc_att = compute_accuracy(preds, labels_ids, ignore_label=-100)
                stats["acc"] = acc_att
            else:
                stats["acc"] = -1

            if not is_all_minus_100_audio:
                preds = torch.argmax(logits_audio, -1)
                acc_att = compute_accuracy(preds, labels_ids_audio, ignore_label=-100)
                stats["acc_audio"] = acc_att
            else:
                stats["acc_audio"] = -1
        
        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (feature_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        # input: single message data
        # output: single data dict
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            if role == "system":
                system.append(item)
            elif role == "user":
                user.append(item)
            elif role == "assistant":
                assistant.append(item)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

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

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        feature_beg = batch["feature_beg"]
        feature_type = batch["feature_type"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, _ = inputs_embeds.shape

        # audio encoder
        speech = batch.get("speech", None)
        if speech is not None:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.dtype)
            image_embeds = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        pixel_values_videos = batch.get("pixel_values_videos", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.vision_encoder.dtype)
            video_embeds = self.vision_encoder(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        speech_idx = 0
        image_idx = 0
        video_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = encoder_out[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1
                    # image
                    elif feature_type[batch_idx, feature_idx] == 2:
                        image_token = image_embeds[image_idx:image_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = image_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        image_idx += feature_length
                    # video
                    elif feature_type[batch_idx, feature_idx] == 3:
                        video_token = video_embeds[video_idx:video_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = video_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        video_idx += feature_length

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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

@tables.register("model_classes", "MinMo_S2S_Channels_SFT_Pack")
class MinMo_S2S_Channels_SFT_Pack(nn.Module):
    """ """

    def __init__(
            self,
            audio_encoder: str = None,
            audio_encoder_conf: dict = None,
            audio_adaptor: str = None,
            audio_adaptor_conf: dict = None,
            vision_encoder: str = None,
            vision_encoder_conf: dict = None,
            llm: str = None,
            llm_conf: dict = None,
            input_size: int = 80,
            length_normalized_loss: bool = False,
            **kwargs,
    ):

        super().__init__()

        # pack block diagonal attention mask
        if llm_conf.get("pack_block_diag_attention_mask", False):
            from funasr.models.minmo.packing import configure_packing
            configure_packing()
            print("pack block diagonal attention mask")

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
        if hub == "ms":
            from funasr import AutoModel
            print (f'The audio encoder is model: {audio_encoder}')
            model = AutoModel(model=audio_encoder, model_revision="master")
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            if audio_encoder == "QwenAudioEncoder":
                from funasr.models.bestrq.model import QwenAudioEncoder
                audio_encoder = QwenAudioEncoder(input_size=input_size, **audio_encoder_conf)
                audio_encoder_output_size = audio_encoder.output_size
            else:
                encoder_class = tables.encoder_classes.get(audio_encoder)
                audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
                audio_encoder_output_size = audio_encoder.output_size() if hasattr(audio_encoder, "output_size") else -1
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        print(f"llm_conf-qwen3moefused: {llm_conf.get('qwen3moefused', False)}")
        self.moe_model = llm_conf.get('qwen3moefused', False)
        if llm_conf.get('qwen3moefused', False):
            print(f"use qwen3moefused")
            import os
            os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
            print (f"load Qwen3MoeFused model from {init_param_path}")
            model = FunAudioMoeFusedForDualChannelModel.from_pretrained(
                init_param_path,
                attn_implementation="flash_attention_2",
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )
        else:
            print(f"not use qwen3moefused")
            model = FunAudioForDualChannelModel.from_pretrained(
                init_param_path,
                attn_implementation="flash_attention_2",
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )


        # if not llm_conf.get("low_cpu", False):
        #     model = FunAudioForDualChannelModel.from_pretrained(
        #         init_param_path,
        #         attn_implementation="flash_attention_2",
        #         load_in_8bit=None,
        #         device_map=None,
        #         use_cache=None,
        #         **llm_load_kwargs,
        #     )
        # else:
        #     import os

        #     if int(os.environ.get("RANK", 0)) == 0:
        #         model = FunAudioForDualChannelModel.from_pretrained(
        #             init_param_path,
        #             attn_implementation="flash_attention_2",
        #             load_in_8bit=None,
        #             device_map="cpu",
        #             use_cache=None,
        #             **llm_load_kwargs,
        #         )
        #     else:
        #         llm_config = AutoConfig.from_pretrained(init_param_path)
        #         model = FunAudioForDualChannelModel.from_config(llm_config)

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
        audio_adaptor_conf["llm_dim"] = llm_dim
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
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.audio_code_book_size = 59049  # 6761 59049
        self.mask_token_id = -60

        # audio embed and head
        # TODO: take care the param initialization
        self.special_tokens_audio_count = 463
        self.audio_embed = nn.Embedding(self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
        self.audio_head = nn.Linear(llm_dim, self.audio_code_book_size + self.special_tokens_audio_count, bias=False)

        audio_embed_ckpt_path = llm_conf.get("audio_embed_ckpt_path", None)
        if audio_embed_ckpt_path is not None:
            audio_embed_weight = torch.load(audio_embed_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
            if audio_embed_weight.shape == expected_shape:
                self.audio_embed.weight.data = audio_embed_weight
                logging.info(f"Loading audio_embed from {audio_embed_ckpt_path}, shape: {audio_embed_weight.shape}")
            else:
                logging.warning(f"audio_embed shape mismatch: expected {expected_shape}, got {audio_embed_weight.shape}")
        else:
            logging.info(f"No audio_embed_ckpt_path provided, using default initialization")
        
        audio_head_ckpt_path = llm_conf.get("audio_head_ckpt_path", None)
        if audio_head_ckpt_path is not None:
            audio_head_weight = torch.load(audio_head_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, llm_dim)
            if audio_head_weight.shape == expected_shape:
                self.audio_head.weight.data = audio_head_weight
                logging.info(f"Loading audio_head weight from {audio_head_ckpt_path}, shape: {audio_head_weight.shape}")
            else:
                logging.warning(f"audio_head weight shape mismatch: expected {expected_shape}, got {audio_head_weight.shape}")
        else:
            logging.info(f"No audio_head_ckpt_path provided, using default initialization")
        
        if llm_conf.get("use_lora", False):
            # self.llm.model.embed_tokens_audio = self.audio_embed
            # self.llm.model.lm_head_audio = self.audio_head

            self.llm.base_model.model.embed_tokens_audio = self.audio_embed
            self.llm.base_model.model.lm_head_audio = self.audio_head

            # self.llm.base_model.model.model.embed_tokens_audio = self.audio_embed
            # self.llm.base_model.model.model.lm_head_audio = self.audio_head
        else:
            self.llm.embed_tokens_audio = self.audio_embed
            self.llm.lm_head_audio = self.audio_head

        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

        self.input_size = input_size


    def forward(
            self,
            speech: torch.Tensor = None,
            speech_lengths: torch.Tensor = None,
            input_ids: torch.Tensor = None,
            labels_ids: torch.Tensor = None,
            input_ids_audio: torch.Tensor = None,
            labels_ids_audio: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            feature_beg: torch.Tensor = None,
            feature_type: torch.Tensor = None,
            output_embeds: bool = False, 
            **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # print (f'[Model forward] input length: {input_ids.shape}')
        
        assert input_ids.max() < self.llm.model.config.vocab_size, f'input_ids.max() {input_ids.max()} >= self.llm.model.config.vocab_size {self.llm.model.config.vocab_size}'
        assert input_ids_audio.max() < self.audio_embed.num_embeddings, f'input_ids_audio.max() {input_ids_audio.max()} >= self.audio_embed.num_embeddings {self.audio_embed.num_embeddings}'

        assert labels_ids.max() < self.llm.model.config.vocab_size, f'labels_ids.max() {labels_ids.max()} >= self.llm.model.config.vocab_size {self.llm.model.config.vocab_size}'
        assert labels_ids_audio.max() < self.audio_embed.num_embeddings, f'labels_ids_audio.max() {labels_ids_audio.max()} >= self.audio_embed.num_embeddings {self.audio_embed.num_embeddings}'

        # set position according to self.mask_token_id to 0
        input_ids_audio_mask = input_ids_audio == self.mask_token_id
        input_ids_audio[input_ids_audio < 0] = 0
        inputs_embeds_audio = self.audio_embed(input_ids_audio)
        inputs_embeds_audio[input_ids_audio_mask] = 0

        stats = {}
        input_ids_mask = input_ids == self.mask_token_id
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        inputs_embeds[input_ids_mask] = 0

        batch_size_speech = 0
        frames = 0
        acoustic_audio_emb = None
        if speech is not None:
            # speech
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
            # print (f"acoustic_audio_emb shape: {acoustic_audio_emb.shape}")
            loss_fake = None
            # print (f'[MODEL FORWARD] with speech data, shape: {speech.shape}')
        else:
            batch_size_speech = 1
            frames = 64
            speech = torch.zeros(1, 64, 128).to(inputs_embeds.device)
            speech_lengths = torch.tensor([64], dtype=torch.int32).to(inputs_embeds.device)
            
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
            acoustic_audio_emb = encoder_out
            loss_fake = encoder_out.mean() * 0.0

        batch_size, token_num, _ = inputs_embeds.shape
        fake_token_len = kwargs.get("fake_token_len")

        speech_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"assign speech token failed. batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = acoustic_audio_emb[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1

        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item() if speech_lengths is not None else 0
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        inputs_embeds = inputs_embeds + inputs_embeds_audio
        if output_embeds:
            return inputs_embeds
        inputs_embeds = inputs_embeds[:, :-1, :]
        with torch.amp.autocast('cuda',
                enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids == -1] = -100
            labels_ids_audio[labels_ids_audio == -1] = -100
            attention_mask[attention_mask < 0] = 0
            attention_mask = attention_mask[:, :-1]

            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask={'full_attention': attention_mask} if not self.moe_model else attention_mask,
                output_hidden_states=True,
            )
            hidden_states = model_outputs.hidden_states[-1]
            logits_text = self.llm.lm_head(hidden_states)
            logits_audio = self.audio_head(hidden_states)

            labels_ids = labels_ids[:, 1:]
            labels_ids_audio = labels_ids_audio[:, 1:]

            is_all_minus_100 = (labels_ids == -100).all() 
            if not is_all_minus_100:
                text_loss = F.cross_entropy(
                            logits_text.view(-1, logits_text.size(-1)),
                            labels_ids.contiguous().view(-1),
                            ignore_index=-100,
                            reduction='none'
                        )
            else:
                # text_loss = 0.0
                logits_text_mean = torch.nan_to_num(
                logits_text.mean(dim=-1), 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
                )
                text_loss = logits_text_mean * 0.0
            is_all_minus_100_audio = (labels_ids_audio == -100).all()
            if not is_all_minus_100_audio:
                audio_loss = F.cross_entropy(
                                logits_audio.view(-1, logits_audio.size(-1)),
                                labels_ids_audio.contiguous().view(-1),
                                ignore_index=-100,
                                reduction='none'
                            )
            else:
                # audio_loss = 0.0
                logits_audio_mean = torch.nan_to_num(
                logits_audio.mean(dim=-1), 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
                )
                audio_loss = logits_audio_mean * 0.0

            # loss = text_loss + audio_loss

            # valid-token level average loss
            loss_mask_text = (labels_ids != -100).float().view(-1)
            loss_mask_audio = (labels_ids_audio != -100).float().view(-1)
            text_loss = torch.sum(text_loss.view(-1) * loss_mask_text) / (loss_mask_text.sum() + loss_mask_audio.sum())
            audio_loss = torch.sum(audio_loss.view(-1) * loss_mask_audio) / (loss_mask_text.sum() + loss_mask_audio.sum())
            loss = text_loss + audio_loss

            # print (f'[MODEL FORWARD] is_all_minus_100: {is_all_minus_100}, is_all_minus_100_audio: {is_all_minus_100_audio}, text_loss: {text_loss}, audio_loss: {audio_loss}')

            if loss_fake is not None:
                loss += loss_fake

        if is_all_minus_100 and is_all_minus_100_audio:
            print (f'[MODEL FORWARD] all minus 100, text and audio')
            print (f'[MODEL FORWARD] input_ids: {input_ids}')
            print (f'[MODEL FORWARD] input_ids_audio: {input_ids_audio}')
            stats["loss"] = 0.0
            stats["text_loss"] = 0.0
            stats["text_loss"] = 0.0
        else:
            stats["loss"] = torch.clone(loss.detach())
            stats["text_loss"] = torch.clone(text_loss.detach())
            stats["audio_loss"] = torch.clone(audio_loss.detach())

        with torch.no_grad():
            if not is_all_minus_100:
                preds = torch.argmax(logits_text, -1)
                acc_att = compute_accuracy(preds, labels_ids, ignore_label=-100)
                stats["acc"] = acc_att
            else:
                stats["acc"] = -1

            if not is_all_minus_100_audio:
                preds = torch.argmax(logits_audio, -1)
                acc_att = compute_accuracy(preds, labels_ids_audio, ignore_label=-100)
                stats["acc_audio"] = acc_att
            else:
                stats["acc_audio"] = -1
        
        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = (attention_mask>0).sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (feature_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)
        # encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        # input: single message data
        # output: single data dict
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            if role == "system":
                system.append(item)
            elif role == "user":
                user.append(item)
            elif role == "assistant":
                assistant.append(item)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

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

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        feature_beg = batch["feature_beg"]
        feature_type = batch["feature_type"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, _ = inputs_embeds.shape

        # audio encoder
        speech = batch.get("speech", None)
        if speech is not None:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.dtype)
            image_embeds = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        pixel_values_videos = batch.get("pixel_values_videos", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.vision_encoder.dtype)
            video_embeds = self.vision_encoder(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        speech_idx = 0
        image_idx = 0
        video_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = encoder_out[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1
                    # image
                    elif feature_type[batch_idx, feature_idx] == 2:
                        image_token = image_embeds[image_idx:image_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = image_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        image_idx += feature_length
                    # video
                    elif feature_type[batch_idx, feature_idx] == 3:
                        video_token = video_embeds[video_idx:video_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = video_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        video_idx += feature_length

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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



@tables.register("model_classes", "MinMo_AV2T")
class MinMo_AV2T(nn.Module):
    """ """

    def __init__(
            self,
            audio_encoder: str = None,
            audio_encoder_conf: dict = None,
            audio_adaptor: str = None,
            audio_adaptor_conf: dict = None,
            vision_encoder: str = None,
            vision_encoder_conf: dict = None,
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # vision encoder
        self.vision_encoder_activation_checkpoint = vision_encoder_conf.get(
            "activation_checkpoint", False
        )
        self.vision_encoder = None
        from transformers import Qwen2_5_VLForConditionalGeneration

        init_param_path = vision_encoder_conf.get("init_param_path", "Qwen2.5-VL-7B-Instruct")
        vision_encoder_load_kwargs = vision_encoder_conf.get("load_kwargs", {})

        if not vision_encoder_conf.get("low_cpu", False):
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                init_param_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
                use_cache=None,
                **vision_encoder_load_kwargs,
            )
        else:
            import os

            if int(os.environ.get("RANK", 0)) == 0:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    init_param_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    use_cache=None,
                    **vision_encoder_load_kwargs,
                )
            else:
                llm_config = AutoConfig.from_pretrained(init_param_path)
                model = Qwen2_5_VLForConditionalGeneration.from_config(llm_config)

        vision_encoder = model.visual
        freeze = vision_encoder_conf.get("freeze", True)
        if freeze:
            for name, param in vision_encoder.named_parameters():
                param.requires_grad = False
            vision_encoder.eval()
        del model.model
        del model.lm_head
        self.vision_encoder = vision_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
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
        audio_adaptor_conf["llm_dim"] = llm_dim
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
            pixel_values: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            pixel_values_videos: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            labels_ids: torch.Tensor = None,
            feature_beg: torch.Tensor = None,
            feature_type: torch.Tensor = None,
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

        # pdb.set_trace()
        stats = {}
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.dtype)
            if self.vision_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint
                image_embeds = checkpoint(
                    self.vision_encoder,
                    pixel_values,
                    image_grid_thw,
                    use_reentrant=False,
                )
            else:
                image_embeds = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.vision_encoder.dtype)
            if self.vision_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint
                video_embeds = checkpoint(
                    self.vision_encoder,
                    pixel_values_videos,
                    video_grid_thw,
                    use_reentrant=False,
                )
            else:
                video_embeds = self.vision_encoder(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        batch_size_speech = 0
        frames = 0
        if speech is not None:
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

        batch_size, token_num, _ = inputs_embeds.shape
        fake_token_len = kwargs.get("fake_token_len")

        speech_idx = 0
        image_idx = 0
        video_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = encoder_out[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            # import pdb;
                            # pdb.set_trace()
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1
                    # image
                    elif feature_type[batch_idx, feature_idx] == 2:
                        image_token = image_embeds[image_idx:image_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = image_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        image_idx += feature_length
                    # video
                    elif feature_type[batch_idx, feature_idx] == 3:
                        video_token = video_embeds[video_idx:video_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = video_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        video_idx += feature_length

        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item() if speech_lengths is not None else 0
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

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
            loss = model_outputs.loss

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (feature_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
        pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        image_processor = kwargs.get("image_processor", None)

        input_ids, labels, fbank, fbank_lens, \
            pixel_values_lst, image_grid_thws, pixel_values_videos_lst, \
            video_grid_thws, video_sencond_per_grids, feature_attention_masks, input_features_lst, \
            feature_beg, feature_type, fake_token_len = (
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            ## add vision_start/vision_end
            user_prompt = user_prompt.replace("<|startofimage|>", "<|vision_start|><|startofimage|>").replace(
                "<|endofimage|>", "<|endofimage|><|vision_end|>")
            user_prompt = user_prompt.replace("<|startofvideo|>", "<|vision_start|><|startofvideo|>").replace(
                "<|endofvideo|>", "<|endofvideo|><|vision_end|>")

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

            splits = pattern.split(source_input)
            source_ids = []
            # audio
            speech = []
            speech_lengths = []
            # image
            pixel_values_turn = []
            image_grid_thw_turn = []
            # video
            pixel_values_videos_turn = []
            video_grid_thw_turn = []
            video_sencond_per_grid_turn = []
            feature_attention_mask_turn = []
            input_features_turn = []
            # audio:1, image:2, video:3
            feature_type_turn = 0
            output_kwargs = {'text_kwargs': {'padding': True, 'padding_side': 'left', 'return_tensors': 'pt'},
                             'images_kwargs': {'return_tensors': 'pt'},
                             'audio_kwargs': {'sampling_rate': 16000, 'padding': 'max_length',
                                              'return_attention_mask': True, 'return_tensors': 'pt'},
                             'videos_kwargs': {'return_tensors': 'pt'},
                             'common_kwargs': {'return_tensors': 'pt'}}
            for k, sub_str in enumerate(splits):
                if sub_str is None or sub_str == "":
                    continue
                fake_token_len_i = 0
                featrue_beg_i = -1
                if not any(sub_str.startswith(prefix) for prefix in
                           ["<|startofspeech|>", "<|startofimage|>", "<|startofvideo|>"]):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                elif sub_str.startswith("<|startofspeech|>"):
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
                        featrue_beg_i = len(source_ids)
                        feature_type_turn = 1
                        source_ids += fake_token
                elif sub_str.startswith("<|startofimage|>") or sub_str.startswith("<|startofvideo|>"):
                    from transformers.image_utils import make_batched_videos
                    import qwen_omni_utils
                    image_inputs = []
                    video_inputs = []
                    video_sample_fps_list = []

                    if sub_str.startswith("<|startofimage|>"):
                        try:
                            sub_str = sub_str.replace("<|startofimage|>", "").replace(
                                "<|endofimage|>", "")
                            image_inputs.append(qwen_omni_utils.fetch_image({"image": sub_str[1:]}))
                            if len(image_inputs) == 0:
                                image_inputs = None

                            images_inputs = image_processor(images=image_inputs, videos=None,
                                                            **output_kwargs["images_kwargs"])
                            image_grid_thw = images_inputs["image_grid_thw"]
                            # image_index = 0
                            fake_token_len_i = image_grid_thw[0].prod() // (image_processor.merge_size ** 2)
                            fake_token = [0] * fake_token_len_i
                            featrue_beg_i = len(source_ids)
                            feature_type_turn = 2
                            source_ids += fake_token
                            pixel_values_turn.extend(images_inputs["pixel_values"])
                            image_grid_thw_turn.extend(image_grid_thw)

                        except Exception as e:
                            logging.warning(
                                f"Loading Image failed! {str(e)}, {traceback.format_exc()}"
                            )
                            badcase_flag = True
                            continue
                    elif sub_str.startswith("<|startofvideo|>"):
                        use_audio_in_video = False
                        try:
                            sub_str = sub_str.replace("<|startofvideo|>", "").replace(
                                "<|endofvideo|>", "")
                            video_input, video_sample_fps = qwen_omni_utils.fetch_video({"video": sub_str[1:]},
                                                                                        return_video_sample_fps=True)
                            video_sample_fps_list.append(video_sample_fps)
                            video_inputs.append(video_input)
                            if len(video_inputs) == 0:
                                video_inputs = None

                            videos = make_batched_videos(video_inputs)
                            videos_inputs = image_processor(images=None, videos=videos,
                                                            **output_kwargs["videos_kwargs"])
                            fps = output_kwargs["videos_kwargs"].pop("fps", None)
                            if fps is None:
                                fps = [2.0] * len(videos)
                            videos_inputs["video_second_per_grid"] = [
                                image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
                            ]
                            video_grid_thw = videos_inputs["video_grid_thw"]

                            if not use_audio_in_video:
                                # video_index = 0
                                fake_token_len_i = video_grid_thw[0].prod() // (image_processor.merge_size ** 2)
                                fake_token = [0] * fake_token_len_i
                                featrue_beg_i = len(source_ids)
                                feature_type_turn = 3
                                source_ids += fake_token
                                pixel_values_videos_turn.extend(videos_inputs["pixel_values_videos"])
                                video_grid_thw_turn.extend(videos_inputs["video_grid_thw"])
                                # video_sencond_per_grid_turn.append(videos_inputs["video_second_per_grid"])
                                # feature_attention_mask_turn.append(videos_inputs.get("feature_attention_mask", None))
                                # input_features_turn.append(videos_inputs.get("input_features", None))
                            else:
                                # TODO: add audio in video
                                pass
                        except Exception as e:
                            logging.warning(
                                f"Loading Image failed! {str(e)}, {traceback.format_exc()}"
                            )
                            badcase_flag = True
                            continue
                if featrue_beg_i > 0:
                    feature_beg += [featrue_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                    feature_type += [feature_type_turn]

            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)
            if len(pixel_values_turn) > 0:
                pixel_values_lst.extend(pixel_values_turn)
                image_grid_thws.extend(image_grid_thw_turn)
            if len(pixel_values_videos_turn) > 0:
                pixel_values_videos_lst.extend(pixel_values_videos_turn)
                video_grid_thws.extend(video_grid_thw_turn)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        # fbank = speech[0, :, :]
        # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
        feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
        feature_type = torch.tensor(feature_type, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        output = {
            "feature_beg": feature_beg[None, :],
            "feature_type": feature_type[None, :],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None, :],
            "attention_mask": attention_mask[None, :],
            "labels_ids": labels[None, :],
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        if len(fbank) > 0:
            output["speech"] = fbank
            output["speech_lengths"] = fbank_lens
        if len(pixel_values_lst) > 0:
            output["pixel_values"] = pixel_values_lst
            output["image_grid_thw"] = image_grid_thws
        if len(pixel_values_videos_turn) > 0:
            output["pixel_values_videos"] = pixel_values_videos_lst
            output["video_grid_thw"] = video_grid_thws

        for key, data_list in output.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                    pad_value = -1
                else:
                    pad_value = 0.0

                output[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

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

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        feature_beg = batch["feature_beg"]
        feature_type = batch["feature_type"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, _ = inputs_embeds.shape

        # audio encoder
        speech = batch.get("speech", None)
        if speech is not None:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.dtype)
            image_embeds = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        pixel_values_videos = batch.get("pixel_values_videos", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.vision_encoder.dtype)
            video_embeds = self.vision_encoder(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        speech_idx = 0
        image_idx = 0
        video_idx = 0
        for batch_idx in range(batch_size):
            for feature_idx in range(feature_beg.shape[1]):
                feature_beg_id = feature_beg[batch_idx, feature_idx].item()
                if feature_beg_id > 0:
                    feature_length = fake_token_len[batch_idx, feature_idx].item()
                    # audio
                    if feature_type[batch_idx, feature_idx] == 1:
                        speech_token = encoder_out[speech_idx, :feature_length, :]

                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, feature_beg_id: {feature_beg_id}, speech_token_len: {feature_length}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            feature_length = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :feature_length, :]
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = speech_token
                        speech_idx += 1
                    # image
                    elif feature_type[batch_idx, feature_idx] == 2:
                        image_token = image_embeds[image_idx:image_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = image_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        image_idx += feature_length
                    # video
                    elif feature_type[batch_idx, feature_idx] == 3:
                        video_token = video_embeds[video_idx:video_idx + feature_length, :]
                        try:
                            inputs_embeds[
                            batch_idx, feature_beg_id: feature_beg_id + feature_length, :
                            ] = video_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            # TODO
                            pass
                        video_idx += feature_length

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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


@tables.register("model_classes", "MinMo_T2S")
class MinMo_T2S(nn.Module):
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
            attn_implementation="flash_attention_2",
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
        self.codec_unit = kwargs.get("codec_unit", 4096) + 1
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
                    codec_i = codec_emb[idx: idx + 1, :codec_len_i, :]
                    codec_i, _ = self.specaug(codec_i, codec_len_i)
                    inputs_embeds[
                    batch_idx: batch_idx + 1, codec_beg_idx: codec_beg_idx + codec_len_i, :
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

    # def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):
    #
    #     system = contents["system"]
    #     user = contents["user"]
    #     assistant = contents["assistant"]
    #     pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
    #
    #     input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #     )
    #     input_source_ids = []
    #     for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
    #         if i >= kwargs.get("multiturn_num_max", 5):
    #             break
    #         if len(input_ids) > kwargs.get("max_token_length", 1500):
    #             break
    #         if isinstance(user_prompt, (list, tuple)):
    #             user_prompt, meta_dict = user_prompt
    #             audio = meta_dict.get("audio", None)
    #
    #             source_input = (
    #                 f"<|im_start|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    #             )
    #             source_input
    #
    #         splits = pattern.split(source_input)
    #         source_ids = []
    #         fbank_i = []
    #         fbank_mask_i = []
    #         fake_token_len_i = 0
    #         fbank_beg_i = -1
    #         fbank_lens_i = []
    #         speech, speech_lengths = [], []
    #         for k, sub_str in enumerate(splits):
    #             if not sub_str.startswith("<|startofspeech|>"):
    #                 sub_token = tokenizer.encode(sub_str)
    #                 source_ids += sub_token
    #                 fbank_mask_i += [0] * len(sub_token)
    #             else:
    #                 sub_str = sub_str.replace("<|startofspeech|>", "").replace(
    #                     "<|endofspeech|>", ""
    #                 )
    #                 if sub_str.startswith("!"):
    #                     sub_str = sub_str[1:]
    #                     if sub_str.startswith("!"):  # !!: audio sample point
    #                         sub_str = audio
    #                     try:
    #                         time1 = time.perf_counter()
    #                         data_src = load_audio_text_image_video(sub_str, fs=frontend.fs)
    #                         time2 = time.perf_counter()
    #                         meta_data["load_data"] = f"{time2 - time1:0.3f}"
    #                     except Exception as e:
    #                         logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
    #
    #                     speech, speech_lengths = extract_fbank(
    #                         data_src,
    #                         data_type=kwargs.get("data_type", "sound"),
    #                         frontend=frontend,
    #                         is_final=True,
    #                     )  # speech: [b, T, d]
    #
    #                     time3 = time.perf_counter()
    #                     meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
    #                     meta_data["batch_data_time"] = (
    #                         speech_lengths.sum().item()
    #                         * frontend.frame_shift
    #                         * frontend.lfr_n
    #                         / 1000
    #                     )
    #
    #                     if kwargs.get("permute", True):
    #                         speech = speech.permute(0, 2, 1)
    #                     if speech_lengths > kwargs.get("max_source_length", 5500):
    #                         # logging.info(
    #                         #     f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
    #                         # )
    #                         badcase_flag = True
    #
    #                     olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
    #                     olens = 1 + (olens - 3 + 2 * 1) // 2
    #                     fake_token_len_i = (olens - 1) // 2 + 1
    #                     fake_token = [0] * fake_token_len_i
    #                     fbank_beg_i = len(source_ids)
    #                     source_ids += fake_token
    #                     fbank_mask_i += [1] * len(fake_token)
    #
    #         fbank_beg += [fbank_beg_i + len(input_ids)]
    #         fake_token_len += [fake_token_len_i]
    #         source_mask = [-100] * len(source_ids)
    #         target_out = f"{target_out}<|im_end|>"
    #         target_ids = tokenizer.encode(target_out)
    #         input_source_ids = input_ids + source_ids
    #         input_ids += source_ids + target_ids
    #         labels += source_mask + target_ids
    #         fbank_mask += fbank_mask_i
    #         if len(speech) > 0:
    #             fbank.append(speech[0, :, :])
    #             fbank_lens.append(speech_lengths)
    #
    #     input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
    #     attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
    #     labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
    #
    #     # fbank = speech[0, :, :]
    #     # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
    #     fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
    #     fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
    #     fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
    #     source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
    #     target_ids = torch.tensor(target_ids, dtype=torch.int64)
    #
    #     if len(fbank) > 0:
    #         speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
    #         speech_lengths = torch.nn.utils.rnn.pad_sequence(
    #             fbank_lens, batch_first=True, padding_value=-1
    #         )
    #     else:
    #         speech = []
    #         speech_lengths = []
    #     output = {
    #         "speech": speech,
    #         "speech_lengths": speech_lengths,
    #         "fbank_mask": fbank_mask[None, :],
    #         "fbank_beg": fbank_beg[None,],
    #         "fake_token_len": fake_token_len[None, :],
    #         "input_ids": input_ids[None,],
    #         "attention_mask": attention_mask[None,],
    #         "labels_ids": labels,
    #         "source_ids": source_ids[None, :],
    #         "target_ids": target_ids[None, :],
    #     }
    #
    #     return output

    def data_load_codec(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

        user_prompt_text = kwargs.get("user_prompt_text", None)

        prompt_codec = kwargs.get("prompt_codec", None)
        if prompt_codec is not None:
            prompt_codec = np.load(prompt_codec)
            prompt_codec = prompt_codec.tolist() + [4096]
            prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)

        prompt_wav = kwargs.get("prompt_wav", None)
        if prompt_wav is not None:
            prompt_wav = load_audio_text_image_video(prompt_wav, fs=frontend.fs)

        spk_emb = kwargs.get("spk_emb", None)
        if spk_emb is not None:
            spk_emb = np.load(spk_emb)
            spk_emb = torch.from_numpy(spk_emb)

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        # pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

        input_ids = []
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):

            user_prompt = user_prompt.replace("语音合成：", "")
            user_prompt = user_prompt.replace("Speech Synthesis:", "")
            if user_prompt_text is not None:
                user_prompt = user_prompt_text + user_prompt

            source_input = f"<|im_start|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            source_ids = tokenizer.encode(source_input)
            # single turn
            input_source_ids = input_ids + source_ids

            ## multi-turn
            # source_mask = [-100] * len(source_ids)
            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)
            # input_source_ids = input_ids + source_ids
            # input_ids += source_ids + target_ids
            # labels += source_mask + target_ids

        input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)

        output = {
            "source_ids": input_source_ids[None, :],
            "user_prompt_text": user_prompt_text,
            "prompt_codec": prompt_codec,
            "prompt_wav": prompt_wav,
            "spk_emb": spk_emb,
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

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_codec(
            contents, tokenizer, frontend, meta_data=meta_data, **kwargs
        )  # offline, 必须包括：[source_ids]

        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]

        source_ids[source_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(source_ids)

        prompt_codec = batch.get("prompt_codec", None)
        prompt_wav = batch.get("prompt_wav", None)
        spk_emb = batch.get("spk_emb", None)
        user_prompt_text = batch.get("user_prompt_text", None)

        if prompt_codec is not None:
            codec_emb = self.codec_embed(prompt_codec[None,])
            inputs_embeds = torch.cat((inputs_embeds, codec_emb), dim=1)

        meta_data = {
            "user_prompt_text": user_prompt_text,
            "prompt_codec": prompt_codec,
            "prompt_wav": prompt_wav,
            "spk_emb": spk_emb,
        }

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
                enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            self.llm.lm_head = self.codec_head
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})

            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                **llm_kwargs,
            )

        # flow
        prompt_codec = meta_data.get("prompt_codec", None)
        prompt_wav = meta_data.get("prompt_wav", None)
        spk_emb = meta_data.get("spk_emb", None)

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {"key": key[0], "text": response, "text_tn": response_clean, "label": label}

        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data


@tables.register("model_classes", "MinMo_T2S_low_rate")
class MinMo_T2S_low(nn.Module):
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
            attn_implementation="flash_attention_2",
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
        self.codec_unit = kwargs.get("codec_unit", 4096) + 1
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

        from .subsampling import Conv1dSubsampling

        sub_sampling_conf = kwargs.get(
            "sub_sampling_conf",
            {
                "kernel_size": 2,
                "stride": 2,
                "pad": [0, 1],
            },
        )
        self.codec_subsampling = Conv1dSubsampling(
            **sub_sampling_conf,
            idim=llm_dim,
            odim=llm_dim,
        )

        from .upsampling import Conv1dUpsampling

        up_sampling_conf = kwargs.get(
            "up_sampling_conf",
            {
                "kernel_size": 2,
                "stride": 2,
            },
        )
        self.codec_upsampling = Conv1dUpsampling(
            **up_sampling_conf,
            idim=self.codec_unit,
            odim=self.codec_unit,
        )

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
        codec_emb_sub, codec_emb_sub_len = self.codec_subsampling(codec_emb, codec_len)
        idx = 0
        for i, batch_idx in enumerate(range(batch_size)):

            for j, turn_id in enumerate(range(codec_beg.shape[1])):
                codec_beg_idx = codec_beg[batch_idx, turn_id].item()
                if codec_beg_idx > 0:
                    codec_len_i = fake_codec_len[batch_idx, turn_id]
                    codec_i = codec_emb_sub[idx: idx + 1, :codec_len_i, :]
                    codec_i, _ = self.specaug(codec_i, codec_len_i)
                    inputs_embeds[
                    batch_idx: batch_idx + 1, codec_beg_idx: codec_beg_idx + codec_len_i, :
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

            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
            logits = logits.float()

            idx = 0
            b, _, d = logits.shape
            _, t = labels_ids.shape
            logits_upsampling = torch.zeros((b, t, d), dtype=logits.dtype).to(logits.device)

            for i, batch_idx in enumerate(range(batch_size)):

                for j, turn_id in enumerate(range(codec_beg.shape[1])):
                    codec_beg_idx = codec_beg[batch_idx, turn_id].item()
                    if codec_beg_idx > 0:
                        codec_len_i = fake_codec_len[batch_idx, turn_id]
                        codec_sub_i = logits[
                                      batch_idx: batch_idx + 1,
                                      codec_beg_idx: codec_beg_idx + codec_len_i,
                                      :,
                                      ]
                        codec_up_i, codec_up_len_i = self.codec_upsampling(
                            codec_sub_i, codec_emb_sub_len[batch_idx: batch_idx + 1]
                        )
                        logits_upsampling[batch_idx: batch_idx + 1, :codec_beg_idx, :] = logits[
                                                                                         batch_idx: batch_idx + 1,
                                                                                         :codec_beg_idx, :
                                                                                         ]
                        codec_up_len_i_raw = codec_len[batch_idx].item()
                        logits_upsampling[
                        batch_idx: batch_idx + 1,
                        codec_beg_idx: codec_beg_idx + codec_up_len_i_raw,
                        :,
                        ] = codec_up_i[:, :codec_up_len_i_raw, :]

            # b, _, d = logits.shape
            # logits_codec_emb = torch.zeros((b, codec_emb_sub.shape[1], d), dtype=logits.dtype).to(logits.device)
            # for i, batch_idx in enumerate(range(batch_size)):
            #
            #     for j, turn_id in enumerate(range(codec_beg.shape[1])):
            #         codec_beg_idx = codec_beg[batch_idx, turn_id].item()
            #         if codec_beg_idx > 0:
            #             codec_len_i = fake_codec_len[batch_idx, turn_id]
            #             codec_sub_i = logits[
            #                 batch_idx : batch_idx + 1,
            #                 codec_beg_idx : codec_beg_idx + codec_len_i,
            #                 :,
            #             ]
            #
            #             logits_codec_emb[batch_idx: batch_idx + 1, :codec_len_i, :] = codec_sub_i
            #
            # codec_up, codec_up_len = self.codec_upsampling(
            #     logits_codec_emb, fake_codec_len
            # )
            #
            # _, t = labels_ids.shape
            # logits_upsampling = torch.zeros((b, t, d), dtype=logits.dtype).to(logits.device)
            # for i, batch_idx in enumerate(range(batch_size)):
            #
            #     for j, turn_id in enumerate(range(codec_beg.shape[1])):
            #         codec_beg_idx = codec_beg[batch_idx, turn_id].item()
            #         if codec_beg_idx > 0:
            #             # codec_up_i, codec_up_len_i = self.codec_upsampling(
            #             #     codec_sub_i, codec_emb_sub_len[batch_idx: batch_idx + 1]
            #             # )
            #             logits_upsampling[batch_idx: batch_idx + 1, :codec_beg_idx, :] = logits[
            #                                                                              batch_idx: batch_idx + 1,
            #                                                                              :codec_beg_idx, :
            #                                                                              ]
            #             codec_up_len_i_raw = codec_len[batch_idx].item()
            #             logits_upsampling[
            #             batch_idx: batch_idx + 1,
            #             codec_beg_idx: codec_beg_idx + codec_up_len_i_raw,
            #             :,
            #             ] = codec_up[batch_idx: batch_idx + 1, :codec_up_len_i_raw, :]

            # Shift so that tokens < n predict n

            shift_logits = logits_upsampling[..., :-2, :].contiguous()
            shift_labels = labels_ids[..., 2:].contiguous()

            loss = self.criterion_ce(shift_logits, shift_labels)

        with torch.no_grad():
            preds = torch.argmax(logits_upsampling, -1)
            acc_att = compute_accuracy(
                preds[:, :-2], labels_ids[:, 2:], ignore_label=self.ignore_id
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

    # def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):
    #
    #     system = contents["system"]
    #     user = contents["user"]
    #     assistant = contents["assistant"]
    #     pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
    #
    #     input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #     )
    #     input_source_ids = []
    #     for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
    #         if i >= kwargs.get("multiturn_num_max", 5):
    #             break
    #         if len(input_ids) > kwargs.get("max_token_length", 1500):
    #             break
    #         if isinstance(user_prompt, (list, tuple)):
    #             user_prompt, meta_dict = user_prompt
    #             audio = meta_dict.get("audio", None)
    #
    #             source_input = (
    #                 f"<|im_start|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    #             )
    #             source_input
    #
    #         splits = pattern.split(source_input)
    #         source_ids = []
    #         fbank_i = []
    #         fbank_mask_i = []
    #         fake_token_len_i = 0
    #         fbank_beg_i = -1
    #         fbank_lens_i = []
    #         speech, speech_lengths = [], []
    #         for k, sub_str in enumerate(splits):
    #             if not sub_str.startswith("<|startofspeech|>"):
    #                 sub_token = tokenizer.encode(sub_str)
    #                 source_ids += sub_token
    #                 fbank_mask_i += [0] * len(sub_token)
    #             else:
    #                 sub_str = sub_str.replace("<|startofspeech|>", "").replace(
    #                     "<|endofspeech|>", ""
    #                 )
    #                 if sub_str.startswith("!"):
    #                     sub_str = sub_str[1:]
    #                     if sub_str.startswith("!"):  # !!: audio sample point
    #                         sub_str = audio
    #                     try:
    #                         time1 = time.perf_counter()
    #                         data_src = load_audio_text_image_video(sub_str, fs=frontend.fs)
    #                         time2 = time.perf_counter()
    #                         meta_data["load_data"] = f"{time2 - time1:0.3f}"
    #                     except Exception as e:
    #                         logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
    #
    #                     speech, speech_lengths = extract_fbank(
    #                         data_src,
    #                         data_type=kwargs.get("data_type", "sound"),
    #                         frontend=frontend,
    #                         is_final=True,
    #                     )  # speech: [b, T, d]
    #
    #                     time3 = time.perf_counter()
    #                     meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
    #                     meta_data["batch_data_time"] = (
    #                         speech_lengths.sum().item()
    #                         * frontend.frame_shift
    #                         * frontend.lfr_n
    #                         / 1000
    #                     )
    #
    #                     if kwargs.get("permute", True):
    #                         speech = speech.permute(0, 2, 1)
    #                     if speech_lengths > kwargs.get("max_source_length", 5500):
    #                         # logging.info(
    #                         #     f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
    #                         # )
    #                         badcase_flag = True
    #
    #                     olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
    #                     olens = 1 + (olens - 3 + 2 * 1) // 2
    #                     fake_token_len_i = (olens - 1) // 2 + 1
    #                     fake_token = [0] * fake_token_len_i
    #                     fbank_beg_i = len(source_ids)
    #                     source_ids += fake_token
    #                     fbank_mask_i += [1] * len(fake_token)
    #
    #         fbank_beg += [fbank_beg_i + len(input_ids)]
    #         fake_token_len += [fake_token_len_i]
    #         source_mask = [-100] * len(source_ids)
    #         target_out = f"{target_out}<|im_end|>"
    #         target_ids = tokenizer.encode(target_out)
    #         input_source_ids = input_ids + source_ids
    #         input_ids += source_ids + target_ids
    #         labels += source_mask + target_ids
    #         fbank_mask += fbank_mask_i
    #         if len(speech) > 0:
    #             fbank.append(speech[0, :, :])
    #             fbank_lens.append(speech_lengths)
    #
    #     input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
    #     attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
    #     labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
    #
    #     # fbank = speech[0, :, :]
    #     # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
    #     fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
    #     fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
    #     fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
    #     source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
    #     target_ids = torch.tensor(target_ids, dtype=torch.int64)
    #
    #     if len(fbank) > 0:
    #         speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
    #         speech_lengths = torch.nn.utils.rnn.pad_sequence(
    #             fbank_lens, batch_first=True, padding_value=-1
    #         )
    #     else:
    #         speech = []
    #         speech_lengths = []
    #     output = {
    #         "speech": speech,
    #         "speech_lengths": speech_lengths,
    #         "fbank_mask": fbank_mask[None, :],
    #         "fbank_beg": fbank_beg[None,],
    #         "fake_token_len": fake_token_len[None, :],
    #         "input_ids": input_ids[None,],
    #         "attention_mask": attention_mask[None,],
    #         "labels_ids": labels,
    #         "source_ids": source_ids[None, :],
    #         "target_ids": target_ids[None, :],
    #     }
    #
    #     return output

    def data_load_codec(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

        user_prompt_text = kwargs.get("user_prompt_text", None)

        prompt_codec = kwargs.get("prompt_codec", None)
        if prompt_codec is not None:
            prompt_codec = np.load(prompt_codec)
            prompt_codec = prompt_codec.tolist() + [4096]
            prompt_codec = torch.tensor(prompt_codec, dtype=torch.int64)

        prompt_wav = kwargs.get("prompt_wav", None)
        if prompt_wav is not None:
            prompt_wav = load_audio_text_image_video(prompt_wav, fs=frontend.fs)

        spk_emb = kwargs.get("spk_emb", None)
        if spk_emb is not None:
            spk_emb = np.load(spk_emb)
            spk_emb = torch.from_numpy(spk_emb)

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        # pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

        input_ids = []
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):

            user_prompt = user_prompt.replace("语音合成：", "")
            user_prompt = user_prompt.replace("Speech Synthesis:", "")
            if user_prompt_text is not None:
                user_prompt = user_prompt_text + user_prompt

            source_input = f"<|im_start|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            source_ids = tokenizer.encode(source_input)
            # single turn
            input_source_ids = input_ids + source_ids

            ## multi-turn
            # source_mask = [-100] * len(source_ids)
            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)
            # input_source_ids = input_ids + source_ids
            # input_ids += source_ids + target_ids
            # labels += source_mask + target_ids

        input_source_ids = torch.tensor(input_source_ids, dtype=torch.int64)

        output = {
            "source_ids": input_source_ids[None, :],
            "user_prompt_text": user_prompt_text,
            "prompt_codec": prompt_codec,
            "prompt_wav": prompt_wav,
            "spk_emb": spk_emb,
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

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_codec(
            contents, tokenizer, frontend, meta_data=meta_data, **kwargs
        )  # offline, 必须包括：[source_ids]

        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]

        source_ids[source_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(source_ids)

        prompt_codec = batch.get("prompt_codec", None)
        prompt_wav = batch.get("prompt_wav", None)
        spk_emb = batch.get("spk_emb", None)
        user_prompt_text = batch.get("user_prompt_text", None)

        if prompt_codec is not None:
            codec_emb = self.codec_embed(prompt_codec[None,])
            inputs_embeds = torch.cat((inputs_embeds, codec_emb), dim=1)

        meta_data = {
            "user_prompt_text": user_prompt_text,
            "prompt_codec": prompt_codec,
            "prompt_wav": prompt_wav,
            "spk_emb": spk_emb,
        }

        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
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

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
                enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            self.llm.lm_head = self.codec_head
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})

            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                **llm_kwargs,
            )

        # flow
        prompt_codec = meta_data.get("prompt_codec", None)
        prompt_wav = meta_data.get("prompt_wav", None)
        spk_emb = meta_data.get("spk_emb", None)

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {"key": key[0], "text": response, "text_tn": response_clean, "label": label}

        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data


@tables.register("model_classes", "MinMo_S2S")
class MinMo_S2S(nn.Module):
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            attn_implementation="flash_attention_2",
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            output_hidden_states=True,
            **llm_load_kwargs,
        )
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

        from .utils.stream_utils import _sample

        model._sample = types.MethodType(_sample, model)

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

        from .utils.stream_utils import TextStatesIteratorStreamer

        self.IteratorStreamer = TextStatesIteratorStreamer
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim
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

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss

        codec_decoder = kwargs.get("codec_decoder")
        codec_decoder_conf = kwargs.get("codec_decoder_conf", {})
        init_param_path = codec_decoder_conf.get("init_param_path", None)
        codec_decoder = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            attn_implementation="flash_attention_2",
            # output_hidden_states=True,
        )

        codec_decoder_dim = codec_decoder.get_input_embeddings().weight.shape[-1]

        del codec_decoder.lm_head
        del codec_decoder.model.embed_tokens

        self.codec_decoder = codec_decoder
        self.codebook_size = codec_decoder_conf.get("codebook_size", 4096)
        self.codec_embed = nn.Embedding(self.codebook_size + 3, codec_decoder_dim, 0)
        self.codec_head = nn.Linear(codec_decoder_dim, self.codebook_size + 3, bias=True)
        self.codec_adaptor = nn.Linear(llm_dim * 2, codec_decoder_dim, bias=False)
        self.codec_sos_eos_embed = torch.nn.Embedding(2, codec_decoder_dim)

        self.sos_eos = codec_decoder_conf.get("sos_eos", 0)
        self.task_id = codec_decoder_conf.get("task_id", 1)
        self.fill_token = codec_decoder_conf.get("fill_token", 2)
        self.text_codec_mix_ratio = codec_decoder_conf.get("text_codec_mix_ratio", [4, 12])
        # self.text_codec_mix_prob = self.text_codec_mix_ratio[0] / self.text_codec_mix_ratio[1]
        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.codebook_size + 3,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        # specaug = kwargs.get("specaug", None)
        # specaug_conf = kwargs.get("specaug_conf", {})
        # if specaug is not None:
        #     specaug_class = tables.specaug_classes.get(specaug)
        #     specaug = specaug_class(**specaug_conf)
        # self.specaug = specaug
        if len(kwargs.get("flow_matching_conf", {})) > 0:

            # build flow mathing model
            self.streaming_flow = kwargs.get("streaming_flow", False)
            flow_conf = kwargs.get("flow_matching_conf", {})
            from .diffusion_text2audio_model import MaskedDiffWithXvec

            self.fm_model = MaskedDiffWithXvec(**flow_conf)
            # build vocoder
            vocoder_conf = kwargs.get("vocoder_conf", {})
            from .hifigan import HifiGan

            self.vocoder = HifiGan(**vocoder_conf)
            xvec_path = kwargs.get("xvec_path")
            if ".ark:" in xvec_path:
                import kaldiio

                xvec = kaldiio.load_mat(xvec_path)
            else:
                xvec = np.load(xvec_path)
            self.register_buffer("xvec", torch.tensor(xvec).unsqueeze(0))
            self.fm_diff_steps = kwargs.get("fm_diff_steps", 10)
            self.fm_temperature = kwargs.get("fm_temperature", 1.0)
            self.chunk_size = self.text_codec_mix_ratio[1]
            self.reback = kwargs.get("reback", 4)  # token reback
            if hasattr(self.fm_model.encoder, "pre_lookahead_len"):
                self.feat_reback_in_token = self.fm_model.encoder.pre_lookahead_len
                self.feat_reback = self.fm_model.calc_target_len(
                    self.feat_reback_in_token
                )  # feat reback
            else:
                self.feat_reback_in_token = 4
                self.feat_reback = kwargs.get("feat_reback", 8)
            self.sample_rate = flow_conf["mel_feat_conf"]["sampling_rate"]
            self.token_rate = flow_conf["input_frame_rate"]
            self.mel_hop_length = self.vocoder.mel_spec_transform.hop_length
            self.mel_frame_rate = int(self.sample_rate / self.mel_hop_length)
            self.vocoder_hop = kwargs.get("vocoder_hop", 1920)
            self.vocoder_hop_in_token = int(self.vocoder_hop / self.sample_rate * self.token_rate)
            self.vocoder_hop_in_feat = int(self.vocoder_hop / self.mel_hop_length)

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

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

            batch_size, token_num, dims = inputs_embeds.shape
            fake_token_len = kwargs.get("fake_token_len")
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            speech_idx = 0
            for batch_idx in range(batch_size):

                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    if fbank_beg_idx > 0 and speech_token_len > 0:

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
            loss_llm = model_outputs.loss
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_llm = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc_llm"] = acc_llm

        stats["loss_llm"] = torch.clone(loss_llm.detach())

        hidden_states = model_outputs.hidden_states[-1].float()
        hidden_states_select = []
        hidden_states_select_len = []
        multiturn_beg = kwargs.get("multiturn_beg")
        multiturn_beg[multiturn_beg < 0] = -1
        multiturn_len = kwargs.get("multiturn_len")
        multiturn_len[multiturn_len < 0] = 0

        for batch_idx in range(multiturn_beg.shape[0]):
            for turn_id in range(multiturn_beg.shape[1]):
                multiturn_beg_idx = multiturn_beg[batch_idx, turn_id].item()
                multiturn_len_idx = multiturn_len[batch_idx, turn_id].item() - 1  # rm eos
                multiturn_end_idx = multiturn_beg_idx + multiturn_len_idx
                if multiturn_beg_idx >= 0:
                    hidden_states_idx = hidden_states[
                                        batch_idx, multiturn_beg_idx - 1: multiturn_end_idx - 1, :
                                        ]
                    inputs_embeds_idx = inputs_embeds[
                                        batch_idx, multiturn_beg_idx:multiturn_end_idx, :
                                        ]
                    hidden_states_select.append(
                        torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                    )
                    hidden_states_select_len.append(multiturn_len_idx)

        # hidden_states_his_select
        device = hidden_states.device
        hidden_states_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_select, batch_first=True, padding_value=0.0
        ).to(device=device)
        hidden_states_select_len = torch.tensor(hidden_states_select_len, dtype=torch.int64).to(
            device=device
        )

        hidden_states_select = self.codec_adaptor(hidden_states_select)
        # codec
        codec = kwargs.get("codec")
        codec[codec < 0] = 0
        codec_len = kwargs.get("codec_len")
        # codec_len[codec_len < 0] = 0

        codec_emb = self.codec_embed(codec)
        sos_emb = self.codec_sos_eos_embed(
            torch.tensor([self.sos_eos], dtype=torch.int64).to(device=device)
        )
        task_id_emb = self.codec_sos_eos_embed(
            torch.tensor([self.task_id], dtype=torch.int64).to(device=device)
        )

        codec_decoder_inputs_embeds = []
        codec_decoder_inputs_len = []
        codec_decoder_targets = []
        codec_decoder_attention_mask = []

        for batch_idx in range(hidden_states_select.shape[0]):
            hidden_states_select_idx = hidden_states_select[
                                       batch_idx, : hidden_states_select_len[batch_idx], :
                                       ]
            codec_emb_idx = codec_emb[batch_idx, : codec_len[batch_idx], :]
            code_idx = codec[batch_idx, : codec_len[batch_idx]]
            (
                codec_decoder_inputs_embeds_idx,
                codec_decoder_targets_idx,
                codec_decoder_inputs_len_idx,
            ) = self.mixup_text_codec(
                hidden_states_select_idx,
                codec_emb_idx,
                sos_emb,
                task_id_emb,
                code_idx,
            )
            codec_decoder_inputs_embeds.append(codec_decoder_inputs_embeds_idx)
            codec_decoder_inputs_len.append(codec_decoder_inputs_len_idx)
            codec_decoder_targets.append(codec_decoder_targets_idx)
            codec_decoder_attention_mask_idx = torch.tensor(
                [1] * codec_decoder_inputs_len_idx, dtype=torch.int64
            )
            codec_decoder_attention_mask.append(codec_decoder_attention_mask_idx)

        codec_decoder_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_inputs_embeds, batch_first=True, padding_value=0.0
        ).to(device=device)
        codec_decoder_inputs_len = torch.tensor(codec_decoder_inputs_len, dtype=torch.int64).to(
            device=device
        )
        codec_decoder_targets = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_targets, batch_first=True, padding_value=-100
        ).to(device=device)
        codec_decoder_attention_mask = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_attention_mask, batch_first=True, padding_value=0.0
        ).to(device=device)

        codec_hidden_states = self.codec_decoder.model(
            inputs_embeds=codec_decoder_inputs_embeds,
            attention_mask=codec_decoder_attention_mask,
        )[0]

        codec_logits = self.codec_head(codec_hidden_states)
        loss = self.criterion_ce(codec_logits, codec_decoder_targets)

        with torch.no_grad():
            preds = torch.argmax(codec_logits, -1)
            acc_att = compute_accuracy(preds, codec_decoder_targets, ignore_label=-100)
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

    def mixup_text_codec(self, text_emb, codec_emb, sos_emb, task_id_emb, codec_token):
        # self.sos_eos:0
        # task_id_emb: 1
        # self.ignore_id = -100
        # self.text_codec_mix_ratio: [4, 12]
        # self.codebook_size: codec本身size，不包含sos
        # self.fill_token: 2
        pos = -1
        prompt_emb = text_emb[: pos + 1]
        prompt_fill_token = torch.ones(pos + 1).fill_(self.ignore_id).to(codec_token)
        text_emb = text_emb[pos + 1:]

        text_len = text_emb.shape[0]
        codec_len = codec_emb.shape[0]
        llm_length = text_len + codec_len + 2 + (pos + 1)
        text_interval, codec_interval = self.text_codec_mix_ratio
        exp_intervals = int(math.floor(text_len / text_interval))
        exp_codec_len = exp_intervals * codec_interval
        if exp_codec_len < codec_len:
            exp_text_len = exp_intervals * text_interval
            # Calculate the length of the mixed sequence
            mixed_length = exp_text_len + exp_codec_len

            # Create an empty tensor for the mixed sequence
            mixed_seq = torch.zeros(
                mixed_length, codec_emb.shape[1], dtype=codec_emb.dtype, device=codec_emb.device
            )

            # Fill the mixed sequence
            for i in range(text_interval):
                mixed_seq[i:: text_interval + codec_interval] = text_emb[
                                                                i:exp_text_len:text_interval
                                                                ]
            for i in range(codec_interval):
                try:
                    mixed_seq[i + text_interval:: text_interval + codec_interval] = codec_emb[
                                                                                    i:exp_codec_len:codec_interval
                                                                                    ]
                except:
                    import pdb

                    pdb.set_trace()

            ret_seq = torch.cat(
                [
                    sos_emb,
                    prompt_emb,
                    mixed_seq,
                    text_emb[exp_text_len:],
                    task_id_emb,
                    codec_emb[exp_codec_len:],
                ],
                dim=0,
            )

            fill_token = self.codebook_size + self.fill_token
            ret_targets = None
            if codec_token is not None:
                targets = torch.zeros(
                    mixed_length, dtype=codec_token.dtype, device=codec_token.device
                )
                for i in range(text_interval):
                    targets[i:: text_interval + codec_interval] = fill_token
                for i in range(codec_interval):
                    targets[i + text_interval:: text_interval + codec_interval] = codec_token[
                                                                                  i:exp_codec_len:codec_interval
                                                                                  ]
                ret_targets = torch.cat(
                    [
                        prompt_fill_token,
                        targets,
                        torch.ones((text_len - exp_text_len)).to(codec_token) * fill_token,
                        torch.tensor([self.ignore_id]).to(codec_token),
                        codec_token[exp_codec_len:],
                        torch.tensor([self.codebook_size + self.sos_eos]).to(codec_token),
                    ],
                    dim=0,
                )

            # ret_seq: audio_decoder, input, 包括sos, 不包括eos
            # ret_targets: audio_decoder, target, 不包括sos, 包括eos
            # llm_length: ret_seq.shape[0]
            return ret_seq, ret_targets, llm_length
        else:
            # offline mode
            ret_seq = torch.cat(
                [
                    sos_emb,
                    prompt_emb,
                    text_emb,
                    task_id_emb,
                    codec_emb,
                ],
                dim=0,
            )
            ret_targets = torch.cat(
                [
                    prompt_fill_token,
                    torch.ones(text_len).to(codec_token) * self.ignore_id,
                    torch.tensor([self.ignore_id]).to(codec_token),
                    codec_token,
                    torch.tensor([self.codebook_size + self.sos_eos]).to(codec_token),
                ],
                dim=0,
            )

            return ret_seq, ret_targets, llm_length

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

            splits = pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech, speech_lengths = [], []
            target_ids = []
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

            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)

            target_out = f"{target_out}<|im_end|>"
            splits = pattern.split(target_out)
            # codec_i = []
            sub_token = []
            codec_i_len = 0

            for k, sub_str in enumerate(splits):
                if len(sub_str) < 1:
                    continue
                if not sub_str.startswith("<|startofspeech|>"):
                    # sub_str = f"{sub_str}<|im_end|>"
                    sub_token = tokenizer.encode(sub_str)
                    target_ids += sub_token
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    # if sub_str.startswith("!"):
                    #     sub_token_codec = np.load(sub_str[1:])
                    #     codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                    #     codec_i_len = len(sub_token_codec)
                    #
                    # else:
                    #     sub_token_codec = []
                    #     for x in sub_str.split("|"):
                    #         if x.startswith("c"):
                    #             sub_token_codec.append(int(x[1:]))
                    #     codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                    #     codec_i_len = len(sub_token_codec)
                    # codec.append(codec_i)
                    # codec_len.append(codec_i_len)

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
                speech_token_len = fake_token_len[batch_idx, turn_id]
                if fbank_beg_idx > 0 and speech_token_len > 0:

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

    @torch.no_grad()
    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        # import pdb; pdb.set_trace()
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

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

            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=kwargs.get("max_length", 512),
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True,
                **llm_kwargs,
            )

            target_ids = generated_ids["sequences"]

            for idx in target_ids[0]:
                print(f"{tokenizer.decode([idx])}")
        response = tokenizer.batch_decode(
            generated_ids["sequences"], skip_special_tokens=kwargs.get("skip_special_tokens", True)
        )[0]

        loss = None

        uttid = key[0]
        hidden_states = generated_ids[
            "hidden_states"
        ]  # hidden_states: (t1, t2, ..., tn, ..., tN), tn=(l1, l2, ..., ln, ..., lN), ln: shape: 1x1x3584

        token_num = len(hidden_states)
        hidden_states_select = torch.zeros((1, token_num, 3584), dtype=torch.float32).to(
            inputs_embeds.device
        )
        hidden_states_out_len = torch.tensor(
            [
                token_num,
            ],
            dtype=torch.int32,
        ).to(inputs_embeds.device)
        for i in range(token_num):
            hidden_states_select[0, i, :] = hidden_states[i][-1][0, 0, :].to(torch.float32)

        target_ids = generated_ids["sequences"]
        target_emb = self.llm.model.get_input_embeddings()(target_ids)

        hidden_states_select = torch.concat(
            (hidden_states_select[:, :-1, :], target_emb[:, :-1, :]), dim=-1
        )
        hidden_states_select = self.codec_adaptor(hidden_states_select)

        codec_decoder_generator_conf = kwargs.get(
            "codec_decoder_generator_conf", {"eos_token": [6563, 6561]}
        )
        if "device" not in codec_decoder_generator_conf:
            codec_decoder_generator_conf["device"] = kwargs.get("device", "cuda:0")
        if "codec_dtype" not in codec_decoder_generator_conf:
            codec_decoder_generator_conf["codec_dtype"] = kwargs.get("llm_dtype", "fp32")
        dtype = dtype_map[codec_decoder_generator_conf["codec_dtype"]]
        device = codec_decoder_generator_conf["device"]
        if not hasattr(self, "codec_decoder_generator"):
            from .utils.llm_decoding import LLMDecoder as LLMDecoder

            self.codec_decoder_generator = LLMDecoder(
                token_embeder=self.codec_embed, **codec_decoder_generator_conf
            ).to(dtype)

            self.codec_decoder.lm_head = self.codec_head.to(dtype)
            self.codec_decoder.to(device)
            self.codec_decoder.to(dtype)
            self.codec_decoder_generator.to(device)

        sos_emb = self.codec_sos_eos_embed(
            torch.tensor([self.sos_eos], dtype=torch.int64).to(device=inputs_embeds.device)
        )
        task_id_emb = self.codec_sos_eos_embed(
            torch.tensor([self.task_id], dtype=torch.int64).to(device=inputs_embeds.device)
        )
        slice_num = (hidden_states_select.shape[1] - 1) // self.text_codec_mix_ratio[0] + 1
        if hidden_states_select.shape[1] % self.text_codec_mix_ratio[0] == 0:
            slice_num = slice_num + 1

        states = kwargs.get("states", {})
        codec_pred = []
        for slice_idx in range(0, slice_num):

            hidden_states_select_slice = hidden_states_select[
                                         :,
                                         slice_idx
                                         * self.text_codec_mix_ratio[0]: (slice_idx + 1)
                                                                         * self.text_codec_mix_ratio[0],
                                         :,
                                         ]
            if slice_idx == 0:
                hidden_states_select_slice = torch.concat(
                    (sos_emb[None, :, :], hidden_states_select_slice), dim=1
                )

            if (slice_idx + 1) * self.text_codec_mix_ratio[0] > hidden_states_select.shape[1]:
                hidden_states_select_slice = torch.concat(
                    (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
                )
            codec_generator = self.codec_decoder_generator(
                hidden_states_select_slice.to(dtype).to(device),
                self.codec_decoder,
                states=states,
                max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
                min_length=kwargs.get("min_length_codec_duration_s", None),
                **codec_decoder_generator_conf,
            )
            response_chunk = tokenizer.decode(
                target_ids[
                0,
                slice_idx
                * self.text_codec_mix_ratio[0]: (slice_idx + 1)
                                                * self.text_codec_mix_ratio[0],
                ]
            )
            for gen_codec in codec_generator:
                codec_pred.extend(gen_codec)
                print(f"{uttid}: response: {response_chunk}, {len(gen_codec)} tokens: {gen_codec}")

        streaming_wav_dir = f'{kwargs.get("output_dir")}/streaming_wav'
        if not os.path.exists(streaming_wav_dir):
            os.makedirs(streaming_wav_dir)
        # synthesize waveform
        set_all_random_seed(kwargs.get("seed", 0))
        device = hidden_states_select.device
        pred_inter = torch.tensor(codec_pred, dtype=torch.int64, device=device)[None, :]
        pred_inter_lens = torch.tensor([pred_inter.shape[1]], device=device, dtype=torch.int64)
        xvec_lens = torch.tensor([1], dtype=torch.int64, device=device)
        if self.streaming_flow:
            fm_voc_states = self.init_fm_voc_states()
            n_chunk = int(np.ceil(pred_inter.shape[1] / self.chunk_size))
            wav_list = []
            for i in range(n_chunk):
                chunk = pred_inter[
                        :, i * self.chunk_size: (i + 1) * self.chunk_size
                        ]  # .squeeze(-1)
                pred_chunk_lens = torch.tensor([chunk.shape[1]], device=device, dtype=torch.int64)
                (wav, _), fm_voc_states = self.streaming_flow_voc(
                    chunk,
                    pred_chunk_lens,
                    self.xvec,
                    xvec_lens,
                    (None, None),
                    (None, None),
                    fm_voc_states,
                    is_last=(i == n_chunk - 1),
                )
                wav_list.append(wav)
            feat = fm_voc_states["cached_feat"]
            wav = torch.cat(wav_list, dim=2)
            wav_path = f"{streaming_wav_dir}/{uttid}.wav"
            torchaudio.save(
                wav_path,
                wav.cpu().squeeze(0),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        else:
            feat = self.fm_model.inference(
                pred_inter,
                pred_inter_lens,
                self.xvec,
                xvec_lens,
                diff_steps=self.fm_diff_steps,
                temperature=self.fm_temperature,
                prompt=dict(prompt_text=(None, None), prompt_audio=(None, None)),
            )
            # np.save(f'{kwargs.get("output_dir")}/{uttid}_feat.npy', feat[0].cpu().numpy())
            # feat in B, C, T
            # offline_wav_dir = f'{kwargs.get("output_dir")}/offline_wav'
            # if not os.path.exists(offline_wav_dir):
            #     os.makedirs(offline_wav_dir)
            # wav = self.vocoder.inference(feat.transpose(1, 2))[0].unsqueeze(0)
            # wav_path_offline = f"{offline_wav_dir}/{uttid}.wav"
            # torchaudio.save(
            #     wav_path_offline,
            #     wav.cpu(),
            #     sample_rate=self.vocoder.sample_rate,
            #     encoding="PCM_S",
            #     bits_per_sample=16,
            # )

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"text"]

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

            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"语音转写：<|startofspeech|>!{wav_path}<|endofspeech|>",
                    },
                    {
                        "role": "assistant",
                        "content": f"{response}".replace("\n", " "),
                    },
                ],
                "key": uttid,
                "task": "TTS",
            }
            ibest_writer["wav_streaming.jsonl"][key[0]] = (
                    json.dumps(data, ensure_ascii=False) + "\n"
            )
            # data = {
            #     "messages": [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {
            #             "role": "user",
            #             "content": f"语音转写：<|startofspeech|>!{wav_path_offline}<|endofspeech|>",
            #         },
            #         {
            #             "role": "assistant",
            #             "content": f"{response}".replace("\n", " "),
            #         },
            #     ],
            #     "key": uttid,
            #     "task": "TTS",
            # }
            # ibest_writer["wav_offline.jsonl"][key[0]] = json.dumps(data, ensure_ascii=False)

        return results, meta_data

    @torch.no_grad()
    def gen_codec(
            self,
            hidden_states_select,
            target_ids,
            states,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                hidden_states_select = torch.concat(
                    (self.sos_emb[None, :, :], hidden_states_select), dim=1
                )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]

        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("llm_dtype", "bf16")]
        if not hasattr(self, "codec_decoder_generator"):
            from .utils.llm_decoding import LLMDecoder

            self.codec_decoder_generator = LLMDecoder(token_embeder=self.codec_embed, **kwargs).to(
                dtype
            )
            self.codec_decoder.lm_head = self.codec_head.to(dtype)
            self.codec_decoder.to(device)
            self.codec_decoder.to(dtype)
            self.codec_decoder_generator.to(device)

        codec_generator = self.codec_decoder_generator(
            hidden_states_select.to(dtype).to(device),
            self.codec_decoder,
            states=states,
            max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
            min_length=kwargs.get("min_length_codec_duration_s", None),
            **kwargs,
        )
        for gen_codec in codec_generator:
            yield gen_codec

    @torch.no_grad()
    def export_codec_vllm(self, model_path, **kwargs):
        pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
        vocab_size = self.codec_embed.num_embeddings
        feature_size = self.codec_embed.embedding_dim
        pad_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to

        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("llm_dtype", "bf16")]
        self.codec_decoder.lm_head = self.codec_head.to(dtype)
        new_lm_head = nn.Linear(in_features=feature_size, out_features=pad_vocab_size, bias=True)
        with torch.no_grad():
            new_lm_head.weight[:vocab_size] = self.codec_decoder.lm_head.weight
            new_lm_head.bias[:vocab_size] = self.codec_decoder.lm_head.bias
            new_lm_head.weight[vocab_size:] = 0
            new_lm_head.bias[vocab_size:] = 0
        self.codec_decoder.lm_head = new_lm_head
        new_codec_embed = nn.Linear(in_features=feature_size, out_features=pad_vocab_size)
        with torch.no_grad():
            new_codec_embed.weight[:vocab_size] = self.codec_embed.weight
            new_codec_embed.weight[vocab_size:] = 0
        self.codec_decoder.set_input_embeddings(new_codec_embed)
        self.codec_decoder.to(device)
        self.codec_decoder.to(dtype)
        tmp_vocab_size = self.codec_decoder.config.vocab_size
        tmp_tie_embedding = self.codec_decoder.config.tie_word_embeddings
        self.codec_decoder.config.vocab_size = pad_vocab_size
        self.codec_decoder.config.tie_word_embeddings = False
        self.codec_decoder.save_pretrained(model_path)
        self.codec_decoder.config.vocab_size = tmp_vocab_size
        self.codec_decoder.config.tie_word_embeddings = tmp_tie_embedding

    @torch.no_grad()
    def gen_codec_vllm_sync(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            states,
            hidden_states_vllm_his,
            request_id,
            chunk=15,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("llm_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                hidden_states_select = torch.concat(
                    (self.sos_emb[None, :, :], hidden_states_select), dim=1
                )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            top_k=sampling,
            stop_token_ids=eos_token,
            max_tokens=kwargs.get("max_length_codec_duration_s", 120) * 25,
        )
        vllm_codec_engine.add_request(
            request_id,
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
        )
        ## generator
        out_token_ids = []
        chunk_num = 0
        while True:
            request_outputs: List[RequestOutput] = vllm_codec_engine.step()
            for request_output in request_outputs:
                if str(request_output.request_id) != str(request_id):
                    continue
                if not request_output.finished:
                    # print(f"Partial request output: {request_output}")
                    out_token_ids.append(list(request_output.outputs[0].token_ids)[-1])
                    if is_last_slice and len(out_token_ids) % chunk == 0:
                        yield out_token_ids[
                              chunk * chunk_num: chunk * (chunk_num + 1)
                              ], hidden_states_select
                        chunk_num += 1
                else:
                    # print(f"Finished request output: {request_output}")
                    if out_token_ids[chunk * chunk_num:]:
                        yield out_token_ids[chunk * chunk_num:], hidden_states_select
                        chunk_num += 1
                    break

            if not vllm_codec_engine.has_unfinished_requests():
                break

    @torch.no_grad()
    def gen_codec_vllm(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            states,
            hidden_states_vllm_his,
            request_id,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("llm_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                hidden_states_select = torch.concat(
                    (self.sos_emb[None, :, :], hidden_states_select), dim=1
                )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            top_k=sampling,
            stop_token_ids=eos_token,
            max_tokens=kwargs.get("max_length_codec_duration_s", 120) * 25,
        )
        codec_generator = vllm_codec_engine.generate(
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
            request_id,
        )
        return codec_generator, hidden_states_select

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
        old_overlap = cached[:, -hop_len:]
        sin_wind = torch.blackman_window(2 * hop_len)[None, :].to(old_overlap)
        cf_overlap = (
                old_overlap * sin_wind[:, -hop_len:] + current[:, :hop_len] * sin_wind[:, :hop_len]
        )
        current[:, :hop_len] = cf_overlap

        return current

    def streaming_flow_voc(
            self,
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            prompt_inter,
            prompt_audio,
            fm_voc_states,
            is_last,
    ):
        cached_token = fm_voc_states.get("cached_token", None)
        cached_feat = fm_voc_states.get("cached_feat", None)
        cached_wav = fm_voc_states.get("cached_wav", None)
        if cached_token is not None:
            tokens = torch.cat([cached_token, tokens], dim=1)
            token_lens = token_lens + cached_token.shape[1]

        diff_out = self.fm_model.inference(
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            diff_steps=self.fm_diff_steps,
            temperature=self.fm_temperature,
            prompt=dict(prompt_text=prompt_inter, prompt_audio=prompt_audio),
        )
        # feat in B,C,T. remove lookahead feat and history, which is inaccurate
        feat = diff_out
        padded_feat_len = 0
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            # involving vocoder_hop for cross fade
            feat = diff_out[:, :, cached_feat_len - self.vocoder_hop_in_feat:]
            # pad history feat for vocoder synthesis consistency
            padded_feat_len = min(self.mel_frame_rate, cached_feat_len)
            feat = torch.cat([cached_feat[:, :, -padded_feat_len:], feat], dim=2)
        if not is_last:
            feat = feat[:, :, : -self.feat_reback]

        wav = self.vocoder.inference(feat.transpose(1, 2)).unsqueeze(1)
        # remove padded history wav
        pad_wav_len = int(padded_feat_len * (self.sample_rate / self.mel_frame_rate))
        wav = wav[:, :, pad_wav_len:]
        if cached_wav is not None:
            wav = self.cross_fade(
                cached_wav.squeeze(1), wav.squeeze(1), self.vocoder_hop
            ).unsqueeze(1)

        # play wav without tail hop length
        cur_wav = wav[:, :, : -self.vocoder_hop] if not is_last else wav

        # update caches
        cached_token = tokens
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            if is_last:
                cached_feat = torch.cat([cached_feat, diff_out[:, :, cached_feat_len:]], dim=2)
            else:
                cached_feat = torch.cat(
                    [cached_feat, diff_out[:, :, cached_feat_len: -self.feat_reback]], dim=2
                )
        else:
            if is_last:
                cached_feat = diff_out[:, :, :]
            else:
                cached_feat = diff_out[:, :, : -self.feat_reback]
        if cached_wav is not None:
            cached_wav = torch.cat([cached_wav[:, :, : -self.vocoder_hop], wav], dim=2)
        else:
            cached_wav = wav

        return (cur_wav, diff_out[:, :, : -self.feat_reback]), dict(
            cached_token=cached_token,
            cached_feat=cached_feat,
            cached_wav=cached_wav,
        )


@tables.register("model_classes", "MinMo_S2S_beta")
class MinMo_S2S_beta(nn.Module):
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            attn_implementation="flash_attention_2",
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            output_hidden_states=True,
            **llm_load_kwargs,
        )
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

        from .utils.stream_utils import _sample

        model._sample = types.MethodType(_sample, model)

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

        from .utils.stream_utils import TextStatesIteratorStreamer

        self.IteratorStreamer = TextStatesIteratorStreamer
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim
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

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss

        codec_decoder = kwargs.get("codec_decoder")
        codec_decoder_conf = kwargs.get("codec_decoder_conf", {})
        init_param_path = codec_decoder_conf.get("init_param_path", None)
        codec_decoder = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            attn_implementation="flash_attention_2",
            # output_hidden_states=True,
        )

        codec_decoder_dim = codec_decoder.get_input_embeddings().weight.shape[-1]

        del codec_decoder.lm_head
        del codec_decoder.model.embed_tokens

        self.codec_decoder = codec_decoder
        self.codebook_size = codec_decoder_conf.get("codebook_size", 4096)
        self.codec_embed = nn.Embedding(self.codebook_size + 3, codec_decoder_dim, 0)
        self.codec_head = nn.Linear(codec_decoder_dim, self.codebook_size + 3, bias=True)
        self.codec_adaptor = nn.Linear(llm_dim * 2, codec_decoder_dim, bias=False)
        self.codec_sos_eos_embed = torch.nn.Embedding(2, codec_decoder_dim)

        self.sos_eos = codec_decoder_conf.get("sos_eos", 0)
        self.task_id = codec_decoder_conf.get("task_id", 1)
        self.fill_token = codec_decoder_conf.get("fill_token", 2)
        self.text_codec_mix_ratio = codec_decoder_conf.get("text_codec_mix_ratio", [4, 12])
        # self.text_codec_mix_prob = self.text_codec_mix_ratio[0] / self.text_codec_mix_ratio[1]
        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.codebook_size + 3,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
        self.concat_input_embedding = codec_decoder_conf.get("concat_input_embedding", False)

        if kwargs.get("flow_matching", True) and len(kwargs.get("flow_matching_conf", {})) > 0:

            # build flow mathing model
            self.streaming_flow = kwargs.get("streaming_flow", False)
            flow_conf = kwargs.get("flow_matching_conf", {})
            from .diffusion_text2audio_model import MaskedDiffWithXvec

            self.fm_model = MaskedDiffWithXvec(**flow_conf)
            # build vocoder
            vocoder_conf = kwargs.get("vocoder_conf", {})
            from .hifigan import HifiGan

            self.vocoder = HifiGan(**vocoder_conf)
            xvec_path = kwargs.get("xvec_path")
            if ".ark:" in xvec_path:
                import kaldiio

                xvec = kaldiio.load_mat(xvec_path)
            else:
                xvec = np.load(xvec_path)
            self.register_buffer("xvec", torch.tensor(xvec).unsqueeze(0))
            self.fm_diff_steps = kwargs.get("fm_diff_steps", 10)
            self.fm_temperature = kwargs.get("fm_temperature", 1.0)
            self.chunk_size = self.text_codec_mix_ratio[1]
            self.reback = kwargs.get("reback", 4)  # token reback
            if hasattr(self.fm_model.encoder, "pre_lookahead_len"):
                self.feat_reback_in_token = self.fm_model.encoder.pre_lookahead_len
                self.feat_reback = self.fm_model.calc_target_len(
                    self.feat_reback_in_token
                )  # feat reback
            else:
                self.feat_reback_in_token = 4
                self.feat_reback = kwargs.get("feat_reback", 8)
            self.sample_rate = flow_conf["mel_feat_conf"]["sampling_rate"]
            self.token_rate = flow_conf["input_frame_rate"]
            self.mel_hop_length = self.vocoder.mel_spec_transform.hop_length
            self.mel_frame_rate = int(self.sample_rate / self.mel_hop_length)
            self.vocoder_hop = kwargs.get("vocoder_hop", 1920)
            self.vocoder_hop_in_token = int(self.vocoder_hop / self.sample_rate * self.token_rate)
            self.vocoder_hop_in_feat = int(self.vocoder_hop / self.mel_hop_length)

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

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

            batch_size, token_num, dims = inputs_embeds.shape
            fake_token_len = kwargs.get("fake_token_len")
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            speech_idx = 0
            for batch_idx in range(batch_size):

                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    if fbank_beg_idx > 0 and speech_token_len > 0:

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
            loss_llm = model_outputs.loss
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_llm = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc_llm"] = acc_llm

        stats["loss_llm"] = torch.clone(loss_llm.detach())

        hidden_states = model_outputs.hidden_states[-1].float()
        hidden_states_select = []
        hidden_states_select_len = []
        multiturn_beg = kwargs.get("multiturn_beg")
        multiturn_beg[multiturn_beg < 0] = -1
        multiturn_len = kwargs.get("multiturn_len")
        multiturn_len[multiturn_len < 0] = 0
        if self.concat_input_embedding:
            multiturn_beg_input = kwargs.get("multiturn_beg_input")
            multiturn_beg_input[multiturn_beg_input < 0] = -1
            multiturn_len_input = kwargs.get("multiturn_len_input")
            multiturn_len_input[multiturn_len_input < 0] = 0
            hidden_states_select_input = []
            hidden_states_select_len_input = []

        for batch_idx in range(multiturn_beg.shape[0]):
            for turn_id in range(multiturn_beg.shape[1]):
                multiturn_beg_idx = multiturn_beg[batch_idx, turn_id].item()
                multiturn_len_idx = multiturn_len[batch_idx, turn_id].item() - 1  # rm eos
                multiturn_end_idx = multiturn_beg_idx + multiturn_len_idx
                if multiturn_beg_idx >= 0:
                    hidden_states_idx = hidden_states[
                                        batch_idx, multiturn_beg_idx - 1: multiturn_end_idx - 1, :
                                        ]
                    inputs_embeds_idx = inputs_embeds[
                                        batch_idx, multiturn_beg_idx:multiturn_end_idx, :
                                        ]
                    hidden_states_select.append(
                        torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                    )
                    hidden_states_select_len.append(multiturn_len_idx)

                if self.concat_input_embedding:
                    multiturn_beg_idx_input = multiturn_beg_input[batch_idx, turn_id].item()
                    multiturn_len_idx_input = multiturn_len_input[batch_idx, turn_id].item()
                    multiturn_end_idx_input = multiturn_beg_idx_input + multiturn_len_idx_input
                    if multiturn_beg_idx_input >= 0:
                        hidden_states_idx = hidden_states[
                                            batch_idx, multiturn_beg_idx_input - 1: multiturn_end_idx_input - 1, :
                                            ]
                        inputs_embeds_idx = inputs_embeds[
                                            batch_idx, multiturn_beg_idx_input:multiturn_end_idx_input, :
                                            ]
                        hidden_states_select_input.append(
                            torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                        )
                        hidden_states_select_len_input.append(multiturn_len_idx_input)

        # hidden_states_his_select
        device = hidden_states.device
        hidden_states_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_select, batch_first=True, padding_value=0.0
        ).to(device=device)
        hidden_states_select_len = torch.tensor(hidden_states_select_len, dtype=torch.int64).to(
            device=device
        )
        if self.concat_input_embedding:
            hidden_states_select_input = torch.nn.utils.rnn.pad_sequence(
                hidden_states_select_input, batch_first=True, padding_value=0.0
            ).to(device=device)
            # hidden_states_select_len_input = torch.tensor(hidden_states_select_len_input, dtype=torch.int64).to(
            #     device=device
            # )
            hidden_states_select_input = self.codec_adaptor(hidden_states_select_input)

        hidden_states_select = self.codec_adaptor(hidden_states_select)

        # codec
        codec = kwargs.get("codec")
        codec[codec < 0] = 0
        codec_len = kwargs.get("codec_len")
        # codec_len[codec_len < 0] = 0

        codec_emb = self.codec_embed(codec)
        sos_emb = self.codec_sos_eos_embed(
            torch.tensor([self.sos_eos], dtype=torch.int64).to(device=device)
        )
        task_id_emb = self.codec_sos_eos_embed(
            torch.tensor([self.task_id], dtype=torch.int64).to(device=device)
        )

        codec_decoder_inputs_embeds = []
        codec_decoder_inputs_len = []
        codec_decoder_targets = []
        codec_decoder_attention_mask = []

        for batch_idx in range(hidden_states_select.shape[0]):
            hidden_states_select_idx = hidden_states_select[
                                       batch_idx, : hidden_states_select_len[batch_idx], :
                                       ]
            codec_emb_idx = codec_emb[batch_idx, : codec_len[batch_idx], :]
            code_idx = codec[batch_idx, : codec_len[batch_idx]]
            (
                codec_decoder_inputs_embeds_idx,
                codec_decoder_targets_idx,
                codec_decoder_inputs_len_idx,
            ) = self.mixup_text_codec(
                hidden_states_select_idx,
                codec_emb_idx,
                sos_emb,
                task_id_emb,
                code_idx,
            )

            if self.concat_input_embedding:
                codec_decoder_inputs_embeds_idx = torch.concat(
                    (
                        hidden_states_select_input[
                        batch_idx, : hidden_states_select_len_input[batch_idx], :
                        ],
                        codec_decoder_inputs_embeds_idx[:, :],
                    ),
                    dim=0,
                )
                codec_decoder_inputs_len_idx = (
                        codec_decoder_inputs_len_idx + hidden_states_select_len_input[batch_idx]
                )
                codec_decoder_targets_idx = torch.concat(
                    (
                        torch.ones(hidden_states_select_len_input[batch_idx], dtype=torch.int64).to(
                            codec_decoder_targets_idx.device
                        )
                        * self.ignore_id,
                        codec_decoder_targets_idx[:],
                    ),
                    dim=0,
                )

            codec_decoder_inputs_embeds.append(codec_decoder_inputs_embeds_idx)
            codec_decoder_inputs_len.append(codec_decoder_inputs_len_idx)
            codec_decoder_targets.append(codec_decoder_targets_idx)

            codec_decoder_attention_mask_idx = torch.tensor(
                [1] * codec_decoder_inputs_len_idx, dtype=torch.int64
            )
            codec_decoder_attention_mask.append(codec_decoder_attention_mask_idx)

        codec_decoder_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_inputs_embeds, batch_first=True, padding_value=0.0
        ).to(device=device)
        codec_decoder_inputs_len = torch.tensor(codec_decoder_inputs_len, dtype=torch.int64).to(
            device=device
        )
        codec_decoder_targets = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_targets, batch_first=True, padding_value=-100
        ).to(device=device)
        codec_decoder_attention_mask = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_attention_mask, batch_first=True, padding_value=0.0
        ).to(device=device)

        codec_hidden_states = self.codec_decoder.model(
            inputs_embeds=codec_decoder_inputs_embeds,
            attention_mask=codec_decoder_attention_mask,
        )[0]

        codec_logits = self.codec_head(codec_hidden_states)
        loss = self.criterion_ce(codec_logits, codec_decoder_targets)

        with torch.no_grad():
            preds = torch.argmax(codec_logits, -1)
            acc_att = compute_accuracy(preds, codec_decoder_targets, ignore_label=-100)
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

    def mixup_text_codec(self, text_emb, codec_emb, sos_emb, task_id_emb, codec_token):
        # self.sos_eos:0
        # task_id_emb: 1
        # self.ignore_id = -100
        # self.text_codec_mix_ratio: [4, 12]
        # self.codebook_size: codec本身size，不包含sos
        # self.fill_token: 2
        pos = -1
        prompt_emb = text_emb[: pos + 1]
        prompt_fill_token = torch.ones(pos + 1).fill_(self.ignore_id).to(codec_token)
        text_emb = text_emb[pos + 1:]

        text_len = text_emb.shape[0]
        codec_len = codec_emb.shape[0]
        llm_length = text_len + codec_len + 2 + (pos + 1)
        text_interval, codec_interval = self.text_codec_mix_ratio
        exp_intervals = int(math.floor(text_len / text_interval))
        exp_codec_len = exp_intervals * codec_interval
        if exp_codec_len < codec_len:
            exp_text_len = exp_intervals * text_interval
            # Calculate the length of the mixed sequence
            mixed_length = exp_text_len + exp_codec_len

            # Create an empty tensor for the mixed sequence
            mixed_seq = torch.zeros(
                mixed_length, codec_emb.shape[1], dtype=codec_emb.dtype, device=codec_emb.device
            )

            # Fill the mixed sequence
            for i in range(text_interval):
                mixed_seq[i:: text_interval + codec_interval] = text_emb[
                                                                i:exp_text_len:text_interval
                                                                ]
            for i in range(codec_interval):
                try:
                    mixed_seq[i + text_interval:: text_interval + codec_interval] = codec_emb[
                                                                                    i:exp_codec_len:codec_interval
                                                                                    ]
                except:
                    import pdb

                    pdb.set_trace()

            ret_seq = torch.cat(
                [
                    sos_emb,
                    prompt_emb,
                    mixed_seq,
                    text_emb[exp_text_len:],
                    task_id_emb,
                    codec_emb[exp_codec_len:],
                ],
                dim=0,
            )

            fill_token = self.codebook_size + self.fill_token
            ret_targets = None
            if codec_token is not None:
                targets = torch.zeros(
                    mixed_length, dtype=codec_token.dtype, device=codec_token.device
                )
                for i in range(text_interval):
                    targets[i:: text_interval + codec_interval] = fill_token
                for i in range(codec_interval):
                    targets[i + text_interval:: text_interval + codec_interval] = codec_token[
                                                                                  i:exp_codec_len:codec_interval
                                                                                  ]
                ret_targets = torch.cat(
                    [
                        prompt_fill_token,
                        targets,
                        torch.ones((text_len - exp_text_len)).to(codec_token) * fill_token,
                        torch.tensor([self.ignore_id]).to(codec_token),
                        codec_token[exp_codec_len:],
                        torch.tensor([self.codebook_size + self.sos_eos]).to(codec_token),
                    ],
                    dim=0,
                )

            # ret_seq: audio_decoder, input, 包括sos, 不包括eos
            # ret_targets: audio_decoder, target, 不包括sos, 包括eos
            # llm_length: ret_seq.shape[0]
            return ret_seq, ret_targets, llm_length
        else:
            # offline mode
            ret_seq = torch.cat(
                [
                    sos_emb,
                    prompt_emb,
                    text_emb,
                    task_id_emb,
                    codec_emb,
                ],
                dim=0,
            )
            ret_targets = torch.cat(
                [
                    prompt_fill_token,
                    torch.ones(text_len).to(codec_token) * self.ignore_id,
                    torch.tensor([self.ignore_id]).to(codec_token),
                    codec_token,
                    torch.tensor([self.codebook_size + self.sos_eos]).to(codec_token),
                ],
                dim=0,
            )

            return ret_seq, ret_targets, llm_length

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
        input_source_ids = []
        multiturn_beg_input = []
        multiturn_len_input = []
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

            splits = pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech, speech_lengths = [], []
            target_ids = []
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

            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)

            target_out = f"{target_out}<|im_end|>"
            splits = pattern.split(target_out)
            # codec_i = []
            sub_token = []
            codec_i_len = 0

            for k, sub_str in enumerate(splits):
                if len(sub_str) < 1:
                    continue
                if not sub_str.startswith("<|startofspeech|>"):
                    # sub_str = f"{sub_str}<|im_end|>"
                    sub_token = tokenizer.encode(sub_str)
                    target_ids += sub_token
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    # if sub_str.startswith("!"):
                    #     sub_token_codec = np.load(sub_str[1:])
                    #     codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                    #     codec_i_len = len(sub_token_codec)
                    #
                    # else:
                    #     sub_token_codec = []
                    #     for x in sub_str.split("|"):
                    #         if x.startswith("c"):
                    #             sub_token_codec.append(int(x[1:]))
                    #     codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                    #     codec_i_len = len(sub_token_codec)
                    # codec.append(codec_i)
                    # codec_len.append(codec_i_len)

            multiturn_beg_input += [len(input_ids)]
            multiturn_len_input += [len(source_ids)]
            if i == 0:
                system_token = tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
                multiturn_beg_input[-1] = len(system_token)
                multiturn_len_input[-1] = multiturn_len_input[-1] - len(system_token)
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
        output["multiturn_beg_input"] = torch.tensor(multiturn_beg_input, dtype=torch.int32)
        output["multiturn_len_input"] = torch.tensor(multiturn_len_input, dtype=torch.int32)
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
                speech_token_len = fake_token_len[batch_idx, turn_id]
                if fbank_beg_idx > 0 and speech_token_len > 0:

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

    @torch.no_grad()
    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        uttid = key[0]
        # import pdb; pdb.set_trace()
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

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

            if kwargs.get("tearchforing", False):

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

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] - 1: -2]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                # loss = model_outputs.loss.item()
                hidden_states = model_outputs.hidden_states[-1].float()
                hidden_states_input = hidden_states
                hidden_states_select = hidden_states[:, source_ids.shape[1] - 1: -2, :]
                target_ids = preds
                target_emb = self.llm.model.get_input_embeddings()(target_ids)
                hidden_states_select = torch.concat(
                    (hidden_states_select[:, :, :], target_emb[:, :, :]), dim=-1
                )

            else:
                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **llm_kwargs,
                )
                target_ids = generated_ids["sequences"]
                for idx in target_ids[0]:
                    print(f"{tokenizer.decode([idx])}")

                response = tokenizer.batch_decode(
                    generated_ids["sequences"],
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]

                loss = None

                hidden_states = generated_ids[
                    "hidden_states"
                ]  # hidden_states: (t1, t2, ..., tn, ..., tN), tn=(l1, l2, ..., ln, ..., lN), ln: shape: 1x1x3584

                hidden_states_input = hidden_states[0][-1]
                token_num = len(hidden_states)
                hidden_states_select = torch.zeros((1, token_num, 3584), dtype=torch.float32).to(
                    inputs_embeds.device
                )
                hidden_states_out_len = torch.tensor(
                    [
                        token_num,
                    ],
                    dtype=torch.int32,
                ).to(inputs_embeds.device)

                # import pdb; pdb.set_trace()
                ## filter special symbols
                special_symbols = ["*", "**", "-", "《", "》", '"', "\n"]
                target_ids_new = []
                hs_count = 0
                text_pre = ""
                assert target_ids.shape[1] == token_num
                for i in range(token_num):
                    text_new = tokenizer.decode([target_ids[0][i]])
                    if text_new.strip() not in special_symbols:
                        if text_new == " " and text_pre == " ":
                            continue
                        hidden_states_select[0, hs_count, :] = hidden_states[i][-1][0, 0, :].to(
                            torch.float32
                        )
                        hs_count += 1
                        target_ids_new.append(target_ids[0][i])
                        text_pre = text_new
                hidden_states_select = hidden_states_select[:, :hs_count, :]
                target_ids_new = torch.tensor(([target_ids_new]), device=target_ids.device)
                target_emb = self.llm.model.get_input_embeddings()(target_ids_new)
                # print(response)
                # response = tokenizer.batch_decode(
                #     target_ids_new,
                #     skip_special_tokens=kwargs.get("skip_special_tokens", True),
                # )[0]
                # print(response)

                # ## old
                # for i in range(token_num):
                #     hidden_states_select[0, i, :] = hidden_states[i][-1][0, 0, :].to(torch.float32)
                # target_ids = generated_ids["sequences"]
                # target_emb = self.llm.model.get_input_embeddings()(target_ids)

                hidden_states_select = torch.concat(
                    (hidden_states_select[:, :-1, :], target_emb[:, :-1, :]), dim=-1
                )
        hidden_states_select = self.codec_adaptor(hidden_states_select)

        multiturn_beg_input = batch["multiturn_beg_input"]
        multiturn_len_input = batch["multiturn_len_input"]
        batch_idx = 0
        turn_id = -1
        hidden_states_select_input = []
        hidden_states_select_len_input = []
        if self.concat_input_embedding:
            multiturn_beg_idx_input = multiturn_beg_input[turn_id].item()
            multiturn_len_idx_input = multiturn_len_input[turn_id].item()
            multiturn_end_idx_input = multiturn_beg_idx_input + multiturn_len_idx_input
            if multiturn_beg_idx_input >= 0:
                hidden_states_idx = hidden_states_input[
                                    batch_idx, multiturn_beg_idx_input - 1: multiturn_end_idx_input - 1, :
                                    ]
                inputs_embeds_idx = inputs_embeds[
                                    batch_idx, multiturn_beg_idx_input:multiturn_end_idx_input, :
                                    ]
                hidden_states_select_input.append(
                    torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                )
                hidden_states_select_len_input.append(multiturn_len_idx_input)

                hidden_states_select_input = torch.nn.utils.rnn.pad_sequence(
                    hidden_states_select_input, batch_first=True, padding_value=0.0
                ).to(device=inputs_embeds.device)
                # hidden_states_select_len_input = torch.tensor(hidden_states_select_len_input, dtype=torch.int64).to(
                #     device=device
                # )
                hidden_states_select_input = self.codec_adaptor(
                    hidden_states_select_input.to(hidden_states_select.dtype)
                )

        codec_decoder_generator_conf = kwargs.get(
            "codec_decoder_generator_conf", {"eos_token": [6563, 6561]}
        )
        if "device" not in codec_decoder_generator_conf:
            codec_decoder_generator_conf["device"] = kwargs.get("device", "cuda:0")
        if "codec_dtype" not in codec_decoder_generator_conf:
            codec_decoder_generator_conf["codec_dtype"] = kwargs.get("llm_dtype", "fp32")
        dtype = dtype_map[codec_decoder_generator_conf["codec_dtype"]]
        device = codec_decoder_generator_conf["device"]
        if not hasattr(self, "codec_decoder_generator"):
            from .utils.llm_decoding import LLMDecoder as LLMDecoder

            self.codec_decoder_generator = LLMDecoder(
                token_embeder=self.codec_embed, **codec_decoder_generator_conf
            ).to(dtype)

            self.codec_decoder.lm_head = self.codec_head.to(dtype)
            self.codec_decoder.to(device)
            self.codec_decoder.to(dtype)
            self.codec_decoder_generator.to(device)

        sos_emb = self.codec_sos_eos_embed(
            torch.tensor([self.sos_eos], dtype=torch.int64).to(device=inputs_embeds.device)
        )
        task_id_emb = self.codec_sos_eos_embed(
            torch.tensor([self.task_id], dtype=torch.int64).to(device=inputs_embeds.device)
        )
        states = kwargs.get("states", {})
        codec_preds = []
        codec_pred = []

        ## new process
        hidden_states_select_list = []
        is_first_slice = True
        cut_puncs = ["。", "？", "！", "；", "：", ".", "?", "!", ";", ".", "?", "!", ";", ":"]
        cut_thres = 100
        cut_base = 0
        for idx in range(hidden_states_select.shape[1]):
            hidden_states_select_list.append(hidden_states_select[:, idx: idx + 1, :])
            text_acc = tokenizer.decode(target_ids[0, :idx], skip_special_tokens=True)
            text_new = text_acc[-1] if len(text_acc) != 0 else ""
            if idx >= cut_base + cut_thres and text_new in cut_puncs:
                print(f"Clear history: {idx}")
                cut_base = idx
                hidden_states_select_slice = torch.cat(hidden_states_select_list, dim=1)
                hidden_states_select_list = []
                ## last_slice
                hidden_states_select_slice = torch.concat(
                    (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
                )
                codec_generator = self.codec_decoder_generator(
                    hidden_states_select_slice.to(dtype).to(device),
                    self.codec_decoder,
                    states=states,
                    max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
                    min_length=kwargs.get("min_length_codec_duration_s", None),
                    **codec_decoder_generator_conf,
                )
                for gen_codec in codec_generator:
                    codec_pred.extend(gen_codec)
                    print(f"{uttid}: {len(gen_codec)} tokens: {gen_codec}")
                ## TO-DO clear
                codec_preds.append(codec_pred)
                codec_pred = []
                states["llm_cache"] = None
                is_first_slice = True

            elif len(hidden_states_select_list) % self.text_codec_mix_ratio[0] == 0:
                hidden_states_select_slice = torch.cat(hidden_states_select_list, dim=1)
                hidden_states_select_list = []
                if is_first_slice:
                    hidden_states_select_slice = torch.concat(
                        (sos_emb[None, :, :], hidden_states_select_slice), dim=1
                    )
                    if self.concat_input_embedding:
                        hidden_states_select_slice = torch.concat(
                            (
                                hidden_states_select_input[
                                batch_idx: batch_idx + 1,
                                : hidden_states_select_len_input[batch_idx],
                                :,
                                ],
                                hidden_states_select_slice[:],
                            ),
                            dim=1,
                        )
                    is_first_slice = False
                ## last_slice
                if idx == hidden_states_select.shape[1] - 1:
                    hidden_states_select_slice = torch.concat(
                        (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
                    )
                codec_generator = self.codec_decoder_generator(
                    hidden_states_select_slice.to(dtype).to(device),
                    self.codec_decoder,
                    states=states,
                    max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
                    min_length=kwargs.get("min_length_codec_duration_s", None),
                    **codec_decoder_generator_conf,
                )
                for gen_codec in codec_generator:
                    codec_pred.extend(gen_codec)
                    print(f"{uttid}: {len(gen_codec)} tokens: {gen_codec}")

                if idx == hidden_states_select.shape[1] - 1:
                    codec_preds.append(codec_pred)
                    codec_pred = []

        ## last_slice
        if hidden_states_select_list:
            hidden_states_select_slice = torch.cat(hidden_states_select_list, dim=1)
            hidden_states_select_slice = torch.concat(
                (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
            )
            codec_generator = self.codec_decoder_generator(
                hidden_states_select_slice.to(dtype).to(device),
                self.codec_decoder,
                states=states,
                max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
                min_length=kwargs.get("min_length_codec_duration_s", None),
                **codec_decoder_generator_conf,
            )
            for gen_codec in codec_generator:
                codec_pred.extend(gen_codec)
                print(f"{uttid}: {len(gen_codec)} tokens: {gen_codec}")
            codec_preds.append(codec_pred)
            codec_pred = []

        ## old process
        # slice_num = (hidden_states_select.shape[1] - 1) // self.text_codec_mix_ratio[0] + 1
        # if hidden_states_select.shape[1] % self.text_codec_mix_ratio[0] == 0:
        #     slice_num = slice_num + 1
        # for slice_idx in range(0, slice_num):

        #     hidden_states_select_slice = hidden_states_select[
        #         :,
        #         slice_idx
        #         * self.text_codec_mix_ratio[0] : (slice_idx + 1)
        #         * self.text_codec_mix_ratio[0],
        #         :,
        #     ]
        #     if slice_idx == 0:
        #         hidden_states_select_slice = torch.concat(
        #             (sos_emb[None, :, :], hidden_states_select_slice), dim=1
        #         )

        #         if self.concat_input_embedding:
        #             hidden_states_select_slice = torch.concat(
        #                 (
        #                     hidden_states_select_input[
        #                         batch_idx : batch_idx + 1,
        #                         : hidden_states_select_len_input[batch_idx],
        #                         :,
        #                     ],
        #                     hidden_states_select_slice[:],
        #                 ),
        #                 dim=1,
        #             )
        #     ## last_slice
        #     if (slice_idx + 1) * self.text_codec_mix_ratio[0] > hidden_states_select.shape[1]:
        #         hidden_states_select_slice = torch.concat(
        #             (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
        #         )
        #     codec_generator = self.codec_decoder_generator(
        #         hidden_states_select_slice.to(dtype).to(device),
        #         self.codec_decoder,
        #         states=states,
        #         max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
        #         min_length=kwargs.get("min_length_codec_duration_s", None),
        #         **codec_decoder_generator_conf,
        #     )
        #     response_chunk = tokenizer.decode(
        #         target_ids[
        #             0,
        #             slice_idx
        #             * self.text_codec_mix_ratio[0] : (slice_idx + 1)
        #             * self.text_codec_mix_ratio[0],
        #         ]
        #     )
        #     for gen_codec in codec_generator:
        #         codec_pred.extend(gen_codec)
        #         print(f"{uttid}: response: {response_chunk}, {len(gen_codec)} tokens: {gen_codec}")

        streaming_wav_dir = f'{kwargs.get("output_dir")}/streaming_wav'
        if not os.path.exists(streaming_wav_dir):
            os.makedirs(streaming_wav_dir)
        # synthesize waveform
        set_all_random_seed(kwargs.get("seed", 0))
        device = hidden_states_select.device
        wav_list = []
        if self.streaming_flow:
            codec_pred = []
            for codec_pred_i in codec_preds:
                codec_pred.extend(codec_pred_i)
            # for codec_pred in codec_preds:
            pred_inter = torch.tensor(codec_pred, dtype=torch.int64, device=device)[None, :]
            pred_inter_lens = torch.tensor([pred_inter.shape[1]], device=device, dtype=torch.int64)
            xvec_lens = torch.tensor([1], dtype=torch.int64, device=device)

            fm_voc_states = self.init_fm_voc_states()
            n_chunk = int(np.ceil(pred_inter.shape[1] / self.chunk_size))

            for i in range(n_chunk):
                chunk = pred_inter[
                        :, i * self.chunk_size: (i + 1) * self.chunk_size
                        ]  # .squeeze(-1)
                pred_chunk_lens = torch.tensor([chunk.shape[1]], device=device, dtype=torch.int64)
                (wav, _), fm_voc_states = self.streaming_flow_voc(
                    chunk,
                    pred_chunk_lens,
                    self.xvec,
                    xvec_lens,
                    (None, None),
                    (None, None),
                    fm_voc_states,
                    is_last=(i == n_chunk - 1),
                )
                wav_list.append(wav)
            # feat = fm_voc_states["cached_feat"]
            wav = torch.cat(wav_list, dim=2)
            wav_path = f"{streaming_wav_dir}/{uttid}.wav"
            torchaudio.save(
                wav_path,
                wav.cpu().squeeze(0),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        else:
            feat = self.fm_model.inference(
                pred_inter,
                pred_inter_lens,
                self.xvec,
                xvec_lens,
                diff_steps=self.fm_diff_steps,
                temperature=self.fm_temperature,
                prompt=dict(prompt_text=(None, None), prompt_audio=(None, None)),
            )
            # np.save(f'{kwargs.get("output_dir")}/{uttid}_feat.npy', feat[0].cpu().numpy())
            # feat in B, C, T
            # offline_wav_dir = f'{kwargs.get("output_dir")}/offline_wav'
            # if not os.path.exists(offline_wav_dir):
            #     os.makedirs(offline_wav_dir)
            # wav = self.vocoder.inference(feat.transpose(1, 2))[0].unsqueeze(0)
            # wav_path_offline = f"{offline_wav_dir}/{uttid}.wav"
            # torchaudio.save(
            #     wav_path_offline,
            #     wav.cpu(),
            #     sample_rate=self.vocoder.sample_rate,
            #     encoding="PCM_S",
            #     bits_per_sample=16,
            # )

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"text"]

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

            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"语音转写：<|startofspeech|>!{wav_path}<|endofspeech|>",
                    },
                    {
                        "role": "assistant",
                        "content": f"{response}".replace("\n", " "),
                    },
                ],
                "key": uttid,
                "task": "TTS",
            }
            ibest_writer["wav_streaming.jsonl"][key[0]] = (
                    json.dumps(data, ensure_ascii=False) + "\n"
            )
            # data = {
            #     "messages": [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {
            #             "role": "user",
            #             "content": f"语音转写：<|startofspeech|>!{wav_path_offline}<|endofspeech|>",
            #         },
            #         {
            #             "role": "assistant",
            #             "content": f"{response}".replace("\n", " "),
            #         },
            #     ],
            #     "key": uttid,
            #     "task": "TTS",
            # }
            # ibest_writer["wav_offline.jsonl"][key[0]] = json.dumps(data, ensure_ascii=False)

        return results, meta_data

    @torch.no_grad()
    def gen_codec(
            self,
            hidden_states_select,
            target_ids,
            states,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                hidden_states_select = torch.concat(
                    (self.sos_emb[None, :, :], hidden_states_select), dim=1
                )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]

        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        if not hasattr(self, "codec_decoder_generator"):
            from .utils.llm_decoding import LLMDecoder

            self.codec_decoder_generator = LLMDecoder(token_embeder=self.codec_embed, **kwargs).to(
                dtype
            )
            self.codec_decoder.lm_head = self.codec_head.to(dtype)
            self.codec_decoder.to(device)
            self.codec_decoder.to(dtype)
            self.codec_decoder_generator.to(device)

        codec_generator = self.codec_decoder_generator(
            hidden_states_select.to(dtype).to(device),
            self.codec_decoder,
            states=states,
            max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
            min_length=kwargs.get("min_length_codec_duration_s", None),
            **kwargs,
        )
        for gen_codec in codec_generator:
            yield gen_codec

    @torch.no_grad()
    def export_codec_vllm(self, model_path, **kwargs):
        if os.path.exists(model_path):
            return
        pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
        vocab_size = self.codec_embed.num_embeddings
        feature_size = self.codec_embed.embedding_dim
        pad_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to

        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        self.codec_decoder.lm_head = self.codec_head.to(dtype)
        new_lm_head = nn.Linear(in_features=feature_size, out_features=pad_vocab_size, bias=True)
        with torch.no_grad():
            new_lm_head.weight[:vocab_size] = self.codec_decoder.lm_head.weight
            new_lm_head.bias[:vocab_size] = self.codec_decoder.lm_head.bias
            new_lm_head.weight[vocab_size:] = 0
            new_lm_head.bias[vocab_size:] = 0
        self.codec_decoder.lm_head = new_lm_head
        new_codec_embed = nn.Linear(in_features=feature_size, out_features=pad_vocab_size)
        with torch.no_grad():
            new_codec_embed.weight[:vocab_size] = self.codec_embed.weight
            new_codec_embed.weight[vocab_size:] = 0
        self.codec_decoder.set_input_embeddings(new_codec_embed)
        self.codec_decoder.to(device)
        self.codec_decoder.to(dtype)
        tmp_vocab_size = self.codec_decoder.config.vocab_size
        tmp_tie_embedding = self.codec_decoder.config.tie_word_embeddings
        self.codec_decoder.config.vocab_size = pad_vocab_size
        self.codec_decoder.config.tie_word_embeddings = False
        self.codec_decoder.config.use_bias = True
        self.codec_decoder.save_pretrained(model_path)
        self.codec_decoder.config.vocab_size = tmp_vocab_size
        self.codec_decoder.config.tie_word_embeddings = tmp_tie_embedding

    @torch.no_grad()
    def gen_codec_vllm_sync(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            states,
            hidden_states_vllm_his,
            request_id,
            chunk=15,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25 if is_last_slice else chunk + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        vllm_codec_engine.add_request(
            request_id,
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
        )
        ## generator
        out_token_ids = []
        chunk_num = 0
        while True:
            request_outputs: List[RequestOutput] = vllm_codec_engine.step()
            for request_output in request_outputs:
                if str(request_output.request_id) != str(request_id):
                    continue
                if not request_output.finished:
                    # print(f"Partial request output: {request_output}")
                    out_token_ids.append(list(request_output.outputs[0].token_ids)[-1])
                    if is_last_slice and len(out_token_ids) % chunk == 0:
                        yield out_token_ids[
                              chunk * chunk_num: chunk * (chunk_num + 1)
                              ], hidden_states_select
                        chunk_num += 1
                else:
                    # print(f"Finished request output: {request_output}")
                    if out_token_ids[chunk * chunk_num:]:
                        yield out_token_ids[chunk * chunk_num:], hidden_states_select
                        chunk_num += 1
                    break

            if not vllm_codec_engine.has_unfinished_requests():
                break

    @torch.no_grad()
    def codec_add_request(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            hidden_states_vllm_his,
            request_id,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25
            if is_last_slice
            else self.text_codec_mix_ratio[1] + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        vllm_codec_engine.add_request(
            request_id,
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
        )
        return hidden_states_select

    @torch.no_grad()
    def gen_codec_vllm(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            states,
            hidden_states_vllm_his,
            request_id,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25
            if is_last_slice
            else self.text_codec_mix_ratio[1] + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        codec_generator = vllm_codec_engine.generate(
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
            request_id,
        )
        return codec_generator, hidden_states_select

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
        old_overlap = cached[:, -hop_len:]
        sin_wind = torch.blackman_window(2 * hop_len)[None, :].to(old_overlap)
        cf_overlap = (
                old_overlap * sin_wind[:, -hop_len:] + current[:, :hop_len] * sin_wind[:, :hop_len]
        )
        current[:, :hop_len] = cf_overlap

        return current

    def streaming_flow_voc(
            self,
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            prompt_inter,
            prompt_audio,
            fm_voc_states,
            is_last,
    ):
        cached_token = fm_voc_states.get("cached_token", None)
        cached_feat = fm_voc_states.get("cached_feat", None)
        cached_wav = fm_voc_states.get("cached_wav", None)
        if cached_token is not None:
            tokens = torch.cat([cached_token, tokens], dim=1)
            token_lens = token_lens + cached_token.shape[1]

        diff_out = self.fm_model.inference(
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            diff_steps=self.fm_diff_steps,
            temperature=self.fm_temperature,
            prompt=dict(prompt_text=prompt_inter, prompt_audio=prompt_audio),
        )
        # feat in B,C,T. remove lookahead feat and history, which is inaccurate
        feat = diff_out
        padded_feat_len = 0
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            # involving vocoder_hop for cross fade
            feat = diff_out[:, :, cached_feat_len - self.vocoder_hop_in_feat:]
            # pad history feat for vocoder synthesis consistency
            padded_feat_len = min(self.mel_frame_rate, cached_feat_len)
            feat = torch.cat([cached_feat[:, :, -padded_feat_len:], feat], dim=2)
        if not is_last:
            feat = feat[:, :, : -self.feat_reback]

        wav = self.vocoder.inference(feat.transpose(1, 2)).unsqueeze(1)
        # remove padded history wav
        pad_wav_len = int(padded_feat_len * (self.sample_rate / self.mel_frame_rate))
        wav = wav[:, :, pad_wav_len:]
        if cached_wav is not None:
            wav = self.cross_fade(
                cached_wav.squeeze(1), wav.squeeze(1), self.vocoder_hop
            ).unsqueeze(1)

        # play wav without tail hop length
        cur_wav = wav[:, :, : -self.vocoder_hop] if not is_last else wav

        # update caches
        cached_token = tokens
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            if is_last:
                cached_feat = torch.cat([cached_feat, diff_out[:, :, cached_feat_len:]], dim=2)
            else:
                cached_feat = torch.cat(
                    [cached_feat, diff_out[:, :, cached_feat_len: -self.feat_reback]], dim=2
                )
        else:
            if is_last:
                cached_feat = diff_out[:, :, :]
            else:
                cached_feat = diff_out[:, :, : -self.feat_reback]
        if cached_wav is not None:
            cached_wav = torch.cat([cached_wav[:, :, : -self.vocoder_hop], wav], dim=2)
        else:
            cached_wav = wav

        return (cur_wav, diff_out[:, :, : -self.feat_reback]), dict(
            cached_token=cached_token,
            cached_feat=cached_feat,
            cached_wav=cached_wav,
        )


@tables.register("model_classes", "MinMo_S2S_beta_b")
class MinMo_S2S_beta_b(nn.Module):
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

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

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            attn_implementation="flash_attention_2",
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            output_hidden_states=True,
            **llm_load_kwargs,
        )
        llm_dim = model.get_input_embeddings().weight.shape[-1]
        llm_tokenizer = AutoTokenizer.from_pretrained(init_param_path)
        self.llm_tokenizer = llm_tokenizer

        if llm_conf.get("reform_vocab_token", False):
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

        from .utils.stream_utils import _sample

        model._sample = types.MethodType(_sample, model)

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

        from .utils.stream_utils import TextStatesIteratorStreamer

        self.IteratorStreamer = TextStatesIteratorStreamer
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim
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

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss

        codec_decoder = kwargs.get("codec_decoder")
        codec_decoder_conf = kwargs.get("codec_decoder_conf", {})
        init_param_path = codec_decoder_conf.get("init_param_path", None)
        codec_decoder = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            attn_implementation="flash_attention_2",
            # output_hidden_states=True,
        )

        codec_tokenizer = AutoTokenizer.from_pretrained(init_param_path)
        self.codec_tokenizer = codec_tokenizer
        codec_decoder_dim = codec_decoder.get_input_embeddings().weight.shape[-1]

        self.codec_embed_tokens = codec_decoder.model.embed_tokens
        del codec_decoder.lm_head
        del codec_decoder.model.embed_tokens

        self.codec_decoder = codec_decoder
        self.codebook_size = codec_decoder_conf.get("codebook_size", 4096)
        self.codec_embed = nn.Embedding(self.codebook_size + 3, codec_decoder_dim, 0)
        self.codec_head = nn.Linear(codec_decoder_dim, self.codebook_size + 3, bias=True)
        self.codec_adaptor = nn.Linear(llm_dim * 2, codec_decoder_dim, bias=False)
        self.codec_sos_eos_embed = torch.nn.Embedding(2, codec_decoder_dim)

        self.sos_eos = codec_decoder_conf.get("sos_eos", 0)
        self.task_id = codec_decoder_conf.get("task_id", 1)
        self.fill_token = codec_decoder_conf.get("fill_token", 2)
        self.text_codec_mix_ratio = codec_decoder_conf.get("text_codec_mix_ratio", [4, 12])
        # self.text_codec_mix_prob = self.text_codec_mix_ratio[0] / self.text_codec_mix_ratio[1]
        self.ignore_id = kwargs.get("ignore_id", -100)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.codebook_size + 3,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
        self.concat_input_embedding = codec_decoder_conf.get("concat_input_embedding", False)
        self.concat_target_embedding = codec_decoder_conf.get("concat_target_embedding", False)
        if kwargs.get("flow_matching", True) and len(kwargs.get("flow_matching_conf", {})) > 0:

            # build flow mathing model
            self.streaming_flow = kwargs.get("streaming_flow", False)
            flow_conf = kwargs.get("flow_matching_conf", {})
            from .diffusion_text2audio_model import MaskedDiffWithXvec

            self.fm_model = MaskedDiffWithXvec(**flow_conf)
            # build vocoder
            vocoder_conf = kwargs.get("vocoder_conf", {})
            from .hifigan import HifiGan

            self.vocoder = HifiGan(**vocoder_conf)
            xvec_path = kwargs.get("xvec_path")
            if ".ark:" in xvec_path:
                import kaldiio

                xvec = kaldiio.load_mat(xvec_path)
            else:
                xvec = np.load(xvec_path)
            self.register_buffer("xvec", torch.tensor(xvec).unsqueeze(0))
            self.fm_diff_steps = kwargs.get("fm_diff_steps", 10)
            self.fm_temperature = kwargs.get("fm_temperature", 1.0)
            self.chunk_size = self.text_codec_mix_ratio[1]
            self.reback = kwargs.get("reback", 4)  # token reback
            if hasattr(self.fm_model.encoder, "pre_lookahead_len"):
                self.feat_reback_in_token = self.fm_model.encoder.pre_lookahead_len
                self.feat_reback = self.fm_model.calc_target_len(
                    self.feat_reback_in_token
                )  # feat reback
            else:
                self.feat_reback_in_token = 4
                self.feat_reback = kwargs.get("feat_reback", 8)
            self.sample_rate = flow_conf["mel_feat_conf"]["sampling_rate"]
            self.token_rate = flow_conf["input_frame_rate"]
            self.mel_hop_length = self.vocoder.mel_spec_transform.hop_length
            self.mel_frame_rate = int(self.sample_rate / self.mel_hop_length)
            self.vocoder_hop = kwargs.get("vocoder_hop", 1920)
            self.vocoder_hop_in_token = int(self.vocoder_hop / self.sample_rate * self.token_rate)
            self.vocoder_hop_in_feat = int(self.vocoder_hop / self.mel_hop_length)

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

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

            batch_size, token_num, dims = inputs_embeds.shape
            fake_token_len = kwargs.get("fake_token_len")
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            speech_idx = 0
            for batch_idx in range(batch_size):

                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    if fbank_beg_idx > 0 and speech_token_len > 0:

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
            loss_llm = model_outputs.loss
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_llm = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc_llm"] = acc_llm

        stats["loss_llm"] = torch.clone(loss_llm.detach())

        hidden_states = model_outputs.hidden_states[-1].float()
        device = hidden_states.device

        hidden_states_select = []
        hidden_states_select_len = []
        preds_select = []
        preds_select_len = []
        multiturn_beg = kwargs.get("multiturn_beg")
        multiturn_beg[multiturn_beg < 0] = -1
        multiturn_len = kwargs.get("multiturn_len")
        multiturn_len[multiturn_len < 0] = 0
        if self.concat_input_embedding:
            multiturn_beg_input = kwargs.get("multiturn_beg_input")
            multiturn_beg_input[multiturn_beg_input < 0] = -1
            multiturn_len_input = kwargs.get("multiturn_len_input")
            multiturn_len_input[multiturn_len_input < 0] = 0
            hidden_states_select_input = []
            hidden_states_select_len_input = []

        for batch_idx in range(multiturn_beg.shape[0]):
            for turn_id in range(multiturn_beg.shape[1]):
                multiturn_beg_idx = multiturn_beg[batch_idx, turn_id].item()
                multiturn_len_idx = multiturn_len[batch_idx, turn_id].item() - 1  # rm eos
                multiturn_end_idx = multiturn_beg_idx + multiturn_len_idx

                if multiturn_beg_idx >= 0:
                    if self.concat_target_embedding:
                        hidden_states_idx = hidden_states[
                                            batch_idx, multiturn_beg_idx - 1: multiturn_end_idx - 1, :
                                            ]
                        inputs_embeds_idx = inputs_embeds[
                                            batch_idx, multiturn_beg_idx:multiturn_end_idx, :
                                            ]
                        hidden_states_select.append(
                            torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                        )
                        hidden_states_select_len.append(multiturn_len_idx)

                    else:
                        preds_idx = labels_ids[batch_idx, multiturn_beg_idx:multiturn_end_idx]
                        preds_text = self.llm_tokenizer.decode(preds_idx)
                        preds_select_idx = self.codec_tokenizer.encode(preds_text)
                        hidden_states_select_len.append(len(preds_select_idx))
                        preds_select_idx = torch.tensor(preds_select_idx, dtype=torch.int64).to(
                            device
                        )
                        preds_select_idx = self.codec_embed_tokens(preds_select_idx[None, :])
                        hidden_states_select.append(preds_select_idx[0, :, :])

                if self.concat_input_embedding:
                    multiturn_beg_idx_input = multiturn_beg_input[batch_idx, turn_id].item()
                    multiturn_len_idx_input = multiturn_len_input[batch_idx, turn_id].item()
                    multiturn_end_idx_input = multiturn_beg_idx_input + multiturn_len_idx_input
                    if multiturn_beg_idx_input >= 0:
                        hidden_states_idx = hidden_states[
                                            batch_idx, multiturn_beg_idx_input - 1: multiturn_end_idx_input - 1, :
                                            ]
                        inputs_embeds_idx = inputs_embeds[
                                            batch_idx, multiturn_beg_idx_input:multiturn_end_idx_input, :
                                            ]
                        hidden_states_select_input.append(
                            torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                        )
                        hidden_states_select_len_input.append(multiturn_len_idx_input)

        # hidden_states_his_select

        hidden_states_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_select, batch_first=True, padding_value=0.0
        ).to(device=device)
        hidden_states_select_len = torch.tensor(hidden_states_select_len, dtype=torch.int64).to(
            device=device
        )
        if self.concat_input_embedding:
            hidden_states_select_input = torch.nn.utils.rnn.pad_sequence(
                hidden_states_select_input, batch_first=True, padding_value=0.0
            ).to(device=device)
            # hidden_states_select_len_input = torch.tensor(hidden_states_select_len_input, dtype=torch.int64).to(
            #     device=device
            # )
            hidden_states_select_input = self.codec_adaptor(hidden_states_select_input)

        if self.concat_target_embedding:
            hidden_states_select = self.codec_adaptor(hidden_states_select)

        # codec
        codec = kwargs.get("codec")
        codec[codec < 0] = 0
        codec_len = kwargs.get("codec_len")
        # codec_len[codec_len < 0] = 0
        is_t2c = kwargs.get("is_t2c")

        codec_emb = self.codec_embed(codec)
        sos_emb = self.codec_sos_eos_embed(
            torch.tensor([self.sos_eos], dtype=torch.int64).to(device=device)
        )
        task_id_emb = self.codec_sos_eos_embed(
            torch.tensor([self.task_id], dtype=torch.int64).to(device=device)
        )

        codec_decoder_inputs_embeds = []
        codec_decoder_inputs_len = []
        codec_decoder_targets = []
        codec_decoder_attention_mask = []

        for batch_idx in range(hidden_states_select.shape[0]):
            hidden_states_select_idx = hidden_states_select[
                                       batch_idx, : hidden_states_select_len[batch_idx], :
                                       ]
            codec_emb_idx = codec_emb[batch_idx, : codec_len[batch_idx], :]
            code_idx = codec[batch_idx, : codec_len[batch_idx]]
            (
                codec_decoder_inputs_embeds_idx,
                codec_decoder_targets_idx,
                codec_decoder_inputs_len_idx,
            ) = self.mixup_text_codec(
                hidden_states_select_idx,
                codec_emb_idx,
                sos_emb,
                task_id_emb,
                code_idx,
            )

            if self.concat_input_embedding and not is_t2c[batch_idx]:
                codec_decoder_inputs_embeds_idx = torch.concat(
                    (
                        hidden_states_select_input[
                        batch_idx, : hidden_states_select_len_input[batch_idx], :
                        ],
                        codec_decoder_inputs_embeds_idx[:, :],
                    ),
                    dim=0,
                )
                codec_decoder_inputs_len_idx = (
                        codec_decoder_inputs_len_idx + hidden_states_select_len_input[batch_idx]
                )
                codec_decoder_targets_idx = torch.concat(
                    (
                        torch.ones(hidden_states_select_len_input[batch_idx], dtype=torch.int64).to(
                            codec_decoder_targets_idx.device
                        )
                        * self.ignore_id,
                        codec_decoder_targets_idx[:],
                    ),
                    dim=0,
                )

            codec_decoder_inputs_embeds.append(codec_decoder_inputs_embeds_idx)
            codec_decoder_inputs_len.append(codec_decoder_inputs_len_idx)
            codec_decoder_targets.append(codec_decoder_targets_idx)

            codec_decoder_attention_mask_idx = torch.tensor(
                [1] * codec_decoder_inputs_len_idx, dtype=torch.int64
            )
            codec_decoder_attention_mask.append(codec_decoder_attention_mask_idx)

        codec_decoder_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_inputs_embeds, batch_first=True, padding_value=0.0
        ).to(device=device)
        codec_decoder_inputs_len = torch.tensor(codec_decoder_inputs_len, dtype=torch.int64).to(
            device=device
        )
        codec_decoder_targets = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_targets, batch_first=True, padding_value=-100
        ).to(device=device)
        codec_decoder_attention_mask = torch.nn.utils.rnn.pad_sequence(
            codec_decoder_attention_mask, batch_first=True, padding_value=0.0
        ).to(device=device)

        codec_hidden_states = self.codec_decoder.model(
            inputs_embeds=codec_decoder_inputs_embeds,
            attention_mask=codec_decoder_attention_mask,
        )[0]

        codec_logits = self.codec_head(codec_hidden_states)
        loss = self.criterion_ce(codec_logits, codec_decoder_targets)

        with torch.no_grad():
            preds = torch.argmax(codec_logits, -1)
            acc_att = compute_accuracy(preds, codec_decoder_targets, ignore_label=-100)
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

    def mixup_text_codec(self, text_emb, codec_emb, sos_emb, task_id_emb, codec_token):
        # self.sos_eos:0
        # task_id_emb: 1
        # self.ignore_id = -100
        # self.text_codec_mix_ratio: [4, 12]
        # self.codebook_size: codec本身size，不包含sos
        # self.fill_token: 2
        pos = -1
        prompt_emb = text_emb[: pos + 1]
        prompt_fill_token = torch.ones(pos + 1).fill_(self.ignore_id).to(codec_token)
        text_emb = text_emb[pos + 1:]

        text_len = text_emb.shape[0]
        codec_len = codec_emb.shape[0]
        llm_length = text_len + codec_len + 2 + (pos + 1)
        text_interval, codec_interval = self.text_codec_mix_ratio
        exp_intervals = int(math.floor(text_len / text_interval))
        exp_codec_len = exp_intervals * codec_interval
        if exp_codec_len < codec_len:
            exp_text_len = exp_intervals * text_interval
            # Calculate the length of the mixed sequence
            mixed_length = exp_text_len + exp_codec_len

            # Create an empty tensor for the mixed sequence
            mixed_seq = torch.zeros(
                mixed_length, codec_emb.shape[1], dtype=codec_emb.dtype, device=codec_emb.device
            )

            # Fill the mixed sequence
            for i in range(text_interval):
                mixed_seq[i:: text_interval + codec_interval] = text_emb[
                                                                i:exp_text_len:text_interval
                                                                ]
            for i in range(codec_interval):
                try:
                    mixed_seq[i + text_interval:: text_interval + codec_interval] = codec_emb[
                                                                                    i:exp_codec_len:codec_interval
                                                                                    ]
                except:
                    import pdb

                    pdb.set_trace()

            ret_seq = torch.cat(
                [
                    sos_emb,
                    prompt_emb,
                    mixed_seq,
                    text_emb[exp_text_len:],
                    task_id_emb,
                    codec_emb[exp_codec_len:],
                ],
                dim=0,
            )

            fill_token = self.codebook_size + self.fill_token
            ret_targets = None
            if codec_token is not None:
                targets = torch.zeros(
                    mixed_length, dtype=codec_token.dtype, device=codec_token.device
                )
                for i in range(text_interval):
                    targets[i:: text_interval + codec_interval] = fill_token
                for i in range(codec_interval):
                    targets[i + text_interval:: text_interval + codec_interval] = codec_token[
                                                                                  i:exp_codec_len:codec_interval
                                                                                  ]
                ret_targets = torch.cat(
                    [
                        prompt_fill_token,
                        targets,
                        torch.ones((text_len - exp_text_len)).to(codec_token) * fill_token,
                        torch.tensor([self.ignore_id]).to(codec_token),
                        codec_token[exp_codec_len:],
                        torch.tensor([self.codebook_size + self.sos_eos]).to(codec_token),
                    ],
                    dim=0,
                )

            # ret_seq: audio_decoder, input, 包括sos, 不包括eos
            # ret_targets: audio_decoder, target, 不包括sos, 包括eos
            # llm_length: ret_seq.shape[0]
            return ret_seq, ret_targets, llm_length
        else:
            # offline mode
            ret_seq = torch.cat(
                [
                    sos_emb,
                    prompt_emb,
                    text_emb,
                    task_id_emb,
                    codec_emb,
                ],
                dim=0,
            )
            ret_targets = torch.cat(
                [
                    prompt_fill_token,
                    torch.ones(text_len).to(codec_token) * self.ignore_id,
                    torch.tensor([self.ignore_id]).to(codec_token),
                    codec_token,
                    torch.tensor([self.codebook_size + self.sos_eos]).to(codec_token),
                ],
                dim=0,
            )

            return ret_seq, ret_targets, llm_length

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

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len, is_t2c = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        multiturn_beg_input = []
        multiturn_len_input = []
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

            splits = pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech, speech_lengths = [], []
            target_ids = []
            is_t2c_i = True
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    is_t2c_i = False
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

            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)

            target_out = f"{target_out}<|im_end|>"
            splits = pattern.split(target_out)
            # codec_i = []
            sub_token = []
            codec_i_len = 0

            for k, sub_str in enumerate(splits):
                if len(sub_str) < 1:
                    continue
                if not sub_str.startswith("<|startofspeech|>"):
                    # sub_str = f"{sub_str}<|im_end|>"
                    sub_token = tokenizer.encode(sub_str)
                    target_ids += sub_token
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    # if sub_str.startswith("!"):
                    #     sub_token_codec = np.load(sub_str[1:])
                    #     codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                    #     codec_i_len = len(sub_token_codec)
                    #
                    # else:
                    #     sub_token_codec = []
                    #     for x in sub_str.split("|"):
                    #         if x.startswith("c"):
                    #             sub_token_codec.append(int(x[1:]))
                    #     codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                    #     codec_i_len = len(sub_token_codec)
                    # codec.append(codec_i)
                    # codec_len.append(codec_i_len)

            multiturn_beg_input += [len(input_ids)]
            multiturn_len_input += [len(source_ids)]
            if i == 0:
                system_token = tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
                multiturn_beg_input[-1] = len(system_token)
                multiturn_len_input[-1] = multiturn_len_input[-1] - len(system_token)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)
            is_t2c.append(is_t2c_i)
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
            "is_t2c": is_t2c,
        }
        output["multiturn_beg_input"] = torch.tensor(multiturn_beg_input, dtype=torch.int32)
        output["multiturn_len_input"] = torch.tensor(multiturn_len_input, dtype=torch.int32)
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
                speech_token_len = fake_token_len[batch_idx, turn_id]
                if fbank_beg_idx > 0 and speech_token_len > 0:

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

    @torch.no_grad()
    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            tokenizer=None,
            frontend=None,
            **kwargs,
    ):

        uttid = key[0]
        # import pdb; pdb.set_trace()

        if not hasattr(self, "fe"):
            self.init_ttsfrd()
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )
        device = inputs_embeds.device
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

            if kwargs.get("tearchforing", False):

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

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] - 1: -2]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                # loss = model_outputs.loss.item()
                hidden_states = model_outputs.hidden_states[-1].float()
                hidden_states_input = hidden_states
                hidden_states_select = hidden_states[:, source_ids.shape[1] - 1: -2, :]
                target_ids = preds
                loss = model_outputs.loss
                if self.concat_target_embedding:
                    target_emb = self.llm.model.get_input_embeddings()(target_ids)
                    hidden_states_select = torch.concat(
                        (hidden_states_select[:, :, :], target_emb[:, :, :]), dim=-1
                    )
                    hidden_states_select = self.codec_adaptor(hidden_states_select)

                else:
                    preds_text = self.llm_tokenizer.decode(preds[0, :].tolist())
                    response = preds_text
                    response_clean = self.fe.do_voicegen_frd(response)
                    response_cleans = json.loads(response_clean)["sentences"]
                    response_clean = ""
                    for response_clean_i in response_cleans:
                        response_clean += response_clean_i["text"]
                    print(f"response: \n{response}\nresponse_clean: \n{response_clean}")
                    preds_select_idx = self.codec_tokenizer.encode(response_clean)
                    preds_select_idx = torch.tensor(preds_select_idx, dtype=torch.int64).to(device)
                    target_ids = preds_select_idx[None, :]
                    hidden_states_select = self.codec_embed_tokens(preds_select_idx[None, :])

            else:
                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **llm_kwargs,
                )
                target_ids = generated_ids["sequences"]
                for idx in target_ids[0]:
                    print(f"{tokenizer.decode([idx])}")

                response = tokenizer.batch_decode(
                    generated_ids["sequences"],
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]

                loss = None

                hidden_states = generated_ids[
                    "hidden_states"
                ]  # hidden_states: (t1, t2, ..., tn, ..., tN), tn=(l1, l2, ..., ln, ..., lN), ln: shape: 1x1x3584

                hidden_states_input = hidden_states[0][-1]
                if self.concat_target_embedding:
                    token_num = len(hidden_states)
                    hidden_states_select = torch.zeros(
                        (1, token_num, 3584), dtype=torch.float32
                    ).to(inputs_embeds.device)
                    hidden_states_out_len = torch.tensor(
                        [
                            token_num,
                        ],
                        dtype=torch.int32,
                    ).to(inputs_embeds.device)

                    # import pdb; pdb.set_trace()
                    ## filter special symbols
                    special_symbols = ["*", "**", "-", "《", "》", '"', "\n"]
                    target_ids_new = []
                    hs_count = 0
                    text_pre = ""
                    assert target_ids.shape[1] == token_num
                    for i in range(token_num):
                        text_new = tokenizer.decode([target_ids[0][i]])
                        if text_new.strip() not in special_symbols:
                            if text_new == " " and text_pre == " ":
                                continue
                            hidden_states_select[0, hs_count, :] = hidden_states[i][-1][0, 0, :].to(
                                torch.float32
                            )
                            hs_count += 1
                            target_ids_new.append(target_ids[0][i])
                            text_pre = text_new
                    hidden_states_select = hidden_states_select[:, :hs_count, :]
                    target_ids_new = torch.tensor(([target_ids_new]), device=target_ids.device)
                    target_emb = self.llm.model.get_input_embeddings()(target_ids_new)

                    hidden_states_select = torch.concat(
                        (hidden_states_select[:, :-1, :], target_emb[:, :-1, :]), dim=-1
                    )
                    hidden_states_select = self.codec_adaptor(hidden_states_select)
                else:
                    response_clean = self.fe.do_voicegen_frd(response)
                    response_cleans = json.loads(response_clean)["sentences"]
                    response_clean = ""
                    for response_clean_i in response_cleans:
                        response_clean += response_clean_i["text"]
                    print(f"response: \n{response}\nresponse_clean: \n{response_clean}")
                    preds_select_idx = self.codec_tokenizer.encode(response_clean)
                    preds_select_idx = torch.tensor(preds_select_idx, dtype=torch.int64).to(device)
                    target_ids = preds_select_idx[None, :]
                    hidden_states_select = self.codec_embed_tokens(preds_select_idx[None, :])

        multiturn_beg_input = batch["multiturn_beg_input"]
        multiturn_len_input = batch["multiturn_len_input"]
        batch_idx = 0
        turn_id = -1
        is_t2c = batch["is_t2c"]
        hidden_states_select_input = []
        hidden_states_select_len_input = []
        if self.concat_input_embedding:
            multiturn_beg_idx_input = multiturn_beg_input[turn_id].item()
            multiturn_len_idx_input = multiturn_len_input[turn_id].item()
            multiturn_end_idx_input = multiturn_beg_idx_input + multiturn_len_idx_input
            if multiturn_beg_idx_input >= 0:
                hidden_states_idx = hidden_states_input[
                                    batch_idx, multiturn_beg_idx_input - 1: multiturn_end_idx_input - 1, :
                                    ]
                inputs_embeds_idx = inputs_embeds[
                                    batch_idx, multiturn_beg_idx_input:multiturn_end_idx_input, :
                                    ]
                hidden_states_select_input.append(
                    torch.concat((hidden_states_idx, inputs_embeds_idx), dim=-1)
                )
                hidden_states_select_len_input.append(multiturn_len_idx_input)

                hidden_states_select_input = torch.nn.utils.rnn.pad_sequence(
                    hidden_states_select_input, batch_first=True, padding_value=0.0
                ).to(device=inputs_embeds.device)
                # hidden_states_select_len_input = torch.tensor(hidden_states_select_len_input, dtype=torch.int64).to(
                #     device=device
                # )
                hidden_states_select_input = self.codec_adaptor(
                    hidden_states_select_input.to(hidden_states_select.dtype)
                )

        codec_decoder_generator_conf = kwargs.get(
            "codec_decoder_generator_conf", {"eos_token": [6563, 6561]}
        )
        if "device" not in codec_decoder_generator_conf:
            codec_decoder_generator_conf["device"] = kwargs.get("device", "cuda:0")
        if "codec_dtype" not in codec_decoder_generator_conf:
            codec_decoder_generator_conf["codec_dtype"] = kwargs.get("llm_dtype", "fp32")
        dtype = dtype_map[codec_decoder_generator_conf["codec_dtype"]]
        device = codec_decoder_generator_conf["device"]
        if not hasattr(self, "codec_decoder_generator"):
            from .utils.llm_decoding import LLMDecoder as LLMDecoder

            self.codec_decoder_generator = LLMDecoder(
                token_embeder=self.codec_embed, **codec_decoder_generator_conf
            ).to(dtype)

            self.codec_decoder.lm_head = self.codec_head.to(dtype)
            self.codec_decoder.to(device)
            self.codec_decoder.to(dtype)
            self.codec_decoder_generator.to(device)

        sos_emb = self.codec_sos_eos_embed(
            torch.tensor([self.sos_eos], dtype=torch.int64).to(device=inputs_embeds.device)
        )
        task_id_emb = self.codec_sos_eos_embed(
            torch.tensor([self.task_id], dtype=torch.int64).to(device=inputs_embeds.device)
        )
        states = kwargs.get("states", {})
        codec_preds = []
        codec_pred = []

        ## new process
        hidden_states_select_list = []
        is_first_slice = True
        cut_puncs = ["。", "？", "！", "；", "：", ".", "?", "!", ";", ".", "?", "!", ";", ":"]
        cut_thres = 100
        cut_base = 0
        for idx in range(hidden_states_select.shape[1]):
            hidden_states_select_list.append(hidden_states_select[:, idx: idx + 1, :])
            text_acc = self.codec_tokenizer.decode(
                target_ids[0, : idx + 1], skip_special_tokens=True
            )
            print(
                f"text: {self.codec_tokenizer.decode(target_ids[0, idx:idx + 1], skip_special_tokens=True)}, text_acc: {text_acc}"
            )
            text_new = text_acc[-1] if len(text_acc) != 0 else ""
            if idx >= cut_base + cut_thres and text_new in cut_puncs:
                print(f"Clear history: {idx}")
                cut_base = idx
                hidden_states_select_slice = torch.cat(hidden_states_select_list, dim=1)
                hidden_states_select_list = []
                ## last_slice
                hidden_states_select_slice = torch.concat(
                    (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
                )
                codec_generator = self.codec_decoder_generator(
                    hidden_states_select_slice.to(dtype).to(device),
                    self.codec_decoder,
                    states=states,
                    max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
                    min_length=kwargs.get("min_length_codec_duration_s", None),
                    **codec_decoder_generator_conf,
                )
                for gen_codec in codec_generator:
                    codec_pred.extend(gen_codec)
                    print(f"{uttid}: {len(gen_codec)} tokens: {gen_codec}")
                ## TO-DO clear
                codec_preds.append(codec_pred)
                codec_pred = []
                states["llm_cache"] = None
                is_first_slice = True

            elif len(hidden_states_select_list) % self.text_codec_mix_ratio[0] == 0:
                hidden_states_select_slice = torch.cat(hidden_states_select_list, dim=1)
                hidden_states_select_list = []
                if is_first_slice:
                    hidden_states_select_slice = torch.concat(
                        (sos_emb[None, :, :], hidden_states_select_slice), dim=1
                    )
                    if self.concat_input_embedding and not is_t2c[0]:
                        hidden_states_select_slice = torch.concat(
                            (
                                hidden_states_select_input[
                                batch_idx: batch_idx + 1,
                                : hidden_states_select_len_input[batch_idx],
                                :,
                                ],
                                hidden_states_select_slice[:],
                            ),
                            dim=1,
                        )
                    is_first_slice = False
                # ## last_slice
                # if idx == hidden_states_select.shape[1] - 1:
                #     hidden_states_select_slice = torch.concat(
                #         (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
                #     )
                codec_generator = self.codec_decoder_generator(
                    hidden_states_select_slice.to(dtype).to(device),
                    self.codec_decoder,
                    states=states,
                    max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
                    min_length=kwargs.get("min_length_codec_duration_s", None),
                    **codec_decoder_generator_conf,
                )
                for gen_codec in codec_generator:
                    codec_pred.extend(gen_codec)
                    print(f"{uttid}: {len(gen_codec)} tokens: {gen_codec}")

                if idx == hidden_states_select.shape[1] - 1:
                    codec_preds.append(codec_pred)
                    codec_pred = []

        ## last_slice
        if hidden_states_select_list:
            hidden_states_select_slice = torch.cat(hidden_states_select_list, dim=1)
            hidden_states_select_slice = torch.concat(
                (hidden_states_select_slice, task_id_emb[None, :, :]), dim=1
            )
        else:
            hidden_states_select_slice = task_id_emb[None, :, :]

        codec_generator = self.codec_decoder_generator(
            hidden_states_select_slice.to(dtype).to(device),
            self.codec_decoder,
            states=states,
            max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
            min_length=kwargs.get("min_length_codec_duration_s", None),
            **codec_decoder_generator_conf,
        )
        for gen_codec in codec_generator:
            codec_pred.extend(gen_codec)
            print(f"{uttid}: {len(gen_codec)} tokens: {gen_codec}")
        codec_preds.append(codec_pred)
        codec_pred = []

        streaming_wav_dir = f'{kwargs.get("output_dir")}/streaming_wav'
        if not os.path.exists(streaming_wav_dir):
            os.makedirs(streaming_wav_dir)
        # synthesize waveform
        set_all_random_seed(kwargs.get("seed", 0))
        device = hidden_states_select.device
        wav_list = []
        if self.streaming_flow:
            codec_pred = []
            for codec_pred_i in codec_preds:
                codec_pred.extend(codec_pred_i)
            pred_inter = torch.tensor(codec_pred, dtype=torch.int64, device=device)[None, :]
            pred_inter_lens = torch.tensor([pred_inter.shape[1]], device=device, dtype=torch.int64)
            xvec_lens = torch.tensor([1], dtype=torch.int64, device=device)

            fm_voc_states = self.init_fm_voc_states()
            n_chunk = int(np.ceil(pred_inter.shape[1] / self.chunk_size))

            for i in range(n_chunk):
                chunk = pred_inter[
                        :, i * self.chunk_size: (i + 1) * self.chunk_size
                        ]  # .squeeze(-1)
                pred_chunk_lens = torch.tensor([chunk.shape[1]], device=device, dtype=torch.int64)
                (wav, _), fm_voc_states = self.streaming_flow_voc(
                    chunk,
                    pred_chunk_lens,
                    self.xvec,
                    xvec_lens,
                    (None, None),
                    (None, None),
                    fm_voc_states,
                    is_last=(i == n_chunk - 1),
                )
                wav_list.append(wav)
            # feat = fm_voc_states["cached_feat"]
            wav = torch.cat(wav_list, dim=2)
            wav_path = f"{streaming_wav_dir}/{uttid}.wav"
            torchaudio.save(
                wav_path,
                wav.cpu().squeeze(0),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        else:
            feat = self.fm_model.inference(
                pred_inter,
                pred_inter_lens,
                self.xvec,
                xvec_lens,
                diff_steps=self.fm_diff_steps,
                temperature=self.fm_temperature,
                prompt=dict(prompt_text=(None, None), prompt_audio=(None, None)),
            )
            # np.save(f'{kwargs.get("output_dir")}/{uttid}_feat.npy', feat[0].cpu().numpy())
            # feat in B, C, T
            # offline_wav_dir = f'{kwargs.get("output_dir")}/offline_wav'
            # if not os.path.exists(offline_wav_dir):
            #     os.makedirs(offline_wav_dir)
            # wav = self.vocoder.inference(feat.transpose(1, 2))[0].unsqueeze(0)
            # wav_path_offline = f"{offline_wav_dir}/{uttid}.wav"
            # torchaudio.save(
            #     wav_path_offline,
            #     wav.cpu(),
            #     sample_rate=self.vocoder.sample_rate,
            #     encoding="PCM_S",
            #     bits_per_sample=16,
            # )

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"text"]

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

            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"语音转写：<|startofspeech|>!{wav_path}<|endofspeech|>",
                    },
                    {
                        "role": "assistant",
                        "content": f"{response}".replace("\n", " "),
                    },
                ],
                "key": uttid,
                "task": "TTS",
            }
            ibest_writer["wav_streaming.jsonl"][key[0]] = (
                    json.dumps(data, ensure_ascii=False) + "\n"
            )
            # data = {
            #     "messages": [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {
            #             "role": "user",
            #             "content": f"语音转写：<|startofspeech|>!{wav_path_offline}<|endofspeech|>",
            #         },
            #         {
            #             "role": "assistant",
            #             "content": f"{response}".replace("\n", " "),
            #         },
            #     ],
            #     "key": uttid,
            #     "task": "TTS",
            # }
            # ibest_writer["wav_offline.jsonl"][key[0]] = json.dumps(data, ensure_ascii=False)

        return results, meta_data

    def init_ttsfrd(self):
        import ttsfrd

        fe = ttsfrd.TtsFrontendEngine()
        fe.initialize("/nfs/hengwu.zty/resources/ttsfrd/cv2_20250109")
        fe.set_lang_type("pinyinvg")
        self.fe = fe

    @torch.no_grad()
    def gen_codec(
            self,
            hidden_states_select,
            target_ids,
            states,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                hidden_states_select = torch.concat(
                    (self.sos_emb[None, :, :], hidden_states_select), dim=1
                )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]

        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        if not hasattr(self, "codec_decoder_generator"):
            from .utils.llm_decoding import LLMDecoder

            self.codec_decoder_generator = LLMDecoder(token_embeder=self.codec_embed, **kwargs).to(
                dtype
            )
            self.codec_decoder.lm_head = self.codec_head.to(dtype)
            self.codec_decoder.to(device)
            self.codec_decoder.to(dtype)
            self.codec_decoder_generator.to(device)

        codec_generator = self.codec_decoder_generator(
            hidden_states_select.to(dtype).to(device),
            self.codec_decoder,
            states=states,
            max_length=kwargs.get("max_length_codec_duration_s", 120) * 25,
            min_length=kwargs.get("min_length_codec_duration_s", None),
            **kwargs,
        )
        for gen_codec in codec_generator:
            yield gen_codec

    @torch.no_grad()
    def export_codec_vllm(self, model_path, **kwargs):
        if os.path.exists(model_path):
            return
        pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
        vocab_size = self.codec_embed.num_embeddings
        feature_size = self.codec_embed.embedding_dim
        pad_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to

        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        self.codec_decoder.lm_head = self.codec_head.to(dtype)
        new_lm_head = nn.Linear(in_features=feature_size, out_features=pad_vocab_size, bias=True)
        with torch.no_grad():
            new_lm_head.weight[:vocab_size] = self.codec_decoder.lm_head.weight
            new_lm_head.bias[:vocab_size] = self.codec_decoder.lm_head.bias
            new_lm_head.weight[vocab_size:] = 0
            new_lm_head.bias[vocab_size:] = 0
        self.codec_decoder.lm_head = new_lm_head
        new_codec_embed = nn.Linear(in_features=feature_size, out_features=pad_vocab_size)
        with torch.no_grad():
            new_codec_embed.weight[:vocab_size] = self.codec_embed.weight
            new_codec_embed.weight[vocab_size:] = 0
        self.codec_decoder.set_input_embeddings(new_codec_embed)
        self.codec_decoder.to(device)
        self.codec_decoder.to(dtype)
        tmp_vocab_size = self.codec_decoder.config.vocab_size
        tmp_tie_embedding = self.codec_decoder.config.tie_word_embeddings
        self.codec_decoder.config.vocab_size = pad_vocab_size
        self.codec_decoder.config.tie_word_embeddings = False
        self.codec_decoder.config.use_bias = True
        self.codec_decoder.save_pretrained(model_path)
        self.codec_decoder.config.vocab_size = tmp_vocab_size
        self.codec_decoder.config.tie_word_embeddings = tmp_tie_embedding

    @torch.no_grad()
    def gen_codec_vllm_sync(
            self,
            vllm_codec_engine,
            target_ids,
            hidden_states_vllm_his,
            request_id,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])
        chunk = self.text_codec_mix_ratio[1]

        if len(target_ids) > 0:
            hidden_states_select = self.codec_embed_tokens(
                torch.tensor([target_ids], dtype=torch.int64, device=device)
            )
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25 if is_last_slice else chunk + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        vllm_codec_engine.add_request(
            request_id,
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
        )
        ## generator
        out_token_ids = []
        chunk_num = 0
        while True:
            request_outputs: List[RequestOutput] = vllm_codec_engine.step()
            for request_output in request_outputs:
                if str(request_output.request_id) != str(request_id):
                    continue
                if not request_output.finished:
                    # print(f"Partial request output: {request_output}")
                    out_token_ids.append(list(request_output.outputs[0].token_ids)[-1])
                    if is_last_slice and len(out_token_ids) % chunk == 0:
                        yield out_token_ids[
                              chunk * chunk_num: chunk * (chunk_num + 1)
                              ], hidden_states_select
                        chunk_num += 1
                else:
                    # print(f"Finished request output: {request_output}")
                    if out_token_ids[chunk * chunk_num:]:
                        yield out_token_ids[chunk * chunk_num:], hidden_states_select
                        chunk_num += 1
                    break

            if not vllm_codec_engine.has_unfinished_requests():
                break

    @torch.no_grad()
    def gen_codec_vllm_sync_hs(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            states,
            hidden_states_vllm_his,
            request_id,
            chunk=15,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        # import pdb; pdb.set_trace()
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25 if is_last_slice else chunk + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        vllm_codec_engine.add_request(
            request_id,
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
        )
        ## generator
        out_token_ids = []
        chunk_num = 0
        while True:
            request_outputs: List[RequestOutput] = vllm_codec_engine.step()
            for request_output in request_outputs:
                if str(request_output.request_id) != str(request_id):
                    continue
                if not request_output.finished:
                    # print(f"Partial request output: {request_output}")
                    out_token_ids.append(list(request_output.outputs[0].token_ids)[-1])
                    if is_last_slice and len(out_token_ids) % chunk == 0:
                        yield out_token_ids[
                              chunk * chunk_num: chunk * (chunk_num + 1)
                              ], hidden_states_select
                        chunk_num += 1
                else:
                    # print(f"Finished request output: {request_output}")
                    if out_token_ids[chunk * chunk_num:]:
                        yield out_token_ids[chunk * chunk_num:], hidden_states_select
                        chunk_num += 1
                    break

            if not vllm_codec_engine.has_unfinished_requests():
                break

    @torch.no_grad()
    def codec_add_request(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            hidden_states_vllm_his,
            request_id,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25
            if is_last_slice
            else self.text_codec_mix_ratio[1] + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        vllm_codec_engine.add_request(
            request_id,
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
        )
        return hidden_states_select

    @torch.no_grad()
    def gen_codec_vllm(
            self,
            vllm_codec_engine,
            hidden_states_select,
            target_ids,
            states,
            hidden_states_vllm_his,
            request_id,
            hidden_states_usr_hs=None,
            is_first_slice=False,
            is_last_slice=False,
            **kwargs,
    ):
        device = kwargs.get("device", "cuda:0")
        dtype = dtype_map[kwargs.get("codec_dtype", "bf16")]
        sampling = kwargs.get("sampling", 25)
        eos_token = kwargs.get("eos_token", [6561, 6563])

        if len(hidden_states_select) > 0:
            hidden_states_select = torch.cat(hidden_states_select, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
            target_emb = self.llm.model.get_input_embeddings()(target_ids)

            hidden_states_select = torch.concat(
                (hidden_states_select[None, :, :], target_emb[None, :, :]), dim=-1
            ).to(torch.float32)
            hidden_states_select = self.codec_adaptor(hidden_states_select)
            if not hasattr(self, "sos_emb"):
                self.sos_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.sos_eos], dtype=torch.int64).to(device=target_emb.device)
                )
                self.task_id_emb = self.codec_sos_eos_embed(
                    torch.tensor([self.task_id], dtype=torch.int64).to(device=target_emb.device)
                )
            if is_first_slice:
                if hidden_states_usr_hs is not None:
                    hidden_states_usr_hs = self.codec_adaptor(hidden_states_usr_hs)
                    hidden_states_select = torch.concat(
                        (hidden_states_usr_hs, self.sos_emb[None, :, :], hidden_states_select),
                        dim=1,
                    )
                else:
                    hidden_states_select = torch.concat(
                        (self.sos_emb[None, :, :], hidden_states_select), dim=1
                    )
            if is_last_slice:
                hidden_states_select = torch.concat(
                    (hidden_states_select, self.task_id_emb[None, :, :]), dim=1
                )
        else:
            hidden_states_select = self.task_id_emb[None, :, :]
        ## concat hidden_states_vllm_his
        if hidden_states_vllm_his is not None:
            hidden_states_select = torch.concat(
                (hidden_states_vllm_his, hidden_states_select.squeeze(0)), dim=0
            )
        else:
            hidden_states_select = hidden_states_select.squeeze(0)

        from vllm import SamplingParams

        max_tokens = (
            kwargs.get("max_length_codec_duration_s", 120) * 25
            if is_last_slice
            else self.text_codec_mix_ratio[1] + 1
        )
        sampling_params = SamplingParams(
            top_k=sampling, stop_token_ids=eos_token, max_tokens=max_tokens
        )
        codec_generator = vllm_codec_engine.generate(
            {"prompt_embeds": hidden_states_select.to(dtype).to(device)},
            sampling_params,
            request_id,
        )
        return codec_generator, hidden_states_select

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
        old_overlap = cached[:, -hop_len:]
        sin_wind = torch.blackman_window(2 * hop_len)[None, :].to(old_overlap)
        cf_overlap = (
                old_overlap * sin_wind[:, -hop_len:] + current[:, :hop_len] * sin_wind[:, :hop_len]
        )
        current[:, :hop_len] = cf_overlap

        return current

    def streaming_flow_voc(
            self,
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            prompt_inter,
            prompt_audio,
            fm_voc_states,
            is_last,
    ):
        cached_token = fm_voc_states.get("cached_token", None)
        cached_feat = fm_voc_states.get("cached_feat", None)
        cached_wav = fm_voc_states.get("cached_wav", None)
        if cached_token is not None:
            tokens = torch.cat([cached_token, tokens], dim=1)
            token_lens = token_lens + cached_token.shape[1]

        diff_out = self.fm_model.inference(
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            diff_steps=self.fm_diff_steps,
            temperature=self.fm_temperature,
            prompt=dict(prompt_text=prompt_inter, prompt_audio=prompt_audio),
        )
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
        if not is_last:
            feat = feat[:, :, : -self.feat_reback]

        wav = self.vocoder.inference(feat.transpose(1, 2)).unsqueeze(1)
        # remove padded history wav
        pad_wav_len = int(padded_feat_len * (self.sample_rate / self.mel_frame_rate))
        wav = wav[:, :, pad_wav_len:]
        if cached_wav is not None:
            wav = self.cross_fade(
                cached_wav.squeeze(1), wav.squeeze(1), self.vocoder_hop
            ).unsqueeze(1)

        # play wav without tail hop length
        cur_wav = wav[:, :, : -self.vocoder_hop] if not is_last else wav

        # update caches
        cached_token = tokens
        if cached_feat is not None:
            cached_feat_len = cached_feat.shape[2]
            if is_last:
                cached_feat = torch.cat([cached_feat, diff_out[:, :, cached_feat_len:]], dim=2)
            else:
                cached_feat = torch.cat(
                    [
                        cached_feat,
                        diff_out[:, :, cached_feat_len: diff_out.shape[-1] - self.feat_reback],
                    ],
                    dim=2,
                )
        else:
            if is_last:
                cached_feat = diff_out[:, :, :]
            else:
                cached_feat = diff_out[:, :, : diff_out.shape[-1] - self.feat_reback]
        if cached_wav is not None:
            cached_wav = torch.cat([cached_wav[:, :, : -self.vocoder_hop], wav], dim=2)
        else:
            cached_wav = wav

        return (cur_wav, diff_out[:, :, : diff_out.shape[-1] - self.feat_reback]), dict(
            cached_token=cached_token,
            cached_feat=cached_feat,
            cached_wav=cached_wav,
        )


if __name__=="__main__":
    device = 'cuda:0'
    speech = torch.rand(16, 16000 * 60).to(device)
    speech_lengths = torch.randint(size=(16,), low=16000, high=16000 * 60).to(device)
    speech_token_extractor = SpeechTokenExtractionONNX()
    speech_token, speech_token_lengths = speech_token_extractor(speech, speech_lengths)
    print(speech_token, speech_token_lengths)
