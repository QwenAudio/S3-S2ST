import logging
import os
import gc
import os.path
import re
import regex
import torch
import random
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video, get_oss_object_size
import math
import numpy as np
import random
import subprocess
import tempfile
import json
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import uuid
from collections import defaultdict, OrderedDict
import soundfile as sf
try:
    import usjon as json
except:
    import json


def _rewrite_audio_path_for_oss_cache(path_cur, use_oss_cache=False):
    if use_oss_cache:
        if path_cur.startswith("/mnt/data"):
            path_cur = path_cur.replace("/mnt/data", "/nfs_cache", 1)
        elif path_cur.startswith("/mnt"):
            path_cur = path_cur.replace("/mnt", "/nfs_cache", 1)
        path_cur = path_cur.replace("oss://example-storage-3", "/nfs_cache", 1)
        path_cur = path_cur.replace("oss://example-storage-4", "/oss-example-storage-4", 1)
        path_cur = path_cur.replace("oss://example-storage-4", "/nls_spider_sh_cache", 1)
        path_cur = path_cur.replace("oss://example-storage-1", "/fun_resource_sh_cache", 1)
        path_cur = path_cur.replace("oss://example-storage-4", "/oss-example-storage-4", 1)
    else:
        path_cur = path_cur.replace("oss://example-storage-3/", "oss://example-storage-1/")
    return path_cur


def _log_slow_audio_batch(load_results, wall_seconds, audio_threads):
    """Log cumulative audio latency that is invisible to per-file slow logs."""
    try:
        threshold = float(os.environ.get("FUNASR_SLOW_AUDIO_BATCH_LOG_THRESHOLD_S", "3.0"))
    except (TypeError, ValueError):
        threshold = 3.0
    if wall_seconds < threshold or not load_results:
        return

    elapsed = [result[1] for result in load_results]
    slowest = max(load_results, key=lambda result: result[1])
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
    failures = sum(result[0] is None for result in load_results)
    logging.warning(
        "[SLOW_AUDIO_BATCH] wall_s=%.3f count=%d sum_file_s=%.3f "
        "avg_file_s=%.3f max_file_s=%.3f failures=%d threads=%d "
        "rank=%s host=%s pid=%d slowest_path=%s",
        wall_seconds,
        len(load_results),
        sum(elapsed),
        sum(elapsed) / len(elapsed),
        slowest[1],
        failures,
        audio_threads,
        rank,
        os.uname().nodename,
        os.getpid(),
        slowest[2],
    )


try:
    from wetext import Normalizer

    normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=True)
except ImportError:
    print("Please install wetext: pip install wetext)")


@tables.register("dataset_classes", "OpenAIDataset")
class OpenAIDataset(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1024)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.audio_adaptor_downsample_rate = kwargs.get("audio_adaptor_downsample_rate", 2)
        self.audio_encoder_downsample_rate = kwargs.get("audio_encoder_downsample_rate", 4)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb;
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg = [], [], [], [], [], []

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):

                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_mask_i = []
                fbank_beg_i = []
                fbank_lens_i = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue
                            if self.audio_encoder_downsample_rate == 4:
                                olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                                olens = 1 + (olens - 3 + 2 * 1) // 2
                            elif self.audio_encoder_downsample_rate == 1:
                                olens = speech_lengths[0].item()

                            sub_token_len = (olens - 1) // self.audio_adaptor_downsample_rate + 1
                            sub_token = [0] * sub_token_len
                            fbank_beg_i = [len(source_ids)]
                            source_ids += sub_token
                            fbank_mask_i += [1] * len(sub_token)

                if badcase_flag:
                    continue
                source_mask = [-100] * len(source_ids)
                target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                fbank_mask += fbank_mask_i
                fbank_beg.append(fbank_beg_i)

            if len(input_ids) > self.max_token_length:
                logging.info(
                    f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                )
                badcase_flag = True
            if badcase_flag:
                continue
            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            fbank = speech[0, :, :]
            fbank_lens = speech_lengths
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)

            output = {
                "speech": fbank,
                "speech_lengths": fbank_lens,
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "OpenAIDatasetMultiTurn")
@tables.register("dataset_classes", "MinMo_S2T")
class MinMo_S2T(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.max_target_length = kwargs.get("max_target_length", 1024)
        self.do_think = kwargs.get("do_think", True)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                if not self.do_think:
                    source_input += "<think>\n\n</think>\n\n"

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                # fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            # fbank_mask_i += [1] * len(fake_token)

                if self.preprocessor_text is not None:
                    has_repeat_tokens = self.preprocessor_text(target_out)
                else:
                    has_repeat_tokens = False
                if has_repeat_tokens:
                    badcase_flag = True
                    logging.info(f"item has_repeat_tokens: {item}")
                    break

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                # source_mask = [-100] * len(source_ids)
                # target_out = f"{target_out}<|im_end|>"
                target_ids = []
                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"

                    splits = self.pattern.split(target_out)
                    sub_token = []
                    for k, sub_str in enumerate(splits):
                        if sub_str is None or len(sub_str) < 1:
                            continue
                        if not sub_str.startswith("<|startofspeech|>"):
                            sub_token = self.tokenizer.encode(sub_str)
                            target_ids += sub_token
                        else:
                            pass
                    # if target_out only have <|im_end|>
                    if len(target_ids) == 1:
                        badcase_flag = True
                if len(target_ids) > self.max_target_length:
                    logging.info(
                        f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {item}"
                    )
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
            if badcase_flag:
                logging.warning(f"badcase_flag: {item}")
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            # fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs

@tables.register("dataset_classes", "BestRQ")
class BestRQ(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.max_target_length = kwargs.get("max_target_length", 1024)
        self.do_think = kwargs.get("do_think", True)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                if not self.do_think:
                    source_input += "<think>\n\n</think>\n\n"

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                # fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            # fbank_mask_i += [1] * len(fake_token)

                if self.preprocessor_text is not None:
                    has_repeat_tokens = self.preprocessor_text(target_out)
                else:
                    has_repeat_tokens = False
                if has_repeat_tokens:
                    badcase_flag = True
                    logging.info(f"item has_repeat_tokens: {item}")
                    break

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                # source_mask = [-100] * len(source_ids)
                # target_out = f"{target_out}<|im_end|>"
                target_ids = []
                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"

                    splits = self.pattern.split(target_out)
                    sub_token = []
                    for k, sub_str in enumerate(splits):
                        if sub_str is None or len(sub_str) < 1:
                            continue
                        if not sub_str.startswith("<|startofspeech|>"):
                            if sub_str.startswith("<|"):
                                sub_str = sub_str[6:]
                            sub_token = self.tokenizer.encode(sub_str)
                            target_ids += sub_token
                        else:
                            pass
                    # if target_out only have <|im_end|>
                    if len(target_ids) == 1:
                        badcase_flag = True
                if len(target_ids) > self.max_target_length:
                    logging.info(
                        f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {item}"
                    )
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
            if badcase_flag:
                logging.warning(f"badcase_flag: {item}")
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            # fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMoSense")
class MinMoSense(torch.utils.data.Dataset):
    """
    MinMoSense dataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        preprocessor_noise = kwargs.get("preprocessor_noise", None)
        if preprocessor_noise:
            preprocessor_noise_class = tables.preprocessor_classes.get(preprocessor_noise)
            preprocessor_noise = preprocessor_noise_class(**kwargs.get("preprocessor_noise_conf"))
        self.preprocessor_noise = preprocessor_noise

        prompt_classes_text = kwargs.get("prompt_classes", None)
        if prompt_classes_text is not None:
            prompt_classes = tables.prompt_classes.get(prompt_classes_text)
            prompt_classes = prompt_classes(**kwargs.get("prompt_conf"))
        else:
            prompt_classes = None
        self.prompt_classes = prompt_classes

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.max_target_length = kwargs.get("max_target_length", 1024)
        self.do_think = kwargs.get("do_think", True)
        self.sys_prompt = kwargs.get("sys_prompt", True)

        # used for dynamic output alignment
        self.use_dynamic_output_ratio = kwargs.get("use_dynamic_output_ratio", 0.0)
        self.min_output_mask_token_len = kwargs.get("min_mask_token_len", 1)
        self.min_output_non_mask_token_len = kwargs.get("min_non_mask_token_len", 6)  # [eos]

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def get_random_user_prompt(self, item, user_prompt):
        tasks = ["语音转写：", "Speech transcription:"]
        language = item.get("language", None)
        # LID in distill data is fake
        language = None
        if language is not None:
            if language.lower() == "zh":
                tasks.append("语音转写成中文：")
                tasks.append("Transcribe speech into Chinese:")
            elif language.lower() == "en":
                tasks.append("语音转写成英文：")
                tasks.append("Transcribe speech into English:")
        if len(tasks) == 2:
            task = random.choice(tasks)
        elif len(tasks) == 4:
            task = random.choices(tasks, weights=[0.4, 0.4, 0.1, 0.1])[0]
        if "语音转写：<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("语音转写：<|startofspeech|>", task + "<|startofspeech|>")
        elif "Speech transcription:<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("Speech transcription:<|startofspeech|>", task + "<|startofspeech|>")
        return user_prompt

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            if idx > 0:
                logging.info(f"retry: {idx}")
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            is_noised = item.get("noised", False)
            if "duration" in item.keys():
                duration = item["duration"]
            else:
                duration = [[0, None],]*len(user)
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break
                # process user_prompt
                # user_prompt = self.get_random_user_prompt(item, user_prompt)

                # chose context_prompt by language
                if self.prompt_classes is not None:
                    asr_prompt = user_prompt.split("<|startofspeech|>")[0]
                    language = self.prompt_classes.detect_language(asr_prompt)
                    user_prompt_all_context = self.prompt_classes.get_prompt(item, language)
                else:
                    user_prompt_all_context = ""

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not self.sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                if not self.do_think:
                    source_input += "<think>\n\n</think>\n\n"
                # print(f"source_input: {source_input}")
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                # fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs, duration=duration[i])
                                if self.preprocessor_noise is not None and not is_noised:
                                    try:
                                        data_src = self.preprocessor_noise(data_src.numpy())
                                    except Exception as e:
                                        logging.error(f"Generate noise audio failed: {e}")

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}\n{item}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            # fbank_mask_i += [1] * len(fake_token)

                if self.preprocessor_text is not None:
                    has_repeat_tokens = self.preprocessor_text(target_out)
                else:
                    has_repeat_tokens = False
                if has_repeat_tokens:
                    badcase_flag = True
                    logging.info(f"item has_repeat_tokens: {item}")
                    break

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                # source_mask = [-100] * len(source_ids)
                # target_out = f"{target_out}<|im_end|>"
                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)

                if len(target_ids) > self.max_target_length:
                    logging.info(
                        f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {item}"
                    )
                #  simulate prev-token fixed output
                if np.random.rand() < self.use_dynamic_output_ratio:
                    max_len = len(target_ids)
                    min_output_mask_token_len = min(self.min_output_mask_token_len, max_len)
                    min_output_non_mask_token_len = min(self.min_output_non_mask_token_len, max_len)
                    if max_len - min_output_non_mask_token_len > min_output_mask_token_len:
                        end_index = np.random.randint(min_output_mask_token_len,
                                                      max_len - min_output_non_mask_token_len)
                    else:
                        end_index = max_len - min_output_non_mask_token_len
                    if end_index > 0:
                        target_ids[:end_index] = [-100] * end_index

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            # fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMoSense_CTC")
class MinMoSense_CTC(torch.utils.data.Dataset):
    """
    MinMoSense_CTC dataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        prompt_classes_text = kwargs.get("prompt_classes", None)
        if prompt_classes_text is not None:
            prompt_classes = tables.prompt_classes.get(prompt_classes_text)
            prompt_classes = prompt_classes(**kwargs.get("prompt_conf"))
        else:
            prompt_classes = None
        self.prompt_classes = prompt_classes

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 1)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.max_target_length = kwargs.get("max_target_length", 1024)
        self.do_think = kwargs.get("do_think", True)

        # used for ctc_labels encoding, whisper tokenizer
        ctc_tokenizer = kwargs.get("ctc_tokenizer", None)
        ctc_tokenizer_conf = kwargs.get("ctc_tokenizer_conf", None)
        if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
            ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
            ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
        elif ctc_tokenizer is None and self.tokenizer is not None:
            ctc_tokenizer = self.tokenizer
            logging.info(f"ctc_tokenizer is set to self.tokenizer")
        else:
            logging.info(f"ctc_tokenizer and tokenizer are not set, please set one of them")
        assert ctc_tokenizer is not None, f"ctc_tokenizer must be set"
        self.ctc_tokenizer = ctc_tokenizer
        self.ctc_target_normalize = kwargs.get("ctc_target_normalize", True)

        # used for dynamic output alignment
        self.use_dynamic_output_ratio = kwargs.get("use_dynamic_output_ratio", 0.0)
        self.min_output_mask_token_len = kwargs.get("min_mask_token_len", 1)
        self.min_output_non_mask_token_len = kwargs.get("min_non_mask_token_len", 6)  # [eos]

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def get_random_user_prompt(self, item, user_prompt):
        tasks = ["语音转写：", "Speech transcription:"]
        language = item.get("language", None)
        # LID in distill data is fake
        language = None
        if language is not None:
            if language.lower() == "zh":
                tasks.append("语音转写成中文：")
                tasks.append("Transcribe speech into Chinese:")
            elif language.lower() == "en":
                tasks.append("语音转写成英文：")
                tasks.append("Transcribe speech into English:")
        if len(tasks) == 2:
            task = random.choice(tasks)
        elif len(tasks) == 4:
            task = random.choices(tasks, weights=[0.4, 0.4, 0.1, 0.1])[0]
        if "语音转写：<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("语音转写：<|startofspeech|>", task + "<|startofspeech|>")
        elif "Speech transcription:<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("Speech transcription:<|startofspeech|>", task + "<|startofspeech|>")
        return user_prompt

    def text_normalize(self, text):
        norm_text = normalizer.normalize(text)
        clean_text = regex.sub(r'\p{P}+', '', norm_text)
        return clean_text

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                # user_prompt = self.get_random_user_prompt(item, user_prompt)

                # chose context_prompt by language
                if self.prompt_classes is not None:
                    asr_prompt = user_prompt.split("<|startofspeech|>")[0]
                    language = self.prompt_classes.detect_language(asr_prompt)
                    user_prompt_all_context = self.prompt_classes.get_prompt(item, language)
                else:
                    user_prompt_all_context = ""

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                if not self.do_think:
                    source_input += "<think>\n\n</think>\n\n"

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                # fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs,
                                                                       data_type=self.data_type)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True

                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            # fbank_mask_i += [1] * len(fake_token)

                if self.preprocessor_text is not None:
                    has_repeat_tokens = self.preprocessor_text(target_out)
                else:
                    has_repeat_tokens = False
                if has_repeat_tokens:
                    badcase_flag = True
                    logging.info(f"item has_repeat_tokens: {item}")
                    break

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                # source_mask = [-100] * len(source_ids)
                # target_out = f"{target_out}<|im_end|>"
                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)

                if len(target_ids) > self.max_target_length:
                    logging.info(
                        f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {item}"
                    )
                if self.ctc_target_normalize:
                    target_out = self.text_normalize(target_out.replace('<|im_end|>', ''))
                else:
                    target_out = target_out.replace('<|im_end|>', '')
                ctc_target_ids = self.ctc_tokenizer.encode(target_out)
                ctc_labels = ctc_target_ids
                ctc_len = len(ctc_labels)

                if np.random.rand() < self.use_dynamic_output_ratio:
                    max_len = len(target_ids)
                    min_output_mask_token_len = min(self.min_output_mask_token_len, max_len)
                    min_output_non_mask_token_len = min(self.min_output_non_mask_token_len, max_len)
                    if max_len - min_output_non_mask_token_len > min_output_mask_token_len:
                        end_index = np.random.randint(min_output_mask_token_len,
                                                      max_len - min_output_non_mask_token_len)
                    else:
                        end_index = max_len - min_output_non_mask_token_len
                    if end_index > 0:
                        target_ids[:end_index] = [-100] * end_index

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
            ctc_labels = torch.tensor(ctc_labels, dtype=torch.int64)
            ctc_labels_len = torch.tensor([ctc_len], dtype=torch.int32)
            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            # fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
                "ctc_labels": ctc_labels,
                "ctc_labels_len": ctc_labels_len,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if fbank_lens[0] // 4 < 2 * ctc_len + 1:
                logging.warning(
                    f"speech_lengths // 4 < 2 * ctc_len + 1: {fbank_lens[0] // 4} < {2 * ctc_len + 1}, {item}"
                )
                continue
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


def remove_punctuation(text):
    # 定义中英文标点符号的正则表达式
    # pattern = r'[\u3000-\u303f\u2010-\u2fff！，。？、；：“”‘’《》【】〖〗～·…——（）〈〉\\\'\"{}<>]'
    pattern = r'[!"#$%&()*+,-./:;<=>?@[\$$^_{|}~！，。？、；：“”‘’《》【】〖〗～·…——（）〈〉\\\'\"{}<>]'

    # 使用 re.sub() 方法替换标点符号为空
    result = re.sub(pattern, '', text)

    return result

@tables.register("dataset_classes", "MinMoSense_CTC_aky_gcf")
class MinMoSense_CTC_aky_gcf(torch.utils.data.Dataset):
    """
    MinMoSense_CTC dataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        prompt_classes_text = kwargs.get("prompt_classes", None)
        if prompt_classes_text is not None:
            prompt_classes = tables.prompt_classes.get(prompt_classes_text)
            prompt_classes = prompt_classes(**kwargs.get("prompt_conf"))
        else:
            prompt_classes = None
        self.prompt_classes = prompt_classes

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 1)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.max_target_length = kwargs.get("max_target_length", 1024)
        self.min_source_length = kwargs.get("min_source_length", 0)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.do_think = kwargs.get("do_think", True)

        # used for ctc_labels encoding, whisper tokenizer
        ctc_tokenizer = kwargs.get("ctc_tokenizer", None)
        ctc_tokenizer_conf = kwargs.get("ctc_tokenizer_conf", None)
        if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
            ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
            ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
        elif ctc_tokenizer is None and self.tokenizer is not None:
            ctc_tokenizer = self.tokenizer
            logging.info(f"ctc_tokenizer is set to self.tokenizer")
        else:
            logging.info(f"ctc_tokenizer and tokenizer are not set, please set one of them")
        assert ctc_tokenizer is not None, f"ctc_tokenizer must be set"
        self.ctc_tokenizer = ctc_tokenizer
        try:
            self.sos_ids = self.ctc_tokenizer.encode(self.sos, allowed_special="all") if isinstance(self.eos,
                                                                                                    str) else [self.sos]
            self.eos_ids = self.ctc_tokenizer.encode(self.eos, allowed_special="all") if isinstance(self.eos,
                                                                                                    str) else [self.eos]
        except:
            self.sos = "<|im_start|>"
            self.eos = "<|im_end|>"
            self.sos_ids = self.ctc_tokenizer.encode(self.sos) if isinstance(self.eos, str) else [self.sos]
            self.eos_ids = self.ctc_tokenizer.encode(self.eos) if isinstance(self.eos, str) else [self.eos]

            # used for dynamic output alignment
        self.use_dynamic_output_ratio = kwargs.get("use_dynamic_output_ratio", 0.0)
        self.min_output_mask_token_len = kwargs.get("min_mask_token_len", 1)
        self.min_output_non_mask_token_len = kwargs.get("min_non_mask_token_len", 6)  # [eos]

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            if self.prompt_classes is not None:
                user_prompt_all_context = self.prompt_classes.get_prompt(item)
            else:
                user_prompt_all_context = ""

            # long_context_user_prompt, one_pass_result_prompt, hotwords_user_prompt = "", "", ""
            # long_context_prefix_en = "Historical transcription result: "
            # long_context_prefix_zh = "历史转写结果："
            # if "hist_context" in item:
            #     long_context_text =  item["hist_context"]
            #     long_context_user_prompt_en  = long_context_prefix_en + long_context_text
            #     long_context_user_prompt_zh = long_context_prefix_zh + long_context_text
            # else:
            #     long_context_user_prompt_en = ""
            #     long_context_user_prompt_zh = ""

            # if np.random.rand() < 0.5:
            #     long_context_user_prompt = long_context_user_prompt_zh
            #     user_prompt_all_context = f'请结合下面提供的上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n{long_context_user_prompt}\n{one_pass_result_prompt}\n{hotwords_user_prompt}\n'
            # else:
            #     long_context_user_prompt = long_context_user_prompt_en
            #     user_prompt_all_context = f'Please combine the context information provided below to complete the speech transcription task more accurately. If there is no relevant information, we will leave it blank.\n{long_context_user_prompt}\n{one_pass_result_prompt}\n{hotwords_user_prompt}\n'

            # # 30% for NULL one pass results
            # if np.random.rand() < self.one_pass_result_prompt_ratio and "one_pass_asr_result" in item:
            #     one_pass_asr_result = item["one_pass_asr_result"]
            # else:
            #     one_pass_asr_result = ""

            # one_pass_user_prompt = f"结合语音的一遍解码结果完成语音转写任务。一遍解码结果为：\"{one_pass_asr_result}\"\n"

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                if not self.do_think:
                    source_input += "<think>\n\n</think>\n\n"

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                # fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs,
                                                                       data_type=self.data_type)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True

                                if speech_lengths < self.min_source_length:
                                    logging.info(
                                        f"speech_lengths < min_source_length: {speech_lengths}<{self.min_source_length}, {item}"
                                    )
                                    badcase_flag = True

                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            # fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                # source_mask = [-100] * len(source_ids)
                # target_out = f"{target_out}<|im_end|>"
                if target_out is not None and any(
                    isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out, allowed_special="all")

                # if len(target_ids) > self.max_target_length:
                #     logging.info(
                #         f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {item}"
                #     )
                # ctc_target_ids = self.ctc_tokenizer.encode(normalize_text_one_line(target_out.replace('<|im_end|>', '')))
                # ctc_target_ids = self.ctc_tokenizer.encode(remove_punctuation(target_out.replace('<|im_end|>', '').lower()))
                try:
                    text_ctc = target_out.replace('<|im_end|>', '').lower()
                    lid, text_raw = text_ctc.split('|>', maxsplit=1)
                    lid = lid + '|>'
                except:
                    text_raw = target_out.replace('<|im_end|>', '').lower()
                    lid = '<|zh|>'
                    badcase_flag = True

                ctc_target_ids = self.ctc_tokenizer.encode('<|ASR|>' + lid + remove_punctuation(text_raw), allowed_special="all")
                ctc_labels = ctc_target_ids
                ctc_len = len(ctc_labels)

                att_labels = self.sos_ids + ctc_labels + self.eos_ids
                att_len = len(att_labels)

                if att_len > self.max_target_length:
                    logging.info(
                        f"text_lengths > max_target_length: {att_len}>{self.max_target_length}, {item}"
                    )
                    badcase_flag = True

                if att_len < self.min_target_length:
                    logging.info(
                        f"text_lengths < min_target_length: {att_len}<{self.min_target_length}, {item}"
                    )
                    badcase_flag = True

                # if np.random.rand() < self.use_dynamic_output_ratio:
                #     max_len = len(target_ids)
                #     min_output_mask_token_len = min(self.min_output_mask_token_len, max_len)
                #     min_output_non_mask_token_len = min(self.min_output_non_mask_token_len, max_len)
                #     if max_len - min_output_non_mask_token_len > min_output_mask_token_len:
                #         end_index = np.random.randint(min_output_mask_token_len,
                #                                       max_len - min_output_non_mask_token_len)
                #     else:
                #         end_index = max_len - min_output_non_mask_token_len
                #     if end_index > 0:
                #         target_ids[:end_index] = [-100] * end_index

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)

                if self.preprocessor_text is not None:
                    has_repeat_tokens = self.preprocessor_text(target_out)
                else:
                    has_repeat_tokens = False
                if has_repeat_tokens:
                    badcase_flag = True
                    logging.info(f"item has_repeat_tokens: {item}")
                    break
            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
            ctc_labels = torch.tensor(ctc_labels, dtype=torch.int64)
            ctc_labels_len = torch.tensor([ctc_len], dtype=torch.int32)

            att_labels = torch.tensor(att_labels, dtype=torch.int64)
            att_labels_len = torch.tensor([att_len], dtype=torch.int32)

            target_mask = ([0] + [1] * (ctc_len) + [1])  # [sos + text + eos]
            target_mask_lengths = len(target_mask)
            target_mask = torch.tensor(target_mask, dtype=torch.float32)
            target_mask_lengths = torch.tensor([target_mask_lengths], dtype=torch.int32)

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            # fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
                "ctc_labels": ctc_labels,
                "ctc_labels_len": ctc_labels_len,
                "att_labels": att_labels,
                "att_labels_len": att_labels_len,
                "target_mask": target_mask,
                "target_mask_lengths": target_mask_lengths,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if fbank_lens[0] // 4 < 2 * ctc_len:
                logging.warning(
                    f"speech_lengths // 4 < 2 * ctc_len + 1: {fbank_lens[0] // 4} < {2 * ctc_len + 1}, {item}"
                )
                continue
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMo_S2S_Interleave")
class MinMo_S2S_Interleave(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, audio_tokens, audio_tokens_lens, \
                pixel_values_lst, image_grid_thws, pixel_values_videos_lst, \
                video_grid_thws, video_sencond_per_grids, feature_attention_masks, input_features_lst, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                splits = self.pattern.split(source_input)
                source_ids = []
                # audio
                speech = []
                speech_lengths = []
                speech_token = []
                speech_token_lens = []
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
                               ["<|startofspeech|>", "<|startofaudiotoken|>", "<|startofimage|>", "<|startofvideo|>"]):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                    elif sub_str.startswith("<|startofspeech|>"):
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            # fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token_len_i = olens
                            fake_token = [0] * fake_token_len_i
                            featrue_beg_i = len(source_ids)
                            feature_type_turn = 1
                            source_ids += fake_token
                    elif sub_str.startswith("<|startofaudiotoken|>"):
                        sub_str = sub_str.replace("<|startofaudiotoken|>", "").replace(
                            "<|endofaudiotoken|>", ""
                        )
                        if sub_str.startswith("!"):
                            # Add offset
                            sub_audio_token = np.load(sub_str[1:]) + self.tokenizer.convert_tokens_to_ids("<|c0|>")
                        else:
                            sub_audio_token = self.tokenizer.encode(sub_str)
                        speech_token = torch.tensor(sub_audio_token, dtype=torch.int64)
                        speech_token_lens = torch.tensor([len(sub_audio_token)], dtype=torch.int64)

                    if badcase_flag:
                        continue
                    if featrue_beg_i > 0:
                        feature_beg += [featrue_beg_i + len(input_ids)]
                        fake_token_len += [fake_token_len_i]
                        feature_type += [feature_type_turn]

                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    # target_out = f"{target_out}<|im_end|>"
                    splits = self.pattern.split(target_out)

                    sub_token = []
                    audio_token = []

                    for k, sub_str in enumerate(splits):
                        if sub_str is None or sub_str == "":
                            continue
                        if len(sub_str) < 1:
                            continue
                        if not sub_str.startswith("<|startofaudiotoken|>"):
                            sub_token = self.tokenizer.encode(sub_str)
                        else:
                            sub_str = sub_str.replace("<|startofaudiotoken|>", "").replace(
                                "<|endofaudiotoken|>", ""
                            )
                            if sub_str.startswith("!"):
                                # Add offset
                                audio_token = np.load(sub_str[1:]) + self.tokenizer.convert_tokens_to_ids("<|c0|>")
                                audio_token = audio_token.tolist()
                            else:
                                audio_token = self.tokenizer.encode(sub_str)
                            # add audio delay
                            if len(audio_token) > 0:
                                audio_token = self.audio_delay_lst + audio_token
                    # mixup_text_audio to target_ids
                    target_ids = self.mixup_text_audio(sub_token, audio_token, self.text_task_id, self.audio_task_id)
                    target_ids.append(self.end_id)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                # audio image video
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
                if len(speech_token) > 0:
                    audio_tokens.append(speech_token)
                    audio_tokens_lens.append(speech_token_lens)
                if len(pixel_values_turn) > 0:
                    pixel_values_lst.extend(pixel_values_turn)
                    image_grid_thws.extend(image_grid_thw_turn)
                if len(pixel_values_videos_turn) > 0:
                    pixel_values_videos_lst.extend(pixel_values_videos_turn)
                    video_grid_thws.extend(video_grid_thw_turn)
                    # video_sencond_per_grids.extend(video_sencond_per_grid_turn)
                    # feature_attention_masks.extend(feature_attention_mask_turn)
                    # input_features_lst.extend(input_features_turn)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
            feature_type = torch.tensor(feature_type, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "feature_beg": feature_beg,
                "feature_type": feature_type,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(audio_tokens) > 0:
                output["audio_tokens"] = audio_tokens
                output["audio_tokens_lens"] = audio_tokens_lens
            if len(pixel_values_lst) > 0:
                output["pixel_values"] = pixel_values_lst
                output["image_grid_thw"] = image_grid_thws
            if len(pixel_values_videos_turn) > 0:
                output["pixel_values_videos"] = pixel_values_videos_lst
                output["video_grid_thw"] = video_grid_thws
                # output["video_sencond_per_grid"] = video_sencond_per_grids
                # output["feature_attention_mask"] = feature_attention_masks
                # output["input_features"] = input_features_lst
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def mixup_text_audio(self, text_token, audio_token, text_task_id, audio_task_id):
        mixed_tokens = []
        text_index = 0
        audio_index = 0
        text_length = len(text_token)
        audio_length = len(audio_token)
        text_ratio, audio_ratio = self.text_codec_mix_ratio

        while text_index < text_length and audio_index < audio_length:
            # Add text tokens according to text_ratio
            mixed_tokens.append(text_task_id)
            for _ in range(text_ratio):
                if text_index < text_length:
                    mixed_tokens.append(text_token[text_index])
                    text_index += 1
                else:
                    break

            # Add audio tokens according to audio_ratio
            mixed_tokens.append(audio_task_id)
            for _ in range(audio_ratio):
                if audio_index < audio_length:
                    mixed_tokens.append(audio_token[audio_index])
                    audio_index += 1
                else:
                    break

        # Append remaining text tokens if any
        if text_index < text_length:
            mixed_tokens.append(text_task_id)
            mixed_tokens.extend(text_token[text_index:])

        # Append remaining audio tokens if any
        if audio_index < audio_length:
            mixed_tokens.append(audio_task_id)
            mixed_tokens.extend(audio_token[audio_index:])

        return mixed_tokens

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


def load_audio_with_duration(args):
    source_wav, duration, fs, turn_idx = args
    if source_wav is None:
        return None, turn_idx
    # return torch.zeros(int((duration[1] - duration[0]) * fs), dtype=torch.float32), turn_idx
    data_src = load_audio_text_image_video(source_wav, fs=fs, duration=duration)
    return data_src, turn_idx

def convert_to_wav(args):
    from funasr.datasets.minmo_datasets.cache_utils.pre_cache_audio_data import ffmpeg_convert_to_wav
    input_file, output_file, sample_rate, channels = args
    return ffmpeg_convert_to_wav(input_file, output_file, sample_rate=sample_rate, channels=channels)


@tables.register("dataset_classes", "MinMo_S2S_Interleave_Pretrain")
class MinMo_S2S_Interleave_Pretrain(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            slice_frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        kwargs['audio_pt_modes_config_name'] = kwargs.get("audio_pt_modes_config_name", "dual_channel")
        self.index_ds = index_ds_class(path, tokenizer=tokenizer, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        else:
            self.preprocessor_speech_conf = None
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.slice_frontend = slice_frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.slice_fs = self.fs if slice_frontend is None else slice_frontend.fs
        assert self.fs == self.slice_fs, f"frontend fs: {self.fs} != slice_frontend fs: {self.slice_fs}"
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        self.slice_permute = self.permute
        if self.slice_frontend is not None:
            self.slice_permute = isinstance(self.slice_frontend, WhisperFrontend)

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.min_token_length = kwargs.get("min_token_length", 1024)
        self.max_slice_duration = kwargs.get("max_merged_slice_duration_sec", 5.0)
        if not isinstance(self.max_slice_duration, float):
            print(f'max_slice_duration range: {self.max_slice_duration}')
            self.max_slice_duration = max(self.max_slice_duration[0], self.max_slice_duration[1])
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")

        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        logging.info(f"init continuous_in: {self.continuous_in}")

        # self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.speech_token_id_offset = 0 # for channels model, no offset needed
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        print(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")

        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")

        # pt mode
        self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-text|>") # unimodal-text
        self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-audio|>") # unimodal-audio
        self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-audio-to-text|>") # mapping-audio-to-text
        self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-text-to-audio|>") # mapping-text-to-audio 
        self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic|>") # interleave-audio-to-semantic 
        self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-text|>") # interleave-audio-to-text
        self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic-and-text|>") # interleave-audio-to-semantic-and-text
        assert self.pt_mode_unimodal_text_token_id is not None

        self.text_vocab_offset = self.tokenizer.vocab_size
        assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print (f'audio code book size: {self.audio_code_book_size}')
        self.mask_token_id = -60

        self.blank_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)

        self.add_task_id = kwargs.get("add_task_id", False)
        self.add_modalidy_id = kwargs.get("add_modalidy_id", False)

    def get_audio_token_id(self, audio_token):
        if isinstance(audio_token, int):
            return audio_token - self.text_vocab_offset + self.audio_code_book_size
        elif isinstance(audio_token, str):
            return self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids(audio_token))
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")


    def get_pt_mode_token_id(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def _extract_audio_segment(self, source_wav, start_time, end_time):
        """使用ffmpeg切分音频文件"""
        try:
            start_time_ffmpeg = time.time()
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)

            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-i', source_wav, '-ss', str(start_time),
                '-t', str(duration), '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logging.error(f"ffmpeg failed: {result.stderr}")
                return None
            end_time_ffmpeg = time.time()
            # print (f"extract_audio_segment time: {end_time_ffmpeg - start_time_ffmpeg}")
            return temp_path
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return None

    def _process_audio_cache(self, segments_dict):
        """
        处理音频缓存逻辑
        
        Args:
            segments_dict: 音频片段字典
            
        Returns:
            tuple: (cache_hit, segment_cache, local_audio_dir)
                - cache_hit: 是否命中缓存
                - segment_cache: 片段缓存字典 {segment_id: local_wav_path}
                - local_audio_dir: 本地音频目录
        """
        segment_cache = {}  # key: segment_id, value: local_wav_path
        cache_hit = False
        local_audio_dir = None
        
        # cache
        from funasr.datasets.minmo_datasets.cache_utils.pre_cache_audio_data import _sharded_dirs, _sha256_hex, _format_time_component, _is_file_intact
        assert self.audio_cache_type == "segments"  # only support segments cache now
        # bucket_and_rest = source_wav[len("oss://"):]
        bucket_name, object_name = 'example-storage-6', 'temp_obj'
        digest = _sha256_hex(object_name)
        remote_cache_dir = _sharded_dirs(self.audio_cache_dir, digest)
        local_audio_dir = os.path.join("/tmp/tmp_wav", uuid.uuid4().hex)

        expected_cache_files = set()
        cache_segment_names = set()
        segment_name2cache_segment_path = {}
        
        for segment_id, segment in segments_dict.items():
            segment_start_time = segment["start_time"]
            segment_end_time = segment["end_time"]
            if segment_start_time is None or segment_end_time is None:
                continue
            prefix = "audio_segment"
            segment_name = f"{prefix}-{segment_id}-{_format_time_component(segment_start_time)}-{_format_time_component(segment_end_time)}.{self.audio_cache_format}"
            if segment["wav_path"] is not None:
                cache_segment_path = segment['wav_path']
                assert os.path.exists(cache_segment_path), f"cache_segment_path not exists: {cache_segment_path}"
            else:
                print ('should not be logged, wav_path will always exist.')
                cache_segment_path = os.path.join(remote_cache_dir, segment_name)

            segment_name2cache_segment_path[segment_name] = cache_segment_path
            cache_segment_names.add((segment_id, segment_name))
            expected_cache_files.add(cache_segment_path)
            
        if all(_is_file_intact(p) for p in expected_cache_files):
            if self.audio_cache_format != "wav":
                # convert to wav and save to local cache
                os.makedirs(local_audio_dir, exist_ok=True)
                convert_args = []
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = segment_name2cache_segment_path[segment_name]
                    local_cache_segment_path = os.path.join(local_audio_dir, segment_name + '.wav')
                    segment_cache[segment_id] = local_cache_segment_path
                    convert_args.append((cache_segment_path, local_cache_segment_path, 16000, 1))
                with ThreadPoolExecutor(max_workers=20) as executor:
                    results = executor.map(convert_to_wav, convert_args)
                if all(results):
                    cache_hit = True
                else:
                    logging.warning(f"cache miss, not all files converted, drop item")
                    cache_hit = False
            else:
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = os.path.join(remote_cache_dir, segment_name)
                    segment_cache[segment_id] = cache_segment_path
                cache_hit = True
        else:
            logging.warning(f"cache miss, not all files are intact, drop item")
            cache_hit = False
        
        # preprocess audio
        if self.preprocessor_speech_conf is not None:
            probatility = self.preprocessor_speech_conf.get('probability', 1.0)
            all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
            use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
            batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            batch_use_6mic = batch_random_source.random() < use_6mic_probability
            conditions = self.preprocessor_speech_conf.get('conditions', [])

            def preprocess_segment(segment_id, segment_audio):
                # if the conditions met
                segment = segments_dict[segment_id]
                conditions_met = True
                for condition in conditions:
                    segment_key = condition.get('segment_key', None)
                    value_min = condition.get('value_min', None)
                    value_max = condition.get('value_max', None)
                    # if segment_key is None or value_min is None or value_max is None:
                    #     continue
                    segment_key_parts = segment_key.split('.')
                    current_item = segment
                    for key in segment_key_parts:
                        if key in current_item:
                            current_item = current_item[key]
                        else:
                            current_item = None
                            break
                    if current_item is None:
                        conditions_met = False
                        break
                    if value_min is not None and current_item < value_min:
                        conditions_met = False
                        break
                    if value_max is not None and current_item > value_max:
                        conditions_met = False
                        break
                if not conditions_met:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met conditions")
                    return segment_audio
                segment_unique_id = segment['unique_id']
                random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
                if random_source.random() > probatility:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met probatility")
                    return segment_audio
                # run the preprocessing
                # print(f"preprocess_segment segment_audio: {segment_audio.shape}")
                segment_audio = self._preprocess_audio(segment_audio, random_source=random_source, use_6mic=batch_use_6mic)
                return segment_audio

        # load and extract fbank
        for segment_id in segments_dict.keys():
            cache_segment_path = segment_cache[segment_id]
            duration = [0, None]
            segment_samples = load_audio_text_image_video(cache_segment_path, fs=self.fs, duration=duration)
            if self.preprocessor_speech_conf is not None:
                segment_samples = preprocess_segment(segment_id, segment_samples)
            segment_fbank, segment_fbank_length = extract_fbank(segment_samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
            segment_cache[segment_id] = {
                'path': cache_segment_path,
                'samples': segment_samples,
                'fbank': segment_fbank,
                'fbank_length': segment_fbank_length,
            }
        
        return cache_hit, segment_cache, local_audio_dir

    def _process_continuous_audio_loading(self, user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant=False):
        """
        提取 slice 对应音频的fbank
        """
        duration_list = []
        
        # test if local cache hit
        cache_duration_list = []
        data_src_list = []
        # cache miss
        if cache_hit:
            if self.audio_cache_type == "slices":
                # prefix = "audio_slice"
                raise NotImplementedError("slices cache is not supported")
            elif self.audio_cache_type == "segments":
                for i, (user_prompt, target_out) in enumerate(
                        zip(user, assistant)
                ):
                    for p_i, prompt in enumerate([user_prompt, target_out]):
                        if prompt is not None:
                            slice_start_time = prompt["start_time"]
                            slice_end_time = prompt["end_time"]
                            slice_mode = prompt["slice_mode"]
                            if 'c' not in slice_mode:
                                data_src_list.append((None, i))
                            else:
                                segment_id = prompt.get("segment_id", None)
                                if segment_id is not None:
                                    segment = segments_dict[segment_id]
                                    segment_start_time = segment["start_time"]
                                    segment_end_time = segment["end_time"]
                                    assert segment_start_time is not None and segment_end_time is not None
                                    offset = slice_start_time - segment_start_time
                                    assert offset >= 0 and offset <= segment_end_time - segment_start_time + 0.0001, f"offset: {offset} is not valid, segment time start: {segment_start_time}, segment time end: {segment_end_time}, slice time start: {slice_start_time}, slice time end: {slice_end_time}, segment: {segment}, slice: {prompt}"
                                    slice_duration = slice_end_time - slice_start_time
                                    assert slice_duration >= 0.01 and slice_duration <= 10000
                                    rel_start_time = offset
                                    rel_end_time = rel_start_time + slice_duration
                                    cache = segment_cache[segment_id]
                                    if isinstance(cache, dict):
                                        if self.slice_frontend is None:
                                            assert 'fbank' in cache and 'fbank_length' in cache, f"segment cache missing fbank key"
                                            segment_fbank = cache['fbank']
                                            segment_fbank_length = cache['fbank_length']
                                            slice_start_frame = int(rel_start_time * 1000 / self.frontend.frame_shift)
                                            slice_num_frames = int(slice_duration * 1000 / self.frontend.frame_shift)
                                            segment_num_frames = segment_fbank_length[0].item()
                                            # assert slice_start_frame + slice_num_frames <= segment_num_frames, f"slice out of segment bounds: {slice_start_frame} + {slice_num_frames} > {segment_num_frames}"
                                            if slice_start_frame + slice_num_frames > segment_num_frames:
                                                # some slightly out-of-bound data (slice_num_frames == segment_num_frames + 1)
                                                slice_num_frames = segment_num_frames - slice_start_frame
                                            slice_fbank = segment_fbank[:, :, slice_start_frame:slice_start_frame + slice_num_frames]
                                            data_src_list.append((slice_fbank, i))
                                        else:
                                            samples = cache['samples']
                                            slice_start_sample = int(rel_start_time * self.slice_fs)
                                            slice_num_samples = int(slice_duration * self.slice_fs)
                                            slice_samples = samples[slice_start_sample:slice_start_sample + slice_num_samples]
                                            data_src_list.append((slice_samples, i))
                                    else:
                                        raise ValueError(f"segment cache is not a dict")
        else:
            raise NotImplementedError("cache miss, not supported yet")

        # 提前多进程调用load_audio_text_image_video，加速训练
        turn_idx2data_src = defaultdict(list)
        for data_src, turn_idx in data_src_list:
            turn_idx2data_src[turn_idx].append(data_src)
        
        return turn_idx2data_src

    
    def extract_fbank_mock(self, data_src, data_type, frontend, is_final):
        assert data_type == "sound"
        fs = frontend.fs
        fps = 1000 // frontend.frame_shift
        num_frames = int(data_src.shape[0] / fs * fps)
        speech = torch.zeros(1, 128, num_frames, dtype=torch.float32)
        speech_lengths = torch.tensor([num_frames], dtype=torch.int32)
        return speech, speech_lengths


    def _process_continuous_audio_slice(self, data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type):

        # if self.preprocessor_speech is not None:
        #     data_src = self._preprocess_audio(data_src)

        # 提取fbank特征
        if isinstance(data_src, torch.Tensor) and self.slice_frontend is None:
            speech = data_src
            speech_lengths = torch.tensor([data_src.shape[2]], dtype=torch.int32)
        else:
            speech, speech_lengths = extract_fbank(
                data_src,
                data_type=self.data_type,
                frontend=self.slice_frontend,
                is_final=True,
            )  # speech: [b, T, d]
        
        if self.slice_permute:
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz
        
        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            if self.add_modalidy_id:
                slice_ids.append(self.audio_start_id)
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type


    def _preprocess_audio(self, audio_tensor: torch.Tensor, random_source=None, use_6mic=True):
        if self.preprocessor_speech is not None:
            processed = self.preprocessor_speech(audio_tensor.numpy(), use_6mic=use_6mic, random_source=random_source)
            processed = torch.from_numpy(processed)
        else:
            processed = audio_tensor
        return processed


    def _preprocess_file(self, input_audio_path: str, output_audio_path: str, random_source=None, use_6mic=True) -> bool:
        import soundfile as sf
        audio_tensor = load_audio_text_image_video(input_audio_path, self.fs)
        processed_audio_tensor = self._preprocess_audio(audio_tensor, random_source=random_source, use_6mic=use_6mic)
        sf.write(output_audio_path, processed_audio_tensor.numpy(), self.fs)
        return True


    def _preprocess_audio_cache(self, segments_dict: dict, segment_cache: dict):
        probatility = self.preprocessor_speech_conf.get('probability', 1.0)
        all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
        use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
        batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
        batch_use_6mic = batch_random_source.random() < use_6mic_probability
        conditions = self.preprocessor_speech_conf.get('conditions', [])

        def process_segment(segment_id):
            # if the conditions met
            segment = segments_dict[segment_id]
            conditions_met = True
            for condition in conditions:
                segment_key = condition.get('segment_key', None)
                value_min = condition.get('value_min', None)
                value_max = condition.get('value_max', None)
                # if segment_key is None or value_min is None or value_max is None:
                #     continue
                segment_key_parts = segment_key.split('.')
                current_item = segment
                for key in segment_key_parts:
                    if key in current_item:
                        current_item = current_item[key]
                    else:
                        current_item = None
                        break
                if current_item is None:
                    conditions_met = False
                    break
                if value_min is not None and current_item < value_min:
                    conditions_met = False
                    break
                if value_max is not None and current_item > value_max:
                    conditions_met = False
                    break
            if not conditions_met:
                return
            segment_unique_id = segment['unique_id']
            random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            if random_source.random() > probatility:
                return
            # run the preprocessing
            segment_path = segment_cache[segment_id]
            preprocessed_path = segment_path + '.preprocessed.wav'
            self._preprocess_file(segment_path, preprocessed_path, random_source=random_source, use_6mic=batch_use_6mic)
            segment_cache[segment_id] = preprocessed_path

        for segment_id in segments_dict.keys():
            process_segment(segment_id)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     results = executor.map(process_segment, segments_dict.keys())   # dingding fe is not friendly to threading


    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()
        time_start = time.time()

        output = None
        timings = OrderedDict()

        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
                user = item["user"]
                assistant = item["assistant"]
                segments_dict = item.get("segments_dict", {})
                pt_mode = item.get("pt_mode", None)
            except Exception as e:
                # logging.error(f"Error getting item: {e}")
                # logging.error(f"Traceback: {traceback.format_exc()}")
                continue

            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            
            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []

            local_audio_dir = None
            segment_cache = {} # key: segment_id, value: local_wav_path
            cache_hit = False
            try:
                # locate cache or download audio
                if pt_mode not in ['unimodal-text']:
                    time_start_audio_cache = time.time()
                    cache_hit, segment_cache, local_audio_dir = self._process_audio_cache(segments_dict)
                    time_end_audio_cache = time.time()
                    timings['process_audio_cache'] = time_end_audio_cache - time_start_audio_cache
    
                    if not cache_hit and self.drop_cache_miss_items:
                        logging.warning(f"cache miss, drop item")
                        continue

                    # if self.preprocessor_speech is not None:
                    #     self._preprocess_audio_cache(segments_dict, segment_cache)

                # extract fbank 
                if pt_mode not in ['unimodal-text', 'unimodal-audio', 'mapping-text-to-audio']:
                    if pt_mode == 'mapping-audio-to-text':
                        both_user_assistant = True
                    else:
                        both_user_assistant = False
                    time_start_process_continuous_audio_loading = time.time()
                    turn_idx2data_src = self._process_continuous_audio_loading(user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant)
                    time_end_process_continuous_audio_loading = time.time()
                    timings['process_continuous_audio_loading'] = time_end_process_continuous_audio_loading - time_start_process_continuous_audio_loading
                
                # add task id
                if self.add_task_id:
                    task_id = self.get_pt_mode_token_id(pt_mode)
                    input_ids.append(task_id)
                    labels.append(-100)

                    input_ids_audio.append(self.get_audio_token_id(task_id))
                    labels_audio.append(-100)

                pre_segment_id = None
                online_speech_tokenize_segment_ids = set()
                slice_modes = defaultdict(int)
                time_start_slice_loop = time.time()
                for i, (user_slice, assistant_slice) in enumerate(zip(user, assistant)):
                    if len(input_ids) > self.max_token_length:
                        logging.info(
                            f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{self.max_token_length}"
                        )
                        break
                    
                    for s_i, slice in enumerate([user_slice, assistant_slice]):
                        if slice is None:
                            continue

                        slice_ids = []
                        speech = []
                        speech_lengths = []
                        
                        text, start_time, end_time, segment_id, audio_tokens, text_tokens, slice_mode = slice["content"], slice["start_time"], slice["end_time"], slice["segment_id"], slice.get("tokens", None), slice.get("text_tokens", None), slice.get("slice_mode", None)

                        if start_time is not None and end_time is not None:
                            slice_duration = end_time - start_time
                            if not (slice_duration >= 0.1 and slice_duration <= self.max_slice_duration + 0.1):
                                logging.warning(f"slice duration is not valid: {slice_duration}, slice: {slice}")
                                continue
                        else:
                            slice_duration = None

                        slice_modes[slice_mode] += 1

                        # audio tokens is empty, skip this data
                        # if pt_mode != 'unimodal-text' and (audio_tokens is  None or len(audio_tokens) == 0):
                        #     logging.warning(f"audio tokens is empty, skip this data: {slice}")
                        #     continue

                        # add tag for segment transition
                        if s_i == 0 and pt_mode != 'unimodal-text':
                            if pre_segment_id is None:
                                pre_segment_id = segment_id
                            elif pre_segment_id != segment_id:
                                input_ids.append(self.end_of_segment_speech_token_id)
                                labels.append(-100)
                                input_ids_audio.append(self.get_audio_token_id(self.end_of_segment_speech_token_id))
                                labels_audio.append(-100)
                                pre_segment_id = segment_id
                        
                        if pt_mode == 'unimodal-text':
                            if not text:
                                continue
                            slice['onlie_discrete_speech_token'] = False
                            if self.add_modalidy_id:
                                slice_ids += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids.extend(text_tokens)
                            else:
                                slice_ids.extend(self.tokenizer.encode(text))
                            input_ids += slice_ids
                            labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                        elif pt_mode == 'unimodal-audio':
                            # discrete audio part
                            if self.add_modalidy_id:
                                slice_ids += [self.audio_start_id]
                            if audio_tokens is not None and len(audio_tokens) > 0:
                                # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                slice['onlie_discrete_speech_token'] = False
                            else:
                                num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                audio_tokens = [-50] * num_tokens
                                slice['onlie_discrete_speech_token'] = True
                                online_speech_tokenize_segment_ids.add(segment_id)
                            slice_ids.extend(audio_tokens)
                            input_ids_audio += slice_ids
                            labels_audio += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids += [self.mask_token_id] * len(slice_ids)
                            labels += [-100] * len(slice_ids)

                        elif pt_mode == 'mapping-audio-to-text':
                            slice['onlie_discrete_speech_token'] = False
                            # continuous audio part
                            data_src = turn_idx2data_src[i][s_i]
                            speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                    data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                )
                            input_ids += slice_ids
                            labels += [-100] * len(slice_ids)
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)
                            # text part
                            slice_ids = []
                            if self.add_modalidy_id:
                                slice_ids += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids.extend(text_tokens)
                            else:
                                slice_ids.extend(self.tokenizer.encode(text))
                            input_ids += slice_ids
                            labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)
                        
                        elif pt_mode == 'mapping-text-to-audio':
                            # text part
                            if self.add_modalidy_id:
                                slice_ids += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids.extend(text_tokens)
                            else:
                                slice_ids.extend(self.tokenizer.encode(text))
                            input_ids += slice_ids
                            labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)
                            # discrete audio part
                            slice_ids = []
                            if self.add_modalidy_id:
                                slice_ids += [self.audio_start_id]
                            if audio_tokens is not None and len(audio_tokens) > 0:
                                # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                slice['onlie_discrete_speech_token'] = False
                            else:
                                num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                audio_tokens = [-50] * num_tokens
                                slice['onlie_discrete_speech_token'] = True
                                online_speech_tokenize_segment_ids.add(segment_id)
                            slice_ids.extend(audio_tokens)
                            input_ids_audio += slice_ids
                            labels_audio += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids += [self.mask_token_id] * len(slice_ids)
                            labels += [-100] * len(slice_ids)

                        
                        elif pt_mode == 'interleave-audio-to-semantic':
                            if slice_mode == 'c':
                                slice['onlie_discrete_speech_token'] = False
                                # continuous audio part
                                data_src = turn_idx2data_src[i][s_i]
                                speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                        data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                    )  
                                input_ids += slice_ids
                                labels += [-100] * len(slice_ids)
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                            elif slice_mode == 'd':
                                # discrete audio part
                                if self.add_modalidy_id:
                                    slice_ids += [self.audio_start_id]
                                if audio_tokens is not None and len(audio_tokens) > 0:
                                    # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                    slice['onlie_discrete_speech_token'] = False
                                else:
                                    num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                    audio_tokens = [-50] * num_tokens
                                    slice['onlie_discrete_speech_token'] = True
                                    online_speech_tokenize_segment_ids.add(segment_id)
                                slice_ids.extend(audio_tokens)
                                input_ids_audio += slice_ids
                                labels_audio += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                                input_ids += [self.mask_token_id] * len(slice_ids)
                                labels += [-100] * len(slice_ids)
                            else:
                                raise ValueError(f"Invalid slice_mode: {slice_mode} for pt_mode: {pt_mode}")
                        
                        elif pt_mode == 'interleave-audio-to-text':
                            if slice_mode == 'c':
                                slice['onlie_discrete_speech_token'] = False
                                # continuous audio part
                                data_src = turn_idx2data_src[i][s_i]
                                speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                        data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                    )  
                                input_ids += slice_ids
                                labels += [-100] * len(slice_ids)
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                            elif slice_mode == 't':
                                slice['onlie_discrete_speech_token'] = False
                                # text part 
                                if self.add_modalidy_id:
                                    slice_ids += [self.text_start_id]
                                if text_tokens is not None:
                                    slice_ids.extend(text_tokens)
                                else:
                                    slice_ids.extend(self.tokenizer.encode(text))
                                input_ids += slice_ids
                                labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                        
                        elif pt_mode == 'interleave-audio-to-semantic-and-text':
                            if slice_mode == 'c':
                                slice['onlie_discrete_speech_token'] = False
                                # continuous audio part
                                data_src = turn_idx2data_src[i][s_i]
                                speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                        data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                    )  
                                input_ids += slice_ids
                                labels += [-100] * len(slice_ids)
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                            elif slice_mode == 'd' or slice_mode == 't':
                                # discrete audio part
                                if self.add_modalidy_id:
                                    slice_ids += [self.audio_start_id]
                                slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
                                if audio_tokens is not None and len(audio_tokens) > 0:
                                    # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                    slice['onlie_discrete_speech_token'] = False
                                else:
                                    num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                    audio_tokens = [-50] * num_tokens
                                    slice['onlie_discrete_speech_token'] = True
                                    online_speech_tokenize_segment_ids.add(segment_id)
                                slice_ids.extend(audio_tokens)
                                num_audio_tokens = len(slice_ids)
                                slice_ids_text = []
                                if self.add_modalidy_id:
                                    slice_ids_text += [self.text_start_id]
                                if text_tokens is not None:
                                    slice_ids_text.extend(text_tokens)
                                else:
                                    slice_ids_text.extend(self.tokenizer.encode(text))
                                num_text_tokens = len(slice_ids_text)
                                if num_text_tokens > num_audio_tokens:
                                    print (f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
                                max_tokens = max(num_text_tokens, num_audio_tokens)

                                input_ids_audio = input_ids_audio + slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
                                labels_audio = labels_audio + [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids] + [-100] * (max_tokens - len(slice_ids))

                                input_ids = input_ids + slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
                                labels = labels +  [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids_text] + [-100] * (max_tokens - len(slice_ids_text))
                            else:
                                raise ValueError(f"Invalid slice_mode: {slice_mode} for pt_mode: {pt_mode}")

                        if len(speech) > 0:
                            fbank.append(speech[0, :, :])
                            fbank_lens.append(speech_lengths)
                time_end_slice_loop = time.time()
                timings['slice_loop'] = time_end_slice_loop - time_start_slice_loop

                if len(input_ids) > self.max_token_length:
                    logging.warning(
                        f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}"
                    )
                    continue

                if len(input_ids) < self.min_token_length:
                    logging.info(
                        f"len(input_ids): {len(input_ids)} < min_token_length: {self.min_token_length}"
                    )
                if len(slice_modes.keys()) == 1 and 'c' in slice_modes.keys():
                    print (f'[DATASET] slice_modes: {slice_modes}')
                    continue

                input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
                labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

                audio_start_id = self.get_audio_token_id(self.audio_start_id)
                text_start_id = self.get_audio_token_id(self.text_start_id)
                input_ids_audio = [audio_start_id if id == self.audio_start_id else id for id in input_ids_audio]
                input_ids_audio = [text_start_id if id == self.text_start_id else id for id in input_ids_audio]
                labels_audio = [audio_start_id if id == self.audio_start_id else id for id in labels_audio]
                labels_audio = [text_start_id if id == self.text_start_id else id for id in labels_audio]
                input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: self.max_token_length]
                labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: self.max_token_length]

                attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)

                feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
                feature_type = torch.tensor(feature_type, dtype=torch.int32)
                fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

                
                segment_speech = []
                segment_speech_lengths = []
                segment_token = []
                segment_token_lengths = []
                segment_fbank = []
                segment_fbank_lengths = []
                if pt_mode in ['unimodal-audio', 'mapping-text-to-audio', 'interleave-audio-to-semantic', 'interleave-audio-to-semantic-and-text']:
                    # for segment-wise speech tokenizer
                    has_bad_audio = False

                    time_start_segment_loop = time.time()
                    for segment_id, segment in item["segments_dict"].items():
                        if segment_id not in online_speech_tokenize_segment_ids:
                            segment['onlie_discrete_speech_token'] = False
                            continue
                        else:
                            segment['onlie_discrete_speech_token'] = True
                        if segment_id in segment_cache:
                            cache = segment_cache[segment_id]
                            duration = [0, None]
                        else:
                            # this shall not happen if cache miss and drop_cache_miss_items is True
                            assert self.drop_cache_miss_items, f"cache miss and drop_cache_miss_items is True, segment_id: {segment_id}"
                    
                        # stime = time.time()
                        if isinstance(cache, dict):
                            one_segment_fbank = cache['fbank']
                            one_segment_fbank_length = cache['fbank_length']
                        else:
                            raise ValueError(f"segment cache is not a dict")
                            try:
                                samples = load_audio_text_image_video(load_audio_path, fs=self.fs, duration=duration)
                                # here the preprocessing was wrongly done once again
                                # if self.preprocessor_speech is not None:
                                #     samples = self._preprocess_audio(samples)
                            except Exception as e:
                                print (f"load_audio_text_image_video error: {e}")
                                has_bad_audio = True
                                break
                            # etime = time.time()
                            # print (f"load_audio_text_image_video time(example): {etime - stime}")
                            # segment_speech.append(samples)
                            # segment_speech_lengths.append(len(samples))
                            # tokens = torch.tensor(self.tokenizer.encode(segment["text"]), dtype=torch.int32)
                            # segment_token.append(tokens)
                            # segment_token_lengths.append(len(tokens))

                            # extract segment level fbank
                            # here the discrete tokenizer shares the same frontend parameters
                            # one_segment_fbank, one_segment_fbank_length = self.extract_fbank_mock(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                            one_segment_fbank, one_segment_fbank_length = extract_fbank(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                        if self.permute:
                            one_segment_fbank = one_segment_fbank.permute(0, 2, 1)
                        segment_fbank.append(one_segment_fbank[0, :, :])
                        segment_fbank_lengths.append(one_segment_fbank_length)
                    time_end_segment_loop = time.time()
                    timings['segment_loop'] = time_end_segment_loop - time_start_segment_loop

                    if has_bad_audio:
                        print (f"has_bad_audio: {has_bad_audio}, item: {item}")
                        continue

                output = {
                    # "segment_speech": segment_speech,
                    # "segment_speech_lengths": segment_speech_lengths,
                    # "segment_token": segment_token,
                    # "segment_token_lengths": segment_token_lengths,
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                    "attention_mask": attention_mask,
                    "item": item,
                    "speech_token_id_offset": self.speech_token_id_offset
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                if len(segment_fbank) > 0:
                    output["segment_fbank"] = segment_fbank
                    output["segment_fbank_lengths"] = segment_fbank_lengths
                break
            except Exception as e:
                print (f"process_continuous_audio_loading error: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
            finally:
                if local_audio_dir is not None:
                    import shutil
                    # print(f"remove local_audio_dir: {local_audio_dir}")
                    try:
                        shutil.rmtree(local_audio_dir)
                    except Exception as e:
                        print (f"remove local_audio_dir: {local_audio_dir} failed: {e}")
                # remove segment wav path
                for segment_id, segment in item["segments_dict"].items():
                    if segment["wav_path"] is not None:
                        os.remove(segment["wav_path"])

        time_end = time.time()
        # timing_string = f"datasets retry {idx}, __getitem__ time: {time_end - time_start}"
        # for key, value in timings.items():
        #     timing_string += f", {key}: {value}"
        # print(timing_string)

        return output


    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs

@tables.register("dataset_classes", "MinMo_S2S_Interleave_Pretrain_S2S")
class MinMo_S2S_Interleave_Pretrain_S2S(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            slice_frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        kwargs['audio_pt_modes_config_name'] = kwargs.get("audio_pt_modes_config_name", "dual_channel_v2")
        self.index_ds = index_ds_class(path, tokenizer=tokenizer, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        else:
            self.preprocessor_speech_conf = None
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.slice_frontend = slice_frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.slice_fs = self.fs if slice_frontend is None else slice_frontend.fs
        assert self.fs == self.slice_fs, f"frontend fs: {self.fs} != slice_frontend fs: {self.slice_fs}"
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        self.slice_permute = self.permute
        if self.slice_frontend is not None:
            self.slice_permute = isinstance(self.slice_frontend, WhisperFrontend)

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.min_token_length = kwargs.get("min_token_length", 1024)
        self.max_slice_duration = kwargs.get("max_merged_slice_duration_sec", 5.0)
        if not isinstance(self.max_slice_duration, float):
            print(f'max_slice_duration range: {self.max_slice_duration}')
            self.max_slice_duration = max(self.max_slice_duration[0], self.max_slice_duration[1])
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")

        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        logging.info(f"init continuous_in: {self.continuous_in}")

        # self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.speech_token_id_offset = 0 # for channels model, no offset needed
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        print(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")

        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")

        # pt mode
        self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-text|>") # unimodal-text
        self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-audio|>") # unimodal-audio
        self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-audio-to-text|>") # mapping-audio-to-text
        self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-text-to-audio|>") # mapping-text-to-audio 
        self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic|>") # interleave-audio-to-semantic 
        self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-text|>") # interleave-audio-to-text
        self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic-and-text|>") # interleave-audio-to-semantic-and-text
        assert self.pt_mode_unimodal_text_token_id is not None

        self.text_vocab_offset = self.tokenizer.vocab_size
        assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print (f'audio code book size: {self.audio_code_book_size}')
        self.mask_token_id = -60

        self.blank_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)

        self.add_task_id = kwargs.get("add_task_id", False)
        self.add_modalidy_id = kwargs.get("add_modalidy_id", False)

    def get_audio_token_id(self, audio_token):
        if isinstance(audio_token, int):
            return audio_token - self.text_vocab_offset + self.audio_code_book_size
        elif isinstance(audio_token, str):
            return self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids(audio_token))
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")


    def get_pt_mode_token_id(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def _extract_audio_segment(self, source_wav, start_time, end_time):
        """使用ffmpeg切分音频文件"""
        try:
            start_time_ffmpeg = time.time()
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)

            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-i', source_wav, '-ss', str(start_time),
                '-t', str(duration), '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logging.error(f"ffmpeg failed: {result.stderr}")
                return None
            end_time_ffmpeg = time.time()
            # print (f"extract_audio_segment time: {end_time_ffmpeg - start_time_ffmpeg}")
            return temp_path
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return None

    def _process_audio_cache(self, segments_dict):
        """
        处理音频缓存逻辑
        
        Args:
            segments_dict: 音频片段字典
            
        Returns:
            tuple: (cache_hit, segment_cache, local_audio_dir)
                - cache_hit: 是否命中缓存
                - segment_cache: 片段缓存字典 {segment_id: local_wav_path}
                - local_audio_dir: 本地音频目录
        """
        segment_cache = {}  # key: segment_id, value: local_wav_path
        cache_hit = False
        local_audio_dir = None
        
        # cache
        from funasr.datasets.minmo_datasets.cache_utils.pre_cache_audio_data import _sharded_dirs, _sha256_hex, _format_time_component, _is_file_intact
        assert self.audio_cache_type == "segments"  # only support segments cache now
        # bucket_and_rest = source_wav[len("oss://"):]
        bucket_name, object_name = 'example-storage-6', 'temp_obj'
        digest = _sha256_hex(object_name)
        remote_cache_dir = _sharded_dirs(self.audio_cache_dir, digest)
        local_audio_dir = os.path.join("/tmp/tmp_wav", uuid.uuid4().hex)

        expected_cache_files = set()
        cache_segment_names = set()
        segment_name2cache_segment_path = {}
        
        for segment_id, segment in segments_dict.items():
            segment_start_time = segment["start_time"]
            segment_end_time = segment["end_time"]
            if segment_start_time is None or segment_end_time is None:
                continue
            prefix = "audio_segment"
            segment_name = f"{prefix}-{segment_id}-{_format_time_component(segment_start_time)}-{_format_time_component(segment_end_time)}.{self.audio_cache_format}"
            if segment["wav_path"] is not None:
                cache_segment_path = segment['wav_path']
                assert os.path.exists(cache_segment_path), f"cache_segment_path not exists: {cache_segment_path}"
            else:
                print ('should not be logged, wav_path will always exist.')
                cache_segment_path = os.path.join(remote_cache_dir, segment_name)

            segment_name2cache_segment_path[segment_name] = cache_segment_path
            cache_segment_names.add((segment_id, segment_name))
            expected_cache_files.add(cache_segment_path)
            
        if all(_is_file_intact(p) for p in expected_cache_files):
            if self.audio_cache_format != "wav":
                # convert to wav and save to local cache
                os.makedirs(local_audio_dir, exist_ok=True)
                convert_args = []
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = segment_name2cache_segment_path[segment_name]
                    local_cache_segment_path = os.path.join(local_audio_dir, segment_name + '.wav')
                    segment_cache[segment_id] = local_cache_segment_path
                    convert_args.append((cache_segment_path, local_cache_segment_path, 16000, 1))
                with ThreadPoolExecutor(max_workers=20) as executor:
                    results = executor.map(convert_to_wav, convert_args)
                if all(results):
                    cache_hit = True
                else:
                    logging.warning(f"cache miss, not all files converted, drop item")
                    cache_hit = False
            else:
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = os.path.join(remote_cache_dir, segment_name)
                    segment_cache[segment_id] = cache_segment_path
                cache_hit = True
        else:
            logging.warning(f"cache miss, not all files are intact, drop item")
            cache_hit = False
        
        # preprocess audio
        if self.preprocessor_speech_conf is not None:
            probatility = self.preprocessor_speech_conf.get('probability', 1.0)
            all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
            use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
            batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            batch_use_6mic = batch_random_source.random() < use_6mic_probability
            conditions = self.preprocessor_speech_conf.get('conditions', [])

            def preprocess_segment(segment_id, segment_audio):
                # if the conditions met
                segment = segments_dict[segment_id]
                conditions_met = True
                for condition in conditions:
                    segment_key = condition.get('segment_key', None)
                    value_min = condition.get('value_min', None)
                    value_max = condition.get('value_max', None)
                    # if segment_key is None or value_min is None or value_max is None:
                    #     continue
                    segment_key_parts = segment_key.split('.')
                    current_item = segment
                    for key in segment_key_parts:
                        if key in current_item:
                            current_item = current_item[key]
                        else:
                            current_item = None
                            break
                    if current_item is None:
                        conditions_met = False
                        break
                    if value_min is not None and current_item < value_min:
                        conditions_met = False
                        break
                    if value_max is not None and current_item > value_max:
                        conditions_met = False
                        break
                if not conditions_met:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met conditions")
                    return segment_audio
                segment_unique_id = segment['unique_id']
                random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
                if random_source.random() > probatility:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met probatility")
                    return segment_audio
                # run the preprocessing
                # print(f"preprocess_segment segment_audio: {segment_audio.shape}")
                segment_audio = self._preprocess_audio(segment_audio, random_source=random_source, use_6mic=batch_use_6mic)
                return segment_audio

        # load and extract fbank
        for segment_id in segments_dict.keys():
            cache_segment_path = segment_cache[segment_id]
            duration = [0, None]
            segment_samples = load_audio_text_image_video(cache_segment_path, fs=self.fs, duration=duration)
            if self.preprocessor_speech_conf is not None:
                segment_samples = preprocess_segment(segment_id, segment_samples)
            segment_fbank, segment_fbank_length = extract_fbank(segment_samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
            segment_cache[segment_id] = {
                'path': cache_segment_path,
                'samples': segment_samples,
                'fbank': segment_fbank,
                'fbank_length': segment_fbank_length,
            }
        
        return cache_hit, segment_cache, local_audio_dir

    def _process_continuous_audio_loading(self, user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant=False):
        """
        提取 slice 对应音频的fbank
        """
        duration_list = []
        
        # test if local cache hit
        cache_duration_list = []
        data_src_list = []
        # cache miss
        if cache_hit:
            if self.audio_cache_type == "slices":
                # prefix = "audio_slice"
                raise NotImplementedError("slices cache is not supported")
            elif self.audio_cache_type == "segments":
                for i, (user_prompt, target_out) in enumerate(
                        zip(user, assistant)
                ):
                    for p_i, prompt in enumerate([user_prompt, target_out]):
                        if prompt is not None:
                            slice_start_time = prompt["start_time"]
                            slice_end_time = prompt["end_time"]
                            slice_mode = prompt["slice_mode"]
                            if 'c' not in slice_mode:
                                data_src_list.append((None, i))
                            else:
                                segment_id = prompt.get("segment_id", None)
                                if segment_id is not None:
                                    segment = segments_dict[segment_id]
                                    segment_start_time = segment["start_time"]
                                    segment_end_time = segment["end_time"]
                                    assert segment_start_time is not None and segment_end_time is not None
                                    offset = slice_start_time - segment_start_time
                                    assert offset >= 0 and offset <= segment_end_time - segment_start_time + 0.0001, f"offset: {offset} is not valid, segment time start: {segment_start_time}, segment time end: {segment_end_time}, slice time start: {slice_start_time}, slice time end: {slice_end_time}, segment: {segment}, slice: {prompt}"
                                    slice_duration = slice_end_time - slice_start_time
                                    assert slice_duration >= 0.01 and slice_duration <= 10000
                                    rel_start_time = offset
                                    rel_end_time = rel_start_time + slice_duration
                                    cache = segment_cache[segment_id]
                                    if isinstance(cache, dict):
                                        if self.slice_frontend is None:
                                            assert 'fbank' in cache and 'fbank_length' in cache, f"segment cache missing fbank key"
                                            segment_fbank = cache['fbank']
                                            segment_fbank_length = cache['fbank_length']
                                            slice_start_frame = int(rel_start_time * 1000 / self.frontend.frame_shift)
                                            slice_num_frames = int(slice_duration * 1000 / self.frontend.frame_shift)
                                            segment_num_frames = segment_fbank_length[0].item()
                                            # assert slice_start_frame + slice_num_frames <= segment_num_frames, f"slice out of segment bounds: {slice_start_frame} + {slice_num_frames} > {segment_num_frames}"
                                            if slice_start_frame + slice_num_frames > segment_num_frames:
                                                # some slightly out-of-bound data (slice_num_frames == segment_num_frames + 1)
                                                slice_num_frames = segment_num_frames - slice_start_frame
                                            slice_fbank = segment_fbank[:, :, slice_start_frame:slice_start_frame + slice_num_frames]
                                            data_src_list.append((slice_fbank, i))
                                        else:
                                            samples = cache['samples']
                                            slice_start_sample = int(rel_start_time * self.slice_fs)
                                            slice_num_samples = int(slice_duration * self.slice_fs)
                                            slice_samples = samples[slice_start_sample:slice_start_sample + slice_num_samples]
                                            data_src_list.append((slice_samples, i))
                                    else:
                                        raise ValueError(f"segment cache is not a dict")
        else:
            raise NotImplementedError("cache miss, not supported yet")

        # 提前多进程调用load_audio_text_image_video，加速训练
        turn_idx2data_src = defaultdict(list)
        for data_src, turn_idx in data_src_list:
            turn_idx2data_src[turn_idx].append(data_src)
        
        return turn_idx2data_src

    
    def extract_fbank_mock(self, data_src, data_type, frontend, is_final):
        assert data_type == "sound"
        fs = frontend.fs
        fps = 1000 // frontend.frame_shift
        num_frames = int(data_src.shape[0] / fs * fps)
        speech = torch.zeros(1, 128, num_frames, dtype=torch.float32)
        speech_lengths = torch.tensor([num_frames], dtype=torch.int32)
        return speech, speech_lengths


    def _process_continuous_audio_slice(self, data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type):

        # if self.preprocessor_speech is not None:
        #     data_src = self._preprocess_audio(data_src)

        # 提取fbank特征
        if isinstance(data_src, torch.Tensor) and self.slice_frontend is None:
            speech = data_src
            speech_lengths = torch.tensor([data_src.shape[2]], dtype=torch.int32)
        else:
            speech, speech_lengths = extract_fbank(
                data_src,
                data_type=self.data_type,
                frontend=self.slice_frontend,
                is_final=True,
            )  # speech: [b, T, d]
        
        if self.slice_permute:
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz
        
        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            if self.add_modalidy_id:
                slice_ids.append(self.get_audio_token_id(self.audio_start_id))
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type


    def _preprocess_audio(self, audio_tensor: torch.Tensor, random_source=None, use_6mic=True):
        if self.preprocessor_speech is not None:
            processed = self.preprocessor_speech(audio_tensor.numpy(), use_6mic=use_6mic, random_source=random_source)
            processed = torch.from_numpy(processed)
        else:
            processed = audio_tensor
        return processed


    def _preprocess_file(self, input_audio_path: str, output_audio_path: str, random_source=None, use_6mic=True) -> bool:
        import soundfile as sf
        audio_tensor = load_audio_text_image_video(input_audio_path, self.fs)
        processed_audio_tensor = self._preprocess_audio(audio_tensor, random_source=random_source, use_6mic=use_6mic)
        sf.write(output_audio_path, processed_audio_tensor.numpy(), self.fs)
        return True


    def _preprocess_audio_cache(self, segments_dict: dict, segment_cache: dict):
        probatility = self.preprocessor_speech_conf.get('probability', 1.0)
        all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
        use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
        batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
        batch_use_6mic = batch_random_source.random() < use_6mic_probability
        conditions = self.preprocessor_speech_conf.get('conditions', [])

        def process_segment(segment_id):
            # if the conditions met
            segment = segments_dict[segment_id]
            conditions_met = True
            for condition in conditions:
                segment_key = condition.get('segment_key', None)
                value_min = condition.get('value_min', None)
                value_max = condition.get('value_max', None)
                # if segment_key is None or value_min is None or value_max is None:
                #     continue
                segment_key_parts = segment_key.split('.')
                current_item = segment
                for key in segment_key_parts:
                    if key in current_item:
                        current_item = current_item[key]
                    else:
                        current_item = None
                        break
                if current_item is None:
                    conditions_met = False
                    break
                if value_min is not None and current_item < value_min:
                    conditions_met = False
                    break
                if value_max is not None and current_item > value_max:
                    conditions_met = False
                    break
            if not conditions_met:
                return
            segment_unique_id = segment['unique_id']
            random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            if random_source.random() > probatility:
                return
            # run the preprocessing
            segment_path = segment_cache[segment_id]
            preprocessed_path = segment_path + '.preprocessed.wav'
            self._preprocess_file(segment_path, preprocessed_path, random_source=random_source, use_6mic=batch_use_6mic)
            segment_cache[segment_id] = preprocessed_path

        for segment_id in segments_dict.keys():
            process_segment(segment_id)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     results = executor.map(process_segment, segments_dict.keys())   # dingding fe is not friendly to threading

    def _get_labels_with_one_pad_loss(self, labels, start_id, pad_id, blank_audio_token_id, max_tokens):
        new_labels = []
        has_pad = False
        for label in labels:
            if label == start_id:
                new_labels.append(-100)
            elif label == pad_id:
                if not has_pad:
                    new_labels.append(label)
                    has_pad = True
                else:
                    new_labels.append(-100)
            elif blank_audio_token_id > 0 and label == blank_audio_token_id:
                new_labels.append(-100)
            else:
                new_labels.append(label)
        new_labels = new_labels + [-100] * (max_tokens - len(new_labels))
        return new_labels

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()
        time_start = time.time()

        output = None
        timings = OrderedDict()

        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
                user = item["user"]
                assistant = item["assistant"]
                segments_dict = item.get("segments_dict", {})
                pt_mode = item.get("pt_mode", None)
            except Exception as e:
                # logging.error(f"Error getting item: {e}")
                # logging.error(f"Traceback: {traceback.format_exc()}")
                continue

            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            
            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []

            local_audio_dir = None
            segment_cache = {} # key: segment_id, value: local_wav_path
            cache_hit = False
            try:
                # locate cache or download audio
                if pt_mode not in ['unimodal-text']:
                    time_start_audio_cache = time.time()
                    cache_hit, segment_cache, local_audio_dir = self._process_audio_cache(segments_dict)
                    time_end_audio_cache = time.time()
                    timings['process_audio_cache'] = time_end_audio_cache - time_start_audio_cache
    
                    if not cache_hit and self.drop_cache_miss_items:
                        logging.warning(f"cache miss, drop item")
                        continue

                # extract fbank 
                if pt_mode not in ['unimodal-text', 'unimodal-audio', 'mapping-text-to-audio']:
                    if pt_mode == 'mapping-audio-to-text':
                        both_user_assistant = True
                    else:
                        both_user_assistant = False
                    time_start_process_continuous_audio_loading = time.time()
                    turn_idx2data_src = self._process_continuous_audio_loading(user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant)
                    time_end_process_continuous_audio_loading = time.time()
                    timings['process_continuous_audio_loading'] = time_end_process_continuous_audio_loading - time_start_process_continuous_audio_loading
                
                # add task id
                if self.add_task_id:
                    task_id = self.get_pt_mode_token_id(pt_mode)
                    input_ids.append(task_id)
                    labels.append(-100)

                    input_ids_audio.append(self.get_audio_token_id(task_id))
                    labels_audio.append(-100)

                pre_segment_id = None
                online_speech_tokenize_segment_ids = set()
                slice_modes = defaultdict(int)
                time_start_slice_loop = time.time()
                for i, (user_slice, assistant_slice) in enumerate(zip(user, assistant)):
                    if len(input_ids) > self.max_token_length:
                        logging.info(
                            f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{self.max_token_length}"
                        )
                        break
                    
                    for s_i, slice in enumerate([user_slice, assistant_slice]):
                        if slice is None:
                            continue

                        slice_ids = []
                        speech = []
                        speech_lengths = []
                        
                        text, start_time, end_time, segment_id, audio_tokens, text_tokens, slice_mode = slice["content"], slice["start_time"], slice["end_time"], slice["segment_id"], slice.get("tokens", None), slice.get("text_tokens", None), slice.get("slice_mode", None)

                        if start_time is not None and end_time is not None:
                            slice_duration = end_time - start_time
                            if not (slice_duration >= 0.1 and slice_duration <= self.max_slice_duration + 0.1):
                                logging.warning(f"slice duration is not valid: {slice_duration}, slice: {slice}")
                                continue
                        else:
                            slice_duration = None

                        slice_modes[slice_mode] += 1

                        # add tag for segment transition
                        if s_i == 0 and pt_mode != 'unimodal-text':
                            if pre_segment_id is None:
                                pre_segment_id = segment_id
                            elif pre_segment_id != segment_id:
                                input_ids.append(self.end_of_segment_speech_token_id)
                                labels.append(-100)
                                input_ids_audio.append(self.get_audio_token_id(self.end_of_segment_speech_token_id))
                                labels_audio.append(-100)
                                pre_segment_id = segment_id
                        
                        if pt_mode == 'unimodal-text':
                            if not text:
                                continue
                            slice['onlie_discrete_speech_token'] = False
                            if self.add_modalidy_id:
                                slice_ids += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids.extend(text_tokens)
                            else:
                                slice_ids.extend(self.tokenizer.encode(text))
                            input_ids += slice_ids
                            labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                        elif pt_mode == 'unimodal-audio':
                            assert slice_mode == 'td'
                            # discrete audio part
                            if self.add_modalidy_id:
                                slice_ids += [self.audio_start_id]
                            slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
                            if audio_tokens is not None and len(audio_tokens) > 0:
                                # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                slice['onlie_discrete_speech_token'] = False
                            else:
                                num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                audio_tokens = [-50] * num_tokens
                                slice['onlie_discrete_speech_token'] = True
                                online_speech_tokenize_segment_ids.add(segment_id)

                            slice_ids.extend(audio_tokens)
                            num_audio_tokens = len(slice_ids)
                            slice_ids_text = []
                            if self.add_modalidy_id:
                                slice_ids_text += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids_text.extend(text_tokens)
                            else:
                                slice_ids_text.extend(self.tokenizer.encode(text))
                            num_text_tokens = len(slice_ids_text)
                            if num_text_tokens > num_audio_tokens:
                                print (f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
                            max_tokens = max(num_text_tokens, num_audio_tokens)

                            input_ids_audio = input_ids_audio + slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
                            # labels_audio = labels_audio + [-100 if id == self.audio_start_id or id == self.text_start_id  else id for id in slice_ids] + [-100] * (max_tokens - len(slice_ids))
                            labels_audio = labels_audio + self._get_labels_with_one_pad_loss(slice_ids, self.get_audio_token_id(self.audio_start_id), self.get_audio_token_id(self.tokenizer.pad_token_id), self.get_audio_token_id(self.blank_audio_token_id), max_tokens)


                            input_ids = input_ids + slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
                            # labels = labels +  [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids_text] + [-100] * (max_tokens - len(slice_ids_text))
                            labels = labels + self._get_labels_with_one_pad_loss(slice_ids_text, self.text_start_id, self.tokenizer.pad_token_id, -1, max_tokens)

                        elif pt_mode == 'mapping-audio-to-text':
                            slice['onlie_discrete_speech_token'] = False
                            # continuous audio part
                            data_src = turn_idx2data_src[i][s_i]
                            speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                    data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                )
                            input_ids += slice_ids
                            labels += [-100] * len(slice_ids)
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                            slice_mode = slice_mode.strip('c')  # remove 'c' from 'ctd'

                            # td part 
                            slice_ids = []
                            if self.add_modalidy_id:
                                slice_ids += [self.audio_start_id]
                            slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
                            if audio_tokens is not None and len(audio_tokens) > 0:
                                # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                slice['onlie_discrete_speech_token'] = False
                            else:
                                num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                audio_tokens = [-50] * num_tokens
                                slice['onlie_discrete_speech_token'] = True
                                online_speech_tokenize_segment_ids.add(segment_id)
                            slice_ids.extend(audio_tokens)
                            num_audio_tokens = len(slice_ids)
                            slice_ids_text = []
                            if self.add_modalidy_id:
                                slice_ids_text += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids_text.extend(text_tokens)
                            else:
                                slice_ids_text.extend(self.tokenizer.encode(text))
                            num_text_tokens = len(slice_ids_text)
                            if num_text_tokens > num_audio_tokens:
                                print (f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
                            max_tokens = max(num_text_tokens, num_audio_tokens)

                            input_ids_audio = input_ids_audio + slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
                            # labels_audio = labels_audio + [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids] + [-100] * (max_tokens - len(slice_ids))
                            labels_audio = labels_audio + self._get_labels_with_one_pad_loss(slice_ids, self.get_audio_token_id(self.audio_start_id), self.get_audio_token_id(self.tokenizer.pad_token_id), self.get_audio_token_id(self.blank_audio_token_id), max_tokens)


                            input_ids = input_ids + slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
                            # labels = labels +  [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids_text] + [-100] * (max_tokens - len(slice_ids_text))
                            labels = labels + self._get_labels_with_one_pad_loss(slice_ids_text, self.text_start_id, self.tokenizer.pad_token_id, -1, max_tokens)
                        
                        elif pt_mode == 'mapping-text-to-audio':
                            # text part
                            if self.add_modalidy_id:
                                slice_ids += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids.extend(text_tokens)
                            else:
                                slice_ids.extend(self.tokenizer.encode(text))
                            input_ids += slice_ids
                            labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                            # td part 
                            slice_ids = []
                            if self.add_modalidy_id:
                                slice_ids += [self.audio_start_id]
                            slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
                            if audio_tokens is not None and len(audio_tokens) > 0:
                                # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                slice['onlie_discrete_speech_token'] = False
                            else:
                                num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                audio_tokens = [-50] * num_tokens
                                slice['onlie_discrete_speech_token'] = True
                                online_speech_tokenize_segment_ids.add(segment_id)
                            slice_ids.extend(audio_tokens)
                            num_audio_tokens = len(slice_ids)
                            slice_ids_text = []
                            if self.add_modalidy_id:
                                slice_ids_text += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids_text.extend(text_tokens)
                            else:
                                slice_ids_text.extend(self.tokenizer.encode(text))
                            num_text_tokens = len(slice_ids_text)
                            if num_text_tokens > num_audio_tokens:
                                print (f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
                            max_tokens = max(num_text_tokens, num_audio_tokens)

                            input_ids_audio = input_ids_audio + slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
                            # labels_audio = labels_audio + [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids] + [-100] * (max_tokens - len(slice_ids))
                            labels_audio = labels_audio + self._get_labels_with_one_pad_loss(slice_ids, self.get_audio_token_id(self.audio_start_id), self.get_audio_token_id(self.tokenizer.pad_token_id), self.get_audio_token_id(self.blank_audio_token_id), max_tokens)


                            input_ids = input_ids + slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
                            # labels = labels +  [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids_text] + [-100] * (max_tokens - len(slice_ids_text))
                            labels = labels + self._get_labels_with_one_pad_loss(slice_ids_text, self.text_start_id, self.tokenizer.pad_token_id, -1, max_tokens)
                        
                        elif pt_mode == 'interleave-audio-to-semantic-and-text':
                            if slice_mode == 'c':
                                slice['onlie_discrete_speech_token'] = False
                                # continuous audio part
                                data_src = turn_idx2data_src[i][s_i]
                                speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                        data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                    )  
                                input_ids += slice_ids
                                labels += [-100] * len(slice_ids)
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                            elif slice_mode == 'td':
                                # discrete audio part
                                if self.add_modalidy_id:
                                    slice_ids += [self.audio_start_id]
                                slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
                                if audio_tokens is not None and len(audio_tokens) > 0:
                                    # audio_tokens = [token + self.speech_token_id_offset for token in audio_tokens]
                                    slice['onlie_discrete_speech_token'] = False
                                else:
                                    num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
                                    audio_tokens = [-50] * num_tokens
                                    slice['onlie_discrete_speech_token'] = True
                                    online_speech_tokenize_segment_ids.add(segment_id)
                                slice_ids.extend(audio_tokens)
                                num_audio_tokens = len(slice_ids)
                                slice_ids_text = []
                                if self.add_modalidy_id:
                                    slice_ids_text += [self.text_start_id]
                                if text_tokens is not None:
                                    slice_ids_text.extend(text_tokens)
                                else:
                                    slice_ids_text.extend(self.tokenizer.encode(text))
                                num_text_tokens = len(slice_ids_text)
                                if num_text_tokens > num_audio_tokens:
                                    print (f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
                                max_tokens = max(num_text_tokens, num_audio_tokens)

                                input_ids_audio = input_ids_audio + slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
                                # labels_audio = labels_audio + [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids] + [-100] * (max_tokens - len(slice_ids))
                                labels_audio = labels_audio + self._get_labels_with_one_pad_loss(slice_ids, self.get_audio_token_id(self.audio_start_id), self.get_audio_token_id(self.tokenizer.pad_token_id), self.get_audio_token_id(self.blank_audio_token_id), max_tokens)


                                input_ids = input_ids + slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
                                # labels = labels +  [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids_text] + [-100] * (max_tokens - len(slice_ids_text))
                                labels = labels + self._get_labels_with_one_pad_loss(slice_ids_text, self.text_start_id, self.tokenizer.pad_token_id, -1, max_tokens)
                            else:
                                raise ValueError(f"Invalid slice_mode: {slice_mode} for pt_mode: {pt_mode}")
                        else:
                            raise ValueError(f"Invalid pt_mode: {pt_mode}")

                        if len(speech) > 0:
                            fbank.append(speech[0, :, :])
                            fbank_lens.append(speech_lengths)
                time_end_slice_loop = time.time()
                timings['slice_loop'] = time_end_slice_loop - time_start_slice_loop

                if len(input_ids) > self.max_token_length:
                    logging.warning(
                        f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}"
                    )
                    continue

                if len(input_ids) < self.min_token_length:
                    logging.info(
                        f"len(input_ids): {len(input_ids)} < min_token_length: {self.min_token_length}"
                    )
                if len(slice_modes.keys()) == 1 and 'c' in slice_modes.keys():
                    print (f'[DATASET] slice_modes: {slice_modes}')
                    continue

                input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
                labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

                audio_start_id = self.get_audio_token_id(self.audio_start_id)
                text_start_id = self.get_audio_token_id(self.text_start_id)
                input_ids_audio = [audio_start_id if id == self.audio_start_id else id for id in input_ids_audio]
                input_ids_audio = [text_start_id if id == self.text_start_id else id for id in input_ids_audio]
                labels_audio = [audio_start_id if id == self.audio_start_id else id for id in labels_audio]
                labels_audio = [text_start_id if id == self.text_start_id else id for id in labels_audio]
                input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: self.max_token_length]
                labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: self.max_token_length]

                attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)

                feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
                feature_type = torch.tensor(feature_type, dtype=torch.int32)
                fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

                segment_fbank = []
                segment_fbank_lengths = []
                if pt_mode in ['unimodal-audio', 'mapping-audio-to-text', 'mapping-text-to-audio', 'interleave-audio-to-semantic', 'interleave-audio-to-semantic-and-text']:
                    # for segment-wise speech tokenizer
                    has_bad_audio = False

                    time_start_segment_loop = time.time()
                    for segment_id, segment in item["segments_dict"].items():
                        if segment_id not in online_speech_tokenize_segment_ids:
                            segment['onlie_discrete_speech_token'] = False
                            continue
                        else:
                            segment['onlie_discrete_speech_token'] = True
                        if segment_id in segment_cache:
                            cache = segment_cache[segment_id]
                            duration = [0, None]
                        else:
                            # this shall not happen if cache miss and drop_cache_miss_items is True
                            assert self.drop_cache_miss_items, f"cache miss and drop_cache_miss_items is True, segment_id: {segment_id}"
                    
                        # stime = time.time()
                        if isinstance(cache, dict):
                            one_segment_fbank = cache['fbank']
                            one_segment_fbank_length = cache['fbank_length']
                        else:
                            raise ValueError(f"segment cache is not a dict")
                            try:
                                samples = load_audio_text_image_video(load_audio_path, fs=self.fs, duration=duration)
                                # here the preprocessing was wrongly done once again
                                # if self.preprocessor_speech is not None:
                                #     samples = self._preprocess_audio(samples)
                            except Exception as e:
                                print (f"load_audio_text_image_video error: {e}")
                                has_bad_audio = True
                                break
                            # etime = time.time()
                            # print (f"load_audio_text_image_video time(example): {etime - stime}")
                            # segment_speech.append(samples)
                            # segment_speech_lengths.append(len(samples))
                            # tokens = torch.tensor(self.tokenizer.encode(segment["text"]), dtype=torch.int32)
                            # segment_token.append(tokens)
                            # segment_token_lengths.append(len(tokens))

                            # extract segment level fbank
                            # here the discrete tokenizer shares the same frontend parameters
                            # one_segment_fbank, one_segment_fbank_length = self.extract_fbank_mock(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                            one_segment_fbank, one_segment_fbank_length = extract_fbank(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                        if self.permute:
                            one_segment_fbank = one_segment_fbank.permute(0, 2, 1)
                        segment_fbank.append(one_segment_fbank[0, :, :])
                        segment_fbank_lengths.append(one_segment_fbank_length)
                    time_end_segment_loop = time.time()
                    timings['segment_loop'] = time_end_segment_loop - time_start_segment_loop

                    if has_bad_audio:
                        print (f"has_bad_audio: {has_bad_audio}, item: {item}")
                        continue

                output = {
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                    "attention_mask": attention_mask,
                    "item": item,
                    "speech_token_id_offset": self.speech_token_id_offset
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                if len(segment_fbank) > 0:
                    output["segment_fbank"] = segment_fbank
                    output["segment_fbank_lengths"] = segment_fbank_lengths
                break
            except Exception as e:
                print (f"process_continuous_audio_loading error: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
            finally:
                if local_audio_dir is not None:
                    import shutil
                    try:
                        shutil.rmtree(local_audio_dir)
                    except Exception as e:
                        print (f"remove local_audio_dir: {local_audio_dir} failed: {e}")
                # remove segment wav path
                for segment_id, segment in item["segments_dict"].items():
                    if segment["wav_path"] is not None:
                        os.remove(segment["wav_path"])

        time_end = time.time()

        return output


    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs

@tables.register("dataset_classes", "MinMo_S2S_Interleave_Pretrain_S2S_Random")
class MinMo_S2S_Interleave_Pretrain_S2S_Random(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            slice_frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        kwargs['audio_pt_modes_config_name'] = kwargs.get("audio_pt_modes_config_name", "dual_channel_v2")
        self.index_ds = index_ds_class(path, tokenizer=tokenizer, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        else:
            self.preprocessor_speech_conf = None
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.slice_frontend = slice_frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.slice_fs = self.fs if slice_frontend is None else slice_frontend.fs
        assert self.fs == self.slice_fs, f"frontend fs: {self.fs} != slice_frontend fs: {self.slice_fs}"
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        self.slice_permute = self.permute
        if self.slice_frontend is not None:
            self.slice_permute = isinstance(self.slice_frontend, WhisperFrontend)

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.min_token_length = kwargs.get("min_token_length", 1024)
        self.max_slice_duration = kwargs.get("max_merged_slice_duration_sec", 5.0)
        if not isinstance(self.max_slice_duration, float):
            print(f'max_slice_duration range: {self.max_slice_duration}')
            self.max_slice_duration = max(self.max_slice_duration[0], self.max_slice_duration[1])
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")

        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        logging.info(f"init continuous_in: {self.continuous_in}")

        # self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.speech_token_id_offset = 0 # for channels model, no offset needed
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        print(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")

        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")

        # pt mode
        self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-text|>") # unimodal-text
        self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-audio|>") # unimodal-audio
        self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-audio-to-text|>") # mapping-audio-to-text
        self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-text-to-audio|>") # mapping-text-to-audio 
        self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic|>") # interleave-audio-to-semantic 
        self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-text|>") # interleave-audio-to-text
        self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic-and-text|>") # interleave-audio-to-semantic-and-text
        assert self.pt_mode_unimodal_text_token_id is not None

        self.text_vocab_offset = self.tokenizer.vocab_size
        assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print (f'audio code book size: {self.audio_code_book_size}')
        self.mask_token_id = -60

        self.blank_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)

        self.add_task_id = kwargs.get("add_task_id", False)
        self.add_modalidy_id = kwargs.get("add_modalidy_id", False)
        self.add_system_prompt = kwargs.get("add_system_prompt", True)

    def get_audio_token_id(self, audio_token):
        if isinstance(audio_token, int):
            return audio_token - self.text_vocab_offset + self.audio_code_book_size
        elif isinstance(audio_token, str):
            return self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids(audio_token))
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")


    def get_pt_mode_token_id(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_pt_mode_system_prompt(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return ''
        elif pt_mode == 'unimodal-audio':
            return ''
        elif pt_mode == 'mapping-audio-to-text':
            return 'This is a mapping-audio-to-text task.'
        elif pt_mode == 'mapping-text-to-audio':
            return 'This is a mapping-text-to-audio task.'
        elif pt_mode == 'interleave-audio-to-semantic':
            return ''
        elif pt_mode == 'interleave-audio-to-text':
            return ''
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return ''
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def _extract_audio_segment(self, source_wav, start_time, end_time):
        """使用ffmpeg切分音频文件"""
        try:
            start_time_ffmpeg = time.time()
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)

            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-i', source_wav, '-ss', str(start_time),
                '-t', str(duration), '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logging.error(f"ffmpeg failed: {result.stderr}")
                return None
            end_time_ffmpeg = time.time()
            # print (f"extract_audio_segment time: {end_time_ffmpeg - start_time_ffmpeg}")
            return temp_path
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return None

    def _process_audio_cache(self, segments_dict):
        """
        处理音频缓存逻辑
        
        Args:
            segments_dict: 音频片段字典
            
        Returns:
            tuple: (cache_hit, segment_cache, local_audio_dir)
                - cache_hit: 是否命中缓存
                - segment_cache: 片段缓存字典 {segment_id: local_wav_path}
                - local_audio_dir: 本地音频目录
        """
        segment_cache = {}  # key: segment_id, value: local_wav_path
        cache_hit = False
        local_audio_dir = None
        
        # cache
        from funasr.datasets.minmo_datasets.cache_utils.pre_cache_audio_data import _sharded_dirs, _sha256_hex, _format_time_component, _is_file_intact
        assert self.audio_cache_type == "segments"  # only support segments cache now
        # bucket_and_rest = source_wav[len("oss://"):]
        bucket_name, object_name = 'example-storage-6', 'temp_obj'
        digest = _sha256_hex(object_name)
        remote_cache_dir = _sharded_dirs(self.audio_cache_dir, digest)
        local_audio_dir = os.path.join("/tmp/tmp_wav", uuid.uuid4().hex)

        expected_cache_files = set()
        cache_segment_names = set()
        segment_name2cache_segment_path = {}
        
        for segment_id, segment in segments_dict.items():
            segment_start_time = segment["start_time"]
            segment_end_time = segment["end_time"]
            if segment_start_time is None or segment_end_time is None:
                continue
            prefix = "audio_segment"
            segment_name = f"{prefix}-{segment_id}-{_format_time_component(segment_start_time)}-{_format_time_component(segment_end_time)}.{self.audio_cache_format}"
            if segment["wav_path"] is not None:
                cache_segment_path = segment['wav_path']
                assert os.path.exists(cache_segment_path), f"cache_segment_path not exists: {cache_segment_path}"
            else:
                print ('should not be logged, wav_path will always exist.')
                cache_segment_path = os.path.join(remote_cache_dir, segment_name)

            segment_name2cache_segment_path[segment_name] = cache_segment_path
            cache_segment_names.add((segment_id, segment_name))
            expected_cache_files.add(cache_segment_path)
            
        if all(_is_file_intact(p) for p in expected_cache_files):
            if self.audio_cache_format != "wav":
                # convert to wav and save to local cache
                os.makedirs(local_audio_dir, exist_ok=True)
                convert_args = []
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = segment_name2cache_segment_path[segment_name]
                    local_cache_segment_path = os.path.join(local_audio_dir, segment_name + '.wav')
                    segment_cache[segment_id] = local_cache_segment_path
                    convert_args.append((cache_segment_path, local_cache_segment_path, 16000, 1))
                with ThreadPoolExecutor(max_workers=20) as executor:
                    results = executor.map(convert_to_wav, convert_args)
                if all(results):
                    cache_hit = True
                else:
                    logging.warning(f"cache miss, not all files converted, drop item")
                    cache_hit = False
            else:
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = os.path.join(remote_cache_dir, segment_name)
                    segment_cache[segment_id] = cache_segment_path
                cache_hit = True
        else:
            logging.warning(f"cache miss, not all files are intact, drop item")
            cache_hit = False
        
        # preprocess audio
        if self.preprocessor_speech_conf is not None:
            probatility = self.preprocessor_speech_conf.get('probability', 1.0)
            all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
            use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
            batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            batch_use_6mic = batch_random_source.random() < use_6mic_probability
            conditions = self.preprocessor_speech_conf.get('conditions', [])

            def preprocess_segment(segment_id, segment_audio):
                # if the conditions met
                segment = segments_dict[segment_id]
                conditions_met = True
                for condition in conditions:
                    segment_key = condition.get('segment_key', None)
                    value_min = condition.get('value_min', None)
                    value_max = condition.get('value_max', None)
                    # if segment_key is None or value_min is None or value_max is None:
                    #     continue
                    segment_key_parts = segment_key.split('.')
                    current_item = segment
                    for key in segment_key_parts:
                        if key in current_item:
                            current_item = current_item[key]
                        else:
                            current_item = None
                            break
                    if current_item is None:
                        conditions_met = False
                        break
                    if value_min is not None and current_item < value_min:
                        conditions_met = False
                        break
                    if value_max is not None and current_item > value_max:
                        conditions_met = False
                        break
                if not conditions_met:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met conditions")
                    return segment_audio
                segment_unique_id = segment['unique_id']
                random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
                if random_source.random() > probatility:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met probatility")
                    return segment_audio
                # run the preprocessing
                # print(f"preprocess_segment segment_audio: {segment_audio.shape}")
                segment_audio = self._preprocess_audio(segment_audio, random_source=random_source, use_6mic=batch_use_6mic)
                return segment_audio

        # load and extract fbank
        for segment_id in segments_dict.keys():
            cache_segment_path = segment_cache[segment_id]
            duration = [0, None]
            segment_samples = load_audio_text_image_video(cache_segment_path, fs=self.fs, duration=duration)
            if self.preprocessor_speech_conf is not None:
                segment_samples = preprocess_segment(segment_id, segment_samples)
            segment_fbank, segment_fbank_length = extract_fbank(segment_samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
            segment_cache[segment_id] = {
                'path': cache_segment_path,
                'samples': segment_samples,
                'fbank': segment_fbank,
                'fbank_length': segment_fbank_length,
            }
        
        return cache_hit, segment_cache, local_audio_dir

    def _process_continuous_audio_loading(self, user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant=False):
        """
        提取 slice 对应音频的fbank
        """
        duration_list = []
        
        # test if local cache hit
        cache_duration_list = []
        data_src_list = []
        # cache miss
        if cache_hit:
            if self.audio_cache_type == "slices":
                # prefix = "audio_slice"
                raise NotImplementedError("slices cache is not supported")
            elif self.audio_cache_type == "segments":
                for i, (user_prompt, target_out) in enumerate(
                        zip(user, assistant)
                ):
                    for p_i, prompt in enumerate([user_prompt, target_out]):
                        if prompt is not None:
                            slice_start_time = prompt["start_time"]
                            slice_end_time = prompt["end_time"]
                            slice_mode = prompt["slice_mode"]
                            if 'c' not in slice_mode:
                                data_src_list.append((None, i))
                            else:
                                segment_id = prompt.get("segment_id", None)
                                if segment_id is not None:
                                    segment = segments_dict[segment_id]
                                    segment_start_time = segment["start_time"]
                                    segment_end_time = segment["end_time"]
                                    assert segment_start_time is not None and segment_end_time is not None
                                    offset = slice_start_time - segment_start_time
                                    assert offset >= 0 and offset <= segment_end_time - segment_start_time + 0.0001, f"offset: {offset} is not valid, segment time start: {segment_start_time}, segment time end: {segment_end_time}, slice time start: {slice_start_time}, slice time end: {slice_end_time}, segment: {segment}, slice: {prompt}"
                                    slice_duration = slice_end_time - slice_start_time
                                    assert slice_duration >= 0.01 and slice_duration <= 10000
                                    rel_start_time = offset
                                    rel_end_time = rel_start_time + slice_duration
                                    cache = segment_cache[segment_id]
                                    if isinstance(cache, dict):
                                        if self.slice_frontend is None:
                                            assert 'fbank' in cache and 'fbank_length' in cache, f"segment cache missing fbank key"
                                            segment_fbank = cache['fbank']
                                            segment_fbank_length = cache['fbank_length']
                                            slice_start_frame = int(rel_start_time * 1000 / self.frontend.frame_shift)
                                            slice_num_frames = int(slice_duration * 1000 / self.frontend.frame_shift)
                                            segment_num_frames = segment_fbank_length[0].item()
                                            # assert slice_start_frame + slice_num_frames <= segment_num_frames, f"slice out of segment bounds: {slice_start_frame} + {slice_num_frames} > {segment_num_frames}"
                                            if slice_start_frame + slice_num_frames > segment_num_frames:
                                                # some slightly out-of-bound data (slice_num_frames == segment_num_frames + 1)
                                                slice_num_frames = segment_num_frames - slice_start_frame
                                            slice_fbank = segment_fbank[:, :, slice_start_frame:slice_start_frame + slice_num_frames]
                                            data_src_list.append((slice_fbank, i))
                                        else:
                                            samples = cache['samples']
                                            slice_start_sample = int(rel_start_time * self.slice_fs)
                                            slice_num_samples = int(slice_duration * self.slice_fs)
                                            slice_samples = samples[slice_start_sample:slice_start_sample + slice_num_samples]
                                            data_src_list.append((slice_samples, i))
                                    else:
                                        raise ValueError(f"segment cache is not a dict")
        else:
            raise NotImplementedError("cache miss, not supported yet")

        # 提前多进程调用load_audio_text_image_video，加速训练
        turn_idx2data_src = defaultdict(list)
        for data_src, turn_idx in data_src_list:
            turn_idx2data_src[turn_idx].append(data_src)
        
        return turn_idx2data_src

    
    def extract_fbank_mock(self, data_src, data_type, frontend, is_final):
        assert data_type == "sound"
        fs = frontend.fs
        fps = 1000 // frontend.frame_shift
        num_frames = int(data_src.shape[0] / fs * fps)
        speech = torch.zeros(1, 128, num_frames, dtype=torch.float32)
        speech_lengths = torch.tensor([num_frames], dtype=torch.int32)
        return speech, speech_lengths


    def _process_continuous_audio_slice(self, data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type):

        # if self.preprocessor_speech is not None:
        #     data_src = self._preprocess_audio(data_src)

        # 提取fbank特征
        if isinstance(data_src, torch.Tensor) and self.slice_frontend is None:
            speech = data_src
            speech_lengths = torch.tensor([data_src.shape[2]], dtype=torch.int32)
        else:
            speech, speech_lengths = extract_fbank(
                data_src,
                data_type=self.data_type,
                frontend=self.slice_frontend,
                is_final=True,
            )  # speech: [b, T, d]
        
        if self.slice_permute:
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz
        
        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            if self.add_modalidy_id:
                slice_ids.append(self.get_audio_token_id(self.audio_start_id))
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type


    def _preprocess_audio(self, audio_tensor: torch.Tensor, random_source=None, use_6mic=True):
        if self.preprocessor_speech is not None:
            processed = self.preprocessor_speech(audio_tensor.numpy(), use_6mic=use_6mic, random_source=random_source)
            processed = torch.from_numpy(processed)
        else:
            processed = audio_tensor
        return processed


    def _preprocess_file(self, input_audio_path: str, output_audio_path: str, random_source=None, use_6mic=True) -> bool:
        import soundfile as sf
        audio_tensor = load_audio_text_image_video(input_audio_path, self.fs)
        processed_audio_tensor = self._preprocess_audio(audio_tensor, random_source=random_source, use_6mic=use_6mic)
        sf.write(output_audio_path, processed_audio_tensor.numpy(), self.fs)
        return True


    def _preprocess_audio_cache(self, segments_dict: dict, segment_cache: dict):
        probatility = self.preprocessor_speech_conf.get('probability', 1.0)
        all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
        use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
        batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
        batch_use_6mic = batch_random_source.random() < use_6mic_probability
        conditions = self.preprocessor_speech_conf.get('conditions', [])

        def process_segment(segment_id):
            # if the conditions met
            segment = segments_dict[segment_id]
            conditions_met = True
            for condition in conditions:
                segment_key = condition.get('segment_key', None)
                value_min = condition.get('value_min', None)
                value_max = condition.get('value_max', None)
                # if segment_key is None or value_min is None or value_max is None:
                #     continue
                segment_key_parts = segment_key.split('.')
                current_item = segment
                for key in segment_key_parts:
                    if key in current_item:
                        current_item = current_item[key]
                    else:
                        current_item = None
                        break
                if current_item is None:
                    conditions_met = False
                    break
                if value_min is not None and current_item < value_min:
                    conditions_met = False
                    break
                if value_max is not None and current_item > value_max:
                    conditions_met = False
                    break
            if not conditions_met:
                return
            segment_unique_id = segment['unique_id']
            random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            if random_source.random() > probatility:
                return
            # run the preprocessing
            segment_path = segment_cache[segment_id]
            preprocessed_path = segment_path + '.preprocessed.wav'
            self._preprocess_file(segment_path, preprocessed_path, random_source=random_source, use_6mic=batch_use_6mic)
            segment_cache[segment_id] = preprocessed_path

        for segment_id in segments_dict.keys():
            process_segment(segment_id)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     results = executor.map(process_segment, segments_dict.keys())   # dingding fe is not friendly to threading

    def _get_labels_with_one_pad_loss(self, labels, start_id, pad_id, blank_audio_token_id, max_tokens):
        new_labels = []
        has_pad = False
        for label in labels:
            if label == start_id:
                new_labels.append(-100)
            elif label == pad_id:
                if not has_pad:
                    new_labels.append(label)
                    has_pad = True
                else:
                    new_labels.append(-100)
            elif blank_audio_token_id > 0 and label == blank_audio_token_id:
                new_labels.append(-100)
            else:
                new_labels.append(label)
        new_labels = new_labels + [-100] * (max_tokens - len(new_labels))
        return new_labels

    def _process_td_slice(
        self,
        audio_tokens,
        slice_duration,
        text_tokens,
        text,
        segment_id,
        slice,
        online_speech_tokenize_segment_ids
    ):
        """
        处理  td slice
        
        返回: (slice_ids, slice_ids_text, input_ids_audio_increment, labels_audio_increment, 
            input_ids_increment, labels_increment)
        """
        slice_ids = []
        if self.add_modalidy_id:
            slice_ids += [self.audio_start_id]
        slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
        if audio_tokens is not None and len(audio_tokens) > 0:
            slice['onlie_discrete_speech_token'] = False
        else:
            num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
            audio_tokens = [-50] * num_tokens
            slice['onlie_discrete_speech_token'] = True
            online_speech_tokenize_segment_ids.add(segment_id)
        slice_ids.extend(audio_tokens)
        num_audio_tokens = len(slice_ids)
        
        slice_ids_text = []
        if self.add_modalidy_id:
            slice_ids_text += [self.text_start_id]
        if text_tokens is not None:
            slice_ids_text.extend(text_tokens)
        else:
            slice_ids_text.extend(self.tokenizer.encode(text))
        num_text_tokens = len(slice_ids_text)
        if num_text_tokens > num_audio_tokens:
            print(f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
        max_tokens = max(num_text_tokens, num_audio_tokens)

        input_ids_audio_increment = slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
        labels_audio_increment = self._get_labels_with_one_pad_loss(
            slice_ids, 
            self.get_audio_token_id(self.audio_start_id), 
            self.get_audio_token_id(self.tokenizer.pad_token_id), 
            self.get_audio_token_id(self.blank_audio_token_id), 
            max_tokens
        )

        input_ids_increment = slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
        labels_increment = self._get_labels_with_one_pad_loss(
            slice_ids_text, 
            self.text_start_id, 
            self.tokenizer.pad_token_id, 
            -1, 
            max_tokens
        )
        
        return (slice_ids, slice_ids_text, input_ids_audio_increment, labels_audio_increment, 
                input_ids_increment, labels_increment)

    def _process_t_slice(
        self,
        text_tokens,
        text,
        slice
    ):
        """
        处理 t slice（纯文本部分）
        
        参数:
            text_tokens: 预编码的文本token列表，如果为None则使用text进行编码
            text: 原始文本
            slice: slice字典，会被修改（设置onlie_discrete_speech_token）
        
        返回: (slice_ids, input_ids_increment, labels_increment, 
            input_ids_audio_increment, labels_audio_increment)
        """
        slice['onlie_discrete_speech_token'] = False
        slice_ids = []
        if self.add_modalidy_id:
            slice_ids += [self.text_start_id]
        if text_tokens is not None:
            slice_ids.extend(text_tokens)
        else:
            slice_ids.extend(self.tokenizer.encode(text))
        
        input_ids_increment = slice_ids
        labels_increment = [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
        input_ids_audio_increment = [self.mask_token_id] * len(slice_ids)
        labels_audio_increment = [-100] * len(slice_ids)
        
        return (slice_ids, input_ids_increment, labels_increment, 
                input_ids_audio_increment, labels_audio_increment)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()
        time_start = time.time()

        output = None
        timings = OrderedDict()

        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
                user = item["user"]
                assistant = item["assistant"]
                segments_dict = item.get("segments_dict", {})
                pt_mode = item.get("pt_mode", None)
            except Exception as e:
                # logging.error(f"Error getting item: {e}")
                # logging.error(f"Traceback: {traceback.format_exc()}")
                continue

            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            
            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []

            local_audio_dir = None
            segment_cache = {} # key: segment_id, value: local_wav_path
            cache_hit = False
            try:
                # locate cache or download audio
                if pt_mode not in ['unimodal-text']:
                    time_start_audio_cache = time.time()
                    cache_hit, segment_cache, local_audio_dir = self._process_audio_cache(segments_dict)
                    time_end_audio_cache = time.time()
                    timings['process_audio_cache'] = time_end_audio_cache - time_start_audio_cache
    
                    if not cache_hit and self.drop_cache_miss_items:
                        logging.warning(f"cache miss, drop item")
                        continue

                # extract fbank 
                if pt_mode not in ['unimodal-text', 'unimodal-audio', 'mapping-text-to-audio']:
                    if pt_mode == 'mapping-audio-to-text':
                        both_user_assistant = True
                    else:
                        both_user_assistant = False
                    time_start_process_continuous_audio_loading = time.time()
                    turn_idx2data_src = self._process_continuous_audio_loading(user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant)
                    time_end_process_continuous_audio_loading = time.time()
                    timings['process_continuous_audio_loading'] = time_end_process_continuous_audio_loading - time_start_process_continuous_audio_loading
                
                # add task id
                if self.add_task_id:
                    task_id = self.get_pt_mode_token_id(pt_mode)
                    input_ids.append(task_id)
                    labels.append(-100)

                    input_ids_audio.append(self.get_audio_token_id(task_id))
                    labels_audio.append(-100)
                if self.add_system_prompt:
                    system_prompt = self.get_pt_mode_system_prompt(pt_mode)
                    if system_prompt:
                        system_prompt_ids = self.tokenizer.encode(system_prompt)
                        input_ids.extend(system_prompt_ids)
                        labels.extend([-100] * len(system_prompt_ids))
                        input_ids_audio.extend([self.mask_token_id] * len(system_prompt_ids))
                        labels_audio.extend([-100] * len(system_prompt_ids))

                pre_segment_id = None
                online_speech_tokenize_segment_ids = set()
                slice_modes = defaultdict(int)
                time_start_slice_loop = time.time()
                for i, (user_slice, assistant_slice) in enumerate(zip(user, assistant)):
                    if len(input_ids) > self.max_token_length:
                        logging.info(
                            f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{self.max_token_length}"
                        )
                        break
                    
                    for s_i, slice in enumerate([user_slice, assistant_slice]):
                        if slice is None:
                            continue

                        slice_ids = []
                        speech = []
                        speech_lengths = []
                        
                        text, start_time, end_time, segment_id, audio_tokens, text_tokens, slice_mode = slice["content"], slice["start_time"], slice["end_time"], slice["segment_id"], slice.get("tokens", None), slice.get("text_tokens", None), slice.get("slice_mode", None)

                        if start_time is not None and end_time is not None:
                            slice_duration = end_time - start_time
                            if not (slice_duration >= 0.1 and slice_duration <= self.max_slice_duration + 0.1):
                                logging.warning(f"slice duration is not valid: {slice_duration}, slice: {slice}")
                                continue
                        else:
                            slice_duration = None

                        slice_modes[slice_mode] += 1

                        # add tag for segment transition
                        if s_i == 0 and pt_mode != 'unimodal-text':
                            if pre_segment_id is None:
                                pre_segment_id = segment_id
                            elif pre_segment_id != segment_id:
                                input_ids.append(self.end_of_segment_speech_token_id)
                                labels.append(-100)
                                input_ids_audio.append(self.get_audio_token_id(self.end_of_segment_speech_token_id))
                                labels_audio.append(-100)
                                pre_segment_id = segment_id
                        
                        if pt_mode == 'unimodal-text':
                            if not text:
                                continue
                            slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                self._process_t_slice(
                                    text_tokens,
                                    text,
                                    slice
                                )
                            
                            input_ids = input_ids + input_ids_inc
                            labels = labels + labels_inc
                            input_ids_audio = input_ids_audio + input_ids_audio_inc
                            labels_audio = labels_audio + labels_audio_inc

                        elif pt_mode == 'unimodal-audio':
                            assert slice_mode == 'td' or slice_mode == 't'
                            if slice_mode == 'td':
                                slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                                    self._process_td_slice(
                                        audio_tokens,
                                        slice_duration,
                                        text_tokens,
                                        text,
                                        segment_id,
                                        slice,
                                        online_speech_tokenize_segment_ids
                                    )
                                
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                            else:
                                slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                    self._process_t_slice(
                                        text_tokens,
                                        text,
                                        slice
                                    )
                                
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc

                        elif pt_mode == 'mapping-audio-to-text':
                            # ASR
                            slice['onlie_discrete_speech_token'] = False
                            # continuous audio part
                            data_src = turn_idx2data_src[i][s_i]
                            speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                    data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                )
                            input_ids += slice_ids
                            labels += [-100] * len(slice_ids)
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                            slice_mode = slice_mode.strip('c')  # remove 'c' from 'ctd'

                            # td part or t part
                            if slice_mode == 'td':
                                slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                                    self._process_td_slice(
                                        audio_tokens,
                                        slice_duration,
                                        text_tokens,
                                        text,
                                        segment_id,
                                        slice,
                                        online_speech_tokenize_segment_ids
                                    )
                                
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                            elif slice_mode == 't':
                                slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                    self._process_t_slice(
                                        text_tokens,
                                        text,
                                        slice
                                    )
                                
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc


                        elif pt_mode == 'mapping-text-to-audio':
                            # text part
                            if self.add_modalidy_id:
                                slice_ids += [self.text_start_id]
                            if text_tokens is not None:
                                slice_ids.extend(text_tokens)
                            else:
                                slice_ids.extend(self.tokenizer.encode(text))
                            input_ids += slice_ids
                            labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                            # td part 
                            slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                                self._process_td_slice(
                                    audio_tokens,
                                    slice_duration,
                                    text_tokens,
                                    text,
                                    segment_id,
                                    slice,
                                    online_speech_tokenize_segment_ids
                                )

                            input_ids_audio = input_ids_audio + input_ids_audio_inc
                            labels_audio = labels_audio + labels_audio_inc
                            input_ids = input_ids + input_ids_inc
                            labels = labels + labels_inc
                        
                        elif pt_mode == 'interleave-audio-to-semantic-and-text':
                            if slice_mode == 'c':
                                slice['onlie_discrete_speech_token'] = False
                                # continuous audio part
                                data_src = turn_idx2data_src[i][s_i]
                                speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                        data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type
                                    )  
                                input_ids += slice_ids
                                labels += [-100] * len(slice_ids)
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                            elif slice_mode == 'td':
                                slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                                    self._process_td_slice(
                                        audio_tokens,
                                        slice_duration,
                                        text_tokens,
                                        text,
                                        segment_id,
                                        slice,
                                        online_speech_tokenize_segment_ids
                                    )
                                
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                            elif slice_mode == 't':
                                slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                    self._process_t_slice(
                                        text_tokens,
                                        text,
                                        slice
                                    )
                                
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc  
                            else:
                                raise ValueError(f"Invalid slice_mode: {slice_mode} for pt_mode: {pt_mode}")
                        else:
                            raise ValueError(f"Invalid pt_mode: {pt_mode}")

                        if len(speech) > 0:
                            fbank.append(speech[0, :, :])
                            fbank_lens.append(speech_lengths)
                time_end_slice_loop = time.time()
                timings['slice_loop'] = time_end_slice_loop - time_start_slice_loop

                if len(input_ids) > self.max_token_length:
                    logging.warning(
                        f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}"
                    )
                    continue

                if len(input_ids) < self.min_token_length:
                    logging.info(
                        f"len(input_ids): {len(input_ids)} < min_token_length: {self.min_token_length}"
                    )
                if len(slice_modes.keys()) == 1 and 'c' in slice_modes.keys():
                    print (f'[DATASET] slice_modes: {slice_modes}')
                    continue

                input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
                labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

                audio_start_id = self.get_audio_token_id(self.audio_start_id)
                text_start_id = self.get_audio_token_id(self.text_start_id)
                input_ids_audio = [audio_start_id if id == self.audio_start_id else id for id in input_ids_audio]
                input_ids_audio = [text_start_id if id == self.text_start_id else id for id in input_ids_audio]
                labels_audio = [audio_start_id if id == self.audio_start_id else id for id in labels_audio]
                labels_audio = [text_start_id if id == self.text_start_id else id for id in labels_audio]
                input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: self.max_token_length]
                labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: self.max_token_length]

                attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)

                feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
                feature_type = torch.tensor(feature_type, dtype=torch.int32)
                fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

                segment_fbank = []
                segment_fbank_lengths = []
                if pt_mode in ['unimodal-audio', 'mapping-audio-to-text', 'mapping-text-to-audio', 'interleave-audio-to-semantic', 'interleave-audio-to-semantic-and-text']:
                    # for segment-wise speech tokenizer
                    has_bad_audio = False

                    time_start_segment_loop = time.time()
                    for segment_id, segment in item["segments_dict"].items():
                        if segment_id not in online_speech_tokenize_segment_ids:
                            segment['onlie_discrete_speech_token'] = False
                            continue
                        else:
                            segment['onlie_discrete_speech_token'] = True
                        if segment_id in segment_cache:
                            cache = segment_cache[segment_id]
                            duration = [0, None]
                        else:
                            # this shall not happen if cache miss and drop_cache_miss_items is True
                            assert self.drop_cache_miss_items, f"cache miss and drop_cache_miss_items is True, segment_id: {segment_id}"
                    
                        # stime = time.time()
                        if isinstance(cache, dict):
                            one_segment_fbank = cache['fbank']
                            one_segment_fbank_length = cache['fbank_length']
                        else:
                            raise ValueError(f"segment cache is not a dict")
                            try:
                                samples = load_audio_text_image_video(load_audio_path, fs=self.fs, duration=duration)
                                # here the preprocessing was wrongly done once again
                                # if self.preprocessor_speech is not None:
                                #     samples = self._preprocess_audio(samples)
                            except Exception as e:
                                print (f"load_audio_text_image_video error: {e}")
                                has_bad_audio = True
                                break
                            # etime = time.time()
                            # print (f"load_audio_text_image_video time(example): {etime - stime}")
                            # segment_speech.append(samples)
                            # segment_speech_lengths.append(len(samples))
                            # tokens = torch.tensor(self.tokenizer.encode(segment["text"]), dtype=torch.int32)
                            # segment_token.append(tokens)
                            # segment_token_lengths.append(len(tokens))

                            # extract segment level fbank
                            # here the discrete tokenizer shares the same frontend parameters
                            # one_segment_fbank, one_segment_fbank_length = self.extract_fbank_mock(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                            one_segment_fbank, one_segment_fbank_length = extract_fbank(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                        if self.permute:
                            one_segment_fbank = one_segment_fbank.permute(0, 2, 1)
                        segment_fbank.append(one_segment_fbank[0, :, :])
                        segment_fbank_lengths.append(one_segment_fbank_length)
                    time_end_segment_loop = time.time()
                    timings['segment_loop'] = time_end_segment_loop - time_start_segment_loop

                    if has_bad_audio:
                        print (f"has_bad_audio: {has_bad_audio}, item: {item}")
                        continue

                output = {
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                    "attention_mask": attention_mask,
                    "item": item,
                    "speech_token_id_offset": self.speech_token_id_offset
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                if len(segment_fbank) > 0:
                    output["segment_fbank"] = segment_fbank
                    output["segment_fbank_lengths"] = segment_fbank_lengths
                break
            except Exception as e:
                print (f"process_continuous_audio_loading error: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
            finally:
                if local_audio_dir is not None:
                    import shutil
                    try:
                        shutil.rmtree(local_audio_dir)
                    except Exception as e:
                        print (f"remove local_audio_dir: {local_audio_dir} failed: {e}")
                # remove segment wav path
                for segment_id, segment in item["segments_dict"].items():
                    if segment["wav_path"] is not None:
                        os.remove(segment["wav_path"])

        time_end = time.time()

        return output


    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs

@tables.register("dataset_classes", "MinMo_S2S_Interleave_Pretrain_S2S_Random_S2T")
class MinMo_S2S_Interleave_Pretrain_S2S_Random_S2T(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """
    ASR_Prompts = [
        # English prompts
        "This task involves converting speech from an audio file into the corresponding text transcript.",
        "You are to perform an audio-to-text transcription, mapping spoken content to written form.",
        "Given an audio clip, produce its accurate textual representation.",
        "Transcribe the speech in the provided audio into text.",
        "This is a speech recognition task: output the text corresponding to the audio input.",
        "Map the spoken words in the audio recording to their written text equivalent.",
        "Your goal is to recognize and render the spoken content of an audio file as text.",
        "Convert the audio signal into its matching textual form.",
        "For each audio segment, generate the precise text that was spoken.",
        "Listen to the provided audio and produce a verbatim text transcript.",
        "Perform ASR (Automatic Speech Recognition) to transform audio to text.",
        "Turn the spoken audio content into an exact text sequence.",
        "The task is to transcribe audio into its corresponding text representation.",
        "Detect the words spoken in the given audio and output them as text.",
        "An audio-to-text mapping is required: produce the written version of the spoken audio.",
        "Process the audio input and deliver the text reflecting what was said.",
        "Transform the given speech audio into its complete text transcript.",
        "Recognize speech from audio and map it directly to text.",
        "Your task: transcribe the speech in the audio recording faithfully into text.",
        "Produce an accurate text rendition of the spoken content in the input audio.",

        # Chinese prompts
        "这个任务是将音频文件中的语音内容转换成对应的文字记录。",
        "你需要执行音频到文本的转写，将口语内容映射为书面形式。",
        "给定一段音频片段，生成它的精确文字表达。",
        "请将所提供音频中的语音转写为文本。",
        "这是一个语音识别任务：输出与输入音频相对应的文本。",
        "将音频录音中的口语转换为等效的书面文字。",
        "你的目标是识别并呈现音频文件中的口语内容为文本。",
        "将音频信号转换为匹配的文字形式。",
        "对每一段音频，生成准确的发言文字记录。",
        "听取给定的音频并生成逐字的文字转录。",
        "执行自动语音识别（ASR），将音频转换为文本。",
        "将口语音频内容转化为精确的文字序列。",
        "任务是把音频内容转写成对应的文字表示。",
        "识别给定音频中所说的词语，并输出为文本。",
        "需要进行音频到文本的映射：生成口语的书面版本。",
        "处理音频输入并输出反映所说内容的文字。",
        "将给定的语音音频转化为完整的文字记录。",
        "从音频中识别语音并直接转换为文本。",
        "你的任务是将录音中的语音内容精准转录为文字。",
        "为输入音频中的口语内容生成精确的文字呈现。"
    ]

    Caption_Prompts = [
        # English prompts
        "This task involves converting an audio clip into a descriptive text output.",
        "Your job is to listen to the provided audio and produce an accurate textual summary.",
        "You will generate a caption or analysis that reflects the content of the given audio.",
        "This is an audio-to-text description exercise — interpret what you hear and write it down.",
        "Analyze the given audio input and return a meaningful text representation.",
        "Listen carefully to the audio file and describe it in words.",
        "This is a task where you map audio information into a written explanation.",
        "The goal is to produce a human-readable description of the provided audio segment.",
        "Hear the sound, and translate it into a concise textual analysis.",
        "Perform semantic mapping from a sound recording to a textual narrative.",
        "Based on the audio provided, create a caption reflecting its main features.",
        "This is a process of generating textual interpretation from an audio sample.",
        "Listen to the audio and produce a linguistic representation of its contents.",
        "Capture the essence of the audio clip through an informative text output.",
        "Convert acoustic information into descriptive written language.",
        "Generate a natural-language caption that encapsulates the audio segment.",
        "Interpret the audio signal and express it as descriptive text.",
        "Transform the auditory input into a coherent written statement.",
        "Process the supplied audio and return an analytic, text-based description.",
        "This task asks you to describe and analyze an audio recording in text form.",
        
        # Chinese prompts
        "该任务需要将音频片段转换为文字描述输出。",
        "你的工作是聆听所提供的音频，并生成准确的文字摘要。",
        "你将生成一个反映给定音频内容的文字说明或分析。",
        "这是一个音频转文字描述的任务——理解你听到的内容并写下来。",
        "分析所给的音频输入并返回有意义的文字表示。",
        "认真聆听音频文件，并用文字描述它。",
        "这是一个将音频信息映射到书面解释的任务。",
        "目标是为所提供的音频片段生成可读的文字描述。",
        "听取声音，并将其转化为简洁的文字分析。",
        "将录音内容进行语义映射，生成文字化的叙述。",
        "根据提供的音频创建一个反映主要特征的文字标题或短句。",
        "这是一个从音频样本生成文字解析的过程。",
        "聆听音频，并用语言表达其内容。",
        "用信息化的文字输出捕捉音频片段的核心内容。",
        "将声学信息转化为描述性的书面语言。",
        "生成一个自然语言的文字说明，概括音频片段。",
        "解读音频信号，并用描述性文字表达。",
        "将听觉输入转化为连贯的文字陈述。",
        "处理所提供的音频，并返回基于文本的分析性描述。",
        "该任务要求你用文字形式描写并分析录音内容。"
    ]



    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            slice_frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        kwargs['audio_pt_modes_config_name'] = kwargs.get("audio_pt_modes_config_name", "dual_channel_v2")
        self.index_ds = index_ds_class(path, tokenizer=tokenizer, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        else:
            self.preprocessor_speech_conf = None
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.slice_frontend = slice_frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.slice_fs = self.fs if slice_frontend is None else slice_frontend.fs
        assert self.fs == self.slice_fs, f"frontend fs: {self.fs} != slice_frontend fs: {self.slice_fs}"
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        self.slice_permute = self.permute
        if self.slice_frontend is not None:
            self.slice_permute = isinstance(self.slice_frontend, WhisperFrontend)

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.min_token_length = kwargs.get("min_token_length", 1024)
        self.max_slice_duration = kwargs.get("max_merged_slice_duration_sec", 5.0)
        if not isinstance(self.max_slice_duration, float):
            print(f'max_slice_duration range: {self.max_slice_duration}')
            self.max_slice_duration = max(self.max_slice_duration[0], self.max_slice_duration[1])
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 6)
        # self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        # self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        # self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.text_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.sample_end_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        logging.info(f"init continuous_in: {self.continuous_in}")

        self.load_audio_num_threads = kwargs.get("load_audio_num_threads", 4)

        # self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.speech_token_id_offset = 0 # for channels model, no offset needed
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        print(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")
        self.continuous_audio_frame_rate = kwargs.get("continuous_audio_frame_rate", 12.5)
        print (f"continuous audio frame rate: {self.continuous_audio_frame_rate} Hz")

        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")

        self.no_audio_vocab = kwargs.get("no_audio_vocab", False)

        if self.no_audio_vocab:
            print(f"no audio vocab is enabled")
        else:
            # pt mode
            self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-text|>") # unimodal-text
            self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-audio|>") # unimodal-audio
            self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-audio-to-text|>") # mapping-audio-to-text
            self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-text-to-audio|>") # mapping-text-to-audio 
            self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic|>") # interleave-audio-to-semantic 
            self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-text|>") # interleave-audio-to-text
            self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic-and-text|>") # interleave-audio-to-semantic-and-text
            assert self.pt_mode_unimodal_text_token_id is not None
        self.text_vocab_offset = self.tokenizer.vocab_size
        # assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print (f'audio code book size: {self.audio_code_book_size}')
        self.mask_token_id = -60

        self.blank_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)

        # self.add_task_id = kwargs.get("add_task_id", False)
        # self.add_modalidy_id = kwargs.get("add_modalidy_id", False)
        # self.add_system_prompt = kwargs.get("add_system_prompt", True)
        self.add_task_id = False
        self.add_modalidy_id = True
        self.add_system_prompt = True

        self.add_segment_end_id = kwargs.get("add_segment_end_id", False)
        self.add_sample_end_id = kwargs.get("add_sample_end_id", True)

        self.gc_counter = 0

    def get_audio_token_id(self, audio_token):
        if self.no_audio_vocab:
            return 0
        if isinstance(audio_token, int):
            return audio_token - self.text_vocab_offset + self.audio_code_book_size
        elif isinstance(audio_token, str):
            return self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids(audio_token))
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")


    def get_pt_mode_token_id(self, pt_mode):
        if self.no_audio_vocab:
            return None
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_pt_mode_system_prompt(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return ''
        elif pt_mode == 'unimodal-audio':
            return ''
        elif pt_mode == 'mapping-audio-to-text':
            return random.choice(MinMo_S2S_Interleave_Pretrain_S2S_Random_S2T.ASR_Prompts)
        elif pt_mode == 'mapping-text-to-audio':
            return 'This is a mapping-text-to-audio task.'
        elif pt_mode == 'mapping-audio-to-caption':
            return random.choice(MinMo_S2S_Interleave_Pretrain_S2S_Random_S2T.Caption_Prompts)
        elif pt_mode == 'interleave-audio-to-semantic':
            return ''
        elif pt_mode == 'interleave-audio-to-text':
            return ''
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return ''
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def _extract_audio_segment(self, source_wav, start_time, end_time):
        """使用ffmpeg切分音频文件"""
        try:
            start_time_ffmpeg = time.time()
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)

            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-i', source_wav, '-ss', str(start_time),
                '-t', str(duration), '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logging.error(f"ffmpeg failed: {result.stderr}")
                return None
            end_time_ffmpeg = time.time()
            # print (f"extract_audio_segment time: {end_time_ffmpeg - start_time_ffmpeg}")
            return temp_path
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return None

    def _process_audio_cache(self, segments_dict):
        """
        处理音频缓存逻辑
        
        Args:
            segments_dict: 音频片段字典
            
        Returns:
            tuple: (cache_hit, segment_cache, local_audio_dir)
                - cache_hit: 是否命中缓存
                - segment_cache: 片段缓存字典 {segment_id: local_wav_path}
                - local_audio_dir: 本地音频目录
        """
        segment_cache = {}  # key: segment_id, value: local_wav_path
        cache_hit = False
        local_audio_dir = None
        
        # cache
        from funasr.datasets.minmo_datasets.cache_utils.pre_cache_audio_data import _sharded_dirs, _sha256_hex, _format_time_component, _is_file_intact
        assert self.audio_cache_type == "segments"  # only support segments cache now
        # bucket_and_rest = source_wav[len("oss://"):]
        bucket_name, object_name = 'example-storage-6', 'temp_obj'
        digest = _sha256_hex(object_name)
        remote_cache_dir = _sharded_dirs(self.audio_cache_dir, digest)
        local_audio_dir = os.path.join("/tmp/tmp_wav", uuid.uuid4().hex)

        expected_cache_files = set()
        cache_segment_names = set()
        segment_name2cache_segment_path = {}
        
        for segment_id, segment in segments_dict.items():
            segment_start_time = segment["start_time"]
            segment_end_time = segment["end_time"]
            if segment_start_time is None or segment_end_time is None:
                continue
            prefix = "audio_segment"
            segment_name = f"{prefix}-{segment_id}-{_format_time_component(segment_start_time)}-{_format_time_component(segment_end_time)}.{self.audio_cache_format}"
            if segment["wav_path"] is not None:
                if segment["wav_path"].startswith("oss://"):
                    segment_cache[segment_id] = segment['wav_path']
                    continue
                else:
                    cache_segment_path = segment['wav_path']
                    assert os.path.exists(cache_segment_path), f"cache_segment_path not exists: {cache_segment_path}"
            else:
                print ('should not be logged, wav_path will always exist.')
                cache_segment_path = os.path.join(remote_cache_dir, segment_name)

            segment_name2cache_segment_path[segment_name] = cache_segment_path
            cache_segment_names.add((segment_id, segment_name))
            expected_cache_files.add(cache_segment_path)
            
        if all(_is_file_intact(p) for p in expected_cache_files):
            if self.audio_cache_format != "wav":
                # convert to wav and save to local cache
                os.makedirs(local_audio_dir, exist_ok=True)
                convert_args = []
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = segment_name2cache_segment_path[segment_name]
                    local_cache_segment_path = os.path.join(local_audio_dir, segment_name + '.wav')
                    segment_cache[segment_id] = local_cache_segment_path
                    convert_args.append((cache_segment_path, local_cache_segment_path, 16000, 1))
                with ThreadPoolExecutor(max_workers=20) as executor:
                    results = executor.map(convert_to_wav, convert_args)
                if all(results):
                    cache_hit = True
                else:
                    logging.warning(f"cache miss, not all files converted, drop item")
                    cache_hit = False
            else:
                for segment_id, segment_name in cache_segment_names:
                    cache_segment_path = os.path.join(remote_cache_dir, segment_name)
                    segment_cache[segment_id] = cache_segment_path
                cache_hit = True
        else:
            logging.warning(f"cache miss, not all files are intact, drop item")
            cache_hit = False
        
        # preprocess audio
        if self.preprocessor_speech_conf is not None:
            probatility = self.preprocessor_speech_conf.get('probability', 1.0)
            all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
            use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
            batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            batch_use_6mic = batch_random_source.random() < use_6mic_probability
            conditions = self.preprocessor_speech_conf.get('conditions', [])

            def preprocess_segment(segment_id, segment_audio):
                # if the conditions met
                segment = segments_dict[segment_id]
                conditions_met = True
                for condition in conditions:
                    segment_key = condition.get('segment_key', None)
                    value_min = condition.get('value_min', None)
                    value_max = condition.get('value_max', None)
                    # if segment_key is None or value_min is None or value_max is None:
                    #     continue
                    segment_key_parts = segment_key.split('.')
                    current_item = segment
                    for key in segment_key_parts:
                        if key in current_item:
                            current_item = current_item[key]
                        else:
                            current_item = None
                            break
                    if current_item is None:
                        conditions_met = False
                        break
                    if value_min is not None and current_item < value_min:
                        conditions_met = False
                        break
                    if value_max is not None and current_item > value_max:
                        conditions_met = False
                        break
                if not conditions_met:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met conditions")
                    return segment_audio
                segment_unique_id = segment['unique_id']
                random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
                if random_source.random() > probatility:
                    # print(f"preprocess_segment segment_audio: {segment_audio.shape} not met probatility")
                    return segment_audio
                # run the preprocessing
                # print(f"preprocess_segment segment_audio: {segment_audio.shape}")
                segment_audio = self._preprocess_audio(segment_audio, random_source=random_source, use_6mic=batch_use_6mic)
                return segment_audio

        # load and extract fbank
        def load_only(segment_id):
            """
            Just load the samples for a given segment_id.
            """
            cache_segment_path = segment_cache[segment_id]
            # check oss segment flac file size
            if cache_segment_path.startswith("oss://"):
                oss_file_size = get_oss_object_size(cache_segment_path)  # 字节，失败为 None
                if oss_file_size is not None and oss_file_size > 20 * 1024 * 1024:
                    print (f"OSS segment file size {oss_file_size} bytes (>20MB) exceeds limit: {cache_segment_path}")
                    raise ValueError(
                        f"OSS segment file size {oss_file_size} bytes (>15MB) exceeds limit: {cache_segment_path}"
                    )
            duration = [0, None]
            segment_samples = load_audio_text_image_video(
                cache_segment_path,
                fs=self.fs,
                duration=duration
            )
            return segment_id, cache_segment_path, segment_samples
        
        # Parallel loading step
        with ThreadPool(self.load_audio_num_threads) as pool:
            for segment_id, cache_segment_path, segment_samples in pool.imap(load_only, segments_dict.keys()):
                # The rest remains sequential
                if self.preprocessor_speech_conf is not None:
                    segment_samples = preprocess_segment(segment_id, segment_samples)

                segment_fbank, segment_fbank_length = extract_fbank(
                    segment_samples,
                    data_type=self.data_type,
                    frontend=self.frontend,
                    is_final=True
                )

                segment_cache[segment_id] = {
                    'path': cache_segment_path,
                    'samples': segment_samples,
                    'fbank': segment_fbank,
                    'fbank_length': segment_fbank_length,
                }
        
        return cache_hit, segment_cache, local_audio_dir

    def _process_continuous_audio_loading(self, user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant=False):
        """
        提取 slice 对应音频的fbank
        """
        duration_list = []
        
        # test if local cache hit
        cache_duration_list = []
        data_src_list = []
        # cache miss
        if cache_hit:
            if self.audio_cache_type == "slices":
                # prefix = "audio_slice"
                raise NotImplementedError("slices cache is not supported")
            elif self.audio_cache_type == "segments":
                for i, (user_prompt, target_out) in enumerate(
                        zip(user, assistant)
                ):
                    for p_i, prompt in enumerate([user_prompt, target_out]):
                        if prompt is not None:
                            slice_start_time = prompt["start_time"]
                            slice_end_time = prompt["end_time"]
                            slice_mode = prompt["slice_mode"]
                            if 'c' not in slice_mode:
                                data_src_list.append((None, i))
                            else:
                                segment_id = prompt.get("segment_id", None)
                                if segment_id is not None:
                                    segment = segments_dict[segment_id]
                                    segment_start_time = segment["start_time"]
                                    segment_end_time = segment["end_time"]
                                    assert segment_start_time is not None and segment_end_time is not None
                                    offset = slice_start_time - segment_start_time
                                    assert offset >= 0 and offset <= segment_end_time - segment_start_time + 0.0001, f"offset: {offset} is not valid, segment time start: {segment_start_time}, segment time end: {segment_end_time}, slice time start: {slice_start_time}, slice time end: {slice_end_time}, segment: {segment}, slice: {prompt}"
                                    slice_duration = slice_end_time - slice_start_time
                                    assert slice_duration >= 0.01 and slice_duration <= 10000
                                    rel_start_time = offset
                                    rel_end_time = rel_start_time + slice_duration
                                    cache = segment_cache[segment_id]
                                    if isinstance(cache, dict):
                                        if self.slice_frontend is None:
                                            assert 'fbank' in cache and 'fbank_length' in cache, f"segment cache missing fbank key"
                                            segment_fbank = cache['fbank']
                                            segment_fbank_length = cache['fbank_length']
                                            slice_start_frame = int(rel_start_time * 1000 / self.frontend.frame_shift)
                                            slice_num_frames = int(slice_duration * 1000 / self.frontend.frame_shift)
                                            segment_num_frames = segment_fbank_length[0].item()
                                            # assert slice_start_frame + slice_num_frames <= segment_num_frames, f"slice out of segment bounds: {slice_start_frame} + {slice_num_frames} > {segment_num_frames}"
                                            if slice_start_frame + slice_num_frames > segment_num_frames:
                                                # some slightly out-of-bound data (slice_num_frames == segment_num_frames + 1)
                                                slice_num_frames = segment_num_frames - slice_start_frame
                                            slice_fbank = segment_fbank[:, :, slice_start_frame:slice_start_frame + slice_num_frames]
                                            data_src_list.append((slice_fbank, i))
                                        else:
                                            samples = cache['samples']
                                            slice_start_sample = int(rel_start_time * self.slice_fs)
                                            slice_num_samples = int(slice_duration * self.slice_fs)
                                            slice_samples = samples[slice_start_sample:slice_start_sample + slice_num_samples]
                                            data_src_list.append((slice_samples, i))
                                    else:
                                        raise ValueError(f"segment cache is not a dict")
        else:
            raise NotImplementedError("cache miss, not supported yet")

        # 提前多进程调用load_audio_text_image_video，加速训练
        turn_idx2data_src = defaultdict(list)
        for data_src, turn_idx in data_src_list:
            turn_idx2data_src[turn_idx].append(data_src)
        
        return turn_idx2data_src

    
    def extract_fbank_mock(self, data_src, data_type, frontend, is_final):
        assert data_type == "sound"
        fs = frontend.fs
        fps = 1000 // frontend.frame_shift
        num_frames = int(data_src.shape[0] / fs * fps)
        speech = torch.zeros(1, 128, num_frames, dtype=torch.float32)
        speech_lengths = torch.tensor([num_frames], dtype=torch.int32)
        return speech, speech_lengths


    def _process_continuous_audio_slice(self, data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type, slice_duration, current_segment):

        # if self.preprocessor_speech is not None:
        #     data_src = self._preprocess_audio(data_src)

        # 提取fbank特征
        if isinstance(data_src, torch.Tensor) and self.slice_frontend is None:
            speech = data_src
            speech_lengths = torch.tensor([data_src.shape[2]], dtype=torch.int32)
        else:
            speech, speech_lengths = extract_fbank(
                data_src,
                data_type=self.data_type,
                frontend=self.slice_frontend,
                is_final=True,
            )  # speech: [b, T, d]
        
        if self.slice_permute:
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz
        
        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]

            # check fake tokens length with slice_duration
            expected_fake_token_len = int(slice_duration * self.continuous_audio_frame_rate)
            if abs(fake_token_len_i - expected_fake_token_len) > 2:
                print (f"fake token length mismatch: {fake_token_len_i} != {expected_fake_token_len}, slice_duration: {slice_duration}; the segment: {json.dumps(current_segment, ensure_ascii=False)}; speech shape: {speech.shape}; speech_lengths: {speech_lengths[0].item()}")
                raise ValueError(f"fake token length mismatch: {fake_token_len_i} != {expected_fake_token_len}, slice_duration: {slice_duration}, continuous_audio_frame_rate: {self.continuous_audio_frame_rate} Hz")
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type


    def _preprocess_audio(self, audio_tensor: torch.Tensor, random_source=None, use_6mic=True):
        if self.preprocessor_speech is not None:
            processed = self.preprocessor_speech(audio_tensor.numpy(), use_6mic=use_6mic, random_source=random_source)
            processed = torch.from_numpy(processed)
        else:
            processed = audio_tensor
        return processed


    def _preprocess_file(self, input_audio_path: str, output_audio_path: str, random_source=None, use_6mic=True) -> bool:
        import soundfile as sf
        audio_tensor = load_audio_text_image_video(input_audio_path, self.fs)
        processed_audio_tensor = self._preprocess_audio(audio_tensor, random_source=random_source, use_6mic=use_6mic)
        sf.write(output_audio_path, processed_audio_tensor.numpy(), self.fs)
        return True


    def _preprocess_audio_cache(self, segments_dict: dict, segment_cache: dict):
        probatility = self.preprocessor_speech_conf.get('probability', 1.0)
        all_segments_id = '_'.join([segments_dict[segment_id]['unique_id'] for segment_id in sorted(segments_dict.keys())])
        use_6mic_probability = self.preprocessor_speech_conf.get('use_6mic_probability', 0.5)
        batch_random_source = random.Random(f"{all_segments_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
        batch_use_6mic = batch_random_source.random() < use_6mic_probability
        conditions = self.preprocessor_speech_conf.get('conditions', [])

        def process_segment(segment_id):
            # if the conditions met
            segment = segments_dict[segment_id]
            conditions_met = True
            for condition in conditions:
                segment_key = condition.get('segment_key', None)
                value_min = condition.get('value_min', None)
                value_max = condition.get('value_max', None)
                # if segment_key is None or value_min is None or value_max is None:
                #     continue
                segment_key_parts = segment_key.split('.')
                current_item = segment
                for key in segment_key_parts:
                    if key in current_item:
                        current_item = current_item[key]
                    else:
                        current_item = None
                        break
                if current_item is None:
                    conditions_met = False
                    break
                if value_min is not None and current_item < value_min:
                    conditions_met = False
                    break
                if value_max is not None and current_item > value_max:
                    conditions_met = False
                    break
            if not conditions_met:
                return
            segment_unique_id = segment['unique_id']
            random_source = random.Random(f"{segment_unique_id}_epoch{self.epoch}_data_split_i{self.data_split_i}")
            if random_source.random() > probatility:
                return
            # run the preprocessing
            segment_path = segment_cache[segment_id]
            preprocessed_path = segment_path + '.preprocessed.wav'
            self._preprocess_file(segment_path, preprocessed_path, random_source=random_source, use_6mic=batch_use_6mic)
            segment_cache[segment_id] = preprocessed_path

        for segment_id in segments_dict.keys():
            process_segment(segment_id)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     results = executor.map(process_segment, segments_dict.keys())   # dingding fe is not friendly to threading

    def _get_labels_with_one_pad_loss(self, labels, start_id, pad_id, blank_audio_token_id, max_tokens):
        new_labels = []
        has_pad = False
        for label in labels:
            if label == start_id:
                new_labels.append(-100)
            elif label == pad_id:
                if not has_pad:
                    new_labels.append(label)
                    has_pad = True
                else:
                    new_labels.append(-100)
            elif blank_audio_token_id > 0 and label == blank_audio_token_id:
                new_labels.append(-100)
            else:
                new_labels.append(label)
        new_labels = new_labels + [-100] * (max_tokens - len(new_labels))
        return new_labels

    def _process_td_slice(
        self,
        audio_tokens,
        slice_duration,
        text_tokens,
        text,
        segment_id,
        slice,
        online_speech_tokenize_segment_ids
    ):
        """
        处理  td slice
        
        返回: (slice_ids, slice_ids_text, input_ids_audio_increment, labels_audio_increment, 
            input_ids_increment, labels_increment)
        """
        slice_ids = []
        if self.add_modalidy_id:
            slice_ids += [self.audio_start_id]
        slice_ids += [self.get_audio_token_id(self.blank_audio_token_id)] * self.blank_audio_token_count
        if audio_tokens is not None and len(audio_tokens) > 0:
            slice['onlie_discrete_speech_token'] = False
        else:
            num_tokens = int(slice_duration * self.discrete_tokenizer_frame_rate)
            audio_tokens = [-50] * num_tokens
            slice['onlie_discrete_speech_token'] = True
            online_speech_tokenize_segment_ids.add(segment_id)
        slice_ids.extend(audio_tokens)
        num_audio_tokens = len(slice_ids)
        
        slice_ids_text = []
        if self.add_modalidy_id:
            slice_ids_text += [self.text_start_id]
        if text_tokens is not None:
            slice_ids_text.extend(text_tokens)
        else:
            slice_ids_text.extend(self.tokenizer.encode(text))
        num_text_tokens = len(slice_ids_text)
        if num_text_tokens > num_audio_tokens:
            print(f"num_text_tokens > num_audio_tokens: {num_text_tokens} > {num_audio_tokens}")
        max_tokens = max(num_text_tokens, num_audio_tokens)

        input_ids_audio_increment = slice_ids + [self.get_audio_token_id(self.tokenizer.pad_token_id)] * (max_tokens - len(slice_ids))
        labels_audio_increment = self._get_labels_with_one_pad_loss(
            slice_ids, 
            self.get_audio_token_id(self.audio_start_id), 
            self.get_audio_token_id(self.tokenizer.pad_token_id), 
            self.get_audio_token_id(self.blank_audio_token_id), 
            max_tokens
        )

        input_ids_increment = slice_ids_text + [self.tokenizer.pad_token_id] * (max_tokens - len(slice_ids_text))
        labels_increment = self._get_labels_with_one_pad_loss(
            slice_ids_text, 
            self.text_start_id, 
            self.tokenizer.pad_token_id, 
            -1, 
            max_tokens
        )
        
        return (slice_ids, slice_ids_text, input_ids_audio_increment, labels_audio_increment, 
                input_ids_increment, labels_increment)

    def _process_t_slice(
        self,
        text_tokens,
        text,
        slice,
        text_only=False
    ):
        """
        处理 t slice（纯文本部分）
        
        参数:
            text_tokens: 预编码的文本token列表，如果为None则使用text进行编码
            text: 原始文本
            slice: slice字典，会被修改（设置onlie_discrete_speech_token）
        
        返回: (slice_ids, input_ids_increment, labels_increment, 
            input_ids_audio_increment, labels_audio_increment)
        """
        slice['onlie_discrete_speech_token'] = False
        slice_ids = []
        if text_tokens is not None:
            slice_ids.extend(text_tokens)
        else:
            slice_ids.extend(self.tokenizer.encode(text))

        if self.add_modalidy_id and not text_only:
            slice_ids += [self.text_end_id]
        
        input_ids_increment = slice_ids
        labels_increment = slice_ids
        input_ids_audio_increment = [self.mask_token_id] * len(slice_ids)
        labels_audio_increment = [-100] * len(slice_ids)
        
        return (slice_ids, input_ids_increment, labels_increment, 
                input_ids_audio_increment, labels_audio_increment)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()
        time_start = time.time()

        output = None
        timings = OrderedDict()

        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
                user = item["user"]
                assistant = item["assistant"]
                segments_dict = item.get("segments_dict", {})
                pt_mode = item.get("pt_mode", None)
                # print(f"pt_mode: {pt_mode}")
            except Exception as e:
                # logging.error(f"Error getting item: {e}")
                # logging.error(f"Traceback: {traceback.format_exc()}")
                continue

            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            
            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []

            local_audio_dir = None
            segment_cache = {} # key: segment_id, value: local_wav_path
            cache_hit = False
            try:
                # locate cache or download audio
                if pt_mode not in ['unimodal-text']:
                    time_start_audio_cache = time.time()
                    cache_hit, segment_cache, local_audio_dir = self._process_audio_cache(segments_dict)
                    time_end_audio_cache = time.time()
                    timings['process_audio_cache'] = time_end_audio_cache - time_start_audio_cache
    
                    if not cache_hit and self.drop_cache_miss_items:
                        logging.warning(f"cache miss, drop item")
                        continue

                # extract fbank 
                if pt_mode not in ['unimodal-text', 'unimodal-audio', 'mapping-text-to-audio']:
                    if pt_mode == 'mapping-audio-to-text' or pt_mode == 'mapping-audio-to-caption':
                        both_user_assistant = True
                    else:
                        both_user_assistant = False
                    time_start_process_continuous_audio_loading = time.time()
                    turn_idx2data_src = self._process_continuous_audio_loading(user, assistant, segments_dict, cache_hit, segment_cache, both_user_assistant)
                    time_end_process_continuous_audio_loading = time.time()
                    timings['process_continuous_audio_loading'] = time_end_process_continuous_audio_loading - time_start_process_continuous_audio_loading
                
                # add task id
                if self.add_task_id:
                    task_id = self.get_pt_mode_token_id(pt_mode)
                    input_ids.append(task_id)
                    labels.append(-100)

                    input_ids_audio.append(self.get_audio_token_id(task_id))
                    labels_audio.append(-100)

                if self.add_system_prompt:
                    system_prompt = self.get_pt_mode_system_prompt(pt_mode)
                    # print(f"system_prompt: {system_prompt}")
                    if system_prompt:
                        system_prompt_ids = self.tokenizer.encode(system_prompt)
                        input_ids.extend(system_prompt_ids)
                        labels.extend([-100] * len(system_prompt_ids))
                        input_ids_audio.extend([self.mask_token_id] * len(system_prompt_ids))
                        labels_audio.extend([-100] * len(system_prompt_ids))

                pre_segment_id = None
                pre_sample_id = None
                online_speech_tokenize_segment_ids = set()
                slice_modes = defaultdict(int)
                time_start_slice_loop = time.time()
                for i, (user_slice, assistant_slice) in enumerate(zip(user, assistant)):
                    if len(input_ids) > self.max_token_length:
                        logging.info(
                            f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{self.max_token_length}"
                        )
                        break
                    
                    for s_i, slice in enumerate([user_slice, assistant_slice]):
                        if slice is None:
                            continue

                        slice_ids = []
                        speech = []
                        speech_lengths = []
                        
                        text, start_time, end_time, segment_id, audio_tokens, text_tokens, slice_mode = slice["content"], slice["start_time"], slice["end_time"], slice["segment_id"], slice.get("tokens", None), slice.get("text_tokens", None), slice.get("slice_mode", None)

                        current_sample_id = segments_dict[segment_id]['raw_audio_path']
                        current_segment = segments_dict[segment_id]

                        if start_time is not None and end_time is not None:
                            slice_duration = end_time - start_time
                            if not (slice_duration >= 0.1 and slice_duration <= self.max_slice_duration + 0.1):
                                logging.warning(f"slice duration is not valid: {slice_duration}, slice: {slice}")
                                continue
                        else:
                            slice_duration = None

                        slice_modes[slice_mode] += 1

                        # add tag for segment transition
                        if s_i == 0 and pt_mode != 'unimodal-text' and self.add_segment_end_id:
                            if pre_segment_id is None:
                                pre_segment_id = segment_id
                            elif pre_segment_id != segment_id:
                                input_ids.append(self.end_of_segment_speech_token_id)
                                labels.append(-100)
                                # input_ids_audio.append(self.get_audio_token_id(self.end_of_segment_speech_token_id))
                                input_ids_audio.append(self.mask_token_id)
                                labels_audio.append(-100)
                                pre_segment_id = segment_id

                        if self.add_sample_end_id and pt_mode != 'unimodal-text':
                            if pre_sample_id is None:
                                pre_sample_id = current_sample_id
                            elif pre_sample_id != current_sample_id:
                                input_ids.append(self.sample_end_id)
                                labels.append(-100)
                                # input_ids_audio.append(self.get_audio_token_id(self.sample_end_id))
                                input_ids_audio.append(self.mask_token_id)
                                labels_audio.append(-100)
                                pre_sample_id = current_sample_id

                        
                        if pt_mode == 'unimodal-text':
                            if not text:
                                continue
                            slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                self._process_t_slice(
                                    text_tokens,
                                    text,
                                    slice,
                                    text_only=True
                                )
                            
                            input_ids = input_ids + input_ids_inc
                            labels = labels + labels_inc
                            input_ids_audio = input_ids_audio + input_ids_audio_inc
                            labels_audio = labels_audio + labels_audio_inc

                        # elif pt_mode == 'unimodal-audio':
                        #     assert slice_mode == 'td' or slice_mode == 't'
                        #     if slice_mode == 'td':
                        #         slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                        #             self._process_td_slice(
                        #                 audio_tokens,
                        #                 slice_duration,
                        #                 text_tokens,
                        #                 text,
                        #                 segment_id,
                        #                 slice,
                        #                 online_speech_tokenize_segment_ids
                        #             )
                                
                        #         input_ids_audio = input_ids_audio + input_ids_audio_inc
                        #         labels_audio = labels_audio + labels_audio_inc
                        #         input_ids = input_ids + input_ids_inc
                        #         labels = labels + labels_inc
                        #     else:
                        #         slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                        #             self._process_t_slice(
                        #                 text_tokens,
                        #                 text,
                        #                 slice
                        #             )
                                
                        #         input_ids = input_ids + input_ids_inc
                        #         labels = labels + labels_inc
                        #         input_ids_audio = input_ids_audio + input_ids_audio_inc
                        #         labels_audio = labels_audio + labels_audio_inc

                        elif pt_mode == 'mapping-audio-to-text' or pt_mode == 'mapping-audio-to-caption':
                            # ASR or Caption
                            slice['onlie_discrete_speech_token'] = False
                            # continuous audio part
                            data_src = turn_idx2data_src[i][s_i]
                            speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                    data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type, slice_duration, current_segment
                                )
                            if self.add_modalidy_id:
                                slice_ids = [self.audio_start_id] + slice_ids + [self.audio_end_id]
                            input_ids += slice_ids
                            labels += [-100] * len(slice_ids)
                            input_ids_audio += [self.mask_token_id] * len(slice_ids)
                            labels_audio += [-100] * len(slice_ids)

                            slice_mode = slice_mode.strip('c')  # remove 'c' from 'ctd'

                            # td part or t part
                            # if slice_mode == 'td':
                            #     slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                            #         self._process_td_slice(
                            #             audio_tokens,
                            #             slice_duration,
                            #             text_tokens,
                            #             text,
                            #             segment_id,
                            #             slice,
                            #             online_speech_tokenize_segment_ids
                            #         )
                                
                            #     input_ids_audio = input_ids_audio + input_ids_audio_inc
                            #     labels_audio = labels_audio + labels_audio_inc
                            #     input_ids = input_ids + input_ids_inc
                            #     labels = labels + labels_inc
                            # elif slice_mode == 't':

                            # text only as default
                            slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                self._process_t_slice(
                                    text_tokens,
                                    text,
                                    slice,
                                    text_only=False
                                )
                            
                            input_ids = input_ids + input_ids_inc
                            labels = labels + labels_inc
                            input_ids_audio = input_ids_audio + input_ids_audio_inc
                            labels_audio = labels_audio + labels_audio_inc


                        # elif pt_mode == 'mapping-text-to-audio':
                        #     # text part
                        #     if self.add_modalidy_id:
                        #         slice_ids += [self.text_start_id]
                        #     if text_tokens is not None:
                        #         slice_ids.extend(text_tokens)
                        #     else:
                        #         slice_ids.extend(self.tokenizer.encode(text))
                        #     input_ids += slice_ids
                        #     labels += [-100 if id == self.audio_start_id or id == self.text_start_id else id for id in slice_ids]
                        #     input_ids_audio += [self.mask_token_id] * len(slice_ids)
                        #     labels_audio += [-100] * len(slice_ids)

                        #     # td part 
                        #     slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                        #         self._process_td_slice(
                        #             audio_tokens,
                        #             slice_duration,
                        #             text_tokens,
                        #             text,
                        #             segment_id,
                        #             slice,
                        #             online_speech_tokenize_segment_ids
                        #         )

                        #     input_ids_audio = input_ids_audio + input_ids_audio_inc
                        #     labels_audio = labels_audio + labels_audio_inc
                        #     input_ids = input_ids + input_ids_inc
                        #     labels = labels + labels_inc
                        
                        elif pt_mode == 'interleave-audio-to-text':
                            if slice_mode == 'c':
                                slice['onlie_discrete_speech_token'] = False
                                # continuous audio part
                                data_src = turn_idx2data_src[i][s_i]
                                speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = self._process_continuous_audio_slice(
                                        data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type, slice_duration, current_segment
                                    )  
                                if self.add_modalidy_id:
                                    slice_ids = [self.audio_start_id] + slice_ids + [self.audio_end_id]
                                input_ids += slice_ids
                                labels += [-100] * len(slice_ids)
                                input_ids_audio += [self.mask_token_id] * len(slice_ids)
                                labels_audio += [-100] * len(slice_ids)
                            # elif slice_mode == 'td':
                            #     slice_ids, slice_ids_text, input_ids_audio_inc, labels_audio_inc, input_ids_inc, labels_inc = \
                            #         self._process_td_slice(
                            #             audio_tokens,
                            #             slice_duration,
                            #             text_tokens,
                            #             text,
                            #             segment_id,
                            #             slice,
                            #             online_speech_tokenize_segment_ids
                            #         )
                                
                            #     input_ids_audio = input_ids_audio + input_ids_audio_inc
                            #     labels_audio = labels_audio + labels_audio_inc
                            #     input_ids = input_ids + input_ids_inc
                            #     labels = labels + labels_inc
                            elif slice_mode == 't':
                                # text only as default
                                slice_ids, input_ids_inc, labels_inc, input_ids_audio_inc, labels_audio_inc = \
                                    self._process_t_slice(
                                        text_tokens,
                                        text,
                                        slice,
                                        text_only=False
                                    )
                                
                                input_ids = input_ids + input_ids_inc
                                labels = labels + labels_inc
                                input_ids_audio = input_ids_audio + input_ids_audio_inc
                                labels_audio = labels_audio + labels_audio_inc  
                            else:
                                raise ValueError(f"Invalid slice_mode: {slice_mode} for pt_mode: {pt_mode}")
                        else:
                            raise ValueError(f"Invalid pt_mode: {pt_mode}")

                        if len(speech) > 0:
                            fbank.append(speech[0, :, :])
                            fbank_lens.append(speech_lengths)
                time_end_slice_loop = time.time()
                timings['slice_loop'] = time_end_slice_loop - time_start_slice_loop

                if len(input_ids) > self.max_token_length:
                    logging.warning(
                        f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}"
                    )
                    continue

                if len(input_ids) < self.min_token_length:
                    logging.info(
                        f"len(input_ids): {len(input_ids)} < min_token_length: {self.min_token_length}"
                    )
                if len(slice_modes.keys()) == 1 and 'c' in slice_modes.keys():
                    print (f'[DATASET] slice_modes: {slice_modes}')
                    continue

                input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
                labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

                input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: self.max_token_length]
                labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: self.max_token_length]

                attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)

                feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
                feature_type = torch.tensor(feature_type, dtype=torch.int32)
                fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

                segment_fbank = []
                segment_fbank_lengths = []
                if pt_mode in ['unimodal-audio', 'mapping-audio-to-text', 'mapping-text-to-audio', 'interleave-audio-to-semantic', 'interleave-audio-to-semantic-and-text']:
                    # for segment-wise speech tokenizer
                    has_bad_audio = False

                    time_start_segment_loop = time.time()
                    for segment_id, segment in item["segments_dict"].items():
                        if segment_id not in online_speech_tokenize_segment_ids:
                            segment['onlie_discrete_speech_token'] = False
                            continue
                        else:
                            segment['onlie_discrete_speech_token'] = True
                        if segment_id in segment_cache:
                            cache = segment_cache[segment_id]
                            duration = [0, None]
                        else:
                            # this shall not happen if cache miss and drop_cache_miss_items is True
                            assert self.drop_cache_miss_items, f"cache miss and drop_cache_miss_items is True, segment_id: {segment_id}"
                    
                        # stime = time.time()
                        if isinstance(cache, dict):
                            one_segment_fbank = cache['fbank']
                            one_segment_fbank_length = cache['fbank_length']
                        else:
                            raise ValueError(f"segment cache is not a dict")
                            try:
                                samples = load_audio_text_image_video(load_audio_path, fs=self.fs, duration=duration)
                                # here the preprocessing was wrongly done once again
                                # if self.preprocessor_speech is not None:
                                #     samples = self._preprocess_audio(samples)
                            except Exception as e:
                                print (f"load_audio_text_image_video error: {e}")
                                has_bad_audio = True
                                break
                            # etime = time.time()
                            # print (f"load_audio_text_image_video time(example): {etime - stime}")
                            # segment_speech.append(samples)
                            # segment_speech_lengths.append(len(samples))
                            # tokens = torch.tensor(self.tokenizer.encode(segment["text"]), dtype=torch.int32)
                            # segment_token.append(tokens)
                            # segment_token_lengths.append(len(tokens))

                            # extract segment level fbank
                            # here the discrete tokenizer shares the same frontend parameters
                            # one_segment_fbank, one_segment_fbank_length = self.extract_fbank_mock(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                            one_segment_fbank, one_segment_fbank_length = extract_fbank(samples, data_type=self.data_type, frontend=self.frontend, is_final=True)
                        if self.permute:
                            one_segment_fbank = one_segment_fbank.permute(0, 2, 1)
                        segment_fbank.append(one_segment_fbank[0, :, :])
                        segment_fbank_lengths.append(one_segment_fbank_length)
                    time_end_segment_loop = time.time()
                    timings['segment_loop'] = time_end_segment_loop - time_start_segment_loop

                    if has_bad_audio:
                        print (f"has_bad_audio: {has_bad_audio}, item: {item}")
                        continue

                output = {
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                    "attention_mask": attention_mask,
                    "item": item,
                    "speech_token_id_offset": self.speech_token_id_offset
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                if len(segment_fbank) > 0:
                    output["segment_fbank"] = segment_fbank
                    output["segment_fbank_lengths"] = segment_fbank_lengths
                break
            except Exception as e:
                print (f"process_continuous_audio_loading error: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
            finally:
                if local_audio_dir is not None:
                    import shutil
                    try:
                        shutil.rmtree(local_audio_dir)
                    except Exception as e:
                        print (f"remove local_audio_dir: {local_audio_dir} failed: {e}")
                # remove segment wav path
                for segment_id, segment in item["segments_dict"].items():
                    if segment["wav_path"] is not None and not segment["wav_path"].startswith("oss://"):
                        os.remove(segment["wav_path"])

        self.gc_counter += 1
        if self.gc_counter > 5:
            gc.collect()
            self.gc_counter = 0

        time_end = time.time()

        return output


    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMo_S2S_Channels_SFT")
class MinMo_S2S_Channels_SFT(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")

        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        logging.info(f"init continuous_in: {self.continuous_in}")

        # self.speech_token_id_offset = kwargs.get("speech_token_id_offset", 151694)
        self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        logging.info(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")

        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")

        # pt mode
        self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-text|>") # unimodal-text
        self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-audio|>") # unimodal-audio
        self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-audio-to-text|>") # mapping-audio-to-text
        self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-text-to-audio|>") # mapping-text-to-audio 
        self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic|>") # interleave-audio-to-semantic 
        self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-text|>") # interleave-audio-to-text
        self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic-and-text|>") # interleave-audio-to-semantic-and-text
        assert self.pt_mode_unimodal_text_token_id is not None

        self.text_vocab_offset = self.tokenizer.vocab_size
        assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print (f'audio code book size: {self.audio_code_book_size}')
        self.mask_token_id = -60

        self.blank_audio_token_id = self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids("<|audio_blank|>"), self.text_vocab_offset, self.audio_code_book_size, self.tokenizer)
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)

    @staticmethod
    def get_audio_token_id(audio_token, text_vocab_offset, audio_code_book_size, tokenizer=None):
        if isinstance(audio_token, int):
            return audio_token - text_vocab_offset + audio_code_book_size
        elif isinstance(audio_token, str):
            if tokenizer is None:
                raise ValueError("tokenizer is required when audio_token is a string")
            return MinMo_S2S_Channels_SFT.get_audio_token_id(
                tokenizer.convert_tokens_to_ids(audio_token),
                text_vocab_offset,
                audio_code_book_size,
                tokenizer
            )
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")

    def get_pt_mode_token_id(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    @staticmethod
    def _process_continuous_audio_slice_static(
        data_src,
        slice_ids,
        input_ids,
        feature_beg,
        fake_token_len,
        feature_type,
        data_type,
        frontend,
        permute,
    ):
        """
        处理连续音频切片的静态方法版本
        
        Args:
            data_src: 音频数据源
            slice_ids: 当前切片的token id列表
            input_ids: 累积的input_ids列表
            feature_beg: 特征开始位置的列表
            fake_token_len: 假token长度的列表
            feature_type: 特征类型的列表
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作
            
        Returns:
            speech: 处理后的音频特征 [b, T, d] 或 [b, d, T] (如果permute=True)
            speech_lengths: 音频长度
            slice_ids: 更新后的slice_ids
            feature_beg: 更新后的feature_beg
            fake_token_len: 更新后的fake_token_len
            feature_type: 更新后的feature_type
        """
        speech, speech_lengths = extract_fbank(
            data_src,
            data_type=data_type,
            frontend=frontend,
            is_final=True,
        )  # speech: [b, T, d]
        
        if permute:
            # if whisper, do this to get [b, T, d]
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz
        
        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type

    @staticmethod
    def _prepare_single_data(
        item,
        tokenizer,
        mask_token_id,
        blank_audio_token_id,
        blank_audio_token_count,
        max_token_length,
        speech_token_id_offset,
        text_vocab_offset,
        audio_code_book_size,
        pattern,
        fs,
        get_audio_token_id_func,
        data_type,
        frontend,
        permute,
        infer_mode=False,
    ):
        """
        处理单个数据项，生成input_ids、labels等
        
        Args:
            item: 数据项，包含system、user、assistant字段
            tokenizer: tokenizer对象
            mask_token_id: mask token的id
            blank_audio_token_id: blank audio token的id
            blank_audio_token_count: blank audio token的数量
            max_token_length: 最大token长度
            speech_token_id_offset: speech token id的偏移量
            pattern: 用于分割文本的正则表达式pattern
            fs: 采样率
            get_audio_token_id_func: 获取audio token id的函数
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作
            
        Returns:
            output: 包含处理后的数据的字典，如果处理失败则返回None
        """
        system = item["system"]
        user = item["user"]
        assistant = item['assistant']
        if len(user) < 1 and len(assistant) < 1:
            logging.warning(f"item is error: {item}")
            return None
        
        # input_ids for text tokens and continuous audio features
        input_ids, labels, fbank, fbank_lens, \
            feature_beg, feature_type, fake_token_len = (
            [], [], [], [], [], [], []
        )
        # for input and output audio channels
        # input_ids_audio for discrete audio tokens
        input_ids_audio, labels_audio = [], []

        bad_data = False

        try:
            for i, (system_slice, user_slice, assistant_slice) in enumerate(zip(system, user, assistant)):
                if len(input_ids) > max_token_length:
                    logging.info(
                        f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{max_token_length}"
                    )
                    break
                if bad_data:
                    break

                for s_i, slice in enumerate([system_slice, user_slice, assistant_slice]):
                    # FIXME make sure system_slice is None, for i != 0
                    if slice is None:
                        continue
                    # system prompt only in the first turn
                    if s_i == 0 and i != 0:
                        continue
                    
                    # text and continuous audio feature
                    slice_ids = []
                    # discrete audio token for feature fusion
                    slice_ids_audio = []
                    speech = []
                    speech_lengths = []
                    
                    # tc(t and continuous audio); tcd(t and continuous audio, plus discrete audio); d(only discrete audio); td(text, plus discrete audio)
                    # fusion is only supported for text and discrete audio
                    slice_mode = slice['mode'] 
                    slice_text = slice['text'] # containing text and audio path(for continuous audio feature)
                    slice_audio = slice['audio'] # only for input fusion mode

                    if 'd' in slice_mode and (slice_audio is None or slice_audio == ''):
                        print (f"slice_audio is None or slice_audio == '' for slice_mode: {slice_mode}")
                        bad_data = True
                        break

                    if s_i == 0:
                        prefix_text_ids = tokenizer.encode('<|im_start|>system\n')
                        # TODO: add more tokens to audio vocab? like system, user, assistant?
                        prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids)) 
                        suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                        suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids)) 
                    elif s_i == 1:
                        prefix_text_ids = tokenizer.encode('<|im_start|>user\n')
                        prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids)) 
                        suffix_text_ids = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n')
                        suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids)) 
                    elif s_i == 2:
                        prefix_text_ids = []
                        prefix_audio_ids = []
                        if slice_text is not None and slice_text != '':
                            suffix_text_ids = tokenizer.encode('<|im_end|>')
                        else:
                            suffix_text_ids = [mask_token_id]
                        if slice_audio is not None and slice_audio != '':
                            suffix_audio_ids = [get_audio_token_id_func(tokenizer.convert_tokens_to_ids("<|im_end|>"), text_vocab_offset, audio_code_book_size, tokenizer)] * (len(suffix_text_ids)) 
                        else:
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids)) 


                    # add prefix ids for both 
                    slice_ids += prefix_text_ids
                    slice_ids_audio += prefix_audio_ids

                    if slice_text is not None and slice_text != '':
                        # slice_ids += prefix_text_ids
                        splits = pattern.split(slice_text)
                        # TODO: handle multiple audio paths in one slice
                        for k, sub_str in enumerate(splits):
                            if sub_str is None or sub_str == '':
                                continue
                            if not sub_str.startswith("<|startofspeech|>"):
                                sub_token = tokenizer.encode(sub_str)
                                slice_ids.extend(sub_token)
                            else:
                                sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>","")
                                # audio path for continuous audio feature
                                if sub_str.startswith("!"):
                                    # if not sub_str[1:].startswith("oss"):
                                    #     bad_data = True
                                    #     print (f"illegal audio path: {sub_str}")
                                    #     break

                                    data_src = load_audio_text_image_video(sub_str[1:], fs=fs, use_ffmpeg=True)

                                    speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = (
                                        MinMo_S2S_Channels_SFT._process_continuous_audio_slice_static(
                                            data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type,
                                            data_type, frontend, permute
                                        )
                                    )

                                    # move the code here to handle multiple audio paths in one slice
                                    if len(speech) > 0:
                                        fbank.append(speech[0, :, :])
                                        fbank_lens.append(speech_lengths)
                                else:
                                    raise ValueError(f"Should be valid audio path starting with !: {sub_str}")
                        # slice_ids += suffix_text_ids

                    if slice_audio is not None and slice_audio != '':
                        # slice_ids_audio += prefix_audio_ids
                        assert slice_mode == 'tcd' or slice_mode == 'td' or slice_mode == 'd', f'slice model is {slice_mode}'
                        # audio latency
                        slice_audio = slice_audio.replace("<|startofaudiotoken|>", "").replace("<|endofaudiotoken|>","")
                        slice_ids_audio += [blank_audio_token_id] * blank_audio_token_count
                        for x in slice_audio.split("|"):
                            if x.startswith("c"):
                                # discrete audio token id should be put in slice_ids_audio
                                slice_ids_audio.append(int(x[1:]))
                        # slice_ids_audio += suffix_audio_ids

                        use_pad_id = True
                    else:
                        use_pad_id = False

                    max_tokens = max(len(slice_ids), len(slice_ids_audio))
                    
                    if use_pad_id:
                        slice_ids += [tokenizer.pad_token_id] * (max_tokens - len(slice_ids))
                        slice_ids_audio += [get_audio_token_id_func(tokenizer.pad_token_id, text_vocab_offset, audio_code_book_size, tokenizer)] * (max_tokens - len(slice_ids_audio))
                    else:
                        slice_ids += [mask_token_id] * (max_tokens - len(slice_ids)) 
                        slice_ids_audio += [mask_token_id] * (max_tokens - len(slice_ids_audio))

                    # add suffix ids for both 
                    slice_ids += suffix_text_ids
                    slice_ids_audio += suffix_audio_ids

                    input_ids += slice_ids
                    if s_i == 2:
                        labels += [id if id not in [mask_token_id] else -100 for id in slice_ids]
                    else:
                        labels += [-100] * len(slice_ids)

                    input_ids_audio += slice_ids_audio
                    if s_i == 2:
                        labels_audio += [id if id not in [mask_token_id] else -100 for id in slice_ids_audio]
                    else:
                        labels_audio += [-100] * len(slice_ids_audio)

                    # print non -100 label audio
                    # for label in labels_audio:
                    #     if label != -100:
                    #         print (f"non -100 label audio: {label}")

                    assert len(input_ids) == len(labels)

            if bad_data:
                return None

            if len(input_ids) > max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {max_token_length}"
                )
                return None

            if not infer_mode and len(input_ids) <= 10:
                print (f"len(input_ids) <= 10: {len(input_ids)}")
                return None

            # 如果labels全是-100，则返回None
            if not infer_mode and all(label == -100 for label in labels) and all(label == -100 for label in labels_audio):
                print (f"all(label == -100 for label in labels) and all(label == -100 for label in labels_audio)")
                return None

            input_ids = torch.tensor(input_ids, dtype=torch.int64) 
            labels = torch.tensor(labels, dtype=torch.int64)  
            input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: max_token_length]
            labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: max_token_length]

            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            assert len(input_ids) == len(labels)
            assert len(input_ids_audio) == len(labels_audio)
            assert len(input_ids) == len(input_ids_audio)
            assert len(labels) == len(labels_audio)

            feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
            feature_type = torch.tensor(feature_type, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "feature_beg": feature_beg,
                "feature_type": feature_type,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "labels_ids": labels,
                "input_ids_audio": input_ids_audio,
                "labels_ids_audio": labels_audio,
                "attention_mask": attention_mask,
                "item": item,
                "speech_token_id_offset": speech_token_id_offset
            }
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            return output
        except Exception as e:
            print (f"process data error: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return None

    def __getitem__(self, index):
        output = None
        timings = OrderedDict()

        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
            except Exception as e:
                logging.error(f"Error getting item in dataset.__getitem__")
                continue

            output = self._prepare_single_data(
                item=item,
                tokenizer=self.tokenizer,
                mask_token_id=self.mask_token_id,
                blank_audio_token_id=self.blank_audio_token_id,
                blank_audio_token_count=self.blank_audio_token_count,
                max_token_length=self.max_token_length,
                speech_token_id_offset=self.speech_token_id_offset,
                text_vocab_offset=self.text_vocab_offset,
                audio_code_book_size=self.audio_code_book_size,
                pattern=self.pattern,
                fs=self.fs,
                get_audio_token_id_func=self.get_audio_token_id,
                data_type=self.data_type,
                frontend=self.frontend,
                permute=self.permute,
            )
            
            if output is not None:
                break             

        return output


    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs
    
    @staticmethod
    def batchify(samples: list = None, retry: int = 100, int_pad_value: int = -1, 
                 float_pad_value: float = 0.0, batch_type: str = 'token', 
                 batch_size_token_max: int = 15000000, device: str = 'cuda'):
        for idx in range(retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = int_pad_value
                    else:
                        pad_value = float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    ).to(device)
            # token
            if batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break
        return outputs

@tables.register("dataset_classes", "MinMo_S2S_Channels_SFT_Pack")
class MinMo_S2S_Channels_SFT_Pack(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", -1)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")

        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        logging.info(f"init continuous_in: {self.continuous_in}")

        # self.speech_token_id_offset = kwargs.get("speech_token_id_offset", 151694)
        self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        logging.info(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")

        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")

        # pt mode
        self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-text|>") # unimodal-text
        self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_unimodal-audio|>") # unimodal-audio
        self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-audio-to-text|>") # mapping-audio-to-text
        self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_mapping-text-to-audio|>") # mapping-text-to-audio 
        self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic|>") # interleave-audio-to-semantic 
        self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-text|>") # interleave-audio-to-text
        self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids("<|pt_mode_interleave-audio-to-semantic-and-text|>") # interleave-audio-to-semantic-and-text
        assert self.pt_mode_unimodal_text_token_id is not None

        self.text_vocab_offset = self.tokenizer.vocab_size
        assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print (f'audio code book size: {self.audio_code_book_size}')
        self.mask_token_id = -60

        self.blank_audio_token_id = self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids("<|audio_blank|>"), self.text_vocab_offset, self.audio_code_book_size, self.tokenizer)
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)

    @staticmethod
    def get_audio_token_id(audio_token, text_vocab_offset, audio_code_book_size, tokenizer=None):
        if isinstance(audio_token, int):
            return audio_token - text_vocab_offset + audio_code_book_size
        elif isinstance(audio_token, str):
            if tokenizer is None:
                raise ValueError("tokenizer is required when audio_token is a string")
            return MinMo_S2S_Channels_SFT.get_audio_token_id(
                tokenizer.convert_tokens_to_ids(audio_token),
                text_vocab_offset,
                audio_code_book_size,
                tokenizer
            )
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")

    def get_pt_mode_token_id(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")

    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    @staticmethod
    def _process_continuous_audio_slice_static(
        data_src,
        slice_ids,
        input_ids,
        feature_beg,
        fake_token_len,
        feature_type,
        data_type,
        frontend,
        permute,
    ):
        """
        处理连续音频切片的静态方法版本
        
        Args:
            data_src: 音频数据源
            slice_ids: 当前切片的token id列表
            input_ids: 累积的input_ids列表
            feature_beg: 特征开始位置的列表
            fake_token_len: 假token长度的列表
            feature_type: 特征类型的列表
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作
            
        Returns:
            speech: 处理后的音频特征 [b, T, d] 或 [b, d, T] (如果permute=True)
            speech_lengths: 音频长度
            slice_ids: 更新后的slice_ids
            feature_beg: 更新后的feature_beg
            fake_token_len: 更新后的fake_token_len
            feature_type: 更新后的feature_type
        """
        speech, speech_lengths = extract_fbank(
            data_src,
            data_type=data_type,
            frontend=frontend,
            is_final=True,
        )  # speech: [b, T, d]
        
        if permute:
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz
        
        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type

    @staticmethod
    def _prepare_single_data(
        packed_items,
        tokenizer,
        mask_token_id,
        blank_audio_token_id,
        blank_audio_token_count,
        max_token_length,
        speech_token_id_offset,
        text_vocab_offset,
        audio_code_book_size,
        pattern,
        fs,
        get_audio_token_id_func,
        data_type,
        frontend,
        permute,
        infer_mode=False,
    ):
        """
        处理单个数据项，生成input_ids、labels等
        
        Args:
            item: 数据项，包含system、user、assistant字段
            tokenizer: tokenizer对象
            mask_token_id: mask token的id
            blank_audio_token_id: blank audio token的id
            blank_audio_token_count: blank audio token的数量
            max_token_length: 最大token长度
            speech_token_id_offset: speech token id的偏移量
            pattern: 用于分割文本的正则表达式pattern
            fs: 采样率
            get_audio_token_id_func: 获取audio token id的函数
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作
            
        Returns:
            output: 包含处理后的数据的字典，如果处理失败则返回None
        """
        samples = []
        online_segment_idx = 0
        for item in packed_items:
            system = item["system"]
            user = item["user"]
            assistant = item['assistant']
            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                return None
            
            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []

            bad_data = False

            try:
                for i, (system_slice, user_slice, assistant_slice) in enumerate(zip(system, user, assistant)):
                    # if len(input_ids) > max_token_length:
                    #     logging.info(
                    #         f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{max_token_length}"
                    #     )
                    #     break
                    if bad_data:
                        break

                    for s_i, slice in enumerate([system_slice, user_slice, assistant_slice]):
                        # FIXME make sure system_slice is None, for i != 0
                        if slice is None:
                            continue
                        # system prompt only in the first turn
                        if s_i == 0 and i != 0:
                            continue
                        
                        # text and continuous audio feature
                        slice_ids = []
                        # discrete audio token for feature fusion
                        slice_ids_audio = []
                        speech = []
                        speech_lengths = []
                        
                        # tc(t and continuous audio); tcd(t and continuous audio, plus discrete audio); d(only discrete audio); td(text, plus discrete audio)
                        # fusion is only supported for text and discrete audio
                        slice_mode = slice['mode'] 
                        slice_text = slice['text'] # containing text and audio path(for continuous audio feature)
                        slice_audio = slice['audio'] # only for input fusion mode

                        if 'd' in slice_mode and (slice_audio is None or slice_audio == ''):
                            print (f"slice_audio is None or slice_audio == '' for slice_mode: {slice_mode}")
                            bad_data = True
                            break

                        if s_i == 0:
                            prefix_text_ids = tokenizer.encode('<|im_start|>system\n')
                            # TODO: add more tokens to audio vocab? like system, user, assistant?
                            prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids))
                            suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))
                        elif s_i == 1:
                            prefix_text_ids = tokenizer.encode('<|im_start|>user\n')
                            prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids))
                            suffix_text_ids = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))
                        elif s_i == 2:
                            prefix_text_ids = []
                            prefix_audio_ids = []
                            if slice_text is not None and slice_text != '':
                                suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                            else:
                                suffix_text_ids = [mask_token_id] * len(tokenizer.encode('<|im_end|>\n'))
                            if slice_audio is not None and slice_audio != '':
                                suffix_audio_ids = [get_audio_token_id_func(
                                    tokenizer.convert_tokens_to_ids("<|im_end|>"), text_vocab_offset,
                                    audio_code_book_size, tokenizer)] * (len(suffix_text_ids))
                            else:
                                suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))


                        # add prefix ids for both 
                        slice_ids += prefix_text_ids
                        slice_ids_audio += prefix_audio_ids

                        if slice_text is not None and slice_text != '':
                            # slice_ids += prefix_text_ids
                            splits = pattern.split(slice_text)
                            # TODO: handle multiple audio paths in one slice
                            for k, sub_str in enumerate(splits):
                                if sub_str is None or sub_str == '':
                                    continue
                                if not sub_str.startswith("<|startofspeech|>"):
                                    sub_token = tokenizer.encode(sub_str)
                                    slice_ids.extend(sub_token)
                                else:
                                    sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>","")
                                    # audio path for continuous audio feature
                                    if sub_str.startswith("!"):
                                        # if not sub_str[1:].startswith("oss"):
                                        #     bad_data = True
                                        #     print (f"illegal audio path: {sub_str}")
                                        #     break

                                        data_src = load_audio_text_image_video(sub_str[1:], fs=fs, use_ffmpeg=True)

                                        speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = (
                                            MinMo_S2S_Channels_SFT._process_continuous_audio_slice_static(
                                                data_src, slice_ids, input_ids, feature_beg, fake_token_len, feature_type,
                                                data_type, frontend, permute
                                            )
                                        )

                                        # move the code here to handle multiple audio paths in one slice
                                        if len(speech) > 0:
                                            fbank.append(speech[0, :, :])
                                            fbank_lens.append(speech_lengths)
                                    else:
                                        raise ValueError(f"Should be valid audio path starting with !: {sub_str}")
                            # slice_ids += suffix_text_ids

                        if slice_audio is not None and slice_audio != '':
                            # slice_ids_audio += prefix_audio_ids
                            assert slice_mode == 'tcd' or slice_mode == 'td' or slice_mode == 'd', f'slice model is {slice_mode}'
                            # audio latency
                            slice_audio = slice_audio.replace("<|startofaudiotoken|>", "").replace("<|endofaudiotoken|>","")
                            slice_ids_audio += [blank_audio_token_id] * blank_audio_token_count
                            for x in slice_audio.split("|"):
                                if x.startswith("c"):
                                    # discrete audio token id should be put in slice_ids_audio
                                    slice_ids_audio.append(int(x[1:]))
                            # slice_ids_audio += suffix_audio_ids

                            use_pad_id = True
                        else:
                            use_pad_id = False

                        max_tokens = max(len(slice_ids), len(slice_ids_audio))
                        
                        if use_pad_id:
                            slice_ids += [tokenizer.pad_token_id] * (max_tokens - len(slice_ids))
                            slice_ids_audio += [get_audio_token_id_func(tokenizer.pad_token_id, text_vocab_offset, audio_code_book_size, tokenizer)] * (max_tokens - len(slice_ids_audio))
                        else:
                            slice_ids += [mask_token_id] * (max_tokens - len(slice_ids)) 
                            slice_ids_audio += [mask_token_id] * (max_tokens - len(slice_ids_audio))

                        # add suffix ids for both 
                        slice_ids += suffix_text_ids
                        slice_ids_audio += suffix_audio_ids

                        input_ids += slice_ids
                        if s_i == 2:
                            labels += [id if id not in [mask_token_id] else -100 for id in slice_ids]
                        else:
                            labels += [-100] * len(slice_ids)

                        input_ids_audio += slice_ids_audio
                        if s_i == 2:
                            labels_audio += [id if id not in [mask_token_id] else -100 for id in slice_ids_audio]
                        else:
                            labels_audio += [-100] * len(slice_ids_audio)

                        # print non -100 label audio
                        # for label in labels_audio:
                        #     if label != -100:
                        #         print (f"non -100 label audio: {label}")

                        assert len(input_ids) == len(labels)

                if bad_data:
                    break

                # if len(input_ids) > max_token_length:
                #     logging.warning(
                #         f"len(input_ids): {len(input_ids)} > max_token_length: {max_token_length}"
                #     )
                #     break

                if not infer_mode and len(input_ids) <= 10:
                    print (f"len(input_ids) <= 10: {len(input_ids)}")
                    break

                # 如果labels全是-100，则返回None
                if not infer_mode and all(label == -100 for label in labels) and all(label == -100 for label in labels_audio):
                    print (f"all(label == -100 for label in labels) and all(label == -100 for label in labels_audio)")
                    break

                # input_ids = torch.tensor(input_ids, dtype=torch.int64) 
                # labels = torch.tensor(labels, dtype=torch.int64)  
                # input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: max_token_length]
                # labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: max_token_length]

                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)

                # feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
                # feature_type = torch.tensor(feature_type, dtype=torch.int32)
                # fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

                output = {
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                else:
                    output["speech"] = []
                    output["speech_lengths"] = []
                samples.append(output)
            except Exception as e:
                print (f"process data error: {e}")
                print("Full traceback:")
                traceback.print_exc()

        # build a packed data
        # if len(samples) <= 1:
        #     # 如果只有一个或没有样本，直接返回
        #     return samples[0]
        
        # 记录每个样本的起始位置和长度，用于构建attention_mask
        sample_boundaries = []  # [(start_pos, end_pos), ...]
        current_pos = 0

        input_ids_pack = []
        labels_ids_pack = []
        input_ids_audio_pack = []
        labels_ids_audio_pack = []
        attention_mask_pack = []
        feature_beg_pack = []
        feature_type_pack = []
        fake_token_len_pack = []
        speech_pack = []
        speech_lengths_pack = []
        cu_seqlens = []

        cu_seqlens.append(0)
        

        # 合并所有样本的序列数据
        for sample_idx, sample in enumerate(samples):
            seq_len = len(sample["input_ids"])
            sample_start = current_pos
            sample_end = current_pos + seq_len
            sample_boundaries.append((sample_start, sample_end))
            
            # 拼接序列数据
            input_ids_pack.extend(sample["input_ids"])
            labels_ids_pack.extend(sample["labels_ids"])
            input_ids_audio_pack.extend(sample["input_ids_audio"])
            labels_ids_audio_pack.extend(sample["labels_ids_audio"])
            
            # 调整feature_beg的位置（加上当前偏移量）
            for beg_pos in sample["feature_beg"]:
                feature_beg_pack.append(beg_pos + current_pos)
            
            feature_type_pack.extend(sample["feature_type"])
            fake_token_len_pack.extend(sample["fake_token_len"])

            # 构建attention mask
            attention_mask_pack.extend([sample_idx + 1] * seq_len)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

            speech_pack.extend(sample["speech"])
            speech_lengths_pack.extend(sample["speech_lengths"])
            
            current_pos = sample_end
        
        # speech_pack size: list of Txd; speech_lengths_pack size: list of T
        # if len(speech_pack) > 0:
        #     speech_pack = torch.nn.utils.rnn.pad_sequence(
        #                     speech_pack, batch_first=True, padding_value=0.0
        #                 )
        # else:
        #     speech_pack = []
        #     speech_lengths_pack = []

        # 在这里进行必要的tensor构建
        input_ids_pack = torch.tensor(input_ids_pack, dtype=torch.int64)
        labels_ids_pack = torch.tensor(labels_ids_pack, dtype=torch.int64)
        input_ids_audio_pack = torch.tensor(input_ids_audio_pack, dtype=torch.int64)
        labels_ids_audio_pack = torch.tensor(labels_ids_audio_pack, dtype=torch.int64)
        feature_beg_pack = torch.tensor(feature_beg_pack, dtype=torch.int32)
        feature_type_pack = torch.tensor(feature_type_pack, dtype=torch.int32)
        fake_token_len_pack = torch.tensor(fake_token_len_pack, dtype=torch.int32)
        attention_mask_pack = torch.tensor(attention_mask_pack, dtype=torch.int32)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        packed_sample = {
            "input_ids": input_ids_pack,
            "labels_ids": labels_ids_pack,
            "input_ids_audio": input_ids_audio_pack,
            "labels_ids_audio": labels_ids_audio_pack,
            "feature_beg": feature_beg_pack,
            "feature_type": feature_type_pack,
            "fake_token_len": fake_token_len_pack,
            "attention_mask": attention_mask_pack,
            "cu_seqlens": cu_seqlens,
        }

        if len(speech_lengths_pack) > 0:
            packed_sample["speech"] = speech_pack
            packed_sample["speech_lengths"] = speech_lengths_pack

        if packed_sample['input_ids'].numel() == 0:
            print (f'meet bad data, {packed_items}')
            return None

        return packed_sample        

    def __getitem__(self, index):
        output = None
        timings = OrderedDict()

        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
            except Exception as e:
                logging.error(f"Error getting item in dataset.__getitem__")
                continue

            output = self._prepare_single_data(
                packed_items=item,
                tokenizer=self.tokenizer,
                mask_token_id=self.mask_token_id,
                blank_audio_token_id=self.blank_audio_token_id,
                blank_audio_token_count=self.blank_audio_token_count,
                max_token_length=self.max_token_length,
                speech_token_id_offset=self.speech_token_id_offset,
                text_vocab_offset=self.text_vocab_offset,
                audio_code_book_size=self.audio_code_book_size,
                pattern=self.pattern,
                fs=self.fs,
                get_audio_token_id_func=self.get_audio_token_id,
                data_type=self.data_type,
                frontend=self.frontend,
                permute=self.permute,
            )
            
            if output is not None:
                break             

        return output


    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            seq_keys = ("input_ids", "labels_ids", "input_ids_audio", "labels_ids_audio", "attention_mask")
            max_len = getattr(self, "max_token_length", -1)
            use_fixed_length = max_len > 0

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    # cu_seqlens 是累积序列长度 [0, len1, len1+len2, ...]，不做 pad，在下面单独合并为 batch 的 (B+1,)
                    if key == "cu_seqlens":
                        continue
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    # max_token_length > 0 时：先截断再 pad 到固定长度，直接传 total_length
                    if use_fixed_length and key in seq_keys:
                        data_list = [
                            t[:max_len] if t.shape[0] > max_len else t
                            for t in data_list
                        ]
                        total_length = max_len
                    else:
                        total_length = None

                    padded = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
                    if total_length is not None and padded.shape[1] < total_length:
                        pad_size = list(padded.shape)
                        pad_size[1] = total_length - padded.shape[1]
                        padded = torch.cat([padded, padded.new_full(pad_size, pad_value)], dim=1)
                    outputs[key] = padded
            # 合并 cu_seqlens：统一按「每样本 cu + offset」合并；无论是否固定长度，都 pad 了，故每样本末尾 append「pad 后长度」（固定用 max_len，否则用 batch 最大长度），再合并
            if "cu_seqlens" in outputs and len(outputs["cu_seqlens"]) > 0:
                device = outputs["cu_seqlens"][0].device
                padded_len = max_len if use_fixed_length else outputs["input_ids"].shape[1]
                cu_list = []
                for t in outputs["cu_seqlens"]:
                    if use_fixed_length:
                        t = t[t <= padded_len]
                    if len(t) == 0 or t[-1] != padded_len:
                        t = torch.cat([t, torch.tensor([padded_len], dtype=torch.int32, device=t.device)])
                    cu_list.append(t)
                batch_cu = [0]
                offset = 0
                for cu in cu_list:
                    for v in cu[1:].tolist():
                        batch_cu.append(offset + v)
                    offset += cu[-1].item()
                outputs["cu_seqlens"] = torch.tensor(batch_cu, dtype=torch.int32, device=device)
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs
    
    @staticmethod
    def batchify(samples: list = None, retry: int = 100, int_pad_value: int = -1, 
                 float_pad_value: float = 0.0, batch_type: str = 'token', 
                 batch_size_token_max: int = 15000, device: str = 'cuda'):
        for idx in range(retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if key == "cu_seqlens":
                        continue
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = int_pad_value
                    else:
                        pad_value = float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    ).to(device)
            if "cu_seqlens" in outputs and len(outputs["cu_seqlens"]) > 0:
                total_lens = [t[-1].item() for t in outputs["cu_seqlens"]]
                batch_cu = torch.zeros(len(total_lens) + 1, dtype=torch.int32, device=device)
                batch_cu[0] = 0
                batch_cu[1:] = torch.tensor(total_lens, dtype=torch.int32, device=device).cumsum(0)
                outputs["cu_seqlens"] = batch_cu
            # token
            if batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs

@tables.register("dataset_classes", "MinMo_S2S_Channels_SFT_Pack_Mega")
class MinMo_S2S_Channels_SFT_Pack_Mega(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """
    
    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_oss_cache = kwargs.get("use_oss_cache", False)
        self.data_split_i = kwargs.get("data_split_i", 0)
        self.epoch = kwargs.get("epoch", 0)
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            self.preprocessor_speech_conf = kwargs.get("preprocessor_speech_conf")
            preprocessor_speech = preprocessor_speech_class(
                **self.preprocessor_speech_conf
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text
        
        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2
        
        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer
        
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)
        
        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend
        
        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        
        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", -1)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.audio_delay_lst = [self.audio_blank_id] * self.audio_delay
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        
        self.continuous_in = kwargs.get("continuous_in", True)
        self.prefetch_audio_to_local = kwargs.get("prefetch_audio_to_local", True)
        self.audio_cache_dir = kwargs.get("audio_cache_dir", None)
        self.audio_cache_type = kwargs.get("audio_cache_type", "segments")
        self.audio_cache_format = kwargs.get("audio_cache_format", "flac")
        self.drop_cache_miss_items = kwargs.get("drop_cache_miss_items", True)
        self.load_audio_num_threads = kwargs.get("load_audio_num_threads", 4)
        logging.info(f"init continuous_in: {self.continuous_in}")
        
        # self.speech_token_id_offset = kwargs.get("speech_token_id_offset", 151694)
        self.speech_token_id_offset = self.tokenizer.convert_tokens_to_ids("<|c0|>")
        self.offline_speech_tokenize = kwargs.get("offline_speech_tokenize", False)
        self.discrete_tokenizer_frame_rate = kwargs.get("discrete_tokenizer_frame_rate", 25)
        logging.info(f"discrete_tokenizer_frame_rate: {self.discrete_tokenizer_frame_rate}")
        
        self.end_of_segment_speech_token_id = self.tokenizer.convert_tokens_to_ids("<|eoss|>")
        
        # pt mode
        self.pt_mode_unimodal_text_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_unimodal-text|>")  # unimodal-text
        self.pt_mode_unimodal_audio_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_unimodal-audio|>")  # unimodal-audio
        self.pt_mode_mapping_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_mapping-audio-to-text|>")  # mapping-audio-to-text
        self.pt_mode_mapping_text_to_audio_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_mapping-text-to-audio|>")  # mapping-text-to-audio
        self.pt_mode_interleave_audio_to_semantic_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_interleave-audio-to-semantic|>")  # interleave-audio-to-semantic
        self.pt_mode_interleave_audio_to_text_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_interleave-audio-to-text|>")  # interleave-audio-to-text
        self.pt_mode_interleave_audio_to_semantic_and_text_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|pt_mode_interleave-audio-to-semantic-and-text|>")  # interleave-audio-to-semantic-and-text
        # assert self.pt_mode_unimodal_text_token_id is not None
        
        self.text_vocab_offset = self.tokenizer.vocab_size
        # assert self.text_vocab_offset == 151643 # remove after first run
        self.audio_code_book_size = kwargs.get('audio_tokenizer_code_book_size', 59049)  # 6761, 59049
        print(f'audio code book size: {self.audio_code_book_size}')
        self.codec_sos_token_id = kwargs.get("codec_sos_token_id", self.audio_code_book_size)
        self.codec_eos_token_id = kwargs.get("codec_eos_token_id", self.audio_code_book_size + 1)
        self.max_codec_label_length = int(kwargs.get("max_codec_label_length", 0) or 0)
        logging.info(f"max_codec_label_length: {self.max_codec_label_length}")
        self.mask_token_id = -60
        
        # self.blank_audio_token_id = self.get_audio_token_id(self.tokenizer.convert_tokens_to_ids("<|audio_blank|>"), self.text_vocab_offset, self.audio_code_book_size, self.tokenizer)
        self.blank_audio_token_id = None
        self.blank_audio_token_count = kwargs.get("blank_audio_token_count", 6)
    
    @staticmethod
    def get_audio_token_id(audio_token, text_vocab_offset, audio_code_book_size, tokenizer=None):
        if isinstance(audio_token, int):
            return audio_token - text_vocab_offset + audio_code_book_size
        elif isinstance(audio_token, str):
            if tokenizer is None:
                raise ValueError("tokenizer is required when audio_token is a string")
            return MinMo_S2S_Channels_SFT.get_audio_token_id(
                tokenizer.convert_tokens_to_ids(audio_token),
                text_vocab_offset,
                audio_code_book_size,
                tokenizer
            )
        else:
            raise ValueError(f"Invalid audio_token: {audio_token}")
    
    def get_pt_mode_token_id(self, pt_mode):
        if pt_mode == 'unimodal-text':
            return self.pt_mode_unimodal_text_token_id
        elif pt_mode == 'unimodal-audio':
            return self.pt_mode_unimodal_audio_token_id
        elif pt_mode == 'mapping-audio-to-text':
            return self.pt_mode_mapping_audio_to_text_token_id
        elif pt_mode == 'mapping-text-to-audio':
            return self.pt_mode_mapping_text_to_audio_token_id
        elif pt_mode == 'interleave-audio-to-semantic':
            return self.pt_mode_interleave_audio_to_semantic_token_id
        elif pt_mode == 'interleave-audio-to-text':
            return self.pt_mode_interleave_audio_to_text_token_id
        elif pt_mode == 'interleave-audio-to-semantic-and-text':
            return self.pt_mode_interleave_audio_to_semantic_and_text_token_id
        else:
            raise ValueError(f"Invalid pt_mode: {pt_mode}")
    
    def get_source_len(self, index):
        if getattr(self.index_ds, "use_file_index", False):
            return self.index_ds.get_source_len(data_dict=None, index=index)
        else:
            item = self.index_ds[index]
            return self.index_ds.get_source_len(item)
    
    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)
    
    def __len__(self):
        return len(self.index_ds)

    @staticmethod
    def _process_continuous_audio_slice_static(
        data_src,
        expected_speech_token_length,
        slice_ids,
        input_ids,
        feature_beg,
        fake_token_len,
        feature_type,
        data_type,
        frontend,
        permute,
    ):
        """
        处理连续音频切片的静态方法版本

        Args:
            data_src: 音频数据源
            slice_ids: 当前切片的token id列表
            input_ids: 累积的input_ids列表
            feature_beg: 特征开始位置的列表
            fake_token_len: 假token长度的列表
            feature_type: 特征类型的列表
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作

        Returns:
            speech: 处理后的音频特征 [b, T, d] 或 [b, d, T] (如果permute=True)
            speech_lengths: 音频长度
            slice_ids: 更新后的slice_ids
            feature_beg: 更新后的feature_beg
            fake_token_len: 更新后的fake_token_len
            feature_type: 更新后的feature_type
        """
        speech, speech_lengths = extract_fbank(
            data_src,
            data_type=data_type,
            frontend=frontend,
            is_final=True,
        )  # speech: [b, T, d]
        
        if permute:
            speech = speech.permute(0, 2, 1)
        
        # 计算输出长度
        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len_i = (olens - 1) // 2 + 1  # 12.5Hz
        # fake_token_len_i = olens  # 25Hz

        len_diff = expected_speech_token_length - fake_token_len_i
        assert len_diff <= 2 and len_diff >= -2, f"len mismatch, please check your speech item. expected_speech_token_length: {expected_speech_token_length}, fake_token_len_i: {fake_token_len_i}"

        if fake_token_len_i > 0:
            # 添加音频开始标记和假token
            fake_token = [0] * fake_token_len_i
            featrue_beg_i = len(slice_ids)
            feature_type_turn = 1
            slice_ids += fake_token
            
            # 更新各种列表
            feature_beg += [featrue_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            feature_type += [feature_type_turn]
        
        return speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type

    @staticmethod
    def _collect_audio_specs_from_packed_items(packed_items, pattern):
        """按处理顺序收集 packed_items 中所有需要加载的音频 (path, duration)"""
        audio_specs = []
        for item in packed_items:
            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 and len(assistant) < 1:
                continue
            for i, (system_slice, user_slice, assistant_slice) in enumerate(zip(system, user, assistant)):
                for s_i, slice in enumerate([system_slice, user_slice, assistant_slice]):
                    if slice is None:
                        continue
                    if s_i == 0 and i != 0:
                        continue
                    slice_text = slice.get("text")
                    # Audio-only targets must occupy a preload slot as well.
                    splits = pattern.split(slice_text or "")
                    for sub_str in splits:
                        if sub_str is None or sub_str == "":
                            continue
                        if not sub_str.startswith("<|startofspeech|>"):
                            continue
                        if "end_time" in slice:
                            duration = [float(slice["start_time"]), float(slice["end_time"])]
                        else:
                            duration = [0, None]
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                        if sub_str.startswith("!"):
                            path_cur = sub_str[1:]
                            # TODO add audio path replacements here, which should be configured in the yaml file, not hardcoded
                            audio_specs.append((path_cur, duration))
                    slice_audio = slice.get("audio")
                    if slice_audio is None or slice_audio == "":
                        continue
                    slice_audio = slice_audio.replace("<|startofaudiotoken|>", "").replace(
                        "<|endofaudiotoken|>", "")
                    if slice_audio.startswith("!"):
                        audio_specs.append((slice_audio[1:], [0, None]))
        return audio_specs

    @staticmethod
    def _prepare_single_data(
        packed_items,
        tokenizer,
        mask_token_id,
        blank_audio_token_id,
        blank_audio_token_count,
        max_token_length,
        speech_token_id_offset,
        text_vocab_offset,
        audio_code_book_size,
        pattern,
        fs,
        get_audio_token_id_func,
        data_type,
        frontend,
        permute,
        infer_mode=False,
        load_audio_num_threads=4,
        use_oss_cache=False,
        discrete_tokenizer_frame_rate=25,
        codec_eos_token_id=None,
        max_codec_label_length=0,
    ):
        """
        处理单个数据项，生成input_ids、labels等

        Args:
            item: 数据项，包含system、user、assistant字段
            tokenizer: tokenizer对象
            mask_token_id: mask token的id
            blank_audio_token_id: blank audio token的id
            blank_audio_token_count: blank audio token的数量
            max_token_length: 最大token长度
            speech_token_id_offset: speech token id的偏移量
            pattern: 用于分割文本的正则表达式pattern
            fs: 采样率
            get_audio_token_id_func: 获取audio token id的函数
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作

        Returns:
            output: 包含处理后的数据的字典，如果处理失败则返回None
        """
        time_start = time.time()

        def audio_duration_too_short(audio_duration, min_samples=400):
            if audio_duration is None or len(audio_duration) < 2 or audio_duration[1] is None:
                return False
            try:
                num_samples = int(round((float(audio_duration[1]) - float(audio_duration[0])) * fs))
            except Exception:
                return False
            return num_samples < min_samples

        max_codec_label_length = int(max_codec_label_length or 0)
        if max_codec_label_length > 0 and not infer_mode:
            packed_codec_label_length = 0
            for item in packed_items:
                for assistant_slice in item.get("assistant", []):
                    if assistant_slice is None:
                        continue
                    slice_audio = assistant_slice.get("audio")
                    if slice_audio is None or slice_audio == "":
                        continue
                    slice_audio = slice_audio.replace("<|startofaudiotoken|>", "").replace(
                        "<|endofaudiotoken|>", "")
                    if not slice_audio.startswith("!"):
                        continue
                    try:
                        packed_codec_label_length += int(assistant_slice.get("speech_length", 0))
                    except Exception:
                        continue
            if packed_codec_label_length > max_codec_label_length:
                logging.warning(
                    f"skip packed sample because total codec label length exceeds limit: "
                    f"{packed_codec_label_length}>{max_codec_label_length}"
                )
                return None

        # 预加载所有音频（多线程）
        audio_specs = MinMo_S2S_Channels_SFT_Pack_Mega._collect_audio_specs_from_packed_items(packed_items, pattern)
        audio_cache = []
        if audio_specs:
            def load_only(spec):
                audio_path, audio_duration = spec

                load_start = time.perf_counter()

                replace_path = audio_path
                replace_path = replace_path.replace("oss://example-storage-2/", "oss://example-storage-4/")
                replace_path = replace_path.replace("oss://example-storage-6/", "oss://example-storage-4/")
                replace_path = replace_path.replace("oss://example-storage-5/", "oss://example-storage-1/")
                replace_path = _rewrite_audio_path_for_oss_cache(replace_path, use_oss_cache)

                if audio_duration_too_short(audio_duration):
                    return None, time.perf_counter() - load_start, replace_path

                try:
                    data_src = load_audio_text_image_video(
                        replace_path, fs=fs, duration=audio_duration, use_ffmpeg=True
                    )
                    return data_src, time.perf_counter() - load_start, replace_path
                except Exception as e:
                    logging.warning(f"加载音频失败: audio_path: {audio_path}, replace_path: {replace_path}, audio_duration: {audio_duration}, 错误: {e}")
                    return None, time.perf_counter() - load_start, replace_path

            audio_batch_start = time.perf_counter()
            with ThreadPool(load_audio_num_threads) as pool:
                load_results = list(pool.imap(load_only, audio_specs))
            audio_batch_wall = time.perf_counter() - audio_batch_start
            # Disabled: SLOW_AUDIO_BATCH warnings are too noisy during training.
            # _log_slow_audio_batch(load_results, audio_batch_wall, load_audio_num_threads)
            audio_cache = [result[0] for result in load_results]
        audio_cache_offset = 0

        samples = []
        retained_item_indices = []
        online_segment_idx = 0
        for item_index, item in enumerate(packed_items):
            # Reserve the entire item before processing: early failure must not
            # leave unread audio belonging to this item for the next item.
            item_audio_count = len(MinMo_S2S_Channels_SFT_Pack_Mega._collect_audio_specs_from_packed_items([item], pattern))
            item_audio_end = audio_cache_offset + item_audio_count
            audio_cache_iter = iter(audio_cache[audio_cache_offset:item_audio_end])
            audio_cache_offset = item_audio_end
            system = item["system"]
            user = item["user"]
            assistant = item['assistant']
            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                return None
            
            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
            feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            segment_fbank, segment_fbank_lens = [], []
            speech_info = []
            codec_label_length = 0
            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []
            online_item = {"user": [], "assistant": [], "segments_dict": {}}
            
            bad_data = False
            
            try:
                for i, (system_slice, user_slice, assistant_slice) in enumerate(zip(system, user, assistant)):
                    # if len(input_ids) > max_token_length:
                    #     logging.info(
                    #         f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{max_token_length}"
                    #     )
                    #     break
                    if bad_data:
                        break

                    online_user_slice = {"onlie_discrete_speech_token": False}
                    online_assistant_slice = {"onlie_discrete_speech_token": False}
                    
                    for s_i, slice in enumerate([system_slice, user_slice, assistant_slice]):
                        # FIXME make sure system_slice is None, for i != 0
                        if slice is None:
                            continue
                        # system prompt only in the first turn
                        if s_i == 0 and i != 0:
                            continue
                        
                        # text and continuous audio feature
                        slice_ids = []
                        # discrete audio token for feature fusion
                        slice_ids_audio = []
                        speech = []
                        speech_lengths = []
                        
                        # tc(t and continuous audio); tcd(t and continuous audio, plus discrete audio); d(only discrete audio); td(text, plus discrete audio)
                        # fusion is only supported for text and discrete audio
                        slice_mode = slice['mode'] if 'mode' in slice else ''
                        slice_text = slice['text'] if 'text' in slice else '' # containing text and audio path(for continuous audio feature)
                        slice_audio = slice['audio'] if 'audio' in slice else '' # only for input fusion mode

                        if 'd' in slice_mode and (slice_audio is None or slice_audio == ''):
                            print(f"slice_audio is None or slice_audio == '' for slice_mode: {slice_mode}")
                            bad_data = True
                            break

                        if s_i == 0:
                            prefix_text_ids = tokenizer.encode('<|im_start|>system\n')
                            # TODO: add more tokens to audio vocab? like system, user, assistant?
                            prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids))
                            suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))
                        elif s_i == 1:
                            prefix_text_ids = tokenizer.encode('<|im_start|>user\n')
                            prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids))
                            suffix_text_ids = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))
                        elif s_i == 2:
                            prefix_text_ids = []
                            prefix_audio_ids = []
                            if slice_text is not None and slice_text != '':
                                suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                            else:
                                suffix_text_ids = [mask_token_id] * len(tokenizer.encode('<|im_end|>\n'))
                            if slice_audio is not None and slice_audio != '':
                                suffix_audio_ids = [get_audio_token_id_func(
                                    tokenizer.convert_tokens_to_ids("<|im_end|>"), text_vocab_offset,
                                    audio_code_book_size, tokenizer)] * (len(suffix_text_ids))
                            else:
                                suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))
                                
                                # add prefix ids for both
                        slice_ids += prefix_text_ids
                        slice_ids_audio += prefix_audio_ids
                        
                        if slice_text is not None and slice_text != '':
                            # slice_ids += prefix_text_ids
                            splits = pattern.split(slice_text)
                            # TODO: handle multiple audio paths in one slice
                            for k, sub_str in enumerate(splits):
                                if sub_str is None or sub_str == '':
                                    continue
                                if not sub_str.startswith("<|startofspeech|>"):
                                    sub_token = tokenizer.encode(sub_str)
                                    slice_ids.extend(sub_token)
                                else:
                                    if "end_time" in slice:
                                        duration = [float(slice["start_time"]), float(slice["end_time"])]
                                    else:
                                        duration = [0, None]
                                    sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                                    # audio path for continuous audio feature
                                    if sub_str.startswith("!"):
                                        path_cur = sub_str[1:]

                                        try:
                                            data_src = next(audio_cache_iter)
                                        except StopIteration:
                                            logging.warning(f"加载音频失败: {sub_str}, 预加载缓存不足")
                                            bad_data = True # ignore current example from batch
                                            break
                                        if data_src is None:
                                            bad_data = True # ignore current example from batch
                                            break
                                        if hasattr(data_src, "numel") and int(data_src.numel()) < 400:
                                            bad_data = True # ignore current example from batch
                                            break
                                        if not hasattr(data_src, "numel") and len(data_src) < 400:
                                            bad_data = True # ignore current example from batch
                                            break

                                        try:
                                            expected_speech_token_length = slice["speech_length"]
                                            # add <|audio_start|> 
                                            slice_ids += tokenizer.encode("<|audio_start|>")
                                            speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = (
                                                MinMo_S2S_Channels_SFT_Pack_Mega._process_continuous_audio_slice_static(
                                                    data_src, expected_speech_token_length, slice_ids, input_ids, feature_beg, fake_token_len,
                                                    feature_type,
                                                    data_type, frontend, permute
                                                )
                                            )
                                            # add <|audio_end|> 
                                            slice_ids += tokenizer.encode("<|audio_end|>")
                                        except Exception as e:
                                            try:
                                                logging.warning(f"extract_fbank error: {e}, data_src: {data_src.shape}, duration: {duration}, item: {item}")
                                            except:
                                                logging.warning(f"extract_fbank error: {e}, data_src: {data_src}, duration: {duration}, item: {item}")
                                            bad_data = True # ignore current example from batch
                                            break

                                        # move the code here to handle multiple audio paths in one slice
                                        if len(speech) > 0:
                                            fbank.append(speech[0, :, :])
                                            fbank_lens.append(speech_lengths)
                                            speech_info.append(path_cur)
                                        else:
                                            logging.warning("speech len is 0")
                                            bad_data = True # ignore current example from batch
                                            break
                                    else:
                                        logging.warning(f"Should be valid audio path starting with !: {sub_str}")
                                        bad_data = True # ignore current example from batch
                                        break
                            # slice_ids += suffix_text_ids
                        
                        if slice_audio is not None and slice_audio != '':
                            # slice_ids_audio += prefix_audio_ids
                            assert slice_mode == 'tcd' or slice_mode == 'td' or slice_mode == 'd', f'slice model is {slice_mode}'
                            # audio latency
                            slice_audio = slice_audio.replace("<|startofaudiotoken|>", "").replace(
                                "<|endofaudiotoken|>", "")
                            blank_id = blank_audio_token_id if blank_audio_token_id is not None else mask_token_id
                            slice_ids_audio += [blank_id] * blank_audio_token_count
                            online_target_audio = False
                            if slice_audio.startswith("!"):
                                online_target_audio = True
                                suffix_audio_ids = [mask_token_id] * len(suffix_audio_ids)
                                try:
                                    data_src = next(audio_cache_iter)
                                except StopIteration:
                                    logging.warning(f"加载音频失败: {slice_audio}, 预加载缓存不足")
                                    bad_data = True
                                    break
                                if data_src is None:
                                    bad_data = True
                                    break
                                if hasattr(data_src, "numel") and int(data_src.numel()) < 400:
                                    bad_data = True
                                    break
                                if not hasattr(data_src, "numel") and len(data_src) < 400:
                                    bad_data = True
                                    break

                                try:
                                    target_speech, target_speech_lengths = extract_fbank(
                                        data_src,
                                        data_type=data_type,
                                        frontend=frontend,
                                        is_final=True,
                                    )
                                    if permute:
                                        target_speech = target_speech.permute(0, 2, 1)
                                except Exception as e:
                                    try:
                                        logging.warning(f"target audio extract_fbank error: {e}, data_src: {data_src.shape}, item: {item}")
                                    except Exception:
                                        logging.warning(f"target audio extract_fbank error: {e}, data_src: {data_src}, item: {item}")
                                    bad_data = True
                                    break

                                expected_codec_len = int(slice.get("speech_length", 0))
                                if expected_codec_len <= 0:
                                    raw_len = int(target_speech_lengths[0].item()) if hasattr(target_speech_lengths[0], "item") else int(target_speech_lengths[0])
                                    expected_codec_len = max(1, int(round(raw_len / 4)))
                                if s_i == 2:
                                    codec_label_length += expected_codec_len
                                    if max_codec_label_length > 0 and codec_label_length > max_codec_label_length:
                                        logging.warning(
                                            f"skip packed sample because codec label length exceeds limit: "
                                            f"{codec_label_length}>{max_codec_label_length}, item: {item}"
                                        )
                                        bad_data = True
                                        break
                                slice_ids_audio.extend([-50] * expected_codec_len)
                                if codec_eos_token_id is not None:
                                    slice_ids_audio.append(int(codec_eos_token_id))

                                segment_id = f"target_audio_{online_segment_idx}"
                                online_segment_idx += 1
                                online_slice = {
                                    "onlie_discrete_speech_token": True,
                                    "segment_id": segment_id,
                                    "start_time": 0.0,
                                    "end_time": expected_codec_len / float(discrete_tokenizer_frame_rate),
                                    # Preserve the online target waveform reference for
                                    # downstream joint codec-LLM -> FM training.  The
                                    # previous online-item conversion retained only the
                                    # segment timing, so the FM loader could not recover
                                    # the assistant target audio from the final batch.
                                    "audio": (
                                        f"<|startofaudiotoken|>{slice_audio}"
                                        "<|endofaudiotoken|>"
                                    ),
                                }
                                if s_i == 1:
                                    online_user_slice = online_slice
                                elif s_i == 2:
                                    online_assistant_slice = online_slice
                                online_item["segments_dict"][segment_id] = {
                                    "segment_id": segment_id,
                                    "start_time": 0.0,
                                    "end_time": expected_codec_len / float(discrete_tokenizer_frame_rate),
                                    "onlie_discrete_speech_token": True,
                                }
                                segment_fbank.append(target_speech[0, :, :])
                                segment_fbank_lens.append(target_speech_lengths)
                            else:
                                for x in slice_audio.split("|"):
                                    if x.startswith("c"):
                                        # discrete audio token id should be put in slice_ids_audio
                                        slice_ids_audio.append(int(x[1:]))
                            # slice_ids_audio += suffix_audio_ids
                            
                            use_pad_id = not online_target_audio
                        else:
                            use_pad_id = False
                        
                        max_tokens = max(len(slice_ids), len(slice_ids_audio))
                        
                        if use_pad_id:
                            slice_ids += [tokenizer.pad_token_id] * (max_tokens - len(slice_ids))
                            slice_ids_audio += [get_audio_token_id_func(tokenizer.pad_token_id, text_vocab_offset,
                                                                        audio_code_book_size, tokenizer)] * (
                                                       max_tokens - len(slice_ids_audio))
                        else:
                            slice_ids += [mask_token_id] * (max_tokens - len(slice_ids))
                            slice_ids_audio += [mask_token_id] * (max_tokens - len(slice_ids_audio))
                        
                        # add suffix ids for both
                        slice_ids += suffix_text_ids
                        slice_ids_audio += suffix_audio_ids
                        
                        input_ids += slice_ids
                        if s_i == 2:
                            labels += [id if id not in [mask_token_id] else -100 for id in slice_ids]
                        else:
                            labels += [-100] * len(slice_ids)
                        
                        input_ids_audio += slice_ids_audio
                        if s_i == 2:
                            labels_audio += [id if id not in [mask_token_id] else -100 for id in slice_ids_audio]
                        else:
                            labels_audio += [-100] * len(slice_ids_audio)
                        
                        # print non -100 label audio
                        # for label in labels_audio:
                        #     if label != -100:
                        #         print (f"non -100 label audio: {label}")
                        
                        assert len(input_ids) == len(labels)

                    online_item["user"].append(online_user_slice)
                    online_item["assistant"].append(online_assistant_slice)
                
                if bad_data:
                    continue
                
                # if len(input_ids) > max_token_length:
                #     logging.warning(
                #         f"len(input_ids): {len(input_ids)} > max_token_length: {max_token_length}"
                #     )
                #     break
                
                if not infer_mode and len(input_ids) <= 10:
                    print(f"len(input_ids) <= 10: {len(input_ids)}")
                    break
                
                # 如果labels全是-100，则返回None
                if not infer_mode and all(label == -100 for label in labels) and all(
                    label == -100 for label in labels_audio):
                    print(f"all(label == -100 for label in labels) and all(label == -100 for label in labels_audio)")
                    break
                
                # input_ids = torch.tensor(input_ids, dtype=torch.int64)
                # labels = torch.tensor(labels, dtype=torch.int64)
                # input_ids_audio = torch.tensor(input_ids_audio, dtype=torch.int64)  # [: max_token_length]
                # labels_audio = torch.tensor(labels_audio, dtype=torch.int64)  # [: max_token_length]
                
                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)
                
                # feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
                # feature_type = torch.tensor(feature_type, dtype=torch.int32)
                # fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
                
                output = {
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                else:
                    output["speech"] = []
                    output["speech_lengths"] = []
                if len(segment_fbank) > 0:
                    output["segment_fbank"] = segment_fbank
                    output["segment_fbank_lengths"] = segment_fbank_lens
                    output["item"] = online_item
                samples.append(output)
                retained_item_indices.append(item_index)
            except Exception as e:
                print(f"process data error: {e}")
                print("Full traceback:")
                traceback.print_exc()
        
        # build a packed data
        # if len(samples) <= 1:
        #     # 如果只有一个或没有样本，直接返回
        #     return samples[0]
        
        # 记录每个样本的起始位置和长度，用于构建attention_mask
        sample_boundaries = []  # [(start_pos, end_pos), ...]
        current_pos = 0
        
        input_ids_pack = []
        labels_ids_pack = []
        input_ids_audio_pack = []
        labels_ids_audio_pack = []
        attention_mask_pack = []
        feature_beg_pack = []
        feature_type_pack = []
        fake_token_len_pack = []
        speech_pack = []
        speech_lengths_pack = []
        cu_seqlens = []
        
        cu_seqlens.append(0)
        
        # 合并所有样本的序列数据
        for sample_idx, sample in enumerate(samples):
            seq_len = len(sample["input_ids"])
            sample_start = current_pos
            sample_end = current_pos + seq_len
            sample_boundaries.append((sample_start, sample_end))
            
            # 拼接序列数据
            input_ids_pack.extend(sample["input_ids"])
            labels_ids_pack.extend(sample["labels_ids"])
            input_ids_audio_pack.extend(sample["input_ids_audio"])
            labels_ids_audio_pack.extend(sample["labels_ids_audio"])
            
            # 调整feature_beg的位置（加上当前偏移量）
            for beg_pos in sample["feature_beg"]:
                feature_beg_pack.append(beg_pos + current_pos)
            
            feature_type_pack.extend(sample["feature_type"])
            fake_token_len_pack.extend(sample["fake_token_len"])
            
            # 构建attention mask
            attention_mask_pack.extend([sample_idx + 1] * seq_len)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            
            speech_pack.extend(sample["speech"])
            speech_lengths_pack.extend(sample["speech_lengths"])
            
            current_pos = sample_end
        
        # speech_pack size: list of Txd; speech_lengths_pack size: list of T
        # if len(speech_pack) > 0:
        #     speech_pack = torch.nn.utils.rnn.pad_sequence(
        #                     speech_pack, batch_first=True, padding_value=0.0
        #                 )
        # else:
        #     speech_pack = []
        #     speech_lengths_pack = []
        
        # 在这里进行必要的tensor构建
        input_ids_pack = torch.tensor(input_ids_pack, dtype=torch.int64)
        labels_ids_pack = torch.tensor(labels_ids_pack, dtype=torch.int64)
        input_ids_audio_pack = torch.tensor(input_ids_audio_pack, dtype=torch.int64)
        labels_ids_audio_pack = torch.tensor(labels_ids_audio_pack, dtype=torch.int64)
        feature_beg_pack = torch.tensor(feature_beg_pack, dtype=torch.int32)
        feature_type_pack = torch.tensor(feature_type_pack, dtype=torch.int32)
        fake_token_len_pack = torch.tensor(fake_token_len_pack, dtype=torch.int32)
        attention_mask_pack = torch.tensor(attention_mask_pack, dtype=torch.int32)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        
        packed_sample = {
            "retained_item_indices": retained_item_indices,
            "input_ids": input_ids_pack,
            "labels_ids": labels_ids_pack,
            "input_ids_audio": input_ids_audio_pack,
            "labels_ids_audio": labels_ids_audio_pack,
            "feature_beg": feature_beg_pack,
            "feature_type": feature_type_pack,
            "fake_token_len": fake_token_len_pack,
            "attention_mask": attention_mask_pack,
            "cu_seqlens": cu_seqlens,
            "item": {
                "user": sum([sample["item"]["user"] for sample in samples if "item" in sample], []),
                "assistant": sum([sample["item"]["assistant"] for sample in samples if "item" in sample], []),
                "segments_dict": {
                    key: value
                    for sample in samples if "item" in sample
                    for key, value in sample["item"]["segments_dict"].items()
                },
            },
            "speech_token_id_offset": speech_token_id_offset,
        }
        #print("input_ids_pack", input_ids_pack.shape)
        
        if len(speech_lengths_pack) > 0:
            packed_sample["speech"] = speech_pack
            packed_sample["speech_lengths"] = speech_lengths_pack
        segment_fbank_pack = []
        segment_fbank_lengths_pack = []
        for sample in samples:
            segment_fbank_pack.extend(sample.get("segment_fbank", []))
            segment_fbank_lengths_pack.extend(sample.get("segment_fbank_lengths", []))
        if len(segment_fbank_lengths_pack) > 0:
            packed_sample["segment_fbank"] = segment_fbank_pack
            packed_sample["segment_fbank_lengths"] = segment_fbank_lengths_pack
        
        if packed_sample['input_ids'].numel() == 0:
            print(f'meet bad data, {packed_items}')
            return None
        
        #print(f"{load_audio_num_threads}_prepare_single_data 执行时长: {time.time() - time_start:.3f}s")
        return packed_sample
    
    def __getitem__(self, index):
        output = None
        timings = OrderedDict()
        
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            try:
                time_start_index_ds = time.time()
                item = self.index_ds[index_cur]
                time_end_index_ds = time.time()
                timings['index_ds'] = time_end_index_ds - time_start_index_ds
            except Exception as e:
                logging.error(f"Error getting item in dataset.__getitem__")
                continue
            
            output = self._prepare_single_data(
                packed_items=item,
                tokenizer=self.tokenizer,
                mask_token_id=self.mask_token_id,
                blank_audio_token_id=self.blank_audio_token_id,
                blank_audio_token_count=self.blank_audio_token_count,
                max_token_length=self.max_token_length,
                speech_token_id_offset=self.speech_token_id_offset,
                text_vocab_offset=self.text_vocab_offset,
                audio_code_book_size=self.audio_code_book_size,
                pattern=self.pattern,
                fs=self.fs,
                get_audio_token_id_func=self.get_audio_token_id,
                data_type=self.data_type,
                frontend=self.frontend,
                permute=self.permute,
                load_audio_num_threads=self.load_audio_num_threads,
                use_oss_cache=self.use_oss_cache,
                discrete_tokenizer_frame_rate=self.discrete_tokenizer_frame_rate,
                codec_eos_token_id=self.codec_eos_token_id,
                max_codec_label_length=self.max_codec_label_length,
            )
            
            if output is not None:
                if "speech_info" in output: # speech_info just for tracing data, when debuging
                    output["speech_info"] = {
                        "index_cur": index_cur,
                        "speech_path": output["speech_info"]
                    }
                else:
                    output["speech_info"] = {
                        "index_cur": index_cur,
                        "speech_path": None
                    }
                break
        
        return output
    
    def collator(self, samples: list = None):
        for idx in range(self.retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])
            
            seq_keys = ("input_ids", "labels_ids", "input_ids_audio", "labels_ids_audio", "attention_mask")
            max_len = getattr(self, "max_token_length", -1)
            use_fixed_length = max_len > 0
            
            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    # cu_seqlens 是累积序列长度 [0, len1, len1+len2, ...]，不做 pad，在下面单独合并为 batch 的 (B+1,)
                    if key == "cu_seqlens":
                        continue
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        
                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    
                    # max_token_length > 0 时：先截断再 pad 到固定长度，直接传 total_length
                    if use_fixed_length and key in seq_keys:
                        data_list = [
                            t[:max_len] if t.shape[0] > max_len else t
                            for t in data_list
                        ]
                        total_length = max_len
                    else:
                        total_length = None
                    
                    padded = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )
                    if total_length is not None and padded.shape[1] < total_length:
                        pad_size = list(padded.shape)
                        pad_size[1] = total_length - padded.shape[1]
                        padded = torch.cat([padded, padded.new_full(pad_size, pad_value)], dim=1)
                    outputs[key] = padded
            # 合并 cu_seqlens：统一按「每样本 cu + offset」合并；无论是否固定长度，都 pad 了，故每样本末尾 append「pad 后长度」（固定用 max_len，否则用 batch 最大长度），再合并
            if "cu_seqlens" in outputs and len(outputs["cu_seqlens"]) > 0:
                device = outputs["cu_seqlens"][0].device
                padded_len = max_len if use_fixed_length else outputs["input_ids"].shape[1]
                cu_list = []
                for t in outputs["cu_seqlens"]:
                    if use_fixed_length:
                        t = t[t <= padded_len]
                    if len(t) == 0 or t[-1] != padded_len:
                        t = torch.cat([t, torch.tensor([padded_len], dtype=torch.int32, device=t.device)])
                    cu_list.append(t)
                batch_cu = [0]
                offset = 0
                for cu in cu_list:
                    for v in cu[1:].tolist():
                        batch_cu.append(offset + v)
                    offset += cu[-1].item()
                outputs["cu_seqlens"] = torch.tensor(batch_cu, dtype=torch.int32, device=device)
            # token
            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue
            
            break
        
        return outputs
    
    @staticmethod
    def batchify(samples: list = None, retry: int = 100, int_pad_value: int = -1,
                 float_pad_value: float = 0.0, batch_type: str = 'token',
                 batch_size_token_max: int = 15000, device: str = 'cuda'):
        for idx in range(retry):
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])
            
            for key, data_list in outputs.items():
                if len(data_list) > 0 and isinstance(data_list[0], torch.Tensor):
                    if key == "cu_seqlens":
                        continue
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        
                        pad_value = int_pad_value
                    else:
                        pad_value = float_pad_value
                    
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    ).to(device)
            if "cu_seqlens" in outputs and len(outputs["cu_seqlens"]) > 0:
                total_lens = [t[-1].item() for t in outputs["cu_seqlens"]]
                batch_cu = torch.zeros(len(total_lens) + 1, dtype=torch.int32, device=device)
                batch_cu[0] = 0
                batch_cu[1:] = torch.tensor(total_lens, dtype=torch.int32, device=device).cumsum(0)
                outputs["cu_seqlens"] = batch_cu
            # token
            if batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue
            
            break
        
        return outputs


@tables.register("dataset_classes", "MinMo_S2S_Channels_SFT_Pack_Mega_UniPrompt0316")
class MinMo_S2S_Channels_SFT_Pack_Mega_UniPrompt0316(MinMo_S2S_Channels_SFT_Pack_Mega):
    @staticmethod
    def _prepare_single_data(
        packed_items,
        tokenizer,
        mask_token_id,
        blank_audio_token_id,
        blank_audio_token_count,
        max_token_length,
        speech_token_id_offset,
        text_vocab_offset,
        audio_code_book_size,
        pattern,
        fs,
        get_audio_token_id_func,
        data_type,
        frontend,
        permute,
        infer_mode=False,
        load_audio_num_threads=4,
        use_oss_cache=False,
    ):
        """
        处理单个数据项，生成input_ids、labels等

        Args:
            item: 数据项，包含system、user、assistant字段
            tokenizer: tokenizer对象
            mask_token_id: mask token的id
            blank_audio_token_id: blank audio token的id
            blank_audio_token_count: blank audio token的数量
            max_token_length: 最大token长度
            speech_token_id_offset: speech token id的偏移量
            pattern: 用于分割文本的正则表达式pattern
            fs: 采样率
            get_audio_token_id_func: 获取audio token id的函数
            data_type: 数据类型，通常是 "sound"
            frontend: 前端处理器
            permute: 是否需要进行permute操作

        Returns:
            output: 包含处理后的数据的字典，如果处理失败则返回None
        """
        time_start = time.time()
        # 预加载所有音频（多线程）
        audio_specs = MinMo_S2S_Channels_SFT_Pack_Mega._collect_audio_specs_from_packed_items(packed_items, pattern)
        audio_cache = []
        if audio_specs:
            def load_only(spec):
                audio_path, audio_duration = spec

                load_start = time.perf_counter()

                # path replace for some old data
                replace_path = audio_path
                replace_path = replace_path.replace("oss://example-storage-2/", "oss://example-storage-4/")
                replace_path = replace_path.replace("oss://example-storage-6/", "oss://example-storage-4/")
                replace_path = replace_path.replace("oss://example-storage-5/", "oss://example-storage-1/")
                replace_path = _rewrite_audio_path_for_oss_cache(replace_path, use_oss_cache)

                try:
                    data_src = load_audio_text_image_video(
                        replace_path, fs=fs, duration=audio_duration, use_ffmpeg=True
                    )
                    return data_src, time.perf_counter() - load_start, replace_path
                except Exception as e:
                    logging.warning(f"加载音频失败: audio_path: {audio_path}, replace_path: {replace_path}, audio_duration: {audio_duration}, 错误: {e}")
                    return None, time.perf_counter() - load_start, replace_path

            audio_batch_start = time.perf_counter()
            with ThreadPool(load_audio_num_threads) as pool:
                load_results = list(pool.imap(load_only, audio_specs))
            audio_batch_wall = time.perf_counter() - audio_batch_start
            # Disabled: SLOW_AUDIO_BATCH warnings are too noisy during training.
            # _log_slow_audio_batch(load_results, audio_batch_wall, load_audio_num_threads)
            audio_cache = [result[0] for result in load_results]
        audio_cache_iter = iter(audio_cache)

        samples = []
        for item in packed_items:
            system = item["system"]
            user = item["user"]
            assistant = item['assistant']
            if len(user) < 1 and len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                return None

            # input_ids for text tokens and continuous audio features
            input_ids, labels, fbank, fbank_lens, \
            feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], []
            )
            speech_info = [] # use to trace speech bad case if given. can be anything you need to print when forward encoder.

            # for input and output audio channels
            # input_ids_audio for discrete audio tokens
            input_ids_audio, labels_audio = [], []

            bad_data = False

            try:
                for i, (system_slice, user_slice, assistant_slice) in enumerate(zip(system, user, assistant)):
                    # if len(input_ids) > max_token_length:
                    #     logging.info(
                    #         f"[TOO LOOONG streaming in] input_ids > max_token_length, break this turn: {len(input_ids)}>{max_token_length}"
                    #     )
                    #     break
                    if bad_data:
                        break

                    for s_i, slice in enumerate([system_slice, user_slice, assistant_slice]):
                        # FIXME make sure system_slice is None, for i != 0
                        if slice is None:
                            continue
                        # system prompt only in the first turn
                        if s_i == 0 and i != 0:
                            continue

                        # text and continuous audio feature
                        slice_ids = []
                        # discrete audio token for feature fusion
                        slice_ids_audio = []
                        speech = []
                        speech_lengths = []

                        # tc(t and continuous audio); tcd(t and continuous audio, plus discrete audio); d(only discrete audio); td(text, plus discrete audio)
                        # fusion is only supported for text and discrete audio
                        slice_mode = slice['mode']
                        slice_text = slice['text']  # containing text and audio path(for continuous audio feature)
                        slice_audio = slice['audio']  # only for input fusion mode

                        if 'd' in slice_mode and (slice_audio is None or slice_audio == ''):
                            print(f"slice_audio is None or slice_audio == '' for slice_mode: {slice_mode}")
                            bad_data = True
                            break

                        if s_i == 0:
                            prefix_text_ids = tokenizer.encode('<|im_start|>system\n')
                            # TODO: add more tokens to audio vocab? like system, user, assistant?
                            prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids))
                            suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))
                        elif s_i == 1:
                            prefix_text_ids = tokenizer.encode('<|im_start|>user\n')
                            prefix_audio_ids = [mask_token_id] * (len(prefix_text_ids))

                            # additional prompt after audio
                            # <user><asr><audio tokens>/n<hotword>/n<context>/n
                            # todo: insert actual prompt here
                            hotword_prompt = ""
                            context_prompt = ""
                            additional_context = f"\nhotword\n{hotword_prompt}\ncontext{context_prompt}"
                            additional_context_ids = tokenizer.encode(additional_context)
                            additional_context_ids_audio = [mask_token_id] * (len(additional_context_ids))
                            # print("additional_context_ids", additional_context_ids)

                            suffix_text_ids = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
                            suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))

                        elif s_i == 2:
                            # todo: insert actual lid here
                            prefix_text_ids = []
                            prefix_audio_ids = []
                            if slice_text is not None and slice_text != '':
                                suffix_text_ids = tokenizer.encode('<|im_end|>\n')
                            else:
                                suffix_text_ids = [mask_token_id] * len(tokenizer.encode('<|im_end|>\n'))
                            if slice_audio is not None and slice_audio != '':
                                suffix_audio_ids = [get_audio_token_id_func(
                                    tokenizer.convert_tokens_to_ids('<|im_end|>'), text_vocab_offset,
                                    audio_code_book_size, tokenizer)] * (len(suffix_text_ids))
                            else:
                                suffix_audio_ids = [mask_token_id] * (len(suffix_text_ids))

                        # add prefix ids for both
                        slice_ids += prefix_text_ids
                        slice_ids_audio += prefix_audio_ids

                        if slice_text is not None and slice_text != '':
                            # slice_ids += prefix_text_ids
                            splits = pattern.split(slice_text)
                            # TODO: handle multiple audio paths in one slice
                            for k, sub_str in enumerate(splits):
                                if sub_str is None or sub_str == '':
                                    continue
                                if not sub_str.startswith("<|startofspeech|>"):
                                    sub_token = tokenizer.encode(sub_str)
                                    slice_ids.extend(sub_token)
                                else:
                                    if "end_time" in slice:
                                        duration = [float(slice["start_time"]), float(slice["end_time"])]
                                    else:
                                        duration = [0, None]
                                    sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                                    # audio path for continuous audio feature
                                    if sub_str.startswith("!"):
                                        path_cur = sub_str[1:]

                                        try:
                                            data_src = next(audio_cache_iter)
                                        except StopIteration:
                                            logging.warning(f"加载音频失败: {sub_str}, 预加载缓存不足")
                                            bad_data = True # ignore current example from batch
                                            break
                                        if data_src is None:
                                            bad_data = True # ignore current example from batch
                                            break

                                        try:
                                            expected_speech_token_length = slice["speech_length"]
                                            # add <|audio_start|>
                                            slice_ids += tokenizer.encode("<|audio_start|>")
                                            speech, speech_lengths, slice_ids, feature_beg, fake_token_len, feature_type = (
                                                MinMo_S2S_Channels_SFT_Pack_Mega._process_continuous_audio_slice_static(
                                                    data_src, expected_speech_token_length, slice_ids, input_ids, feature_beg, fake_token_len,
                                                    feature_type,
                                                    data_type, frontend, permute
                                                )
                                            )
                                            # add <|audio_end|>
                                            slice_ids += tokenizer.encode("<|audio_end|>")
                                        except Exception as e:
                                            try:
                                                logging.warning(f"extract_fbank error: {e}, data_src: {data_src.shape}, duration: {duration}, item: {item}")
                                            except:
                                                logging.warning(f"extract_fbank error: {e}, data_src: {data_src}, duration: {duration}, item: {item}")
                                            bad_data = True # ignore current example from batch
                                            break

                                        # additional prompt after audio
                                        slice_ids += additional_context_ids
                                        slice_ids_audio += additional_context_ids_audio

                                        # move the code here to handle multiple audio paths in one slice
                                        if len(speech) > 0:
                                            fbank.append(speech[0, :, :])
                                            fbank_lens.append(speech_lengths)
                                            speech_info.append(path_cur)
                                        else:
                                            logging.warning("speech len is 0")
                                            bad_data = True # ignore current example from batch
                                            break
                                    else:
                                        logging.warning(f"Should be valid audio path starting with !: {sub_str}")
                                        bad_data = True # ignore current example from batch
                                        break

                        if slice_audio is not None and slice_audio != '':
                            # slice_ids_audio += prefix_audio_ids
                            assert slice_mode == 'tcd' or slice_mode == 'td' or slice_mode == 'd', f'slice model is {slice_mode}'
                            # audio latency
                            slice_audio = slice_audio.replace("<|startofaudiotoken|>", "").replace(
                                "<|endofaudiotoken|>", "")
                            slice_ids_audio += [blank_audio_token_id] * blank_audio_token_count
                            for x in slice_audio.split("|"):
                                if x.startswith("c"):
                                    # discrete audio token id should be put in slice_ids_audio
                                    slice_ids_audio.append(int(x[1:]))

                            use_pad_id = True
                        else:
                            use_pad_id = False

                        max_tokens = max(len(slice_ids), len(slice_ids_audio))

                        if use_pad_id:
                            slice_ids += [tokenizer.pad_token_id] * (max_tokens - len(slice_ids))
                            slice_ids_audio += [get_audio_token_id_func(tokenizer.pad_token_id, text_vocab_offset,
                                                                        audio_code_book_size, tokenizer)] * (
                                                       max_tokens - len(slice_ids_audio))
                        else:
                            slice_ids += [mask_token_id] * (max_tokens - len(slice_ids))
                            slice_ids_audio += [mask_token_id] * (max_tokens - len(slice_ids_audio))

                        # add suffix ids for both
                        slice_ids += suffix_text_ids
                        slice_ids_audio += suffix_audio_ids

                        input_ids += slice_ids
                        if s_i == 2:
                            labels += [id if id not in [mask_token_id] else -100 for id in slice_ids]
                        else:
                            labels += [-100] * len(slice_ids)

                        input_ids_audio += slice_ids_audio
                        if s_i == 2:
                            labels_audio += [id if id not in [mask_token_id] else -100 for id in slice_ids_audio]
                        else:
                            labels_audio += [-100] * len(slice_ids_audio)

                        assert len(input_ids) == len(labels)

                if bad_data:
                    continue

                if not infer_mode and len(input_ids) <= 10:
                    print(f"len(input_ids) <= 10: {len(input_ids)}")
                    break

                # 如果labels全是-100，则返回None
                if not infer_mode and all(label == -100 for label in labels) and all(
                    label == -100 for label in labels_audio):
                    print(f"all(label == -100 for label in labels) and all(label == -100 for label in labels_audio)")
                    break

                assert len(input_ids) == len(labels)
                assert len(input_ids_audio) == len(labels_audio)
                assert len(input_ids) == len(input_ids_audio)
                assert len(labels) == len(labels_audio)

                # for s2t training, always set input_ids_audio as mask_token_id, and labels_ids_audio as -100
                input_ids_audio = [mask_token_id] * len(input_ids_audio)
                labels_audio = [-100] * len(labels_audio)

                output = {
                    "feature_beg": feature_beg,
                    "feature_type": feature_type,
                    "fake_token_len": fake_token_len,
                    "input_ids": input_ids,
                    "labels_ids": labels,
                    "input_ids_audio": input_ids_audio,
                    "labels_ids_audio": labels_audio,
                }
                if len(fbank) > 0:
                    output["speech"] = fbank
                    output["speech_lengths"] = fbank_lens
                    output["speech_info"] = speech_info
                else:
                    output["speech"] = []
                    output["speech_lengths"] = []
                    output["speech_info"] = []
                samples.append(output)
            except Exception as e:
                print(f"process data error: {e}")
                print("Full traceback:")
                traceback.print_exc()


        # 记录每个样本的起始位置和长度，用于构建attention_mask
        sample_boundaries = []  # [(start_pos, end_pos), ...]
        current_pos = 0

        input_ids_pack = []
        labels_ids_pack = []
        input_ids_audio_pack = []
        labels_ids_audio_pack = []
        attention_mask_pack = []
        feature_beg_pack = []
        feature_type_pack = []
        fake_token_len_pack = []
        speech_pack = []
        speech_lengths_pack = []
        speech_info_pack = []
        cu_seqlens = []

        cu_seqlens.append(0)

        # 合并所有样本的序列数据
        for sample_idx, sample in enumerate(samples):
            seq_len = len(sample["input_ids"])
            sample_start = current_pos
            sample_end = current_pos + seq_len
            sample_boundaries.append((sample_start, sample_end))

            # 拼接序列数据
            input_ids_pack.extend(sample["input_ids"])
            labels_ids_pack.extend(sample["labels_ids"])
            input_ids_audio_pack.extend(sample["input_ids_audio"])
            labels_ids_audio_pack.extend(sample["labels_ids_audio"])

            # 调整feature_beg的位置（加上当前偏移量）
            for beg_pos in sample["feature_beg"]:
                feature_beg_pack.append(beg_pos + current_pos)

            feature_type_pack.extend(sample["feature_type"])
            fake_token_len_pack.extend(sample["fake_token_len"])

            # 构建attention mask
            attention_mask_pack.extend([sample_idx + 1] * seq_len)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

            speech_pack.extend(sample["speech"])
            speech_lengths_pack.extend(sample["speech_lengths"])
            speech_info_pack.extend(sample["speech_info"])

            current_pos = sample_end

        # 在这里进行必要的tensor构建
        input_ids_pack = torch.tensor(input_ids_pack, dtype=torch.int64)
        labels_ids_pack = torch.tensor(labels_ids_pack, dtype=torch.int64)
        input_ids_audio_pack = torch.tensor(input_ids_audio_pack, dtype=torch.int64)
        labels_ids_audio_pack = torch.tensor(labels_ids_audio_pack, dtype=torch.int64)
        feature_beg_pack = torch.tensor(feature_beg_pack, dtype=torch.int32)
        feature_type_pack = torch.tensor(feature_type_pack, dtype=torch.int32)
        fake_token_len_pack = torch.tensor(fake_token_len_pack, dtype=torch.int32)
        attention_mask_pack = torch.tensor(attention_mask_pack, dtype=torch.int32)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        packed_sample = {
            "input_ids": input_ids_pack,
            "labels_ids": labels_ids_pack,
            "input_ids_audio": input_ids_audio_pack,
            "labels_ids_audio": labels_ids_audio_pack,
            "feature_beg": feature_beg_pack,
            "feature_type": feature_type_pack,
            "fake_token_len": fake_token_len_pack,
            "attention_mask": attention_mask_pack,
            "cu_seqlens": cu_seqlens,
            "item": None,
            "speech_token_id_offset": speech_token_id_offset,
        }

        if len(speech_lengths_pack) > 0:
            packed_sample["speech"] = speech_pack
            packed_sample["speech_lengths"] = speech_lengths_pack
            packed_sample["speech_info"] = speech_info_pack

        if packed_sample['input_ids'].numel() == 0:
            print(f'meet bad data, {packed_items}')
            return None

        #print(f"{load_audio_num_threads}_prepare_single_data 执行时长: {time.time() - time_start:.3f}s")
        return packed_sample



@tables.register("dataset_classes", "MinMo_S2S_Parallel")
class MinMo_S2S_Parallel(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        self.image_processor = None
        self.merge_length = 0
        preprocessor_path = kwargs.get("preprocessor_path", None)
        if preprocessor_path is not None:
            image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
            self.image_processor = image_processor
            self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofaudiotoken\|>.*?<\|endofaudiotoken\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [13, 52])
        self.audio_delay = kwargs.get("audio_delay", 26)
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.text_task_id = self.tokenizer.convert_tokens_to_ids("<|text_token|>")
        self.audio_task_id = self.tokenizer.convert_tokens_to_ids("<|audio_token|>")
        self.audio_blank_id = self.tokenizer.convert_tokens_to_ids("<|audio_blank|>")
        self.text_blank_id = self.tokenizer.convert_tokens_to_ids("<|text_blank|>")
        self.audio_delay_lst = [self.text_blank_id] * self.audio_delay

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, audio_tokens, audio_tokens_lens, \
                pixel_values_lst, image_grid_thws, pixel_values_videos_lst, \
                video_grid_thws, video_sencond_per_grids, feature_attention_masks, input_features_lst, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                splits = self.pattern.split(source_input)
                source_ids = []
                # audio
                speech = []
                speech_lengths = []
                speech_token = []
                speech_token_lens = []
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
                               ["<|startofspeech|>", "<|startofaudiotoken|>", "<|startofimage|>", "<|startofvideo|>"]):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                    elif sub_str.startswith("<|startofspeech|>"):
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            # fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token_len_i = olens
                            fake_token = [0] * fake_token_len_i
                            featrue_beg_i = len(source_ids)
                            feature_type_turn = 1
                            source_ids += fake_token
                    elif sub_str.startswith("<|startofaudiotoken|>"):
                        sub_str = sub_str.replace("<|startofaudiotoken|>", "").replace(
                            "<|endofaudiotoken|>", ""
                        )
                        if sub_str.startswith("!"):
                            # Add offset
                            sub_audio_token = np.load(sub_str[1:]) + self.tokenizer.convert_tokens_to_ids("<|c0|>")
                        else:
                            sub_audio_token = self.tokenizer.encode(sub_str)
                        speech_token = torch.tensor(sub_audio_token, dtype=torch.int64)
                        speech_token_lens = torch.tensor([len(sub_audio_token)], dtype=torch.int64)

                    if badcase_flag:
                        continue
                    if featrue_beg_i > 0:
                        feature_beg += [featrue_beg_i + len(input_ids)]
                        fake_token_len += [fake_token_len_i]
                        feature_type += [feature_type_turn]

                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    splits = self.pattern.split(target_out)

                    sub_token = []
                    audio_token = []

                    for k, sub_str in enumerate(splits):
                        if sub_str is None or sub_str == "":
                            continue
                        if len(sub_str) < 1:
                            continue
                        if not sub_str.startswith("<|startofaudiotoken|>"):
                            sub_token = self.tokenizer.encode(sub_str)
                        else:
                            sub_str = sub_str.replace("<|startofaudiotoken|>", "").replace(
                                "<|endofaudiotoken|>", ""
                            )
                            if sub_str.startswith("!"):
                                # Add offset
                                audio_token = np.load(sub_str[1:]) + self.tokenizer.convert_tokens_to_ids("<|c0|>")
                                audio_token = audio_token.tolist()
                            else:
                                audio_token = self.tokenizer.encode(sub_str)
                            # add audio delay
                            if len(audio_token) > 0:
                                audio_token = self.audio_delay_lst + audio_token
                    # mixup_text_audio to target_ids
                    target_ids = self.mixup_text_audio(sub_token, audio_token, self.text_task_id, self.audio_task_id)
                    target_ids.append(self.end_id)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                # audio image video
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
                if len(speech_token) > 0:
                    audio_tokens.append(speech_token)
                    audio_tokens_lens.append(speech_token_lens)
                if len(pixel_values_turn) > 0:
                    pixel_values_lst.extend(pixel_values_turn)
                    image_grid_thws.extend(image_grid_thw_turn)
                if len(pixel_values_videos_turn) > 0:
                    pixel_values_videos_lst.extend(pixel_values_videos_turn)
                    video_grid_thws.extend(video_grid_thw_turn)
                    # video_sencond_per_grids.extend(video_sencond_per_grid_turn)
                    # feature_attention_masks.extend(feature_attention_mask_turn)
                    # input_features_lst.extend(input_features_turn)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
            feature_type = torch.tensor(feature_type, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "feature_beg": feature_beg,
                "feature_type": feature_type,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(audio_tokens) > 0:
                output["audio_tokens"] = audio_tokens
                output["audio_tokens_lens"] = audio_tokens_lens
            if len(pixel_values_lst) > 0:
                output["pixel_values"] = pixel_values_lst
                output["image_grid_thw"] = image_grid_thws
            if len(pixel_values_videos_turn) > 0:
                output["pixel_values_videos"] = pixel_values_videos_lst
                output["video_grid_thw"] = video_grid_thws
                # output["video_sencond_per_grid"] = video_sencond_per_grids
                # output["feature_attention_mask"] = feature_attention_masks
                # output["input_features"] = input_features_lst
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMo_AV2T")
class MinMo_AV2T(torch.utils.data.Dataset):
    """
    Image/Video/Audio/TextDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        ## TODO: load image processor from conf
        from transformers import AutoProcessor
        preprocessor_path = kwargs.get("preprocessor_path", None)
        image_processor = AutoProcessor.from_pretrained(preprocessor_path).image_processor
        self.image_processor = image_processor
        self.merge_length = image_processor.merge_size ** 2

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(
            r"(<\|startofspeech\|>.*?<\|endofspeech\|>)|(<\|startofimage\|>.*?<\|endofimage\|>)|(<\|startofvideo\|>.*?<\|endofvideo\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, \
                pixel_values_lst, image_grid_thws, pixel_values_videos_lst, \
                video_grid_thws, video_sencond_per_grids, feature_attention_masks, input_features_lst, \
                feature_beg, feature_type, fake_token_len = (
                [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                ## add vision_start/vision_end
                user_prompt = user_prompt.replace("<|startofimage|>", "<|vision_start|><|startofimage|>").replace(
                    "<|endofimage|>", "<|endofimage|><|vision_end|>")
                user_prompt = user_prompt.replace("<|startofvideo|>", "<|vision_start|><|startofvideo|>").replace(
                    "<|endofvideo|>", "<|endofvideo|><|vision_end|>")

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                splits = self.pattern.split(source_input)
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
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                    elif sub_str.startswith("<|startofspeech|>"):
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
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

                                images_inputs = self.image_processor(images=image_inputs, videos=None,
                                                                     **output_kwargs["images_kwargs"])
                                image_grid_thw = images_inputs["image_grid_thw"]
                                # image_index = 0
                                fake_token_len_i = image_grid_thw[0].prod() // self.merge_length
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
                                videos_inputs = self.image_processor(images=None, videos=videos,
                                                                     **output_kwargs["videos_kwargs"])
                                fps = output_kwargs["videos_kwargs"].pop("fps", None)
                                if fps is None:
                                    fps = [2.0] * len(videos)
                                videos_inputs["video_second_per_grid"] = [
                                    self.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
                                ]
                                video_grid_thw = videos_inputs["video_grid_thw"]

                                if not use_audio_in_video:
                                    # video_index = 0
                                    fake_token_len_i = video_grid_thw[0].prod() // self.merge_length
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

                    if badcase_flag:
                        continue
                    if featrue_beg_i > 0:
                        feature_beg += [featrue_beg_i + len(input_ids)]
                        fake_token_len += [fake_token_len_i]
                        feature_type += [feature_type_turn]

                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                # audio image video
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
                if len(pixel_values_turn) > 0:
                    pixel_values_lst.extend(pixel_values_turn)
                    image_grid_thws.extend(image_grid_thw_turn)
                if len(pixel_values_videos_turn) > 0:
                    pixel_values_videos_lst.extend(pixel_values_videos_turn)
                    video_grid_thws.extend(video_grid_thw_turn)
                    # video_sencond_per_grids.extend(video_sencond_per_grid_turn)
                    # feature_attention_masks.extend(feature_attention_mask_turn)
                    # input_features_lst.extend(input_features_turn)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            feature_beg = torch.tensor(feature_beg, dtype=torch.int32)
            feature_type = torch.tensor(feature_type, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "feature_beg": feature_beg,
                "feature_type": feature_type,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(pixel_values_lst) > 0:
                output["pixel_values"] = pixel_values_lst
                output["image_grid_thw"] = image_grid_thws
            if len(pixel_values_videos_turn) > 0:
                output["pixel_values_videos"] = pixel_values_videos_lst
                output["video_grid_thw"] = video_grid_thws
                # output["video_sencond_per_grid"] = video_sencond_per_grids
                # output["feature_attention_mask"] = feature_attention_masks
                # output["input_features"] = input_features_lst
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMo_S2S")
class MinMo_S2S(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.text_codec_mix_ratio = kwargs.get("text_codec_mix_ratio", [4, 12])

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            (
                input_ids,
                labels,
                fbank,
                fbank_lens,
                fbank_mask,
                fbank_beg,
                fake_token_len,
                codec,
                codec_len,
                multiturn_beg,
                multiturn_len,
                multiturn_beg_input,
                multiturn_len_input,
                is_t2c,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                # fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                target_ids = []
                is_t2c_i = True
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        is_t2c_i = False

                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                                if self.permute:
                                    speech = speech.permute(0, 2, 1)
                                # if speech_lengths > self.batch_size:
                                #     continue
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            # fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                # # source_mask = [-100] * len(source_ids)
                # # target_out = f"{target_out}<|im_end|>"
                # if any(isinstance(item, dict) and "prev_content" in item for item in target_out):
                #     prev_value = next(
                #         (
                #             item["prev_content"]
                #             for item in target_out
                #             if isinstance(item, dict) and "prev_content" in item
                #         ),
                #         None,
                #     )
                #     source_ids += self.tokenizer.encode(prev_value)
                #     source_mask = [-100] * len(source_ids)
                #     target_out = f"{target_out[0]}<|im_end|>"
                # else:
                #     source_mask = [-100] * len(source_ids)
                #     target_out = f"{target_out}<|im_end|>"
                # target_ids = self.tokenizer.encode(target_out)
                #
                source_mask = [-100] * len(source_ids)
                target_out = f"{target_out}<|im_end|>"
                splits = self.pattern.split(target_out)
                # codec_i = []
                sub_token = []
                codec_i_len = 0

                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        # sub_str = f"{sub_str}<|im_end|>"
                        sub_token = self.tokenizer.encode(sub_str)
                        target_ids += sub_token
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            sub_token_codec = np.load(sub_str[1:])
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)

                        else:
                            sub_token_codec = []
                            for x in sub_str.split("|"):
                                if x.startswith("c"):
                                    sub_token_codec.append(int(x[1:]))
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)
                        codec.append(codec_i)
                        codec_len.append(codec_i_len)
                if codec_i_len > 0 and (
                        codec_i_len / (len(target_ids) - 1)
                        <= self.text_codec_mix_ratio[1] / self.text_codec_mix_ratio[0]
                ):
                    logging.warning(
                        f"item: {item}\ncodec_i_len / (len(target_ids) - 1) <= self.text_codec_mix_ratio[1] / self.text_codec_mix_ratio[0]: {codec_i_len / (len(target_ids) - 1)}<={self.text_codec_mix_ratio[1] / self.text_codec_mix_ratio[0]}"
                    )
                    # badcase_flag = True

                # for tts-model
                multiturn_beg += [len(input_ids) + len(source_ids)]
                multiturn_len += [len(target_ids)]

                multiturn_beg_input += [len(input_ids)]
                multiturn_len_input += [len(source_ids)]
                if i == 0:
                    system_token = self.tokenizer.encode(
                        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    )
                    multiturn_beg_input[-1] = len(system_token)
                    multiturn_len_input[-1] = multiturn_len_input[-1] - len(system_token)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
                is_t2c.append(is_t2c_i)
            if badcase_flag:
                logging.warning(f"badcase_flag: {badcase_flag}, skip it, item: {item}")
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            output = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
            output["fbank_beg"] = fbank_beg
            output["fake_token_len"] = fake_token_len

            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens

            if len(codec) > 0:
                # codec_len = torch.tensor(codec_len, dtype=torch.int32)
                output["codec"] = codec
                output["codec_len"] = codec_len
            if len(multiturn_beg) > 0:
                output["multiturn_beg"] = torch.tensor(multiturn_beg, dtype=torch.int32)
                output["multiturn_len"] = torch.tensor(multiturn_len, dtype=torch.int32)
                output["multiturn_beg_input"] = torch.tensor(multiturn_beg_input, dtype=torch.int32)
                output["multiturn_len_input"] = torch.tensor(multiturn_len_input, dtype=torch.int32)
            if len(is_t2c) > 0:
                output["is_t2c"] = is_t2c
            if item.get("key", None) is not None:
                output["key"] = item.get("key")
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "OpenAIDatasetMultiTurnCodec")
class OpenAIDatasetMultiTurnCodec(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            (
                input_ids,
                labels,
                fbank,
                fbank_lens,
                fbank_mask,
                fbank_beg,
                fake_token_len,
                codec,
                codec_len,
            ) = ([], [], [], [], [], [], [], [], [])

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                # inputs
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if speech_lengths > self.max_source_length:
                                logging.info(
                                    f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                )
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                # targets
                # target_out = f"{target_out}<|im_end|>"
                splits = self.pattern.split(target_out)
                codec_i = []
                sub_token = []
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_str = f"{sub_str}<|im_end|>"
                        sub_token = self.tokenizer.encode(sub_str)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if not sub_str.startswith("!"):
                            sub_token_codec = []
                            for x in sub_str.split("|"):
                                if x.startswith("c"):
                                    sub_token_codec.append(int(x[1:]))
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)

                target_ids = sub_token
                if len(codec_i) > 0:
                    codec.append(codec_i)
                    codec_len.append(codec_i_len)
                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

                fbank_mask += fbank_mask_i
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            if len(codec) > 0:
                codec_len = torch.tensor(codec_len, dtype=torch.int32)
                output["codec"] = codec
                output["codec_len"] = codec_len
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "OpenAIDatasetMultiTurnCodecMel")
class OpenAIDatasetMultiTurnCodecMel(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            key = item.get("key", None)
            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            (
                input_ids,
                labels,
                fbank,
                fbank_lens,
                fbank_mask,
                fbank_beg,
                fake_token_len,
                codec,
                codec_len,
                audio,
                audio_len,
                input_mask,
                input_mask_beg,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])

            multiturn_num = len(system)
            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                # inputs
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                input_mask_i = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if speech_lengths > self.max_source_length:
                                logging.info(
                                    f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                )
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                # targets
                # target_out = f"{target_out}<|im_end|>"
                meta = None
                if isinstance(target_out, (list, tuple)):
                    target_out, meta = target_out
                    if isinstance(meta, dict) and "wav_path" in meta:
                        wav_path = meta["wav_path"]
                        wav_samples = load_audio_text_image_video(wav_path, fs=22050)
                        audio.append(wav_samples)
                        audio_len.append(len(wav_samples))

                splits = self.pattern.split(target_out)
                codec_i = []
                sub_token = []
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_str = f"{sub_str}<|im_end|>"
                        sub_token = self.tokenizer.encode(sub_str)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if not sub_str.startswith("!"):
                            sub_token_codec = []
                            for x in sub_str.split("|"):
                                if x.startswith("c"):
                                    sub_token_codec.append(int(x[1:]))
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)

                target_ids = sub_token
                if len(codec_i) > 0:
                    codec.append(codec_i)
                    codec_len.append(codec_i_len)
                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)

                if i == 0:
                    sys_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    sys_prompt_len = self.tokenizer.encode(sys_prompt)
                    input_mask_i = (
                            [1] * len(sys_prompt_len) + [0] * len(source_ids) + [0] * len(target_ids)
                    )
                else:
                    input_mask_i = (
                            [1] * len(input_ids) + [0] * len(source_ids) + [0] * len(target_ids)
                    )
                input_mask_i = torch.tensor(input_mask_i, dtype=torch.int64)
                input_mask_beg.append(input_mask_i)

                input_mask_i = [1] * len(input_ids) + [1] * len(source_ids) + [0] * len(target_ids)
                input_mask_i = torch.tensor(input_mask_i, dtype=torch.int64)
                input_mask.append(input_mask_i)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

                fbank_mask += fbank_mask_i
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            if len(codec) > 0:
                codec_len = torch.tensor(codec_len, dtype=torch.int32)
                output["codec"] = codec
                output["codec_len"] = codec_len
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(audio) > 0:
                output["audio"] = audio
                output["audio_len"] = audio_len  # torch.tensor(audio_len, dtype=torch.int32)
            if len(input_mask) > 0:
                output["input_mask"] = input_mask
                output["input_mask_beg"] = input_mask_beg

            if key is not None:
                output["key"] = key
            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "OpenAIDatasetMultiTurnCodecMel2")
class OpenAIDatasetMultiTurnCodecMel2(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            (
                input_ids,
                labels,
                fbank,
                fbank_lens,
                fbank_mask,
                fbank_beg,
                fake_token_len,
                codec,
                codec_len,
                audio,
                audio_len,
                input_mask,
                turn_targets,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])

            multiturn_num = len(system)
            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                # inputs
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                input_mask_i = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if speech_lengths > self.max_source_length:
                                logging.info(
                                    f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                )
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                # targets
                # target_out = f"{target_out}<|im_end|>"
                meta = None
                if isinstance(target_out, (list, tuple)):
                    target_out, meta = target_out
                    if isinstance(meta, dict) and "wav_path" in meta:
                        wav_path = meta["wav_path"]
                        wav_samples = load_audio_text_image_video(wav_path, fs=22050)
                        audio.append(wav_samples)
                        audio_len.append(len(wav_samples))

                splits = self.pattern.split(target_out)
                codec_i = []
                sub_token = []
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        turn_targets.append(sub_str)
                        token_num_tts = self.index_ds.tts_text_tokenizer.text2tokens(sub_str)

                        sub_str = f"{sub_str}<|im_end|>"
                        sub_token = self.tokenizer.encode(sub_str)

                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if not sub_str.startswith("!"):
                            sub_token_codec = []
                            for x in sub_str.split("|"):
                                if x.startswith("c"):
                                    sub_token_codec.append(int(x[1:]))
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)

                target_ids = sub_token
                if len(codec_i) > 0:
                    codec.append(codec_i)
                    codec_len.append(codec_i_len)
                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)

                # if i == multiturn_num - 1:
                input_mask_i = [1] * len(input_ids) + [1] * len(source_ids) + [0] * len(target_ids)
                input_mask_i = torch.tensor(input_mask_i, dtype=torch.int64)
                input_mask.append(input_mask_i)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

                fbank_mask += fbank_mask_i
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)

                # # filter
                # # if i == multiturn_num - 1:
                # ratio = codec_i_len / len(token_num_tts)
                # if ratio < 1 or ratio > 7:
                #     badcase_flag = True
                # if codec_i_len + len(token_num_tts) > 1500:
                #     badcase_flag = True

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            if len(codec) > 0:
                codec_len = torch.tensor(codec_len, dtype=torch.int32)
                output["codec"] = codec
                output["codec_len"] = codec_len
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(audio) > 0:
                output["audio"] = audio
                output["audio_len"] = audio_len  # torch.tensor(audio_len, dtype=torch.int32)
            if len(input_mask) > 0:
                output["input_mask"] = input_mask
            if len(turn_targets) > 0:
                output["turn_targets"] = turn_targets
            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "OpenAIDatasetMultiTurnForFullDuplexVAD")
class OpenAIDatasetMultiTurnForFullDuplexVAD(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            task = item["task"]
            true_time_span = item["true_time_span"]
            last_time_span = item["last_total_time"]

            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                elif i == len(system) - 1:
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                # self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if speech_lengths > self.max_source_length:
                                logging.info(
                                    f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                )
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)
                # target_out = f"{target_out}<|im_end|>"
                # target_ids = self.tokenizer.encode(target_out)
                target_ids = []
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                fbank.append(speech[0, :, :])
                fbank_mask += fbank_mask_i
                fbank_lens.append(speech_lengths)

            if badcase_flag:
                continue

            turn_taking_labels = [-100] * len(labels)
            barge_in_labels = [-100] * len(labels)
            last_vad = [0] * fake_token_len[-1]
            pos_vad = math.ceil(fake_token_len[-1] * (true_time_span / last_time_span))
            assert pos_vad <= fake_token_len[-1]
            if pos_vad > 0:
                last_vad[-pos_vad:] = [1] * pos_vad

            if task == "turn-taking":
                turn_taking_labels[-fake_token_len[-1]:] = last_vad
            elif task == "barge-in":
                # print(f'barge-in: {last_vad}')
                barge_in_labels[-fake_token_len[-1]:] = last_vad

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
            turn_taking_labels = torch.tensor(
                turn_taking_labels, dtype=torch.int64
            )  # [: self.max_token_length]
            barge_in_labels = torch.tensor(
                barge_in_labels, dtype=torch.int64
            )  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "speech": fbank,
                "speech_lengths": fbank_lens,
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
                "turn_taking_labels": turn_taking_labels,
                "barge_in_labels": barge_in_labels,
            }
            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


# @tables.register("dataset_classes", "MinMo_T2S")
# class MinMo_T2S(torch.utils.data.Dataset):
#     """
#     SenseVoiceDataset
#     """
#
#     def __init__(
#         self,
#         path,
#         index_ds: str = None,
#         frontend=None,
#         tokenizer=None,
#         int_pad_value: int = -1,
#         float_pad_value: float = 0.0,
#         **kwargs,
#     ):
#         super().__init__()
#         index_ds_class = tables.index_ds_classes.get(index_ds)
#         self.index_ds = index_ds_class(path, **kwargs)
#         preprocessor_speech = kwargs.get("preprocessor_speech", None)
#         if preprocessor_speech:
#             preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
#             preprocessor_speech = preprocessor_speech_class(
#                 **kwargs.get("preprocessor_speech_conf")
#             )
#         self.preprocessor_speech = preprocessor_speech
#         preprocessor_text = kwargs.get("preprocessor_text", None)
#         if preprocessor_text:
#             preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
#             preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
#         self.preprocessor_text = preprocessor_text
#
#         self.frontend = frontend
#         self.fs = 16000 if frontend is None else frontend.fs
#         self.data_type = "sound"
#         self.tokenizer = tokenizer
#
#         self.int_pad_value = int_pad_value
#         self.float_pad_value = float_pad_value
#         self.sos = kwargs.get("sos", "<|startoftranscript|>")
#         self.eos = kwargs.get("eos", "<|endoftext|>")
#         self.batch_size = kwargs.get("batch_size")
#         self.batch_type = kwargs.get("batch_type")
#         self.prompt_ids_len = 0
#         self.retry = kwargs.get("retry", 100)
#
#         self.permute = False
#         from funasr.frontends.whisper_frontend import WhisperFrontend
#
#         if isinstance(self.frontend, WhisperFrontend):
#             self.permute = True
#
#         self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
#         # self.kwargs = kwargs
#         self.max_token_length = kwargs.get("max_token_length", 1500)
#         self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
#         self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
#         self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
#         self.max_source_length = kwargs.get("max_source_length", 3000)
#
#     def get_source_len(self, index):
#         item = self.index_ds[index]
#         return self.index_ds.get_source_len(item)
#
#     def get_target_len(self, index):
#         item = self.index_ds[index]
#         return self.index_ds.get_target_len(item)
#
#     def __len__(self):
#         return len(self.index_ds)
#
#     def __getitem__(self, index):
#         # import pdb
#         #
#         # pdb.set_trace()
#
#         output = None
#
#         for idx in range(self.retry):
#             badcase_flag = False
#             if idx == 0:
#                 index_cur = index
#             else:
#                 index_cur = torch.randint(0, len(self.index_ds), ()).item()
#
#             item = self.index_ds[index_cur]
#
#             system = item["system"]
#             user = item["user"]
#             assistant = item["assistant"]
#
#             input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
#                 [],
#                 [],
#                 [],
#                 [],
#                 [],
#                 [],
#                 [],
#             )
#
#             for i, (system_prompt, user_prompt, target_out) in enumerate(
#                 zip(system, user, assistant)
#             ):
#                 if i >= self.multiturn_num_max:
#                     break
#                 if len(input_ids) > self.max_token_length:
#                     logging.info(
#                         f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
#                     )
#                     break
#
#                 if i == 0:
#                     source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
#                 else:
#                     source_input = (
#                         f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
#                     )
#
#                 splits = self.pattern.split(source_input)
#                 source_ids = []
#                 fbank_i = []
#                 # fbank_mask_i = []
#                 fake_token_len_i = 0
#                 fbank_beg_i = -1
#                 fbank_lens_i = []
#                 speech = []
#                 speech_lengths = []
#                 for k, sub_str in enumerate(splits):
#                     if not sub_str.startswith("<|startofspeech|>"):
#                         sub_token = self.tokenizer.encode(sub_str)
#                         source_ids += sub_token
#                         # fbank_mask_i += [0] * len(sub_token)
#                     else:
#                         sub_str = sub_str.replace("<|startofspeech|>", "").replace(
#                             "<|endofspeech|>", ""
#                         )
#                         if sub_str.startswith("!"):
#                             try:
#                                 data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
#
#                                 speech, speech_lengths = extract_fbank(
#                                     data_src,
#                                     data_type=self.data_type,
#                                     frontend=self.frontend,
#                                     is_final=True,
#                                 )  # speech: [b, T, d]
#                                 if speech_lengths > self.max_source_length:
#                                     logging.info(
#                                         f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
#                                     )
#                                     badcase_flag = True
#                                 if self.permute:
#                                     speech = speech.permute(0, 2, 1)
#                                 # if speech_lengths > self.batch_size:
#                                 #     continue
#                             except Exception as e:
#                                 logging.warning(
#                                     f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
#                                 )
#                                 badcase_flag = True
#                                 continue
#                             olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
#                             olens = 1 + (olens - 3 + 2 * 1) // 2
#                             fake_token_len_i = (olens - 1) // 2 + 1
#                             fake_token = [0] * fake_token_len_i
#                             fbank_beg_i = len(source_ids)
#                             source_ids += fake_token
#                             # fbank_mask_i += [1] * len(fake_token)
#
#                 if badcase_flag:
#                     continue
#                 if fbank_beg_i > 0:
#                     fbank_beg += [fbank_beg_i + len(input_ids)]
#                     fake_token_len += [fake_token_len_i]
#                 else:
#                     fbank_beg += [-1]
#                     fake_token_len += [0]
#
#                 source_mask = [-100] * len(source_ids)
#
#                 target_out = f"{target_out}<|im_end|>"
#                 target_ids = self.tokenizer.encode(target_out)
#                 input_ids += source_ids + target_ids
#                 labels += source_mask + target_ids
#                 if len(speech) > 0:
#                     fbank.append(speech[0, :, :])
#                     fbank_lens.append(speech_lengths)
#             if badcase_flag:
#                 continue
#
#             input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
#             attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
#             labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
#
#             # fbank = speech[0, :, :]
#             # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
#             # fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
#             fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
#             fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
#
#             output = {
#                 "fbank_beg": fbank_beg,
#                 "fake_token_len": fake_token_len,
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#                 "labels_ids": labels,
#             }
#             output["item"] = item
#             if len(fbank) > 0:
#                 output["speech"] = fbank
#                 output["speech_lengths"] = fbank_lens
#             break
#
#         return output
#
#     def collator(self, samples: list = None):
#
#         for idx in range(self.retry):
#             badcase_flag = False
#
#             outputs = {}
#             for sample in samples:
#                 if sample is None:
#                     continue
#                 for key in sample.keys():
#                     if key not in outputs:
#                         outputs[key] = []
#                     if isinstance(sample[key], (list, tuple)):
#                         outputs[key].extend(sample[key])
#                     else:
#                         outputs[key].append(sample[key])
#
#             for key, data_list in outputs.items():
#                 if isinstance(data_list[0], torch.Tensor):
#                     if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
#
#                         pad_value = self.int_pad_value
#                     else:
#                         pad_value = self.float_pad_value
#
#                     outputs[key] = torch.nn.utils.rnn.pad_sequence(
#                         data_list, batch_first=True, padding_value=pad_value
#                     )
#
#             if self.batch_type != "example":
#                 b, t = outputs["input_ids"].shape
#                 if b > 1 and b * t > self.batch_size_token_max:
#                     logging.info(
#                         f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
#                     )
#                     samples = samples[:-1]
#                     continue
#
#             break
#
#         return outputs


@tables.register("dataset_classes", "MinMo_T2S")
class MinMo_T2S(torch.utils.data.Dataset):
    """ """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.eos_idx = kwargs.get("eos_idx", 4096)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            key = item.get("key", None)
            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            (
                input_ids,
                labels,
                fbank,
                fbank_lens,
                fbank_mask,
                fbank_beg,
                fake_token_len,
                codec,
                codec_len,
                audio,
                audio_len,
                input_mask,
                input_mask_beg,
                spk_emb,
                spk_emb_len,
                codec_beg,
                fake_codec_len,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

            multiturn_num = len(system)
            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                # inputs
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                input_mask_i = []
                target_ids = []
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if speech_lengths > self.max_source_length:
                                logging.info(
                                    f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                )
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                # targets

                meta = None
                if isinstance(target_out, (list, tuple)):
                    target_out, meta = target_out
                    if isinstance(meta, dict):
                        if "wav_path" in meta:
                            wav_path = meta["wav_path"]
                            wav_samples = load_audio_text_image_video(wav_path, fs=self.fs)
                            audio.append(wav_samples)
                            audio_len.append(len(wav_samples))
                        if "spk_emb":
                            spk_emb_path = meta["spk_emb"]
                            sub_spk_emb = np.load(spk_emb_path)
                            sub_spk_emb = torch.from_numpy(sub_spk_emb)
                            spk_emb.append(sub_spk_emb)
                            spk_emb_len.append(len(sub_spk_emb))

                target_out = f"{target_out}<|im_end|>"
                splits = self.pattern.split(target_out)
                codec_i = []
                sub_token = []
                codec_beg_i = ([],)
                fake_codec_len_i = ([],)
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        # sub_str = f"{sub_str}<|im_end|>"
                        sub_token = self.tokenizer.encode(sub_str)
                        # target_ids += sub_token
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            sub_token_codec = np.load(sub_str[1:])
                            fake_token = sub_token_codec.tolist() + [self.eos_idx]
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)

                            target_ids += fake_token

                if len(codec_i) > 0:
                    codec.append(codec_i)
                    codec_len.append(codec_i_len)
                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)

                codec_beg_i = len(source_ids)
                fake_codec_len_i = len(codec_i)
                codec_beg += [codec_beg_i + len(input_ids)]
                fake_codec_len += [fake_codec_len_i]

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

                fbank_mask += fbank_mask_i
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                # "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            if len(codec) > 0:
                codec_len = torch.tensor(codec_len, dtype=torch.int32)
                codec_beg = torch.tensor(codec_beg, dtype=torch.int32)
                fake_codec_len = torch.tensor(fake_codec_len, dtype=torch.int32)
                output["codec"] = codec
                output["codec_len"] = codec_len
                output["codec_beg"] = codec_beg
                output["fake_codec_len"] = fake_codec_len

            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(audio) > 0:
                output["audio"] = audio
                output["audio_len"] = audio_len  # torch.tensor(audio_len, dtype=torch.int32)
            if len(spk_emb) > 0:
                output["spk_emb"] = spk_emb
                output["spk_emb_len"] = torch.tensor(spk_emb_len, dtype=torch.int32)
            # if len(input_mask) > 0:
            #     output["input_mask"] = input_mask
            #     output["input_mask_beg"] = input_mask_beg

            if key is not None:
                output["key"] = key
            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("dataset_classes", "MinMo_T2S_low_rate")
class MinMo_T2S_low_rate(torch.utils.data.Dataset):
    """ """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.eos_idx = kwargs.get("eos_idx", 4096)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            key = item.get("key", None)
            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            (
                input_ids,
                labels,
                fbank,
                fbank_lens,
                fbank_mask,
                fbank_beg,
                fake_token_len,
                codec,
                codec_len,
                audio,
                audio_len,
                input_mask,
                input_mask_beg,
                spk_emb,
                spk_emb_len,
                codec_beg,
                fake_codec_len,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

            multiturn_num = len(system)
            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                # inputs
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                input_mask_i = []
                target_ids = []
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if speech_lengths > self.max_source_length:
                                logging.info(
                                    f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                )
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                # targets

                meta = None
                if isinstance(target_out, (list, tuple)):
                    target_out, meta = target_out
                    if isinstance(meta, dict):
                        if "wav_path" in meta:
                            wav_path = meta["wav_path"]
                            wav_samples = load_audio_text_image_video(wav_path, fs=self.fs)
                            audio.append(wav_samples)
                            audio_len.append(len(wav_samples))
                        if "spk_emb":
                            spk_emb_path = meta["spk_emb"]
                            sub_spk_emb = np.load(spk_emb_path)
                            sub_spk_emb = torch.from_numpy(sub_spk_emb)
                            spk_emb.append(sub_spk_emb)
                            spk_emb_len.append(len(sub_spk_emb))

                target_out = f"{target_out}<|im_end|>"
                splits = self.pattern.split(target_out)
                codec_i = []
                sub_token = []
                codec_beg_i = ([],)
                fake_codec_len_i = ([],)
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith("<|startofspeech|>"):
                        # sub_str = f"{sub_str}<|im_end|>"
                        sub_token = self.tokenizer.encode(sub_str)
                        # target_ids += sub_token
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            sub_token_codec = np.load(sub_str[1:])
                            codec_token = sub_token_codec.tolist() + [self.eos_idx]
                            codec_i = torch.tensor(codec_token, dtype=torch.int64)
                            codec_i_len = len(codec_token)

                            target_ids += codec_token

                if len(codec_i) > 0:
                    codec.append(codec_i)
                    codec_len.append(codec_i_len)
                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)

                codec_beg_i = len(source_ids)
                # fake_codec_len_i = len(codec_i)
                fake_codec_token_len_i = (codec_i_len + 1) // 2
                fake_codec_token_i = [0] * fake_codec_token_len_i
                codec_beg += [codec_beg_i + len(input_ids)]
                fake_codec_len += [fake_codec_token_len_i]

                input_ids += source_ids + fake_codec_token_i
                labels += source_mask + target_ids

                fbank_mask += fbank_mask_i
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                # "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            if len(codec) > 0:
                codec_len = torch.tensor(codec_len, dtype=torch.int32)
                codec_beg = torch.tensor(codec_beg, dtype=torch.int32)
                fake_codec_len = torch.tensor(fake_codec_len, dtype=torch.int32)
                output["codec"] = codec
                output["codec_len"] = codec_len
                output["codec_beg"] = codec_beg
                output["fake_codec_len"] = fake_codec_len

            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(audio) > 0:
                output["audio"] = audio
                output["audio_len"] = audio_len  # torch.tensor(audio_len, dtype=torch.int32)
            if len(spk_emb) > 0:
                output["spk_emb"] = spk_emb
                output["spk_emb_len"] = torch.tensor(spk_emb_len, dtype=torch.int32)
            # if len(input_mask) > 0:
            #     output["input_mask"] = input_mask
            #     output["input_mask_beg"] = input_mask_beg

            if key is not None:
                output["key"] = key
            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


# @tables.register("dataset_classes", "MinMo_S2S")
# class MinMo_S2S(torch.utils.data.Dataset):
#     """ """
#
#     def __init__(
#         self,
#         path,
#         index_ds: str = None,
#         frontend=None,
#         tokenizer=None,
#         int_pad_value: int = -1,
#         float_pad_value: float = 0.0,
#         **kwargs,
#     ):
#         """
#
#         :param path:
#         :param index_ds:
#         :param frontend:
#         :param tokenizer:
#         :param int_pad_value:
#         :param float_pad_value:
#         :param kwargs:
#         """
#         super().__init__()
#         index_ds_class = tables.index_ds_classes.get(index_ds)
#         self.index_ds = index_ds_class(path, **kwargs)
#         preprocessor_speech = kwargs.get("preprocessor_speech", None)
#         if preprocessor_speech:
#             preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
#             preprocessor_speech = preprocessor_speech_class(
#                 **kwargs.get("preprocessor_speech_conf")
#             )
#         self.preprocessor_speech = preprocessor_speech
#         preprocessor_text = kwargs.get("preprocessor_text", None)
#         if preprocessor_text:
#             preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
#             preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
#         self.preprocessor_text = preprocessor_text
#
#         self.frontend = frontend
#         self.fs = 16000 if frontend is None else frontend.fs
#         self.data_type = "sound"
#         self.tokenizer = tokenizer
#
#         self.int_pad_value = int_pad_value
#         self.float_pad_value = float_pad_value
#         self.sos = kwargs.get("sos", "<|startoftranscript|>")
#         self.eos = kwargs.get("eos", "<|endoftext|>")
#         self.batch_size = kwargs.get("batch_size")
#         self.batch_type = kwargs.get("batch_type")
#         self.prompt_ids_len = 0
#         self.retry = kwargs.get("retry", 100)
#
#         self.permute = False
#         from funasr.frontends.whisper_frontend import WhisperFrontend
#
#         if isinstance(self.frontend, WhisperFrontend):
#             self.permute = True
#
#         self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
#         # self.kwargs = kwargs
#         self.max_token_length = kwargs.get("max_token_length", 1500)
#         self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
#         self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
#         self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
#         self.max_source_length = kwargs.get("max_source_length", 3000)
#         self.eos_idx = kwargs.get("eos_idx", 4096)
#
#     def get_source_len(self, index):
#         item = self.index_ds[index]
#         return self.index_ds.get_source_len(item)
#
#     def get_target_len(self, index):
#         item = self.index_ds[index]
#         return self.index_ds.get_target_len(item)
#
#     def __len__(self):
#         return len(self.index_ds)
#
#     def __getitem__(self, index):
#         # import pdb
#         #
#         # pdb.set_trace()
#
#         output = None
#
#         for idx in range(self.retry):
#             badcase_flag = False
#             if idx == 0:
#                 index_cur = index
#             else:
#                 index_cur = torch.randint(0, len(self.index_ds), ()).item()
#
#             item = self.index_ds[index_cur]
#
#             key = item.get("key", None)
#             system = item["system"]
#             user = item["user"]
#             assistant = item["assistant"]
#
#             (
#                 input_ids,
#                 labels,
#                 fbank,
#                 fbank_lens,
#                 fbank_mask,
#                 fbank_beg,
#                 fake_token_len,
#                 codec,
#                 codec_len,
#                 audio,
#                 audio_len,
#                 input_mask,
#                 input_mask_beg,
#                 spk_emb,
#                 spk_emb_len,
#                 codec_beg,
#                 fake_codec_len,
#                 multiturn_beg,
#                 multiturn_len,
#             ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
#
#             multiturn_num = len(system)
#             for i, (system_prompt, user_prompt, target_out) in enumerate(
#                 zip(system, user, assistant)
#             ):
#                 if i >= self.multiturn_num_max:
#                     break
#                 if len(input_ids) > self.max_token_length:
#                     logging.info(
#                         f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
#                     )
#                     break
#
#                 if i == 0:
#                     source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
#                 else:
#                     source_input = (
#                         f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
#                     )
#
#                 # inputs
#                 splits = self.pattern.split(source_input)
#                 source_ids = []
#                 fbank_i = []
#                 fbank_mask_i = []
#                 fake_token_len_i = 0
#                 fbank_beg_i = -1
#                 fbank_lens_i = []
#                 speech = []
#                 speech_lengths = []
#                 input_mask_i = []
#                 target_ids = []
#                 for k, sub_str in enumerate(splits):
#                     if len(sub_str) < 1:
#                         continue
#                     if not sub_str.startswith("<|startofspeech|>"):
#                         sub_token = self.tokenizer.encode(sub_str)
#                         source_ids += sub_token
#                         fbank_mask_i += [0] * len(sub_token)
#                     else:
#                         sub_str = sub_str.replace("<|startofspeech|>", "").replace(
#                             "<|endofspeech|>", ""
#                         )
#                         if sub_str.startswith("!"):
#                             try:
#                                 data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
#                             except Exception as e:
#                                 logging.warning(
#                                     f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
#                                 )
#                                 badcase_flag = True
#                                 continue
#                             speech, speech_lengths = extract_fbank(
#                                 data_src,
#                                 data_type=self.data_type,
#                                 frontend=self.frontend,
#                                 is_final=True,
#                             )  # speech: [b, T, d]
#                             if speech_lengths > self.max_source_length:
#                                 logging.info(
#                                     f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
#                                 )
#                                 badcase_flag = True
#                             if self.permute:
#                                 speech = speech.permute(0, 2, 1)
#                             # if speech_lengths > self.batch_size:
#                             #     continue
#
#                             olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
#                             olens = 1 + (olens - 3 + 2 * 1) // 2
#                             fake_token_len_i = (olens - 1) // 2 + 1
#                             fake_token = [0] * fake_token_len_i
#                             fbank_beg_i = len(source_ids)
#                             source_ids += fake_token
#                             fbank_mask_i += [1] * len(fake_token)
#
#                 if badcase_flag:
#                     continue
#
#                 # targets
#
#                 meta = None
#                 if isinstance(target_out, (list, tuple)):
#                     target_out, meta = target_out
#                     if isinstance(meta, dict):
#                         if "wav_path" in meta:
#                             wav_path = meta["wav_path"]
#                             wav_samples = load_audio_text_image_video(wav_path, fs=self.fs)
#                             audio.append(wav_samples)
#                             audio_len.append(len(wav_samples))
#                         if "spk_emb":
#                             spk_emb_path = meta["spk_emb"]
#                             sub_spk_emb = np.load(spk_emb_path)
#                             sub_spk_emb = torch.from_numpy(sub_spk_emb)
#                             spk_emb.append(sub_spk_emb)
#                             spk_emb_len.append(len(sub_spk_emb))
#
#                 target_out = f"{target_out}<|im_end|>"
#                 splits = self.pattern.split(target_out)
#                 codec_i = []
#                 sub_token = []
#                 codec_beg_i = ([],)
#                 fake_codec_len_i = ([],)
#                 for k, sub_str in enumerate(splits):
#                     if len(sub_str) < 1:
#                         continue
#                     if not sub_str.startswith("<|startofspeech|>"):
#                         # sub_str = f"{sub_str}<|im_end|>"
#                         sub_token = self.tokenizer.encode(sub_str)
#                         # target_ids += sub_token
#                     else:
#                         sub_str = sub_str.replace("<|startofspeech|>", "").replace(
#                             "<|endofspeech|>", ""
#                         )
#                         if sub_str.startswith("!"):
#                             sub_token_codec = np.load(sub_str[1:])
#                             fake_token = sub_token_codec.tolist() + [self.eos_idx]
#                             codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
#                             codec_i_len = len(sub_token_codec)
#
#                             target_ids += fake_token
#
#                 if len(codec_i) > 0:
#                     codec.append(codec_i)
#                     codec_len.append(codec_i_len)
#                 fbank_beg += [fbank_beg_i + len(input_ids)]
#                 fake_token_len += [fake_token_len_i]
#                 source_mask = [-100] * len(source_ids)
#
#                 codec_beg_i = len(source_ids)
#                 fake_codec_len_i = len(codec_i)
#                 codec_beg += [codec_beg_i + len(input_ids)]
#                 fake_codec_len += [fake_codec_len_i]
#
#                 # for tts-model
#                 multiturn_beg += len(input_ids)
#                 multiturn_len += [len(source_ids) + len(target_ids)]
#
#                 input_ids += source_ids + target_ids
#                 labels += source_mask + target_ids
#
#                 fbank_mask += fbank_mask_i
#                 if len(speech) > 0:
#                     fbank.append(speech[0, :, :])
#                     fbank_lens.append(speech_lengths)
#
#             if badcase_flag:
#                 continue
#
#             input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
#             attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
#             labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
#
#             # fbank = speech[0, :, :]
#             # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
#             fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
#             fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
#             fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
#
#             output = {
#                 # "fbank_mask": fbank_mask,
#                 "fbank_beg": fbank_beg,
#                 "fake_token_len": fake_token_len,
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#                 "labels_ids": labels,
#             }
#             if len(codec) > 0:
#                 codec_len = torch.tensor(codec_len, dtype=torch.int32)
#                 codec_beg = torch.tensor(codec_beg, dtype=torch.int32)
#                 fake_codec_len = torch.tensor(fake_codec_len, dtype=torch.int32)
#                 output["codec"] = codec
#                 output["codec_len"] = codec_len
#                 output["codec_beg"] = codec_beg
#                 output["fake_codec_len"] = fake_codec_len
#
#             if len(fbank) > 0:
#                 output["speech"] = fbank
#                 output["speech_lengths"] = fbank_lens
#             if len(audio) > 0:
#                 output["audio"] = audio
#                 output["audio_len"] = audio_len  # torch.tensor(audio_len, dtype=torch.int32)
#             if len(spk_emb) > 0:
#                 output["spk_emb"] = spk_emb
#                 output["spk_emb_len"] = torch.tensor(spk_emb_len, dtype=torch.int32)
#             # if len(input_mask) > 0:
#             #     output["input_mask"] = input_mask
#             #     output["input_mask_beg"] = input_mask_beg
#
#             if len(multiturn_beg) > 0:
#                 output["multiturn_beg"] = torch.tensor(multiturn_beg, dtype=torch.int32)
#                 output["multiturn_len"] = torch.tensor(multiturn_len, dtype=torch.int32)
#
#             if key is not None:
#                 output["key"] = key
#             break
#
#         return output
#
#     def collator(self, samples: list = None):
#
#         for idx in range(self.retry):
#             badcase_flag = False
#
#             outputs = {}
#             for sample in samples:
#                 if sample is None:
#                     continue
#                 for key in sample.keys():
#                     if key not in outputs:
#                         outputs[key] = []
#                     if isinstance(sample[key], (list, tuple)):
#                         outputs[key].extend(sample[key])
#                     else:
#                         outputs[key].append(sample[key])
#
#             for key, data_list in outputs.items():
#                 if isinstance(data_list[0], torch.Tensor):
#                     if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
#
#                         pad_value = self.int_pad_value
#                     else:
#                         pad_value = self.float_pad_value
#
#                     outputs[key] = torch.nn.utils.rnn.pad_sequence(
#                         data_list, batch_first=True, padding_value=pad_value
#                     )
#
#             if self.batch_type != "example":
#                 b, t = outputs["input_ids"].shape
#                 if b > 1 and b * t > self.batch_size_token_max:
#                     logging.info(
#                         f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
#                     )
#                     samples = samples[:-1]
#                     continue
#
#             break
#
#         return outputs



import hydra
import logging
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    if kwargs.get("debug", False):
        import pdb

        pdb.set_trace()

    from index_ds import OpenAIIndexDSJsonlForFullDuplexVAD

    index_ds = OpenAIIndexDSJsonlForFullDuplexVAD(
        path="/path/to/vad_test_samples.jsonl"
    )
    print(index_ds.contents)

    kwargs = OmegaConf.to_container(kwargs, resolve=True)
    # build frontend if frontend is none None
    frontend = kwargs.get("frontend", None)
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend = frontend_class(**kwargs["frontend_conf"])
        kwargs["frontend"] = frontend
        kwargs["input_size"] = frontend.output_size()

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = (
                tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            )
            kwargs["token_list"] = (
                tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            )
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer

    # dataset
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
    dataset_train = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=False,
        **kwargs.get("dataset_conf"),
    )

    # dataloader
    batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
    batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
    dataset_conf = kwargs.get("dataset_conf")
    print(f"dataset_conf:\n{dataset_conf}")
    batch_sampler_train = batch_sampler_class(dataset_train, is_training=False, **dataset_conf)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, collate_fn=dataset_train.collator, **batch_sampler_train
    )

    for batch_idx, batch in enumerate(dataloader_train):
        print(batch_idx)
        for key in batch.keys():
            print(key)
            print(batch[key].size())
            if key != "speech":
                print(batch[key].tolist())

        break


if __name__ == "__main__":
    main_hydra()
    """
    local_path="/path/to/vad"
    config="Speech2Text_Align_val_exp3_last8.yaml"

    python funasr/datasets/openai_datasets/datasets.py \
    --config-path "${local_path}" \
    --config-name "${config}" \
    ++train_data_set_list="/path/to/vad_test_samples.jsonl"
    """
