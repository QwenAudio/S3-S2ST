import logging
import os.path
import re
import regex
import torch
import random
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
import math
import numpy as np
import random

try:
    from wetext import Normalizer

    normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=True)
except ImportError:
    print("Please install wetext: pip install wetext)")


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
