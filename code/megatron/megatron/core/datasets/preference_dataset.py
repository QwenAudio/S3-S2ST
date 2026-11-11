from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset
import json

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.chat_format_utils import make_context
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training.global_vars import get_args

MAX_INT = 1024*1024

def open_jsonl(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    opt = []
    for line in lines:
        opt.append(json.loads(line))
    return opt

@dataclass
class PreferenceDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: bool = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: bool = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: bool = None
    """Option to enable the EOD mask loss"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    use_dynamic_collate: bool = None
    """Option to use dynamic collate. Enable to reduce pad token."""

    chat_format: str = "chatml"
    """chat format"""

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None

class PreferenceDataset(Dataset):
    """Torch dataset"""

    def __init__(
            self,
            input_file,
            config: PreferenceDatasetConfig):
        self.input_file = input_file
        self.dataset = open_jsonl(self.input_file)
        self.config = config
        self.tokenizer = config.tokenizer
        self.im_end_id = self.tokenizer.tokenize("<|im_end|>")  #should be 151645
        self.eod = self.tokenizer.tokenize("<|endoftext|>")[0]  #should be 151643
        self.max_length = config.sequence_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # 处理query
        raw_text, context_tokens, raw_text_len, context_length = make_context(
            message=data["message"], history=data["history"], tokenizer=self.tokenizer,
            max_window_size=MAX_INT,    #设一个大数，不做过滤
            system=data["system"],
            response_prefix=data.get("response_prefix", ""),
            chat_format=self.config.chat_format,
            verbose=False,
        )
        query = context_tokens

        # 处理response
        responses = []
        for res in data['response']:
            if self.config.chat_format == "chatml":
                nl_tokens = self.tokenizer.tokenize('\n')

                res_tokens = self.tokenizer.tokenize(res) + \
                             self.im_end_id + nl_tokens + [self.eod]
            else:
                raise NotImplementedError()

            responses.append(res_tokens)

        scores = list(map(int, data['score']))

        batch = batchify_to_max_length(scores, query, responses, self.eod, self.max_length, self.config.use_dynamic_collate)
        attention_mask, _, position_ids = _get_ltor_masks_and_position_ids(
            batch["tokens"],
            self.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )

        if self.config.create_attention_mask:
            return {
                **batch,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                **batch,
                "position_ids": position_ids,
            }

def build_train_valid_test_datasets_for_preference(input_file, config):
    train_ds = PreferenceDataset(input_file[0], config)
    valid_ds = None
    test_ds = None
    return (train_ds, valid_ds, test_ds)


def batchify_to_max_length(scores, query, responses, eod, max_length, use_dynamic_collate):
    """Creates `TrainingInstance`"""

    num_responses = len(responses)
    assert len(scores) == num_responses
    assert num_responses >= 2

    query_len = len(query)

    # 拼接query+response, 若超出max length，则报错
    out_sequences = []
    labels = []
    loss_masks = []
    for response in responses:
        assert query_len + len(response) <= max_length, f"prompt({query_len}) + response({len(response)}) must <= max_length({max_length})"
        token = query + response
        out_sequences.append(token)
        label = token[1:]
        labels.append(label)
        loss_mask = [0] * (query_len - 1) + [1] * len(response)
        loss_masks.append(loss_mask)

    max_size = max_length
    if use_dynamic_collate:
        pad_size = get_args().tensor_model_parallel_size
        max_size = (max([len(x) for x in out_sequences]) + pad_size - 1) // pad_size * pad_size

    # pad labels, out_sequences to max length
    labels = torch.tensor([pad_and_trunc(t, max_size, eod) for t in labels], dtype=torch.int64)
    tokens = torch.tensor([pad_and_trunc(t, max_size, eod) for t in out_sequences], dtype=torch.int64)
    loss_masks = torch.tensor([pad_and_trunc(t, max_size, 0) for t in loss_masks], dtype=torch.float32)

    output_batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_masks,
        "scores": torch.tensor(scores, dtype=torch.int64),
        "num_responses": torch.tensor(num_responses, dtype=torch.int64),
    }

    return output_batch

def pad_and_trunc(text, max_len, pad_id):
    if len(text) > max_len:
        raise Exception(f"token length({len(text)}) > max_len({max_len})")
    # right padding
    return text + [pad_id] * (max_len - len(text))
