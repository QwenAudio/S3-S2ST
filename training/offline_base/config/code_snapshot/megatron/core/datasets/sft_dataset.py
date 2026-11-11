# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""SFT dataset."""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split

logger = logging.getLogger(__name__)


@dataclass
class SFTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for SFT datasets"""

    reset_position_ids: bool = None
    """Option to reset the position IDs in the dataset at an interval"""

    use_dynamic_collate: bool = None
    """Option to use dynamic collate. Enable to reduce pad token."""

    use_smart_batching: bool = False
    """using smart batching optimization for sft training"""

    use_chunkflow: bool = False
    """using chunkflow optimization for sft training"""

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.use_dynamic_collate is not None

        assert not(self.use_chunkflow and self.use_smart_batching), 'chunkflow and smart batching cannot be enabled together'

class SFTDataset(MegatronDataset):
    """SFT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (SFTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices,
        num_samples: Optional[int],
        index_split: Split,
        config: SFTDatasetConfig,
        mask_indexed_dataset: IndexedDataset = None,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        self.mask_indexed_dataset = mask_indexed_dataset
        self.pad_token_id = self.config.tokenizer.eod

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.document_indices.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: SFTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (SFTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return len(self.dataset)

    def _get_sample_idx(self, idx):
        # TODO: Add support to --mix-pretrain-sft.

        return idx

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        sample_idx = self._get_sample_idx(idx)

        sample = self.dataset[sample_idx]
        init_mask = (
            None
            if self.mask_indexed_dataset is None
            else self.mask_indexed_dataset[sample_idx]
        )

        train_sample = build_training_sample(
            sample,
            self.config.sequence_length,
            self.pad_token_id,
            self.config.use_dynamic_collate,
            init_mask=init_mask,
            reset_position_ids=self.config.reset_position_ids,
            packing_batch=(self.config.use_chunkflow or self.config.use_smart_batching)
        )
        return train_sample

    @staticmethod
    def get_mask_dataset_prefix(data_prefix: str) -> Optional[str]:
        """Get mask dataset prefix.

        Tokens dataset: /path/to/dataset/v7_text_document_qwen_15w
        Mask dataset:   /path/to/dataset/v7_mask_qwen_15w
        """
        parts = data_prefix.split("/")
        last_part = parts[-1]

        if "_text_document_" not in last_part:
            return None

        mask_last_part = last_part.replace("_text_document_", "_mask_")
        parts[-1] = mask_last_part

        return "/".join(parts)


def build_training_sample(
    sample,
    max_seq_length,
    pad_id,
    use_dynamic_collate: bool,
    init_mask: Optional[np.ndarray] = None,
    reset_position_ids: bool = False,
    packing_batch: bool = False
):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        max_seq_length: Maximum length of the sequence. All values are padded to this length.
        pad_id: Padding token id.
        use_dynamic_collate: Enable dynamic collate (will disable padding to max_seq_length).
        init_mask: Optional initial mask from dataset.
    """
    # flatten sentences into one list
    origin_tokens = [token for token in sample]

    # Truncate to `target_sequence_length`.
    tokens = origin_tokens[:max_seq_length]
    labels = origin_tokens[1:max_seq_length+1]

    if init_mask is not None:
        init_mask = init_mask[1:max_seq_length+1].tolist()

    # Padding.
    tokens, labels, loss_mask, position_ids = pad_and_convert_to_tensor(
        tokens,
        labels,
        pad_id,
        max_seq_length,
        use_dynamic_collate,
        pad_multiplier=128,
        init_mask=init_mask,
        reset_position_ids=reset_position_ids,
        packing_batch=packing_batch
    )

    train_sample = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }

    return train_sample


def pad_and_convert_to_tensor(
    tokens,
    labels,
    pad_id,
    max_seq_length,
    use_dynamic_collate: bool,
    pad_multiplier: int,
    init_mask: Optional[List[int]] = None,
    reset_position_ids: bool = False,
    packing_batch:bool =False
):
    """Pad sequences and convert them to numpy."""

    num_tokens = len(tokens)
    assert max_seq_length >= num_tokens

    if packing_batch:
        padding_length = 0
    elif use_dynamic_collate:
        # Pad sequence length to `pad_multiplier` for tensor parallel.
        assert max_seq_length % pad_multiplier == 0
        target_seq_length = num_tokens
        while target_seq_length % pad_multiplier != 0:
            target_seq_length += 1
        assert target_seq_length <= max_seq_length
        padding_length = target_seq_length - num_tokens
    else:
        padding_length = max_seq_length - num_tokens

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens = torch.tensor(tokens + filler, dtype=torch.int64)

    label_filler = [pad_id] * (padding_length + num_tokens - len(labels))
    labels = torch.tensor(labels + label_filler, dtype=torch.int64)

    if init_mask is None:
        mask = [1] * num_tokens
    else:
        mask = init_mask

    # Loss mask.
    loss_mask = torch.tensor(mask + [0] * (padding_length + num_tokens - len(mask)), dtype=torch.float32)

    position_ids = torch.arange(tokens.shape[0], dtype=torch.int64)

    # We need to clone as the ids will be modifed based on batch index.

    if reset_position_ids:
        # Find indices where EOD token is.
        eod_index = position_ids[tokens == pad_id]

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    assert not torch.all(loss_mask == 0), f"All-zero mask is detected: {loss_mask}"

    return tokens, labels, loss_mask, position_ids

