# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""
from typing import Union
import random
import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain
from torch.utils.data import Dataset
from megatron import get_args, get_tokenizer
from megatron.core import mpu


def collate_tokens(
    values,
    pad_idx,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    for i in range(len(values)):
        if isinstance(values[i], np.ndarray):
            values[i] = torch.from_numpy(values[i])

    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][: len(v)])
    return res


def dynamic_collate(samples):
    samples = [s for s in samples if 'error' not in s]
    if len(samples) == 0:
        return {}

    def merge(key, pad):
        return collate_tokens([s[key] for s in samples], pad_idx=pad, pad_to_multiple=1)

    def has_key(key):
        for s in samples:
            if key in s:
                return True
        return False

    batch = {}

    if has_key("text"):
        tokenizer = get_tokenizer()
        batch['text'] = merge("text", pad=tokenizer.eod_id)

    if has_key("image"):
        images = []
        for sample in samples:
            if len(sample.get('image', [])) > 0:
                images.extend(sample['image'])
        if len(images) > 0:
            batch['image'] = torch.stack(images, dim=0)

    if has_key("raw_text"):
        batch['raw_text'] = [sample.get('raw_text', None) for sample in samples]

    if has_key("source"):
        batch['source'] = [sample.get('source', None) for sample in samples]

    if has_key("loss_mask"):
        batch['loss_mask'] = merge("loss_mask", pad=0)

    return batch


def collate_tokens_vl(datas: list[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
    for i in range(len(datas)):
        if isinstance(datas[i], np.ndarray):
            datas[i] = torch.from_numpy(datas[i])

    return torch.cat(datas, dim=0).unsqueeze(0)


def dynamic_collate_qwenvl(samples):
    samples = [s for s in samples if 'error' not in s]
    if len(samples) == 0:
        return {}

    def merge(key):
        return collate_tokens_vl([s[key] for s in samples])

    def has_key(key):
        for s in samples:
            if key in s:
                return True
        return False

    batch = {}

    if has_key("text"):
        batch["text"] = merge("text")

    if has_key("labels"):
        batch["labels"] = merge("labels")

    if has_key("scores"):
        batch["scores"] = torch.stack([sample["scores"] for sample in samples], dim=0)

    if has_key("num_responses"):
        batch["num_responses"] = torch.stack([sample["num_responses"] for sample in samples], dim=0)

    if has_key("image"):
        images = list(chain.from_iterable(x["image"] for x in samples if "image" in x))
        if len(images) > 0:
            batch['image'] = torch.cat(images, dim=0)

    if has_key("raw_text"):
        batch['raw_text'] = [sample.get('raw_text', None) for sample in samples]

    if has_key("source"):
        batch['source'] = [sample.get('source', None) for sample in samples]

    if has_key("loss_mask"):
        batch['loss_mask'] = merge("loss_mask")

    if has_key("grid_ts"):
        batch["grid_ts"] = list(
            chain.from_iterable(x["grid_ts"] for x in samples if "grid_ts" in x)
        )

    if has_key("grid_hs"):
        batch["grid_hs"] = list(
            chain.from_iterable(x["grid_hs"] for x in samples if "grid_hs" in x)
        )

    if has_key("grid_ws"):
        batch["grid_ws"] = list(
            chain.from_iterable(x["grid_ws"] for x in samples if "grid_ws" in x)
        )

    if has_key("vision_seqlens"):
        batch["vision_seqlens"] = list(
            chain.from_iterable(x["vision_seqlens"] for x in samples if "vision_seqlens" in x)
        )

    return batch


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )
    else:
        raise Exception('{} dataloader type is not supported.'.format(args.dataloader_type))

    if args.use_dynamic_collate or args.data_impl == 'multimodal':
        collate_fn = dynamic_collate_qwenvl if args.qwenvl or args.qwenclip else dynamic_collate
    else:
        collate_fn = None

    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=10
    )


class MegatronPretrainingSampler:

    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert (
            self.consumed_samples < self.total_samples
        ), 'no samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
