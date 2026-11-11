# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""
from typing import Union
import functools
import os
import random
import time
import torch
import numpy as np
from itertools import chain
from torch.utils.data import Dataset
from megatron.training import get_args, get_tokenizer
from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_current_global_batch_size, get_num_microbatches


def _codex_tensor_stats(value):
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    stats = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
    try:
        data = value.detach().cpu().reshape(-1)
        stats["numel"] = int(data.numel())
        if data.numel() > 0 and value.dtype in (
            torch.float16, torch.float32, torch.float64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool,
        ):
            data_f = data.float()
            stats.update({
                "min": round(float(data_f.min().item()), 4),
                "max": round(float(data_f.max().item()), 4),
                "mean": round(float(data_f.mean().item()), 4),
            })
    except Exception as exc:
        stats["error"] = repr(exc)
    return stats


def _codex_sample_summary(samples, limit=3):
    summary = []
    for sample in (samples or [])[:limit]:
        if not isinstance(sample, dict):
            summary.append({"type": type(sample).__name__})
            continue
        item = sample.get("item")
        if isinstance(item, list) and item:
            item = item[0]
        if isinstance(item, dict):
            users = item.get("user", [])
            assistants = item.get("assistant", [])
            summary.append({
                "key": item.get("key"),
                "source_len": item.get("source_len"),
                "duration": item.get("duration"),
                "user_prefix": str(users[0])[:120] if users else None,
                "assistant_len": len(str(assistants[0])) if assistants else None,
            })
        else:
            summary.append({
                "keys": sorted(sample.keys())[:12],
                "input_ids": _codex_tensor_stats(sample.get("input_ids")) if "input_ids" in sample else None,
                "speech_lengths": _codex_tensor_stats(sample.get("speech_lengths")) if "speech_lengths" in sample else None,
            })
    return summary


def _codex_log_slow_collate(name, elapsed_s, samples, outputs=None, pad_times=None):
    threshold_s = float(os.environ.get("CODEX_SLOW_COLLATE_LOG_SEC", "5"))
    if elapsed_s < threshold_s:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else -1
    except Exception:
        rank = -1
    worker = torch.utils.data.get_worker_info()
    summary = {
        "name": name,
        "rank": rank,
        "worker_id": worker.id if worker is not None else None,
        "elapsed_s": round(elapsed_s, 3),
        "num_samples": len(samples or []),
        "samples": _codex_sample_summary(samples),
        "pad_times": pad_times or {},
    }
    if outputs:
        summary["outputs"] = {
            key: (_codex_tensor_stats(value) if isinstance(value, torch.Tensor) else {"type": type(value).__name__, "len": len(value) if hasattr(value, "__len__") else None})
            for key, value in outputs.items()
        }
    # print(f"[CODEX_SLOW_COLLATE] {summary}", flush=True)


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


def dynamic_collate(samples, pad_id: int, force_loss_mask: bool = False):
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

    # [NOTE]: Add loss_mask for all samples (when using mix_pretrain_sft).
    if force_loss_mask:
        all_loss_masks = []
        for sample in samples:
            loss_mask = sample.get('loss_mask', None)
            if loss_mask is None:
                loss_mask = np.ones_like(sample['tokens'])
            all_loss_masks.append(loss_mask)
        loss_mask = collate_tokens(all_loss_masks, pad_idx=0)
    else:
        loss_mask = None
        if samples[0].get("loss_mask", None) is not None:
            loss_mask = merge("loss_mask", pad=0)
    if loss_mask is not None:
        batch['loss_mask'] = loss_mask

    for key in samples[0].keys():
        if key == 'loss_mask':
            continue
        batch[key] = merge(key, pad=pad_id)

    return batch
def flatten_collate(samples, pad_id=-100, pad_to_multiple=1, force_loss_mask: bool = False, pad_to_length=None):
    """Flatten a list of documents to a single sample, the result batchsize will always be 1"""
    if len(samples) == 0:
        return {}
    assert pad_to_length is None or pad_to_length >= sum([s['tokens'].shape[0]  for s in samples]), "pad_to_length should be larger than the total length of all samples"
    loss_mask = None
    position_ids = None
    total_length = sum([s['tokens'].shape[0] for s in samples])
    # Making sure total_length % pad_to_multiple == 0
    padding_length = 0 if (total_length % pad_to_multiple == 0) else (pad_to_multiple - (total_length % pad_to_multiple))
    padding_length = padding_length if pad_to_length is None else max(0, pad_to_length - total_length)
    # Padding
    samples[-1]['tokens'] = np.concatenate([samples[-1]['tokens'], np.array([pad_id] * padding_length, dtype=np.int64)])
    samples[-1]['loss_mask'] = np.concatenate([samples[-1]['loss_mask'], np.array([0] * padding_length, dtype=np.float32)])
    samples[-1]['labels'] = np.concatenate([samples[-1]['labels'], np.array([pad_id] * padding_length, dtype=np.int64)])
    samples[-1]['position_ids'] = np.arange(samples[-1]['position_ids'].shape[0] + padding_length, dtype=np.int64)
    tokens = np.concatenate([s["tokens"] for s in samples], axis=0, dtype=np.int64)
    loss_mask = np.concatenate(
        [s["loss_mask"] for s in samples], axis=0, dtype=np.float32)
    position_ids = np.concatenate(
        [s["position_ids"] for s in samples], axis=0, dtype=np.int64)
    label = np.concatenate([s["labels"] for s in samples], axis=0, dtype=np.int64)
    batch = {
        "tokens": torch.from_numpy(tokens).unsqueeze(0),
        "labels": torch.from_numpy(label).unsqueeze(0),
        "position_ids": torch.from_numpy(position_ids).unsqueeze(0),
        "loss_mask": torch.from_numpy(loss_mask).unsqueeze(0)
        }
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

def collate_fun_asr(samples):
    """Collator function - 与 MinMoSense 保持一致"""

    # collate_start = time.time()
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

    # pad_times = {}
    for key, data_list in outputs.items():
        if isinstance(data_list[0], torch.Tensor):
            if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                pad_value = -1  # int_pad_value
            else:
                pad_value = 0.0  # float_pad_value

            # pad_start = time.time()
            outputs[key] = torch.nn.utils.rnn.pad_sequence(
                data_list, batch_first=True, padding_value=pad_value
            )
            # pad_times[key] = round(time.time() - pad_start, 3)

    # CODEX slow-collate debug logging disabled.
    # _codex_log_slow_collate("collate_fun_asr", time.time() - collate_start, samples, outputs, pad_times)
    return outputs


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

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
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        if args.use_smart_batching:
            batch_sampler = MegatronPretrainingSmartBatchingSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                enable_bucket_optimization=args.use_smart_batching_bucket,
                chunk_size=args.chunk_size,
                balance_strategy=args.smart_batching_balance_strategy,
            )
        else:
            batch_sampler = MegatronPretrainingRandomSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                data_sharding=args.data_sharding)

    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    print (f'args.use_smart_batching: {args.use_smart_batching}; args.use_dynamic_collate: {args.use_dynamic_collate}; args.qwenvl: {args.qwenvl}; args.qwenclip: {args.qwenclip}; args.mix_pretrain_sft: {args.mix_pretrain_sft}')
    if args.use_smart_batching:
        pad_to_multiple = mpu.get_tensor_model_parallel_world_size()
        # moe interleaving require sb/tp % 2 == 0
        if args.flash_moe_interleaving:
            pad_to_multiple *= 2
        if args.dynamic_padding_full:
            pad_to_length = args.seq_length
        else:
            pad_to_length = None

        collate_fn = functools.partial(flatten_collate, pad_id=dataset.pad_token_id, pad_to_multiple=pad_to_multiple,pad_to_length= pad_to_length, force_loss_mask=args.mix_pretrain_sft)
    elif args.use_dynamic_collate:
        collate_fn = dynamic_collate_qwenvl if args.qwenvl or args.qwenclip else functools.partial(
            dynamic_collate, pad_id=dataset.pad_token_id, force_loss_mask=args.mix_pretrain_sft
        )
    else:
        # TODO: Apply `force_loss_mask` when `args.use_dynamic_collate` is False and `args.mix_pretrain_sft` is True.
        # collate_fn = None
        collate_fn = collate_fun_asr

    # ChunkFlowFlowDataset
    if args.use_chunkflow:
        dataset = ChunkFlowFlowDataset(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                enable_bucket_optimization=args.use_chunkflow_bucket,
                balance_strategy=args.chunkflow_balance_strategy
            )
        return torch.utils.data.DataLoader(dataset,
                                       collate_fn=None,
                                       batch_sampler=None,
                                       num_workers=1,
                                       prefetch_factor=get_num_microbatches(), # Prefetch more data for smart batching.
                                       pin_memory=True)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       collate_fn=collate_fn,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       prefetch_factor=get_num_microbatches() if args.use_chunkflow or args.use_smart_batching else args.prefetch_factor
                                       # remove persistent_workers=True for vpp
                                       # TODO: wait for further optim.
                                       # persistent_workers=True if args.num_workers > 0 else False,
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

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

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

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
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronPretrainingSmartBatchingSampler:
    """
    Construct a dataloader that can try-best to balance total data length processed across all dp ranks
    When using MegatronPretrainingSmartBatchingSampler:
    - Dynamic collate must be disabled
    - FlashAtten must be enabled
    """
    # TODO: Compatible with --mix-pretrain-sft
    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, enable_bucket_optimization, chunk_size, balance_strategy='global-batch'):
        self.args = get_args()
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.balance_strategy = balance_strategy
        self.global_batch_size = get_current_global_batch_size()
        self.num_micro_batches = get_num_microbatches()
        self.enable_bucket_optimization = enable_bucket_optimization
        self.last_batch_size = self.total_samples % self.global_batch_size if balance_strategy == "global-batch" else \
            self.total_samples % self.micro_batch_times_data_parallel_size
        self.chunk_size = chunk_size
        # Sanity checks.
        assert not self.args.use_dynamic_collate, \
            'dynamic collate must be disabled for MegatronPretrainingSmartBatchingSampler'
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)
        assert self.balance_strategy in ['micro-batch', 'global-batch', "none"], \
            'invalid balance_strategy: {}, only {} and {} are supported'.format(self.balance_strategy, 'micro-batch', 'global-batch')
    def __len__(self):
        return self.total_samples

    def binpack_to_constant_bin_number_with_max_weight_limit(self, packages, max_length, bin_num):
        """A bin-packing algorithm to pack the packages into bins with constant bin number and max length limit""" \
        """Returns None if the max weight limit cannot be satisfied"""
        packages.sort(key=lambda item: item[1], reverse=True)
        bins = [[] for _ in range(bin_num)]
        bin_sum = [0] * bin_num
        package_idx = 0
        bin_idx_list = list(range(0, bin_num, 1)) + list(range(bin_num - 1, -1, -1))
        while True:
            processed = False
            for bin_idx in bin_idx_list:
                if bin_sum[bin_idx] + packages[package_idx][1] <= max_length:
                    bins[bin_idx].append(packages[package_idx])
                    bin_sum[bin_idx] += packages[package_idx][1]
                    package_idx += 1
                    processed = True
                    if package_idx == len(packages):
                        break
            if not processed or package_idx == len(packages):
                break
        return bins if package_idx == len(packages) else None

    def construct_balanced_batch(self, batch):
        # No balancing, just flatten the batch
        if self.balance_strategy == "none":
            return batch[self.data_parallel_rank::self.data_parallel_size]
        # Micro-batch level balancing
        if self.balance_strategy == "micro-batch":
            import binpacking
            packages = {}
            for idx, sample_idx in enumerate(batch):
                packages[idx] = self.dataset.dataset.sequence_lengths[sample_idx]
            bins = binpacking.to_constant_bin_number(packages, self.data_parallel_size)
            current_batch = []
            for idx in bins[self.data_parallel_rank].keys():
                current_batch.append(batch[idx])
            return current_batch

        # Global-batch level balancing
        # Special process for sequences longer than chunk_size
        standalone_batches = [[] for _ in range(self.data_parallel_size)]
        dp_packages = []
        idx_list = list(range(self.global_batch_size))
        for dp_rank in range(self.data_parallel_size):
            packages = []
            for idx, sample_idx in zip(idx_list[dp_rank::self.data_parallel_size], batch[dp_rank::self.data_parallel_size]):
                if self.dataset.dataset.sequence_lengths[sample_idx] > self.chunk_size:
                    standalone_batches[dp_rank].append([batch[idx]])
                    continue
                packages.append((idx, self.dataset.dataset.sequence_lengths[sample_idx]))
            dp_packages.append(packages)

        minimum_bin_num = max([len(standalone_batch) for standalone_batch in standalone_batches])
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            minimum_bin_num = max(minimum_bin_num, mpu.get_pipeline_model_parallel_world_size() * 2) # pp-bubble-estimation < 0.5
        minimum_bin_num = min(minimum_bin_num, self.num_micro_batches)
        # Packing documents into same number of samples across dp ranks
        result_bins = None
        for bin_num in range(minimum_bin_num, self.num_micro_batches + 1, 1):
            pack_success = True
            result_bins = None
            for rank in range(self.data_parallel_size):
                pack_res = self.binpack_to_constant_bin_number_with_max_weight_limit(dp_packages[rank], self.chunk_size, bin_num - len(standalone_batches[rank]))
                pack_success = pack_success and (pack_res is not None)
                result_bins = pack_res if rank == self.data_parallel_rank else result_bins
            if pack_success:
                break

        # Construct micro batches
        micro_batches = []
        for bin_idx in range(len(result_bins)):
            micro_batch = [batch[item[0]] for item in result_bins[bin_idx]]
            micro_batches.append(micro_batch)
        micro_batches += standalone_batches[self.data_parallel_rank]
        return micro_batches

    def __iter__(self):
        # Sanity checks:
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0
        if self.enable_bucket_optimization:
            # Same balancing logic with MegatronPretrainingBucketSampler when data_sharding is enabled
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                          * self.micro_batch_size
            dp_rank_shards = []
            for dp_rank in range(self.data_parallel_size):
                start_idx = dp_rank * bucket_size
                g = torch.Generator()
                g.manual_seed(self.epoch)
                # get the data shard
                sequential_idx = list(range(bucket_size))
                idx_range = [start_idx + x for x in sequential_idx]
                current_bucket_size = len(idx_range)
                # Divide the bucket into micro batches and ensure the integrity of the last bucket through padding
                micro_bucket_size = self.micro_batch_size
                last_micro_bucket = current_bucket_size % micro_bucket_size
                idx_range_pad = []
                if last_micro_bucket != 0:
                    # need pad for the last extra micro bucket
                    assert micro_bucket_size-last_micro_bucket <= bucket_size
                    idx_range_pad = idx_range[:micro_bucket_size-last_micro_bucket]
                idx_range.extend(idx_range_pad)

                # sort ids by data length
                idx_range = np.array(idx_range)
                idx_range = idx_range[np.argsort(self.dataset.dataset.sequence_lengths[idx_range], kind='mergesort')]
                current_bucket_size = len(idx_range)
                assert current_bucket_size % micro_bucket_size == 0

                # shuffle all micro buckets
                buckets_num = current_bucket_size//micro_bucket_size
                random_idx = torch.randperm(buckets_num, generator=g).tolist()
                idx_range_shuffled = []
                for index in random_idx:
                    idx_range_shuffled.extend(idx_range[index*micro_bucket_size:(index+1)*micro_bucket_size])
                idx_range = idx_range_shuffled
                dp_rank_shards.append(idx_range)
            shuffle_samples = [0] * sum([len(dp_rank_shard) for dp_rank_shard in dp_rank_shards])
            for dp_rank, dp_rank_shard in enumerate(dp_rank_shards):
                shuffle_samples[dp_rank::self.data_parallel_size] = dp_rank_shard
        else:
            # Continue train from where it left
            g = torch.Generator()
            g.manual_seed(self.epoch)
            shuffle_samples = torch.randperm(self.total_samples, generator=g).tolist()
        shuffle_samples = shuffle_samples[current_epoch_samples: ]
        # Get one batch
        batch = []
        for idx in shuffle_samples:
            batch.append(idx)
            # Balance micro-batch across data parallel ranks
            if (self.balance_strategy == "micro-batch" or self.balance_strategy == "none") and \
                len(batch) == self.micro_batch_times_data_parallel_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield self.construct_balanced_batch(batch)
                batch.clear()
            # Balance micro-batch across data parallel ranks and also across micro-steps
            if self.balance_strategy == "global-batch" and \
                len(batch) == self.global_batch_size:
                micro_batches_for_current_rank = self.construct_balanced_batch(batch)
                self.consumed_samples += self.global_batch_size
                for micro_batch in micro_batches_for_current_rank:
                    yield micro_batch
                batch.clear()

class ChunkFlowFlowDataset(torch.utils.data.IterableDataset):
    """
    A dataset wrapper that splits sequences into smaller mini-sequences for efficient training.

    This class takes a dataset and splits each sequence into smaller mini-sequences of a specified length.
    It ensures that the mini-sequences are properly padded and aligned for parallel processing.
    """
    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, enable_bucket_optimization, balance_strategy='global-batch'):
        self.dataset = dataset
        self.args = get_args()
        self.chunk_size = self.args.chunk_size
        self.batch_sampler = MegatronPretrainingSmartBatchingSampler(
            dataset,
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
            enable_bucket_optimization,
            self.chunk_size,
            balance_strategy,
        )

    def split_batch_into_chunks(self, original_micro_batch):
        """Split a batch containing single sequence into K chunks by $self.chunk_size$"""
        assert len(original_micro_batch) == 1, "Only support micro batch size of 1"
        total_length = sum([s['tokens'].shape[0] for s in original_micro_batch])
        last_chunk_size = total_length % self.chunk_size
        padding_size = 0
        if last_chunk_size % mpu.get_tensor_model_parallel_world_size() != 0:
            padding_size = ((last_chunk_size // mpu.get_tensor_model_parallel_world_size() + 1) * mpu.get_tensor_model_parallel_world_size()) - last_chunk_size
            pad_to_length=total_length + padding_size

        batch = flatten_collate(original_micro_batch, pad_id=self.dataset.pad_token_id, pad_to_length=pad_to_length, force_loss_mask=self.args.mix_pretrain_sft)
        total_tokens = int(batch['loss_mask'].view(-1).float().sum())
        new_batches = []
        for idx in range(0, batch['tokens'].size(1), self.chunk_size):
            new_batch = {}
            for key in batch:
                new_batch[key] = batch[key][0, idx:idx + self.chunk_size]
            new_batch["total_tokens"] = torch.tensor(total_tokens)
            new_batch["seq_length"] = torch.tensor(total_length + padding_size)
            new_batch["alloc_state_store"] = torch.tensor(0)
            new_batches.append(new_batch)

        if len(new_batches) > self.args.chunks_kept:
            # By default, we will execute: F1F2F3F4B4B3B2B1
            # We will execute: F1F2F3F4B4F3B3F2B2F1B1
            new_batch = {}
            for key in batch:
                new_batch[key] = batch[key][0, :self.chunk_size]
            new_batch["total_tokens"] = torch.tensor(total_tokens)
            new_batch["seq_length"] = torch.tensor(total_length + padding_size)
            new_batch["alloc_state_store"] = torch.tensor(1)
            new_batches = new_batches + list(reversed(new_batches[:-self.args.chunks_kept]))
            new_batches[0] = new_batch

        new_batches[0]["alloc_state_store"] = torch.tensor(1)
        return new_batches
    def construct_chunks(self, original_micro_batch):
        total_length = sum([s['tokens'].shape[0]  for s in original_micro_batch])
        if total_length <= self.chunk_size:
            pad_to_multiple = mpu.get_tensor_model_parallel_world_size()
            if self.args.flash_moe_interleaving:
                pad_to_multiple *= 2
            if self.args.dynamic_padding_full:
                pad_to_length = self.chunk_size
            else:
                pad_to_length = None
            batch = flatten_collate(original_micro_batch, pad_id=self.dataset.pad_token_id, pad_to_multiple=mpu.get_tensor_model_parallel_world_size(), force_loss_mask=self.args.mix_pretrain_sft, pad_to_length=pad_to_length)
            for key in batch:
                batch[key] = batch[key][0]
            total_tokens = int(batch['loss_mask'].view(-1).float().sum())
            batch["total_tokens"] = torch.tensor(total_tokens)
            batch["seq_length"] = torch.tensor(total_length)
            batch["alloc_state_store"] = torch.tensor(1)
            return [batch]
        return self.split_batch_into_chunks(original_micro_batch)
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.num_workers <= 1:  # single-process data loading, return the full iterator
            for batch_sample_idx in self.batch_sampler:
                current_batch = [self.dataset[sample_idx] for sample_idx in batch_sample_idx]
                constructed_batches = self.construct_chunks(current_batch)
                for constructed_batch in constructed_batches:
                    yield constructed_batch
        else:  # in a worker process
            # split workload, currently only support single worker
            raise NotImplementedError

    def __len__(self):
        return len(self.batch_sampler)
