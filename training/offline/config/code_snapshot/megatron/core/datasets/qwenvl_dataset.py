import os
import json
import math
import random
import contextlib
import requests
import oss2
import re
import time
import torch
import torchvision.transforms.v2 as T
import numpy as np
import bisect
from collections import defaultdict, OrderedDict
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
from typing import Optional
from urllib.parse import unquote
from dataclasses import dataclass

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.tokenizer.tokenizer import AbstractTokenizer
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.utils import Split

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
OSS_CONFIG_PATH = os.environ.get("OSS_CONFIG_PATH", "./.oss_config.json")


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_timestamp_index(timestamp, frame_timestamps):
    position = bisect.bisect_left(frame_timestamps, timestamp)
    if position == 0:
        return 0
    elif position == len(frame_timestamps):
        return len(frame_timestamps) - 1
    elif abs(frame_timestamps[position] - timestamp) < abs(frame_timestamps[position - 1] - timestamp):
        return position
    else:
        return position - 1


def tokenize_timestamps(timestamps_info, frame_timestamps, tokenizer):
    """
    Inputs:
        timestamps_info: [
            (0, 5, [3.5, 4.6]),
            (6, 11, [8.3, 11.2]),
        ]
    Outputs:
        tokenize_result: [
            (0, 5, [id1, id2]),
            (6, 11, [id3, id4]),
        ]
    """

    tokenize_result = []
    for timestamps_item in timestamps_info:
        start_idx, end_idx, timestamps = timestamps_item
        item = []
        for timestamp in timestamps:
            timestamp_index = get_timestamp_index(timestamp, frame_timestamps)
            quantize_timestamp = round((timestamp_index / len(frame_timestamps)) * 999)
            quantize_timestamp = str(quantize_timestamp).zfill(3)
            item.extend([tokenizer.time_start_id] + tokenizer.tokenize(quantize_timestamp) + [tokenizer.time_end_id])
        assert len(item) == (end_idx - start_idx)
        tokenize_result.append((start_idx, end_idx, item))

    return tokenize_result


def _bbox_format(bbox) -> (str, int):
    """
    Output:
        ref_str: bbox format str
        count: the point of polygon
    [1, 2, 3, 4] -> "(1,2),(3,4)", 2
    [[1, 2], [3, 4]] -> "(1,2),(3,4)", 2
    [[1, 2], [3, 4], [5, 6], [7, 8]] -> "(1,2),(3,4),(5,6),(7,8)", 4
    """
    assert isinstance(bbox, (tuple, list))
    if isinstance(bbox[0], (tuple, list)):
        bbox = list(chain.from_iterable(bbox))
    bbox = list(map(int, bbox))
    assert len(bbox) % 2 == 0
    s = []
    for i in range(len(bbox) // 2):
        s.append('(%d,%d)' % (bbox[i * 2], bbox[i * 2 + 1]))
    return ','.join(s), len(bbox) // 2


def _remove_overlap_interval(refexps):
    """
    Remove overlap intervals but allow completely equal intervals
        Refer: https://leetcode.com/problems/non-overlapping-intervals
    Inputs:
        refexps: [
            {"chunk": [0, 4], "bbox": 1},
            {"chunk": [0, 3], "bbox": 2},
            {"chunk": [4, 7], "bbox": 3},
            {"chunk": [0, 3], "bbox": 4},
            {"chunk": [3, 7], "bbox": 5},
        ]
    Outpus:
        chunk2boxes: {
            (0, 3): [2, 4],
            (3, 7): [5],
        }
    """
    # remove overlap intervals
    refexps = list(sorted(refexps, key=lambda x: x['chunk']))
    refexps_no_overlap = [refexps[0]]
    refexps_exclude = []
    right = refexps[0]['chunk'][1]
    for i in range(1, len(refexps)):
        if right <= refexps[i]['chunk'][0]:
            refexps_no_overlap.append(refexps[i])
            right = refexps[i]['chunk'][1]
        else:
            refexps_exclude.append(refexps[i])

    # build chunk2boxes
    chunk2boxes = defaultdict(list)
    for ref in refexps_no_overlap:
        bbox = ref['bbox']
        if isinstance(bbox[0], (tuple, list)):
            bbox = list(chain.from_iterable(bbox))
        chunk2boxes[tuple(ref['chunk'])].append(tuple(bbox))

    # allow completely equal intervals
    for ref in refexps_exclude:
        if tuple(ref['chunk']) in chunk2boxes:
            bbox = ref['bbox']
            if isinstance(bbox[0], (tuple, list)):
                bbox = list(chain.from_iterable(ref['bbox']))
            chunk2boxes[tuple(ref['chunk'])].append(tuple(bbox))

    # remove duplicate boxes
    for chunk, boxes in chunk2boxes.items():
        chunk2boxes[chunk] = list(set(boxes))
    return chunk2boxes


def _refexps2segments(text, refexps):
    """
    Inputs:
        text: "I am a Chinese student."
        refexps: [
            {"chunk": [0, 4], "bbox": [1, 2, 3, 4]},
            {"chunk": [7, 14], "bbox": [5, 6, 7, 8]},
            {"chunk": [7, 14], "bbox": [9, 0, 1, 2]},
        ]
    Outputs:
        segments: [
            ("I am", [[1, 2, 3, 4]]),
            (" a", None),
            (" Chinese", [[5, 6, 7, 8], [9, 0, 1, 2]]),
            (" student.", None),
        ]
    """
    # filter invalid interval
    refexps = [
        ref
        for ref in refexps
        if (ref['chunk'][0] >= 0 and ref['chunk'][1] >= 0 and ref['chunk'][0] < ref['chunk'][1])
    ]
    # remove overlap intervals but allow completely equal intervals
    chunk2boxes = _remove_overlap_interval(refexps)

    segments = []
    last_end = 0
    for (start, end), bbox in sorted(chunk2boxes.items(), key=lambda x: x[0][0]):
        # move whitesapce behind, change ["I am ", "student"] to ["I am", " student"]
        if start > 0 and text[start - 1] == ' ':
            start -= 1
        # check no overlap chunk
        assert last_end <= start and start < end, text + str(refexps)
        # insert each segment
        if start > last_end:
            segments.append((text[last_end:start], None))
        segments.append((text[start:end], bbox))
        last_end = end
    if len(text) > last_end:
        segments.append((text[last_end : len(text)], None))

    return segments


def _segment2tokens(segments, tokenizer):
    """
    Inputs:
        segments: [
            ("I am", [[1, 2, 3, 4]]),
            (" a", None),
            (" Chinese", [[5, 6, 7, 8], [9, 0, 1, 2]]),
            (" student.", None),
        ]
        tokenizer: Qwen MM Tokenizer
    Outputs:
        tokens:
        loss_mask:
    """
    tokens = []
    for seg_text, bboxes in segments:
        src_item = tokenizer.tokenize(seg_text)
        if bboxes is None:
            tokens.extend(src_item)
        else:
            tokens.extend(
                [
                    tokenizer.ref_start_id,
                    *src_item,
                    tokenizer.ref_end_id,
                ]
            )
            for bbox in bboxes:
                ref_str, cnt = _bbox_format(bbox)
                bbox_item = tokenizer.tokenize(ref_str)
                if cnt == 2:
                    tokens.extend(
                        [
                            tokenizer.box_start_id,
                            *bbox_item,
                            tokenizer.box_end_id,
                        ]
                    )
                elif cnt >= 3:
                    tokens.extend(
                        [
                            tokenizer.quad_start_id,
                            *bbox_item,
                            tokenizer.quad_end_id,
                        ]
                    )
                else:
                    raise ValueError("bbox format error: {}".format(bbox))
    return tokens


def tokenize_list_format(content, tokenizer, path, name):
    args = get_args()
    eod = tokenizer.eod
    tokens, loss_mask, images = [], [], []
    timestamps_info = []
    for ele in content:
        if ele is None:
            tokens.append(eod)
            loss_mask.append(1)
            if timestamps_info != []:
                images[-1]['timestamps_info'] = timestamps_info
                timestamps_info = []
            continue

        if isinstance(ele, str):
            ele = {'text': ele}

        if 'refexps' in ele and len(ele['refexps']) > 0:
            if 'text' in ele:
                text, loss_value = ele['text'], 1
            else:
                text, loss_value = ele['prompt'], 0
            segments = _refexps2segments(text, ele['refexps'])
            src_item = _segment2tokens(segments, tokenizer)
            tokens.extend(src_item)
            if '/rg/' in path:
                assert src_item[0] == tokenizer.ref_start_id
                assert src_item[-1] == tokenizer.box_end_id
                box_idx = src_item.index(tokenizer.box_start_id)
                loss_mask.extend([0] * box_idx + [1] * (len(src_item) - box_idx))
            else:
                loss_mask.extend([loss_value] * len(src_item))
        elif 'text' in ele:
            src_item = tokenizer.tokenize(ele['text'])
            tokens.extend(src_item)
            loss_mask.extend([1] * len(src_item))
        elif 'prompt' in ele:
            src_item = tokenizer.tokenize(ele['prompt'])
            tokens.extend(src_item)
            loss_mask.extend([0] * len(src_item))
        elif 'image' in ele or "video" in ele:
            seq_len = ele["seq_len"]
            if "video" in ele:
                spatial_tokens = (
                        ele["resized_height"]
                        * ele["resized_width"]
                        // args.spatial_patch_size
                        // args.spatial_patch_size
                        // (args.spatial_merge_size**2)
                )
                seq_len = ele["num_frames"] // args.temporal_patch_size * spatial_tokens
            src_item = [tokenizer.image_modality_token_id] * seq_len
            tokens.extend(src_item)
            loss_mask.extend([0] * seq_len)
            images.append(ele)
        elif 'timestamps' in ele:
            src_item = [tokenizer.time_start_id] * (len(ele['timestamps']) * 5)
            tokens.extend(src_item)
            loss_mask.extend([1] * len(src_item))
            timestamps_info.append((len(tokens) - len(src_item), len(tokens), ele['timestamps']))
        else:
            raise ValueError("Unknown ele {}, only accept str or list".format(ele))

    if len(tokens) > 0 and tokens[-1] != eod:
        tokens.append(eod)
        loss_mask.append(1)
        if timestamps_info != []:
            images[-1]['timestamps_info'] = timestamps_info
    return tokens, loss_mask, images


class OssReader:
    def __init__(self):
        self.bucket = {}
        self.config = {}
        # with open(OSS_CONFIG_PATH) as fin:
        #     self.config = json.load(fin)

    def _parse(self, oss_path):
        assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
        bucket_name = oss_path[len('oss://') :].split('/', 1)[0]
        if bucket_name not in self.bucket:
            bucket_meta = self.config[bucket_name]
            auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
            self.bucket[bucket_name] = oss2.Bucket(
                auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint']
            )
        key_path = oss_path[len('oss://') + len(bucket_name) + 1 :]

        byte_range = None
        if "offset" in key_path and "size" in key_path:
            offset_size = key_path.split('#')[-1]
            start = int(offset_size.split("&")[0][len("offset="):])
            size = int(offset_size.split("&")[-1][len("size="):])
            byte_range = (start, start + size - 1)
            key_path = key_path.split('#')[0]
        elif '#' in key_path:
            start = int(key_path.split('#')[1].split('-')[0])
            end = int(key_path.split('#')[1].split('-')[1])
            size = end - start
            byte_range = (start, end - 1)
            key_path = key_path.split('#')[0]
        return bucket_name, key_path, byte_range

    def read(self, oss_path, retry=3):
        # try:
        #     bucket_name, key_path, byte_range = self._parse(oss_path)
        #     for i in range(retry):
        #         try:
        #             data = self.bucket[bucket_name].get_object(key_path, byte_range=byte_range).read()
        #             break
        #         except Exception as e:
        #             print('retry=', i, e, oss_path)
        #             time.sleep(1)
        #             data = None
        # except Exception as e:
        # 通过url下载
        try:
            data = requests.get(oss_path).content
        except Exception as e:
            print('retry=', i, e, oss_path)
            data = None

        if data is None:
            print('OSS Read File Error {}'.format(oss_path))
        return data

    def exist(self, oss_path):
        bucket_name, key_path, byte_range = self._parse(oss_path)
        return self.bucket[bucket_name].object_exists(key_path)

    # TODO QT delete
    # def get_public_url(self, oss_path):
    #     assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
    #     bucket_name = oss_path[len('oss://') :].split('/', 1)[0]
    #     if bucket_name not in self.bucket:
    #         bucket_meta = self.config[bucket_name]
    #         auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
    #         self.bucket[bucket_name] = oss2.Bucket(
    #             auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint']
    #         )
    #     key_path = oss_path[len('oss://') + len(bucket_name) + 1 :]
    #     if not self.bucket[bucket_name].object_exists(key_path):
    #         raise ValueError(f"{key_path} not exists")
    #     url = self.bucket[bucket_name].sign_url('GET', key_path, 300, slash_safe=True)
    #     return url


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    if height < 10 or width < 10:
        raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
    elif max(height, width) / min(height, width) > 100:
        raise ValueError(f'absolute aspect ratio must be smaller than 100, got {height} / {width}')
    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def get_video_frames_by_full_fps24_cache(oss_reader, oss_url, video_start=0, video_end=-1, n_frames=1):
    """
    Inputs:
        oss_reader: OssReader
        oss_url: a video oss_url like `oss://bucket_name/video_path`
        video_start (float): video start second, inclusive
        video_end (float): video end second, exclusive, any non-positive value (<=0) means the end of full video
        n_frames (int or float):
            if > 0: n_frames (int)
            if < 0: -fps (float)
    Outputs:
        if no fps24 cache, return None
        else return a np.array with shape [n_frames, height, width, n_channels]
    """
    # load video fps24 meta
    info = oss_reader.read(f'{oss_url}.fps24.txt').decode('utf-8')
    info = np.array([d.split() for d in info.splitlines()], dtype=np.int64)

    # sanity check & build frame_indices
    video_start, video_end = video_start * 24, video_end * 24
    if video_end <= 0 or video_end >= len(info):
        video_end = len(info) - 1
    assert video_start >= 0 and video_start < len(info) and video_end >= 0 and video_end <= len(
        info) and video_start < video_end, \
        f"{oss_url} {video_start} {video_end} {n_frames}"
    if n_frames < 0:
        fps = -n_frames
        n_frames = int((video_end - video_start) / 24 * fps)
    assert n_frames <= video_end - video_start, f"{oss_url} {video_start} {video_end} {n_frames}"
    idx = np.linspace(video_start, video_end, n_frames, endpoint=True).round().astype(np.int64).clip(0, len(info) - 1)
    frame_timestamps = idx / 24
    frame_indices = info[idx]
    assert len(frame_indices) == n_frames, f"{oss_url} {video_start} {video_end} {n_frames}, {len(frame_indices)}"

    # load video frames
    frames = []
    for offset, size in frame_indices:
        data = oss_reader.read(f'{oss_url}.fps24.tar#{offset}-{offset + size}')
        try:
            frames.append(np.array(Image.open(BytesIO(data)).convert("RGB")))
        except Exception as e:
            print(f"Error read oss_url: {oss_url} offset: {offset} size: {size}")
            raise ValueError(f"Error read oss_url: {oss_url} offset: {offset} size: {size}")
    frames = np.stack(frames)
    return frames, frame_timestamps



class JsonlDataset(torch.utils.data.Dataset):
    # lowlevel jsonl dataset
    def __init__(self, path):
        super().__init__()
        self.contents = []
        with open(path, encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line.strip())

                if isinstance(data, dict) and data.get('type', '') == 'chatml':
                    self.contents.append(data)
                elif isinstance(data, list):
                    self.contents.append(data)
                elif 'image' in data:
                    self.contents.append([{'image': data['image']}, {'text': data['text']}])
                else:
                    self.contents.append(data['text'])

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index]

@dataclass
class VLSFTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for VLSFT datasets"""


    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

class QwenVLDataset(MegatronDataset):
    def __init__(
            self,
            indexed_dataset: JsonlDataset,
            dataset_path: Optional[str],
            indexed_indices: np.ndarray,
            num_samples: int,
            index_split: Split,
            config: VLSFTDatasetConfig,
    ):
        self.args = get_args()
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        if dataset_path.endswith("final.jsonl"):
            self.source = '/'.join('/'.join(dataset_path.split('/')[-3:-1]).split('.')[:1])
        else:
            self.source = '/'.join('/'.join(dataset_path.split('/')[-2:]).split('.')[:1])

        indexed_indices = np.arange(indexed_indices[0], indexed_indices[1])
        assert np.min(indexed_indices) >= 0, indexed_indices
        assert np.max(indexed_indices) <= len(indexed_dataset)

        iter_cnt = int(math.ceil(num_samples / len(indexed_indices)))
        shuffle_idx = [indexed_indices.copy() for _ in range(iter_cnt)]
        if index_split == Split.train:
            with numpy_seed(self.config.random_seed):
                for i in range(iter_cnt):
                    np.random.shuffle(shuffle_idx[i])
        self.shuffle_idx = np.array(shuffle_idx).flatten()

        self.tokenizer = get_tokenizer()
        self.eod = self.tokenizer.eod

        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=self.args.image_mean, std=self.args.image_std),
            ]
        )
        self.oss_reader = OssReader()

        self.spatial_patch_size = self.args.spatial_patch_size
        self.temporal_patch_size = self.args.temporal_patch_size
        self.spatial_merge_size = self.args.spatial_merge_size
        self.temporal_merge_size = self.args.temporal_merge_size

        self.sequence_parallel = self.args.sequence_parallel
        self.pad_size = self.args.tensor_model_parallel_size
        self.num_workers = self.args.num_workers

        self.use_shanghai_oss = self.args.use_shanghai_oss
        self.min_pixels = self.args.min_pixels
        self.max_pixels = self.args.max_pixels

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: JsonlDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return len(low_level_dataset)
    
    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: VLSFTDatasetConfig) -> JsonlDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (SFTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        return JsonlDataset(dataset_path)

    def __len__(self):
        return self.num_samples

    def get_patch_image(self, image_info, min_pixels=None, max_pixels=None):
        image_path = image_info["image"]
        # TODO QT delete
        #     image_path = image_path.replace(
        #     )
        #     image_path = image_path.replace(
        #         "oss://example-storage-12/zheluo/vl_sft_data"
        #     )
        #
        #                                     'oss://example-storage-9/tansinan.tsn/data')
        #     image_path = image_path.replace(
        #         'oss://example-storage-11/data_hub/coco/train2017')
        #                                     'oss://example-storage-9/tansinan.tsn/data/benchmark')
        #                                     'oss://example-storage-9/tansinan.tsn/data_v1/pretrain')
        #                                     'oss://example-storage-9/tansinan.tsn/Data')
        #                                     'oss://example-storage-9/tansinan.tsn/fanzhihao/blob/datasets')
        #                                     'oss://example-storage-9/tansinan.tsn/data')
        #                                     'oss://example-storage-9/tansinan.tsn/data')
        #     image_path = image_path.replace(
        #         'oss://example-storage-11/data_hub/coco/train2017')
        #     image_path = image_path.replace(
        #         'oss://example-storage-9/tansinan.tsn/data/benchmark')
        #     image_path = image_path.replace(
        #         'oss://example-storage-9/tansinan.tsn/data_v1/pretrain')
        #                                     'oss://example-storage-9/tansinan.tsn/Data')
        #     image_path = image_path.replace(
        #         'oss://example-storage-9/tansinan.tsn/fanzhihao/blob/datasets')
        #                                     'oss://example-storage-9/tansinan.tsn/data')
        #     image_path = image_path.replace(
        #         'oss://example-storage-9/tansinan.tsn/data')

        if image_path.startswith('oss'):
            # # TODO QT delete
            # if image_path.startswith("oss://example-storage-13/data_hub/wanjuan"):
            #     image_path = image_path.replace(
            #         "oss://example-storage-13/data_hub/wanjuan", "oss://example-storage-9/data_hub/wanjuan"
            #     )
            # if image_path.startswith("oss://example-storage-7/data_hub/coco"):
            #     image_path = image_path.replace(
            #         "oss://example-storage-7/data_hub/coco", "oss://example-storage-11/data_hub/coco"
            #     )
            # if image_path.startswith("oss://example-storage-7/data/objects365"):
            #     image_path = image_path.replace(
            #         "oss://example-storage-7/data/objects365", "oss://example-storage-11/data/objects365"
            #     )
            # if image_path.startswith("oss://example-storage-11/data_hub/coco"):
            #     image_path = image_path.replace(
            #         "data_hub/coco/train2014", "data/coco/2014/train2014"
            #     )
            #
            # if self.use_shanghai_oss:
            #     image_path = map_oss_path_to_shanghai(image_path)

            data = self.oss_reader.read(image_path)
        elif image_path.startswith('file:'):
            if "offset" in image_path and "size" in image_path:
                offset_size = image_path.split('#')[-1]
                start = int(offset_size.split("&")[0][len("offset=") :])
                size = int(offset_size.split("&")[-1][len("size=") :])
            else:
                start = int(image_path.split('#')[1].split('-')[0])
                end = int(image_path.split('#')[1].split('-')[1])
                size = end - start
            # # TODO QT delete
            # if self.data_prefix.startswith(
            # ):
            #     image_data_payload = open(
            #         os.path.join(
            #             self.data_prefix.replace(
            #             )
            #             .replace('.jsonl', '')
            #             .replace('_pack8k', '')
            #             .replace('_shuf', '')
            #             .replace('_filter', ''),
            #             image_path.split('#')[0].split('://')[1],
            #         ),
            #         'rb',
            #     )
            # elif self.data_prefix.startswith(
            # ):
            #     image_data_payload = open(
            #         os.path.join(
            #             self.data_prefix.replace(
            #             )
            #             .replace('.jsonl', '')
            #             .replace('_pack8k', '')
            #             .replace('_shuf', '')
            #             .replace('_filter', ''),
            #             image_path.split('#')[0].split('://')[1],
            #         ),
            #         'rb',
            #     )
            # else:
            #     image_data_payload = open(
            #         os.path.join(image_path.split('#')[0].split('://')[1]), 'rb'
            #     )

            image_data_payload = open(
                os.path.join(image_path.split('#')[0].split('://')[1]), 'rb'
            )

            image_data_payload.seek(start)
            data = image_data_payload.read(size)
            image_data_payload.close()
        elif image_path.startswith('http'):
            # TODO delete QT
            last_slash_index = image_path.rfind('/')
            convert_path = "/path/to/images/" + image_path[last_slash_index + 1:]
            if os.path.exists(convert_path):
                with open(convert_path, 'rb') as fin:
                    data = fin.read()
                try:
                    image = Image.open(BytesIO(data)).convert("RGB")
                except Exception as e:
                    raise ValueError(f"{image_path} {str(e)}")
            else:
                image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            # # TODO QT delete
            #     image_path = image_path.replace(
            #     )
            with open(image_path, 'rb') as fin:
                data = fin.read()
            try:
                image = Image.open(BytesIO(data)).convert("RGB")
            except Exception as e:
                raise ValueError(f"{image_path} {str(e)}")
        if 'crop' in image_info:
            crop_info = image_info['crop']
            x1, y1, x2, y2 = crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']
            image = image.crop((x1, y1, x2, y2))

        if 'resized_height' not in image_info or 'resized_width' not in image_info:
            w, h = image.size
            resized_height, resized_width = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
            image_info['resized_height'] = resized_height
            image_info['resized_width'] = resized_width

        image = T.functional.resize(
            image,
            [image_info['resized_height'], image_info['resized_width']],
            interpolation=InterpolationMode.BICUBIC,
        )
        patch_image = self.transform(image)
        grid_t = 1
        grid_h = image_info['resized_height'] // self.spatial_patch_size
        grid_w = image_info['resized_width'] // self.spatial_patch_size
        patch_image = patch_image.unsqueeze(1).expand(-1, 2, -1, -1)
        flatten_image = patch_image.reshape(
            patch_image.size(0),
            grid_t,
            self.temporal_patch_size,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.spatial_patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.spatial_patch_size,
        )
        flatten_image = flatten_image.permute(1, 3, 6, 4, 7, 0, 2, 5, 8).reshape(grid_t * grid_h * grid_w, -1)
        seq_len = grid_h * grid_w // (self.spatial_merge_size**2)
        return flatten_image, seq_len, grid_t, grid_h, grid_w, None

    def get_patch_video_from_image_list(self, video_info: dict):
        video_path = video_info["video"]
        assert isinstance(video_path, list)

        num_frames = len(video_path)
        max_video_seq_len = self.max_pixels / 28 / 28
        min_pixels = 16*28*28
        max_pixels = min(768, max_video_seq_len / num_frames * self.temporal_patch_size)*28*28
        num_frames = round(num_frames / self.temporal_patch_size) * self.temporal_patch_size

        frame_indices = np.linspace(0, len(video_path), num_frames, endpoint=True).round().astype(np.int64).clip(0, len(video_path)-1)

        frames = []
        for index in frame_indices:
            frame_path = video_path[index]
            # # TODO QT delete
            # if self.use_shanghai_oss:
            #     frame_path = map_oss_path_to_shanghai(frame_path)

            data = self.oss_reader.read(frame_path)
            frames.append(np.array(Image.open(BytesIO(data)).convert("RGB")))

        frames = np.stack(frames)

        height, width = frames.shape[1], frames.shape[2]
        resized_height, resized_width = smart_resize(
            height,
            width,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        video_info['num_frames'] = num_frames
        video_info['resized_height'] = resized_height
        video_info['resized_width'] = resized_width

        tokenized_timestamps = None
        if 'timestamps_info' in video_info:
            tokenized_timestamps = tokenize_timestamps(
                video_info['timestamps_info'], frame_timestamps, self.tokenizer
            )

        # TxHxWxC->TxCxHxW
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        frames = T.functional.resize(
            frames,
            [video_info['resized_height'], video_info['resized_width']],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        # TxCxHxW->CxTxHxW
        frames = frames.permute(1, 0, 2, 3).contiguous()
        if frames.dtype == torch.uint8:
            frames = frames.float()
            frames = frames / 255.0
        mean = torch.tensor(self.args.image_mean)
        std = torch.tensor(self.args.image_std)
        frames = frames - mean[:, None, None, None]
        frames = frames / std[:, None, None, None]
        grid_t = video_info['num_frames'] // self.temporal_patch_size
        grid_h = video_info["resized_height"] // self.spatial_patch_size
        grid_w = video_info["resized_width"] // self.spatial_patch_size

        frames = (
            frames.reshape(
                frames.size(0),
                grid_t,
                self.temporal_patch_size,
                grid_h // self.spatial_merge_size,
                self.spatial_merge_size,
                self.spatial_patch_size,
                grid_w // self.spatial_merge_size,
                self.spatial_merge_size,
                self.spatial_patch_size,
            )
            .permute(1, 3, 6, 4, 7, 0, 2, 5, 8)
            .reshape(grid_t * grid_h * grid_w, -1)
        )
        seq_len = grid_t * grid_h * grid_w // (self.spatial_merge_size**2)
        return frames, seq_len, grid_t, grid_h, grid_w, tokenized_timestamps

    def get_patch_video(self, video_info: dict):
        video_path = video_info["video"]
        if isinstance(video_path, list):
            return self.get_patch_video_from_image_list(video_info)

        # # TODO QT delete
        #
        # video_path = video_path.replace('example-storage-10/', 'example-storage-9/')
        # video_path = video_path.replace('example-storage-8/', 'example-storage-9/')
        #                                 'oss://example-storage-9/data_hub')
        # video_path = video_path.replace(
        #     'oss://example-storage-9/data_hub/ego4d_videos')
        #
        # if self.use_shanghai_oss:
        #     video_path = map_oss_path_to_shanghai(video_path)

        if 'num_frames' not in video_info:
            height, width = float(video_info['height']), float(video_info['width'])
            max_num_frames = video_info.get("max_num_frames", 1024)
            min_num_frames = video_info.get("min_num_frames", 4)
            max_video_seq_len = video_info.get("max_video_seq_len", self.max_pixels / 28 / 28)
            fps = video_info.get("fps", 2)
            if 'video_start' in video_info and 'video_end' in video_info:
                duration = float(video_info['video_end']) - float(video_info['video_start'])
            else:
                duration = float(video_info["duration"])
            num_frames = round(duration * fps / self.temporal_patch_size) * self.temporal_patch_size
            num_frames = max(min(num_frames, max_num_frames), min_num_frames)
            if video_info.get('total_frames', None):
                num_frames = min(num_frames, video_info['total_frames'])
            min_pixels = 16 * 28 * 28
            max_pixels = min(768, max_video_seq_len / num_frames * self.temporal_patch_size) * 28 * 28
            resized_height, resized_width = smart_resize(
                height,
                width,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            video_info['num_frames'] = num_frames
            video_info['resized_height'] = resized_height
            video_info['resized_width'] = resized_width

        if 'video_start' in video_info and 'video_end' in video_info:
            video_start = float(video_info['video_start'])
            video_end = float(video_info['video_end'])
        else:
            video_start = 0
            video_end = 0
        num_frames = int(video_info["num_frames"])

        try:
            frames, frame_timestamps = get_video_frames_by_full_fps24_cache(
                self.oss_reader, video_path, video_start, video_end, num_frames
            )
        except Exception as e:
            raise ValueError(f'cant read video {video_path}, {e}, {video_info["duration"]}')

        tokenized_timestamps = None
        if 'timestamps_info' in video_info:
            tokenized_timestamps = tokenize_timestamps(
                video_info['timestamps_info'], frame_timestamps, self.tokenizer
            )

        # TxHxWxC->TxCxHxW
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        frames = T.functional.resize(
            frames,
            [video_info['resized_height'], video_info['resized_width']],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        # TxCxHxW->CxTxHxW
        frames = frames.permute(1, 0, 2, 3).contiguous()
        if frames.dtype == torch.uint8:
            frames = frames.float()
            frames = frames / 255.0
        mean = torch.tensor(self.args.image_mean)
        std = torch.tensor(self.args.image_std)
        frames = frames - mean[:, None, None, None]
        frames = frames / std[:, None, None, None]
        grid_t = num_frames // self.temporal_patch_size
        grid_h = video_info["resized_height"] // self.spatial_patch_size
        grid_w = video_info["resized_width"] // self.spatial_patch_size

        frames = (
            frames.reshape(
                frames.size(0),
                grid_t,
                self.temporal_patch_size,
                grid_h // self.spatial_merge_size,
                self.spatial_merge_size,
                self.spatial_patch_size,
                grid_w // self.spatial_merge_size,
                self.spatial_merge_size,
                self.spatial_patch_size,
            )
            .permute(1, 3, 6, 4, 7, 0, 2, 5, 8)
            .reshape(grid_t * grid_h * grid_w, -1)
        )
        seq_len = grid_t * grid_h * grid_w // (self.spatial_merge_size**2)
        return frames, seq_len, grid_t, grid_h, grid_w, tokenized_timestamps

    def __getitem__(self, idx: int) -> dict:
        try:
            ds_idx = self.shuffle_idx[idx]
            text = self.dataset[ds_idx]

            src_item, loss_mask, images = self.parse_rich_text(text)

            # training input and label is token[:-1] and token[1:], so we add 1 here
            trunc_pos = self.config.sequence_length + 1
            src_item = src_item[:trunc_pos]
            loss_mask = loss_mask[:trunc_pos]

            if len(images) == 0:
                return {
                    "text": src_item,
                    "loss_mask": loss_mask,
                    "source": [self.source, 'multi_modal'],
                }

            vision_patches = []
            grid_ts = []
            grid_hs = []
            grid_ws = []
            vision_seqlens = []
            with ThreadPoolExecutor(max_workers=10) as pool:
                func = self.get_patch_image if "image" in images[0] else self.get_patch_video
                results = pool.map(func, images)
                for flatten_image, seq_len, grid_t, grid_h, grid_w, tokenized_timestamps in results:
                    vision_patches.append(flatten_image)
                    grid_ts.append(grid_t)
                    grid_hs.append(grid_h)
                    grid_ws.append(grid_w)
                    vision_seqlens.append(seq_len)
                    if tokenized_timestamps is not None:
                        for start_idx, end_idx, tokenized_timestamp in tokenized_timestamps:
                            src_item[start_idx:end_idx] = torch.tensor(
                                tokenized_timestamp, dtype=torch.long
                            )
        except Exception as e:
            # print(self.dataset_path, idx, ds_idx, text, str(e))
            return self.__getitem__(idx - 1 if idx > 0 else idx + 1)

        return {
            "text": src_item,
            "loss_mask": loss_mask,
            "image": vision_patches,
            "grid_ts": grid_ts,
            "grid_hs": grid_hs,
            "grid_ws": grid_ws,
            "vision_seqlens": vision_seqlens,
            "raw_text": text,
            "source": [self.source, 'multi_modal'],
        }

    def parse_rich_text(self, content):
        if isinstance(content, list):
            tokens, loss_mask, images = tokenize_list_format(
                content, self.tokenizer, self.data_prefix, self.name
            )
            if self.sequence_parallel:
                seq_len = len(tokens)
                pad_len = (
                        (seq_len + self.pad_size - 2) // self.pad_size * self.pad_size + 1 - seq_len
                )
                if pad_len > 0:
                    # this is slightly faster than self.eod due to flash-attn
                    tokens.extend([self.eod - 1] * pad_len)
                    loss_mask.extend([0.0] * pad_len)
            tokens = torch.tensor(tokens)
            loss_mask = torch.tensor(loss_mask, dtype=torch.float32)
            return tokens, loss_mask, images
        elif isinstance(content, str):
            img_id = self.tokenizer.image_modality_token_id
            tokens, images = self.tokenizer.tokenize_rich(content)
            tokens = torch.tensor(tokens + [self.eod])
            loss_mask = torch.ones_like(tokens, dtype=torch.float32)
            loss_mask[tokens == img_id] = 0.0
            return tokens, loss_mask, images
        else:
            raise ValueError("Unknown content {}, only accept str or list".format(content))


class QwenVLSFTDataset(QwenVLDataset):
    def __init__(
            self,
            indexed_dataset: JsonlDataset,
            dataset_path: Optional[str],
            indexed_indics: np.ndarray,
            num_samples: int,
            index_split: Split,
            config: VLSFTDatasetConfig,
    ):
        super(QwenVLSFTDataset, self).__init__(
            indexed_dataset, dataset_path, indexed_indics,  num_samples, index_split, config
        )
        self.VISION_TAG = '✿VISION✿'
        self.image_modality_token_id = self.tokenizer.image_modality_token_id
        self.im_start_token = self.tokenizer.im_start_id
        self.im_end_token = self.tokenizer.im_end_id
        self.vision_start_id = self.tokenizer.vision_start_id
        self.vision_end_id = self.tokenizer.vision_end_id
        self.nl_token = self.tokenizer.tokenize("\n")
        self.vision_prefix_len = len(self.tokenizer.tokenize(self.VISION_TAG))
        self.mask_special_token = True
        self.mask_prompt = self.args.mask_prompt
        self.mask_chatml_role = self.args.mask_chatml_role
        self.mask_vision_response = self.args.mask_vision_response

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        try:
            ds_idx = self.shuffle_idx[idx]
            text = self.dataset[ds_idx]
            if isinstance(text, list):
                text = {'messages': text}
            src_item, loss_mask, images = self.parse_rich_text(text)
            if not any(loss_mask[1 : self.config.sequence_length + 1]):
                raise RuntimeError(
                    f"All-zero mask detected under {self.config.sequence_length} truncation. "
                    f"Data: {str(text)}"
                )

            # training input and label is token[:-1] and token[1:], so we add 1 here
            trunc_pos = self.config.sequence_length + 1
            src_item = src_item[:trunc_pos]
            loss_mask = loss_mask[:trunc_pos]
            if len(images) == 0:
                return {
                    "text": src_item,
                    "loss_mask": loss_mask,
                    "raw_text": text,
                    "source": [self.source, "text"],
                }

            vision_patches = []
            grid_ts = []
            grid_hs = []
            grid_ws = []
            vision_seqlens = []
            for image in images:
                vision_patches.append(image["flatten_image"])
                grid_ts.append(image["grid_t"])
                grid_hs.append(image["grid_h"])
                grid_ws.append(image["grid_w"])
                vision_seqlens.append(image["vision_seqlen"])
            return {
                "text": src_item,
                "loss_mask": loss_mask,
                "image": vision_patches,
                "grid_ts": grid_ts,
                "grid_hs": grid_hs,
                "grid_ws": grid_ws,
                "vision_seqlens": vision_seqlens,
                "raw_text": text,
                "source": [self.source, 'multi_modal'],
            }
        except Exception as e:
            print(self.dataset_path, idx, ds_idx, str(e))
            return self.__getitem__(idx - 1 if idx > 0 else idx + 1)

    def parse_rich_text(self, data: dict, add_generation_prompt: bool = False) -> tuple[torch.Tensor, torch.Tensor, list]:
        tokens = []
        images = []
        loss_mask = []
        source = data.get('source', '')
        mask_options = data.get('mask_options', {})
        num_last_messages_to_keep = mask_options.get('num_last_messages_to_keep', 9999999)
        if (
            source == 'en:api.agent.opensource.code_feedback'
            and num_last_messages_to_keep != 9999999
        ):
            assert data['messages'][0]['role'] == 'system'
            assert num_last_messages_to_keep == len(data['messages']) - 3, data
        elif source == 'en:api.agent.opensource.code_act' and num_last_messages_to_keep != 9999999:
            assert (
                    num_last_messages_to_keep < len(data['messages'])
                    and not num_last_messages_to_keep % 2
            ), data
        elif source == "online_feedback" or source == "online_repetition" or "other:" in source or "sharegpt" in source:
            num_last_messages_to_keep = 1
        n_turns = len(data['messages'])

        num_images = 0
        for turn in data['messages']:
            if turn is not None:
                content = turn['content']
                num_images += sum(1 if "image" in ele else 0 for ele in content) if isinstance(content, list) else 0
        min_pixels = self.min_pixels
        max_pixels = self.max_pixels / num_images if num_images > 1 else self.max_pixels

        if data['messages'][-1] is None:
            data['messages'] = data['messages'][:-1]

        for turn_id, turn in enumerate(data['messages']):
            if turn is None:
                tokens = tokens + [self.eod]
                loss_mask = loss_mask + [0]
                continue
            turn_tokens, turn_masks = [], []
            role, content = turn['role'], turn['content']
            role_tokens = self.tokenizer.tokenize(role)
            content_tokens: list = []
            vision_info: list = []
            if isinstance(content, list):
                for ele in content:
                    if "video" in ele or "image" in ele:
                        if 'num_frames' in ele and "keepnframe" not in self.data_prefix:
                            del ele['num_frames']
                        flatten_image, vision_seqlen, grid_t, grid_h, grid_w, _ = (
                            self.get_patch_video(ele) if "video" in ele
                            else self.get_patch_image(ele, min_pixels, max_pixels)
                        )
                        vision_info.append(
                            {
                                "flatten_image": flatten_image,
                                "grid_t": grid_t,
                                "grid_h": grid_h,
                                "grid_w": grid_w,
                                "vision_seqlen": vision_seqlen,
                            }
                        )
                        content_tokens.extend(self.vision_start_id + [self.image_modality_token_id] * vision_seqlen + self.vision_end_id)
                    elif "text" in ele:
                        _content_tokens, _ = self.tokenizer.tokenize_rich(ele["text"].rstrip(), self.oss_reader)
                        content_tokens += _content_tokens
                    else:
                        raise ValueError(
                            "Unknown ele {}, only accept video or text now".format(ele)
                        )
            elif isinstance(content, str):
                # deprecated in the near future.
                text = content.rstrip()
                _content_tokens, _vision_info = self.tokenizer.tokenize_rich(text.rstrip(), self.oss_reader)
                content_tokens += _content_tokens
                vision_info += _vision_info
            else:
                NotImplementedError(f"Unknown chat format {content}")
            if vision_info:
                images.extend(vision_info)
            turn_tokens = (
                    self.im_start_token
                    + role_tokens
                    + self.nl_token
                    + content_tokens
                    + self.im_end_token
                    + self.nl_token
            )
            mask_content = (turn_id < n_turns - num_last_messages_to_keep) or (
                    self.mask_prompt and (role == "user" or role == "system")
            )
            # for user, system turn and masked history, mask all special and content tokens
            if mask_content and self.mask_special_token:
                turn_masks = [0] * len(turn_tokens)
            else:
                # Common mask (always keep special tokens).
                turn_masks.append(1)  # im_start
                turn_masks.extend(
                    [int(not self.mask_chatml_role)] * (len(role_tokens) + 1)
                )  # +1 for '\n' behind role
                # Modify content masks.
                if self.mask_vision_response and (
                        role == "assistant"
                        and isinstance(content, str)
                        and content.startswith(self.VISION_TAG)
                ):
                    # [1] * self.vision_prefix_len indicates VISION_TAG's mask
                    content_masks = [1] * self.vision_prefix_len + [0] * (
                            len(content_tokens) - self.vision_prefix_len
                    )
                    # mask imend to avoid learning wrong stop
                    mask_imend = True
                else:
                    content_masks = [1] * len(content_tokens)
                    mask_imend = False
                # Mask the whole content (without special tokens).
                if mask_content and not self.mask_special_token:
                    content_masks = [0] * len(content_masks)
                turn_masks.extend(content_masks)  # content
                if mask_imend:
                    turn_masks.append(0)  # im_end
                    turn_masks.extend([0] * len(self.nl_token))  # '\n'
                else:
                    turn_masks.append(1)  # im_end
                    turn_masks.extend([1] * len(self.nl_token))  # '\n'
            tokens.extend(turn_tokens)
            loss_mask.extend(turn_masks)
        
        if add_generation_prompt:
            tokens = tokens + self.im_start_token + self.tokenizer.tokenize("assistant") + self.nl_token
            loss_mask = []
            return tokens, loss_mask, images
        elif "finished" in data['messages'][-1] and not data['messages'][-1]["finished"]:
            # if last turn is not finished, DO NOT learn imend token
            tokens = tokens[:-2]  # pop the last imend and '\n'
            loss_mask = loss_mask[:-2]
        else:
            tokens = tokens + [self.eod]
            loss_mask = loss_mask + [0]

        if self.sequence_parallel:
            seq_len = len(tokens)
            pad_len = (seq_len + self.pad_size - 2) // self.pad_size * self.pad_size + 1 - seq_len
            if pad_len > 0:
                # this is slightly faster than self.eod due to flash-attn
                tokens.extend([self.eod - 1] * pad_len)
                loss_mask.extend([0.0] * pad_len)
        tokens = torch.tensor(tokens)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)
        loss_mask[tokens == self.image_modality_token_id] = 0.0
        return tokens, loss_mask, images


def pad_and_trunc(text: list[int], max_len: int, pad_id: int) -> list[int]:
    if len(text) > max_len:
        raise Exception(f"text length({len(text)}) > max_len({max_len})")
    return text + [pad_id] * (max_len - len(text))


def batchify_to_max_length(
    scores: list[float],
    query: list[int],
    responses: list[int],
    tokenizer: AbstractTokenizer,
    max_length: int,
    data,
) -> dict:
    LABEL_PAD_ID = -100
    pad_size = get_args().tensor_model_parallel_size
    num_responses = len(responses)
    assert len(scores) == num_responses
    assert num_responses >= 2

    query_len = len(query)

    out_sequences: list = []
    labels: list = []

    for response in responses:
        assert (
            query_len + len(response) <= max_length
        ), f"prompt({query_len}) + response({len(response)}) must <= max_length({max_length}), data: {json.dumps(data, ensure_ascii=False)}"
        out_sequences.append(query + response)
        label = [LABEL_PAD_ID] * (query_len - 1) + response
        labels.append(label)

    max_size = max_length
    use_dynamic_collate = get_args().use_dynamic_collate or get_args().data_impl == 'multimodal'
    if use_dynamic_collate:
        max_size = (max([len(x) for x in out_sequences]) + pad_size - 1) // pad_size * pad_size

    # pad labels, out_sequences to max length
    labels = [pad_and_trunc(t, max_size, LABEL_PAD_ID) for t in labels]
    text = [pad_and_trunc(t, max_size, tokenizer.eod) for t in out_sequences]

    output_batch = {
        "text": np.array(text),
        "labels": np.array(labels),
        "scores": torch.tensor(scores),
        "query_len": query_len,
        "num_responses": torch.tensor(num_responses),
    }
    return output_batch
