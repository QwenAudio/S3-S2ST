# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import math
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import torch
from PIL import Image
from transformers import BatchFeature

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def round_by_factor(number: int, factor: int) -> int:
    """ 返回最接近 number 的且能被 factor 整除的整数 """
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """ 返回大于等于 number 的且能被 factor 整除的整数 """
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """ 返回小于等于 number 的且能被 factor 整除的整数 """
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = 28,
                 min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 16384):
    if height < 10 or width < 10:
        raise RuntimeError(
            f'height:{height} or width:{width} must be larger than 10')
    elif max(height, width) / min(height, width) > 200:
        raise RuntimeError(
            f'absolute aspect ratio must be smaller than 200, got {height} / {width}'
        )
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


class PreProcessingCache:

    def __init__(self):
        self.video_cache_queue_size = envs.VLLM_PREPROCESS_MMD_CACHE_SIZE
        self.video_cache_ptr = 0
        self.cached_image_hash_table = [None] * self.video_cache_queue_size
        self.cached_image_features = [None] * self.video_cache_queue_size

        self.audio_cache_queue_size = envs.VLLM_PREPROCESS_MMD_CACHE_SIZE
        self.audio_cache_ptr = 0
        self.cached_audio_hash_table = [None] * self.audio_cache_queue_size
        self.cached_audio_features = [None] * self.audio_cache_queue_size
        self.max_workers = 20
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.video_lock = threading.Lock()
        self.audio_lock = threading.Lock()

    def update_cache(
        self,
        processor_data: Mapping[str, object],
        features: BatchFeature,
        image_use_cache_list,
        audio_use_cache_list,
    ):
        with self.video_lock:
            if 'videos' in processor_data:
                # eacho video chunk response has one video_grid_thw
                assert (len(processor_data['videos']) == len(
                    features['video_grid_thw']))
                image_pixel_start = 0
                for i, image_batch in enumerate(processor_data['videos']):
                    image_pixel_length = int(
                        features['video_grid_thw'][i].prod())
                    if image_use_cache_list[i][0] is not None:
                        image_pixel_start += image_pixel_length
                        continue
                    if len(image_batch):
                        # img_in_batch = len(image_batch)
                        hash_code = image_use_cache_list[i][1]
                        if hash_code is None:
                            break
                        self.cached_image_hash_table[
                            self.video_cache_ptr] = hash_code
                        image_cache = {}
                        image_cache['pixel_values_videos'] = features[
                            'pixel_values_videos'][
                                image_pixel_start:image_pixel_start +
                                image_pixel_length].detach()
                        image_pixel_start += image_pixel_length
                        image_cache['video_grid_thw'] = features[
                            'video_grid_thw'][i].detach()
                        image_cache['video_second_per_grid'] = features[
                            'video_second_per_grid'][i]
                        self.cached_image_features[
                            self.video_cache_ptr] = image_cache
                        self.video_cache_ptr += 1
                        self.video_cache_ptr %= self.video_cache_queue_size
        with self.audio_lock:
            if 'audio' in processor_data:
                for i, audio in enumerate(processor_data['audio']):
                    if audio_use_cache_list[i][0] is not None:
                        continue
                    hash_code = audio_use_cache_list[i][1]
                    self.cached_audio_hash_table[
                        self.audio_cache_ptr] = hash_code
                    audio_cache = {}
                    audio_cache['input_features'] = features['input_features'][
                        i].detach().unsqueeze(0)
                    audio_cache['feature_attention_mask'] = features[
                        'feature_attention_mask'][i].detach().unsqueeze(0)
                    self.cached_audio_features[
                        self.audio_cache_ptr] = audio_cache
                    self.audio_cache_ptr += 1
                    self.audio_cache_ptr %= self.audio_cache_queue_size

    def get_image_batch_hash_idx(self, hash_code):
        for i in range(self.video_cache_queue_size):
            if self.cached_image_hash_table[i] == hash_code:
                return i, self.cached_image_features[i], hash_code
        return -1, None, hash_code

    def get_audio_hash_idx(self, hash_code):
        for i in range(self.audio_cache_queue_size):
            if self.cached_audio_hash_table[i] == hash_code:
                return i, self.cached_audio_features[i], hash_code
        return -1, None, hash_code

    def load_image(self, path, resized_width, resized_height):
        if path.startswith('file://'):
            path = path[7:]
        image = Image.open(path).convert('RGB').resize(
            (resized_width, resized_height))
        return image

    def get_use_cached_feature_lists(
        self,
        processor_data: Mapping[str, object],
    ):
        new_processor_data = {}
        # calculate hash_code for all images and audios in multi-threads
        # list[list[image]]
        hash_list = []
        all_images_path = []
        image_use_cache_list = []
        if 'videos' in processor_data:
            for i, image_batch in enumerate(processor_data['videos']):
                all_images_path.append(image_batch)
                hash_code = ''
                for image_path in image_batch:
                    hash_code += hashlib.md5(
                        image_path.encode("utf-8")).hexdigest()
                hash_list.append(hash_code)
            with self.video_lock:
                for hash_code in hash_list:
                    img_cache_idx, img_cache_fea, hash_code = \
                        self.get_image_batch_hash_idx(hash_code)
                    if img_cache_idx < 0:
                        image_use_cache_list.append([None, hash_code])
                    else:
                        image_use_cache_list.append([img_cache_fea, hash_code])

            logger.info('preprocessor got all hash, rid')

            images_future = []
            first_image = None
            resize_h = 1
            resize_w = 1
            load_count = 0
            for i in range(len(image_use_cache_list)):
                futures = []
                if image_use_cache_list[i][0] is None:
                    for image_path in all_images_path[i]:
                        if first_image is None:
                            real_url = image_path
                            if real_url.startswith('file://'):
                                real_url = real_url[7:]
                            first_image = Image.open(real_url)
                            resize_h, resize_w = smart_resize(
                                first_image.size[1],
                                first_image.size[0],
                                factor=32,
                                min_pixels=32 * 32 * 4,
                                max_pixels=32 * 32 * 768)
                            logger.info(
                                'preprocess image from w:%d, h:%d resize to w:%d, h:%d',
                                first_image.size[1], first_image.size[0],
                                resize_w, resize_h)
                        futures.append(
                            self.executor.submit(self.load_image, image_path,
                                                 resize_w, resize_h))
                        load_count += 1
                    images_future.append(futures)
                else:
                    images_future.append(None)

            logger.info('preprocessor loaded all image: %d', load_count)

            new_video = []

            for i in range(len(image_use_cache_list)):
                if image_use_cache_list[i][0] is None:
                    new_video.append([x.result() for x in images_future[i]])
            if new_video != []:
                new_processor_data['videos'] = new_video

        audio_use_cache_list = []

        if 'audio' in processor_data:
            hash_list = []
            for i, audio in enumerate(processor_data['audio']):
                hash_list.append(hashlib.md5(audio.data).hexdigest())
            with self.audio_lock:
                for hash_code in hash_list:
                    audio_cache_idx, audio_cache_fea, hash_code = \
                        self.get_audio_hash_idx(hash_code)
                    if audio_cache_idx < 0:
                        audio_use_cache_list.append([None, hash_code])
                    else:
                        audio_use_cache_list.append(
                            [audio_cache_fea, hash_code])

        image_cache_hit_rate = 0
        if len(image_use_cache_list) > 0:
            image_cache_hit_rate = sum(
                [x[0] is not None
                 for x in image_use_cache_list]) / len(image_use_cache_list)
        audio_cache_hit_rate = 0
        if len(audio_use_cache_list) > 0:
            audio_cache_hit_rate = sum(
                [x[0] is not None
                 for x in audio_use_cache_list]) / len(audio_use_cache_list)

        if len(audio_use_cache_list) > 0 and 'audio' in processor_data:
            index_to_reserve = [
                i for i, cache in enumerate(audio_use_cache_list)
                if cache[0] is None
            ]
            new_audios = [processor_data['audio'][i] for i in index_to_reserve]
            if new_audios != []:
                new_processor_data['audio'] = new_audios

        if 'inference_kwargs' in processor_data:
            new_processor_data['inference_kwargs'] = processor_data[
                'inference_kwargs']

        new_text_str = ''
        if 'audio' in new_processor_data:
            new_text_str += '<|AUDIO|>' * len(new_processor_data['audio'])
        if 'videos' in new_processor_data:
            new_text_str += '<|VIDEO|>' * len(new_processor_data['videos'])
        new_processor_data['text'] = new_text_str

        return (
            image_use_cache_list,
            audio_use_cache_list,
            image_cache_hit_rate,
            audio_cache_hit_rate,
            new_processor_data
        )

    def get_hf_processor_result_with_cache(
        self,
        processor_data: Mapping[str, object],
        features: BatchFeature,
        image_cache_list,
        audio_cache_list,
    ) -> BatchFeature:
        if 'videos' in processor_data:
            image_fea_segments = []
            video_grids = []
            video_seconds = []
            batch_idx = 0
            image_pixel_start = 0
            # 如果没有image，不会进入循环
            # eacho video chunk response has one video_grid_thw
            assert (len(processor_data['videos']) == len(
                features['video_grid_thw']))
            for cache, hash_code in image_cache_list:
                if cache is not None:
                    # 命中的部分，包含全部命中的情况
                    image_fea_segments.append(cache['pixel_values_videos'])
                    video_grids.append(cache['video_grid_thw'])
                    video_seconds.append(cache['video_second_per_grid'])
                else:
                    # 未命中的部分，填入hp processor输出，
                    # 在全部处理后更新cache，防止中间cache被覆盖
                    # image_batch = processor_data['videos'][batch_idx]
                    image_pixel_length = int(
                        features['video_grid_thw'][batch_idx].prod())
                    image_fea_segments.append(
                        features['pixel_values_videos']
                        [image_pixel_start:image_pixel_start +
                         image_pixel_length])
                    image_pixel_start += image_pixel_length
                    video_grids.append(features['video_grid_thw'][batch_idx])
                    video_seconds.append(
                        features['video_second_per_grid'][batch_idx])
                    batch_idx += 1
            features['pixel_values_videos'] = torch.cat(image_fea_segments,
                                                        dim=0).detach()
            features['video_grid_thw'] = torch.stack(video_grids,
                                                     dim=0).detach()
            features['video_second_per_grid'] = torch.tensor(video_seconds)
        else:
            if len(image_cache_list) > 0:
                # 如果全部命中，需要单独处理
                image_fea_segments = []
                video_grids = []
                video_seconds = []
                # 如果没有image，不会进入循环
                for cache, hash_code in image_cache_list:
                    if cache is not None:
                        # 命中的部分，包含全部命中的情况
                        image_fea_segments.append(cache['pixel_values_videos'])
                        video_grids.append(cache['video_grid_thw'])
                        video_seconds.append(cache['video_second_per_grid'])
                    else:
                        raise ValueError(
                            'should not be here, all cached but cache not found'
                        )

                features['pixel_values_videos'] = torch.cat(image_fea_segments,
                                                            dim=0).detach()
                features['video_grid_thw'] = torch.stack(video_grids,
                                                         dim=0).detach()
                features['video_second_per_grid'] = video_seconds

        if 'audio' in processor_data:
            audio_mask_segments = []
            audio_fea_segments = []
            batch_idx = 0
            # 如果没有image，不会进入循环
            for cache, hash_code in audio_cache_list:
                if cache is not None:
                    # 命中的部分，包含全部命中的情况
                    audio_mask_segments.append(cache['feature_attention_mask'])
                    audio_fea_segments.append(cache['input_features'])
                else:
                    # 未命中的部分，填入hp processor输出，
                    # 在全部处理后更新cache，防止中间cache被覆盖
                    audio_mask_segments.append(
                        features['feature_attention_mask']
                        [batch_idx].unsqueeze(0))
                    audio_fea_segments.append(
                        features['input_features'][batch_idx].unsqueeze(0))
                    batch_idx += 1
            features['feature_attention_mask'] = torch.cat(audio_mask_segments,
                                                           dim=0).detach()
            features['input_features'] = torch.cat(audio_fea_segments,
                                                   dim=0).detach()
        else:
            if len(audio_cache_list) > 0:
                # 如果全部命中，需要单独处理
                audio_mask_segments = []
                audio_fea_segments = []
                # 如果没有image，不会进入循环
                for cache, hash_code in audio_cache_list:
                    if cache is not None:
                        # 命中的部分，包含全部命中的情况
                        audio_mask_segments.append(
                            cache['feature_attention_mask'])
                        audio_fea_segments.append(cache['input_features'])
                    else:
                        raise ValueError(
                            'should not be here, all cached but cache not found'
                        )
                features['feature_attention_mask'] = torch.cat(
                    audio_mask_segments, dim=0).detach()
                features['input_features'] = torch.cat(audio_fea_segments,
                                                       dim=0).detach()

        new_input_ids = []
        # Fix me:
        #     1. use <|VIDEO|> <|AUDIO|> token id from config file
        #     2. calculate audio_ids_len & video_ids_len when updating cache
        if 'feature_attention_mask' in features:
            audio_ids_len = features['feature_attention_mask'].sum() // 4
            new_input_ids.extend(repeat(151672, audio_ids_len))
        if 'video_grid_thw' in features:
            video_ids_len = torch.prod(features['video_grid_thw'],
                                       dim=1).sum() // 4
            new_input_ids.extend(repeat(151673, video_ids_len))
        features['input_ids'] = torch.tensor([new_input_ids])
        features['attention_mask'] = torch.ones(features['input_ids'].shape)
        # for k, v in features.items():
        #     print('{}, {}, {}'.format(k, v, v.shape))

        return features

    def get_usage_rate(self):
        image_cache_usage_rate = sum(
            [x is not None for x in self.cached_image_hash_table])
        audio_cache_usage_rate = sum(
            [x is not None for x in self.cached_audio_hash_table])
        return image_cache_usage_rate, audio_cache_usage_rate
