#!/usr/bin/env python3

'''
Build a index for a big jsonl file.
For each line in a jsonl file, the index contains:
1. the file path, "file_path"
2. the line number, "line_no"
3. the offset and length of the line in bytes, "offset", "length"
4. extra customizable fields, "extra"

The index is saved as a jsonl file, with the same name as the input jsonl file, but with the suffix ".index".

Usage:
python build_jsonl_index.py <input_jsonl_file> <output_jsonl_file>
'''

import os
try:
    import ujson as json
except ImportError:
    import json
import tqdm
import argparse
import random
from functools import partial
from copy import deepcopy
from collections import defaultdict
import hashlib
import math
import numpy as np


def build_jsonl_index(input_jsonl_file, output_index_file, extra_fields_handler: callable = None):
    num_lines = 0
    with open(input_jsonl_file, 'r', encoding='utf-8') as fp_in, \
         open(output_index_file, 'w', encoding='utf-8') as fp_out_index:

        line_no = 0
        num_drop = 0
        offset = 0
        pbar = tqdm.tqdm(total=os.path.getsize(input_jsonl_file) / 1024**2, desc="Building index", unit='M')
        while True:
            # Capture offset before reading line
            # offset = fp_in.tell()
            line = fp_in.readline()
            end_offset = fp_in.tell()
            length = end_offset - offset
            # print(offset, end_offset, length, len(line))
            if not line:  # EOF
                break

            try:
                data = json.loads(line)
            except:
                print(f"Error loading line: {line[:100]}")
                continue
            if extra_fields_handler:
                should_drop, extra_fields = extra_fields_handler(line_no, offset, line, data)
            else:
                extra_fields = None
                should_drop = False
            if should_drop:
                num_drop += 1
            else:
                index = {
                    'file_path': input_jsonl_file,
                    'line_no': line_no,
                    'offset': offset,
                    'length': length,
                    'extra': extra_fields
                }
                fp_out_index.write(json.dumps(index) + '\n')
                num_lines += 1
            line_no += 1
            pbar.update(length / 1024**2)
            pbar.set_postfix(line_no=line_no, num_drop=num_drop, refresh=False)
            offset = end_offset
        pbar.close()

    # Write stats file next to output index file (e.g. output.index -> output.index.stats)
    stats_path = output_index_file + ".stats"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({"num_lines": num_lines}, f)


MAX_TOKENS = 8192
# MAX_TOKENS = 6000
TOO_SHORT_THRESHOLD = 0.9
TEXT_PADDING_TOKENS = 10
MAX_TEXT_TOKENS_PER_SECOND = 19
AUDIO_PADDING_TOKENS = 20
AUDIO_PADDING_TOKENS_SEGMENT = 1
AUDIO_PADDING_TOKENS_SLICE = 0
AUDIO_PADDING_TOKENS_PER_MODAL = 1
AUDIO_PADDING_TOKENS_PER_SPLIT = 0
AUDIO_TOKEN_RATE_CONTINUOUS = 12.5

# Lidong
# AUDIO_TOKEN_RATE_DISCRETE = 25
# DUAL_CHANNEL = False

# Xiaoxue
AUDIO_TOKEN_RATE_DISCRETE = 12.5
DUAL_CHANNEL = True
DUAL_CHANNEL_V2 = False

DUAL_CHANNEL_V2_P1_ALL_TD = False
# DUAL_CHANNEL_V2_P1_ALL_TD = True

DUAL_CHANNEL_V2_P2_4TYPE = False
# DUAL_CHANNEL_V2_P2_4TYPE = True

# DUAL_CHANNEL_V3_S2T = False
DUAL_CHANNEL_V3_S2T = True

# AUDIO_MERGE_SLICES = False
AUDIO_MERGE_SLICES = True

# AUDIO_MERGE_SLICES_MAX_DURATION_SEC = [5.0, 50.0]
AUDIO_MERGE_SLICES_MAX_DURATION_SEC = [5.0, 300.0] # for caption data
AUDIO_MERGE_SLICES_MAX_DURATION_SEC_MAX = max(AUDIO_MERGE_SLICES_MAX_DURATION_SEC) if isinstance(AUDIO_MERGE_SLICES_MAX_DURATION_SEC, list) else AUDIO_MERGE_SLICES_MAX_DURATION_SEC
DUAL_CHANNEL_AUDIO_DELAY = 6

# AUDIO_MAX_SLICE_DURATION_SEC = 5.0
AUDIO_MAX_SLICE_DURATION_SEC = 50.0

print(f"AUDIO_TOKEN_RATE_DISCRETE: {AUDIO_TOKEN_RATE_DISCRETE}, DUAL_CHANNEL: {DUAL_CHANNEL}")

if DUAL_CHANNEL_V3_S2T:
    EOD = "<|endoftext|>"
else:
    EOD = "<|im_end|>"

TEMPLATE_MESSAGE = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": ""
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]
}
AUDIO_MAX_AVG_WER = 2.0

def pack_text_jsonl(input_jsonl_file, output_jsonl_file, output_index_file, pack_mode: str = "concat", data_preprocessor: callable = None, pack_items_handler: callable = None, extra_fields_handler: callable = None):
    # if success file exists, skip
    if os.path.exists(output_jsonl_file + '.successQ35'):
        print(f"Success file exists: {output_jsonl_file + '.successQ35'}, skipping...")
        return

    from multiprocessing import Pool
    print(f"Loading and tokenizing text data from {input_jsonl_file}...")
    if pack_mode == "concat":
        split_mode = None
    elif pack_mode == "binpacking":
        split_mode = "split"
    else:
        raise ValueError(f"Invalid pack mode: {pack_mode}")
    num_dropped = 0
    with open(input_jsonl_file, 'r', encoding='utf-8') as fp_in, Pool(processes=4) as pool:
        items = []  # each is (item, num_tokens)
        for item in tqdm.tqdm(pool.imap_unordered(partial(data_preprocessor, split_mode=split_mode), fp_in, chunksize=100)):
            # print(item)
            # break
            data, num_tokens = item
            if data is not None and num_tokens is not None:
                items.append(item)
            else:
                num_dropped += 1
        print(f"Loaded {len(items)} items, dropped {num_dropped} items")

    if pack_mode == "concat":
        text_tokenizer = get_text_tokenizer()
        eod_token = text_tokenizer.encode(EOD, add_special_tokens=False)[0]
        def new_item(token_buffer):
            split_item = deepcopy(TEMPLATE_MESSAGE)
            decoded_text = text_tokenizer.decode(token_buffer)
            split_item["messages"][2]["content"] = decoded_text
            return split_item
        # concat then split
        token_buffer = []
        bins = []
        item_idx = 0
        item_token_offset = 0
        print(f"Concating text data into bins...")
        pbar = tqdm.tqdm(total=len(items), desc="Concating text data into bins")
        while True:
            if item_idx >= len(items):
                break
            item, num_tokens = items[item_idx]
            if item is None or num_tokens is None:
                item_idx += 1
                continue
            buffer_empty_space = MAX_TOKENS - len(token_buffer) - TEXT_PADDING_TOKENS
            current_item_remaining_tokens = num_tokens - item_token_offset
            if buffer_empty_space <= 0:
                split_item = new_item(token_buffer)
                bins.append([(split_item, len(token_buffer))])
                token_buffer = []
                continue
            if current_item_remaining_tokens <= 0:
                token_buffer.append(eod_token)
                item_idx += 1
                item_token_offset = 0
                pbar.update(1)
                continue
            num_tokens_to_add = min(buffer_empty_space, current_item_remaining_tokens)
            token_buffer.extend(item["tokens"][item_token_offset:item_token_offset + num_tokens_to_add])
            item_token_offset += num_tokens_to_add
        if len(token_buffer) > 0:
            split_item = new_item(token_buffer)
            bins.append([(split_item, len(token_buffer))])
        print(f"Concated into {len(bins)} bins")
        pbar.close()
    elif pack_mode == "binpacking":
        split_items = []
        for item, num_tokens in items:
            if isinstance(item, list):
                split_items.extend(item)
            else:
                split_items.append((item, num_tokens))
        print(f"Split items: {len(split_items)}")

        print(f"Packing text data into bins...")
        import binpacking
        bins = binpacking.to_constant_volume(split_items, MAX_TOKENS - TEXT_PADDING_TOKENS, weight_pos=1)
        # for bin_idx, bin in enumerate(bins):
        #     bin_num_tokens = sum(num_tokens if num_tokens < MAX_TOKENS else MAX_TOKENS for item, num_tokens in bin)
            # for item, num_tokens in bin:
            #     print(item, num_tokens)
            # print(f"bin {bin_idx} num_tokens: {bin_num_tokens}")
    print(f"Packed into {len(bins)} bins")
    
    print(f"Writing packed data to {output_jsonl_file}...")
    with open(output_jsonl_file, 'wb') as fp_out_jsonl, \
         open(output_index_file, 'w', encoding='utf-8') as fp_out_index:

        line_no = 0
        num_drop = 0
        offset = 0
        for bin_idx, bin in enumerate(tqdm.tqdm(bins, desc="Writing packed data")):
            offset = fp_out_jsonl.tell()

            assert pack_items_handler is not None
            packed_data = pack_items_handler(bin)

            if extra_fields_handler:
                should_drop, extra_fields = extra_fields_handler(line_no, offset, None, packed_data)
            else:
                extra_fields = None
                should_drop = False

            if should_drop:
                num_drop += 1
            else:
                data_encoded = (json.dumps(packed_data) + '\n').encode('utf-8')
                length = len(data_encoded)
                fp_out_jsonl.write(data_encoded)

                index = {
                    'file_path': output_jsonl_file,
                    'line_no': line_no,
                    'offset': offset,
                    'length': length,
                    'extra': extra_fields
                }
                fp_out_index.write(json.dumps(index) + '\n')
            line_no += 1
    print(f"Written {line_no} lines to {output_jsonl_file}")
    # write a .success file
    with open(output_jsonl_file + '.successQ35', 'w', encoding='utf-8') as fp_out_success:
        fp_out_success.write(f"Successfully packed {line_no - num_drop} lines to {output_jsonl_file}, dropped {num_drop} lines.")


def pack_audio_jsonl(input_index_files, output_index_file, pack_mode: str = "binpacking", data_preprocessor: callable = None, pack_items_handler: callable = None, extra_fields_handler: callable = None):
    from multiprocessing import Pool
    print(f"Loading audio data from index file {input_index_files}...")
    if pack_mode == "concat":
        raise ValueError("concat mode is not supported for audio data")
    elif pack_mode == "binpacking":
        split_mode = "split"
    else:
        raise ValueError(f"Invalid pack mode: {pack_mode}")
    audios = defaultdict(list)
    num_segments = 0
    file_path_exists = {}
    for input_index_file in input_index_files:
        with open(input_index_file, 'r', encoding='utf-8', errors='ignore') as fp_in:
            # collect segments of same audio
            for line in tqdm.tqdm(fp_in, desc="Loading audio data"):
                try:
                    data = json.loads(line)
                except:
                    print(f"Error loading line: {line[:100]}")
                    continue
                try:
                    # check file path exists
                    file_path = data["file_path"]
                    if file_path not in file_path_exists:
                        exists = os.path.exists(file_path)
                        file_path_exists[file_path] = exists
                    if not file_path_exists[file_path]:
                        print(f"Warning: file path {file_path} does not exist")
                        continue
                    
                    extra_fields = data["extra"]
                    raw_audio_path = extra_fields["segment_info"]["raw_audio_path"]
                    start_time = extra_fields["segment_info"]["start_time"]
                except Exception as e:
                    print(f"Error parsing extra fields: {extra_fields}")
                    continue
                audios[raw_audio_path].append(data)
                num_segments += 1
    print(f"Loaded {num_segments} segments from {len(audios)} audios")

    if pack_mode == "binpacking":
        per_mode_split_items = defaultdict(list)
        for raw_audio_path, audio_segments in audios.items():
            # print(key)
            # for segment in value:
            #     print(segment)
            # run preprocessor
            # print(len(audio_segments))
            item, num_tokens = data_preprocessor(None, data=audio_segments, split_mode=split_mode)
            if isinstance(item, list):
                for item, num_tokens in item:
                    pt_mode = item["pt_mode"]
                    per_mode_split_items[pt_mode].append((item, num_tokens))
            else:
                pt_mode = item["pt_mode"]
                per_mode_split_items[pt_mode].append((item, num_tokens))
        bins = defaultdict(list)
        for pt_mode in sorted(per_mode_split_items.keys()):
            split_items = per_mode_split_items[pt_mode]
            print(f"Split items for pt_mode {pt_mode}: {len(split_items)}")
            print(f"Packing per mode audio data into bins...")
            import binpacking
            per_mode_bins = binpacking.to_constant_volume(split_items, MAX_TOKENS - AUDIO_PADDING_TOKENS, weight_pos=1)
            # for bin_idx, bin in enumerate(bins):
            #     bin_num_tokens = sum(num_tokens if num_tokens < MAX_TOKENS else MAX_TOKENS for item, num_tokens in bin)
            #     print(f" ====================== bin_idx num_tokens: {bin_num_tokens}, {bin_idx}")
            #     for item, num_tokens in bin:
            #         print(f"   ==================== sub_audio num_tokens: {num_tokens}, {item}")
            bins[pt_mode].extend(per_mode_bins)
            print(f"Packed {len(per_mode_bins)} bins for pt_mode {pt_mode}")
    print(f"Packed into {len(bins)} bins")
    
    print(f"Writing packed data to {output_index_file}...")
    with open(output_index_file, 'w', encoding='utf-8') as fp_out_index:

        line_no = 0
        num_drop = 0
        offset = 0
        for pt_mode in sorted(per_mode_split_items.keys()):
            per_mode_bins = bins[pt_mode]
            for bin_idx, bin in enumerate(tqdm.tqdm(per_mode_bins, desc=f"Writing packed data for pt_mode {pt_mode}")):

                assert pack_items_handler is not None
                index = pack_items_handler(bin, pt_mode=pt_mode)
                # print(' ====================== index: ', index)

                if extra_fields_handler:
                    should_drop, extra_fields = extra_fields_handler(line_no, offset, None, index)
                else:
                    extra_fields = None
                    should_drop = False

                # should_drop = False
                if should_drop:
                    num_drop += 1
                else:
                    fp_out_index.write(json.dumps(index, ensure_ascii=False) + '\n')
                line_no += 1
    print(f"Written {line_no - num_drop} lines to {output_index_file}, dropped {num_drop} lines.")


# copied. to avoid huge FunASR import overhead
def MinMo_S2S_Interleave_Pretrain_index_ds_load_line(line: str,
                min_slice_duration_sec,
                min_slice_duration_drop_audio,
                text_data_random_drop_ratio):
    if isinstance(line, str):
        try:
            data_dict_ = json.loads(line.strip())
        except:
            return None
    else:
        data_dict_ = line
    # transform Matterhorn format to chatml format
    if 'raw_audio_path' in data_dict_ and 'messages' not in data_dict_:
        try:
            data_dict = {}
            segments_chatml = []
            messages_chatml = []
            for segment in data_dict_["segments"]:
                if len(segment['slices']) == 0:
                    return None
                segment_chatml = {
                    # 'lid': segment['postprocess']["lid"],
                    # 'text': segment['postprocess']["text"],
                    'start_time': float(segment['start_time']),
                    'end_time': float(segment['end_time']),
                    'segment_id': len(segments_chatml),
                    'wav_path': segment['slices'][0]['messages'][1]['content'].replace('<|startofspeech|>!', '').replace('<|endofspeech|>', '')
                }
                for slice in segment['slices']:
                    messages = slice['messages']
                    start_time = messages[1]['start_time'] + float(segment['start_time'])
                    end_time = messages[1]['end_time'] + float(segment['start_time'])
                    content = messages[2]['content']
                    tokens = messages[2]['slice_tokens']
                    segment_id = len(segments_chatml)

                    role = 'user' if len(messages_chatml) % 2 == 0 else 'assistant'
                    messages_chatml.append({
                        'role': role,
                        'content': content,
                        'start_time': start_time,
                        'end_time': end_time,
                        'segment_id': segment_id,
                        'tokens': tokens
                    })
                segments_chatml.append(segment_chatml)
            data_dict = {
                'messages': messages_chatml,
                'source': data_dict_['raw_audio_path'],
                'segments': segments_chatml,
            }
        except Exception as e:
            # logging.error(f"Error loading jsonl file: {e}; data_dict_: {line}")
            return None
    else:
        data_dict = data_dict_

    data = data_dict["messages"]
    wav_path = data_dict.get("source", None)
    segments = data_dict.get("segments", [])
    segments_dict = {}
    for segment in segments:
        segments_dict[segment["segment_id"]] = segment

    system, user, assistant = [], [], []
    referenced_segment_ids = set()
    should_drop = False
    is_audio = False
    source_len = 0
    for i, item in enumerate(data):
        role = item["role"]
        content = item["content"]
        if role != "system":
            start_time = item.get("start_time", None)
            end_time = item.get("end_time", None)
            segment_id = item.get("segment_id", None)
            if segment_id is not None:
                referenced_segment_ids.add(segment_id)
            # check for too short slice
            if start_time is not None and end_time is not None:
                is_audio = True

                segment = segments_dict[segment_id]
                segment_start_time = segment["start_time"]
                segment_end_time = segment["end_time"]
                if segment_start_time is None or segment_end_time is None:
                    continue
                if start_time - segment_start_time < -0.0001 or end_time - segment_end_time > 0.0001:
                    print(f"! slice out of segment, slice: {item}, segment: {segment}")
                    should_drop = True

                duration = end_time - start_time
                if duration < min_slice_duration_sec:
                    print(f"! too short slice, slice duration: {duration} < {min_slice_duration_sec}, item: {item}")
                    if min_slice_duration_drop_audio:
                        should_drop = True
                        break
                    else:
                        continue

        if role == "system":
            system.append({"content": content, "start_time": None, "end_time": None})
        elif role == "user":
            user.append({"content": content, "start_time": start_time, "end_time": end_time, "segment_id": segment_id, "tokens": item['tokens'] if 'tokens' in item else None})
            if is_audio:
                source_len += int((end_time - start_time) * 25)
            else:
                source_len += int(len(content) / 2)
        elif role == "assistant":
            if "prev_content" in item:
                prev_content = item["prev_content"]
                assistant.append([content, {"prev_content": prev_content}])
                raise ValueError("prev_content is not supported")
            else:
                assistant.append({"content": content, "start_time": start_time, "end_time": end_time, "segment_id": segment_id, "tokens": item['tokens'] if 'tokens' in item else None})
            # assume assistant is always text
            source_len += int(len(content) / 2)

    if is_audio:
        # num_audio += 1
        # if should_drop_audio:
        # 	# logging.info(f"!! drop audio, should_drop_audio: {should_drop_audio}")
        # 	num_audio_drop += 1
            
        # else:
        # 	num_audio_keep += 1
        pass
    else:
        # num_text += 1
        if text_data_random_drop_ratio and random.random() < text_data_random_drop_ratio:
            # num_text_drop += 1
            should_drop = True
        else:
            # num_text_keep += 1
            should_drop = False

    # user， assistant并没有区分含义，
    assert len(user) - len(assistant) <= 1, f"user turn does not match assistant turn"
    if len(user) < len(assistant):
        user.append(None)
    elif len(user) > len(assistant):
        assistant.append(None)

    if len(system) == 0:
        system = [{"content": "You are a helpful assistant.", "start_time": None, "end_time": None}]
    system = system * len(user)

    referenced_segments = []
    for segment_id in sorted(referenced_segment_ids):
        referenced_segments.append(segments_dict[segment_id])

    contents_i = {
        "system": system,
        "user": user,
        "assistant": assistant,
        "source_wav": wav_path,
        "segments": referenced_segments,
        "segments_dict": segments_dict,
        "source_len": source_len,
    }
    if "key" in data_dict:
        contents_i["key"] = data_dict["key"]
    return is_audio, should_drop, contents_i


def MinMo_S2S_Interleave_Pretrain_handler(line_no: int, offset: int, line: str, data: dict):
    min_slice_duration_sec = 0.1
    min_slice_duration_drop_audio = True
    text_data_random_drop_ratio = 0.0
    line_result = MinMo_S2S_Interleave_Pretrain_index_ds_load_line(data, min_slice_duration_sec, min_slice_duration_drop_audio, text_data_random_drop_ratio)
    if line_result is None:
        return True, None
    is_audio, should_drop, contents_i = line_result
    source_len = contents_i["source_len"]
    extra_fields = {
        "source_len": source_len,
    }
    return should_drop, extra_fields


text_tokenizer = None
def get_text_tokenizer():
    global text_tokenizer
    if text_tokenizer is None:
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained("/path/to/Qwen3.5-397B-A17B")
    return text_tokenizer


def pack_text_preprocessor(line: str, split_mode: str=None) -> tuple:
    text_tokenizer = get_text_tokenizer()

    data = json.loads(line)
    try:
        text = data["messages"][2]["content"]
    except:
        print(f"Error loading line: {line[:1000]}")
        return None, None
    tokens = text_tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(tokens)

    num_tokens_each_split = MAX_TOKENS - TEXT_PADDING_TOKENS
    if split_mode == "truncate":
        if num_tokens > num_tokens_each_split:
            truncated_tokens = tokens[:num_tokens_each_split]
            text = text_tokenizer.decode(truncated_tokens)
            data["messages"][2]["content"] = text
            num_tokens = num_tokens_each_split
            data["tokens"] = truncated_tokens
            data["num_tokens"] = num_tokens
    elif split_mode == "split":
        if num_tokens > num_tokens_each_split:
            splits = []
            for i in range(0, num_tokens, num_tokens_each_split):
                split_tokens = tokens[i:i+num_tokens_each_split]
                text = text_tokenizer.decode(split_tokens)
                split_data = deepcopy(TEMPLATE_MESSAGE)
                split_data["messages"][2]["content"] = text
                split_data["tokens"] = split_tokens
                split_data["num_tokens"] = len(split_tokens)
                splits.append((split_data, len(split_tokens)))
            data = splits
    elif split_mode is None:
        data["tokens"] = tokens
        data["num_tokens"] = num_tokens
    else:
        raise ValueError(f"Invalid split mode: {split_mode}")
    return data, num_tokens


# mapping from pt_mode to (weight, slice_mode_list)
# slice_mode_list:
#   'c': continuous audio, num_tokens = ceil(speech_duration * 25)
#   'd': discrete audio, num_tokens = num_speech_tokens
#   't': text, num_tokens = num_text_tokens
#   ['c', 't']: continuous audio and text
if DUAL_CHANNEL_V2:
    if DUAL_CHANNEL_V2_P1_ALL_TD:
        AUDIO_PT_MODES = {
            # "unimodal-text": (7, ['t']), # unimodal-text
            "unimodal-audio": (1, [['td']]), # unimodal-audio
            # "mapping-audio-to-text": (1, [['c', 'td']]), # mapping-audio-to-text
            # "mapping-text-to-audio": (1, [['t', 'td']]), # mapping-text-to-audio 
            # # "interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
            # # "interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
            # "interleave-audio-to-semantic-and-text": (4, [['c'], ['td']]), # interleave-audio-to-semantic-and-text
        }
    elif DUAL_CHANNEL_V2_P2_4TYPE:
        AUDIO_PT_MODES = {
            # "unimodal-text": (7, ['t']), # unimodal-text
            "unimodal-audio": (1, [[{'t': 0.2, 'td': 0.8}]]), # unimodal-audio
            "mapping-audio-to-text": (1, [['c', {'t': 0.2, 'td': 0.8}]], {"system_tokens": 20}), # mapping-audio-to-text
            "mapping-text-to-audio": (1, [['t', 'td']], {"system_tokens": 20}), # mapping-text-to-audio 
            # "interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
            # "interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
            "interleave-audio-to-semantic-and-text": (4, [[{'c': 1/3, 't': 1/3, 'td': 1/3}]], {"max_segment_gap_to_allow_end_in_c": 5.0}), # interleave-audio-to-semantic-and-text
        }
    else:
        AUDIO_PT_MODES = {
            # "unimodal-text": (7, ['t']), # unimodal-text
            "unimodal-audio": (1, [['td']]), # unimodal-audio
            "mapping-audio-to-text": (1, [['c', 'td']]), # mapping-audio-to-text
            "mapping-text-to-audio": (1, [['t', 'td']]), # mapping-text-to-audio 
            # "interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
            # "interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
            "interleave-audio-to-semantic-and-text": (4, [['c'], ['td']]), # interleave-audio-to-semantic-and-text
        }
elif DUAL_CHANNEL_V3_S2T:
    AUDIO_PT_MODES = {
        # "unimodal-text": (7, ['t']), # unimodal-text
        # "unimodal-audio": (1, [[{'t': 1.0}]]), # unimodal-audio
        "mapping-audio-to-text": (1, [['c', {'t': 1.0}]], {"system_tokens": 20, "allowed_data_types": set(["asr", "asr,interleave"])}), # mapping-audio-to-text
        "mapping-audio-to-caption": (1, [['c', {'t': 1.0}]], {"system_tokens": 20, "allowed_data_types": set(["caption"])}), # mapping-audio-to-text
        "interleave-audio-to-text": (1, [['c'], ['t']],
            {
                "allowed_data_types": set(["interleave", "asr,interleave"]),
                # "audio_padding_tokens_slice": {
                #     "c": 2,
                #     "t": 1,
                # }
            }
        ), # interleave-audio-to-semantic-and-text
    }
    AUDIO_PADDING_TOKENS_SEGMENT = 0
    AUDIO_PADDING_TOKENS_SLICE = 0
    AUDIO_PADDING_TOKENS_PER_MODAL = {"c": 2, "t": 1}
    AUDIO_PADDING_TOKENS_PER_SPLIT = 1
else:
    if DUAL_CHANNEL:
        INTERLEAVE_AUDIO_TO_SEMANTIC_AND_TEXT_MODE = [['c'], ['d']]
    else:
        INTERLEAVE_AUDIO_TO_SEMANTIC_AND_TEXT_MODE = [['c'], ['d'], ['c'], ['t']]

    AUDIO_PT_MODES = {
        # "unimodal-text": (7, ['t']), # unimodal-text
        "unimodal-audio": (1, [['d']]), # unimodal-audio
        "mapping-audio-to-text": (1, [['c', 't']]), # mapping-audio-to-text
        "mapping-text-to-audio": (1, [['t', 'd']]), # mapping-text-to-audio 
        "interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
        "interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
        "interleave-audio-to-semantic-and-text": (2, INTERLEAVE_AUDIO_TO_SEMANTIC_AND_TEXT_MODE), # interleave-audio-to-semantic-and-text
    }

def _pt_mode_allows_data_type(pt_mode: str, data_type: str | None) -> bool:
    """Check if pt_mode allows the given data_type. None (default) is allowed by all pt_modes."""
    if data_type is None:
        return True
    entry = AUDIO_PT_MODES[pt_mode]
    if len(entry) <= 2:
        return True
    allowed = entry[2].get("allowed_data_types")
    if allowed is None: #or len(allowed) == 0:
        return True
    return data_type in allowed


# Build DATA_TYPE_TO_PT_MODES: for each data_type (including None = default), (modes, weights).
# Default data_type None is allowed for all pt_modes.
_all_data_types = {None}
for entry in AUDIO_PT_MODES.values():
    if len(entry) > 2:
        allowed = entry[2].get("allowed_data_types")
        if allowed is not None:
            _all_data_types.update(allowed)

DATA_TYPE_TO_PT_MODES = {}
for data_type in _all_data_types:
    allowed_modes = [m for m in AUDIO_PT_MODES if _pt_mode_allows_data_type(m, data_type)]
    if not allowed_modes:
        continue
    total_weight = sum(AUDIO_PT_MODES[m][0] for m in allowed_modes)
    weights = [AUDIO_PT_MODES[m][0] / total_weight for m in allowed_modes]
    DATA_TYPE_TO_PT_MODES[data_type] = (allowed_modes, weights)


def select_audio_pt_mode(data: list[dict]) -> str:
    first_segment_postprocess = data[0]['extra']['segment_info']['postprocess']
    data_type = first_segment_postprocess.get("data_type", None)
    modes, weights = DATA_TYPE_TO_PT_MODES.get(data_type, DATA_TYPE_TO_PT_MODES[None])
    return random.choices(modes, weights=weights, k=1)[0], data_type


def random_choice_slice_mode(slice_mode_dict: dict) -> str:
    return random.choices(list(slice_mode_dict.keys()), weights=list(slice_mode_dict.values()), k=1)[0]


def get_slice_num_tokens(pt_mode: str, slice_idx: int, slice: dict, slice_mode=None) -> int:
    if slice_mode is None:
        slice_mode_list = AUDIO_PT_MODES[pt_mode][1]
        slice_mode = slice_mode_list[slice_idx % len(slice_mode_list)]
        # print(f"slice_mode: {slice_mode}")
        if isinstance(slice_mode, dict):
            slice_mode = random_choice_slice_mode(slice_mode)
            # print(f"random_choice_slice_mode: {slice_mode_dict} -> {slice_mode}")
        elif isinstance(slice_mode, list):
            chosen_modes = []
            for one_mode in slice_mode:
                if isinstance(one_mode, dict):
                    one_chosen_mode = random_choice_slice_mode(one_mode)
                    # print(f"random_choice_slice_mode: {one_mode} -> {one_chosen_mode}")
                    chosen_modes.append(one_chosen_mode)
                else:
                    chosen_modes.append(one_mode)
            slice_mode = chosen_modes
        else:
            raise ValueError(f"Invalid slice mode: {slice_mode}")
    num_tokens = 0
    # print(f"slice_idx: {slice_idx}, slice_mode: {slice_mode}")

    # check for bad slice
    num_speech_tokens = slice.get("num_speech_tokens", 0)
    if num_speech_tokens >= 0 and num_speech_tokens < 5:
        # = 0: failed to extract
        # < 5: too short, won't happen in theory
        return None, None
    num_text_tokens = slice.get("num_text_tokens", 0)
    if num_text_tokens >= 100000:
        return None, None
    speech_duration = slice.get("speech_duration", 0)
    if speech_duration <= 0 or speech_duration > 10000:
        return None, None
    # text_tokens_per_second = num_text_tokens / speech_duration
    # if text_tokens_per_second > MAX_TEXT_TOKENS_PER_SECOND:
    #     return None, None

    if slice["num_text_tokens"] < 0:
        text_tokenizer = get_text_tokenizer()
        text_tokens = text_tokenizer.encode(slice["text"], add_special_tokens=False)
        slice["num_text_tokens"] = len(text_tokens)

    for one_mode in slice_mode:
        if one_mode == 'c':
            num_tokens += int(math.ceil(slice["speech_duration"] * AUDIO_TOKEN_RATE_CONTINUOUS))
            # print(f"one_mode {one_mode}, num_tokens {num_tokens}")
        elif one_mode == 'td' or (pt_mode == "interleave-audio-to-semantic-and-text" and one_mode in {'d', 't'} and DUAL_CHANNEL and not DUAL_CHANNEL_V2):
            if num_speech_tokens == -1:
                # need to extract online
                num_speech_tokens = int(math.ceil(slice["speech_duration"] * AUDIO_TOKEN_RATE_DISCRETE)) + DUAL_CHANNEL_AUDIO_DELAY
            # assert slice["num_text_tokens"] < num_speech_tokens, f"slice['num_text_tokens'] {slice['num_text_tokens']} >= num_speech_tokens {num_speech_tokens}, slice: {slice}"
            num_tokens += max(num_speech_tokens, slice["num_text_tokens"])
        elif one_mode == 'd':
            if num_speech_tokens == -1:
                # need to extract online
                num_speech_tokens = int(math.ceil(slice["speech_duration"] * AUDIO_TOKEN_RATE_DISCRETE))
            num_tokens += num_speech_tokens
            # print(f"one_mode {one_mode}, num_tokens {num_tokens}")
        elif one_mode == 't':
            num_tokens += slice["num_text_tokens"]
            # print(f"one_mode {one_mode}, num_tokens {num_tokens}")
        else:
            raise ValueError(f"Invalid slice mode: {one_mode}")
        num_tokens += AUDIO_PADDING_TOKENS_PER_MODAL if not isinstance(AUDIO_PADDING_TOKENS_PER_MODAL, dict) else AUDIO_PADDING_TOKENS_PER_MODAL[one_mode]
    num_tokens += AUDIO_PADDING_TOKENS_SLICE
    return num_tokens, slice_mode


def merge_slices(slices, pt_mode, current_slice_mode_idx, max_duration_sec=30, shall_not_end_in_mode_c=False):
    '''
    max_duration_sec: float for a fixed value, or tuple of float meaning min and max values for random sampling
    '''
    if not isinstance(max_duration_sec, float):
        max_duration_sec = random.uniform(max_duration_sec[0], max_duration_sec[1])
    merged_slices = []
    merged_slice_modes = []

    current_slice = None

    for slice_idx, slice in enumerate(slices):
        # print(f'slice: {slice}')
        # {
        #     "speech_duration": 1.08,
        #     "num_speech_tokens": -1,
        #     "num_text_tokens": 4,
        #     "start_time": 0.08,
        #     "end_time": 1.16,
        #     "text": "thing how to find"
        # }
        text = slice['text']
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
            slice['text'] = text
        start_time = slice['start_time'] # + float(segment['start_time'])
        end_time = slice['end_time'] # + float(segment['start_time'])

        is_first_slice = slice_idx == 0
        is_last_slice = slice_idx == len(slices) - 1

        if current_slice is not None:
            current_slice_duration = current_slice['end_time'] - current_slice['start_time']
            shall_break = shall_not_end_in_mode_c and (not is_first_slice and is_last_slice) and 'c' in slice_mode
            # if shall_not_end_in_mode_c and 'c' in slice_mode:
            #     if shall_break:
            #         print(shall_break)
            if start_time - current_slice['end_time'] < 0.1 and current_slice_duration + (end_time - start_time) < max_duration_sec \
                    and not shall_break:
                # merge with current slice
                current_slice['end_time'] = end_time
                current_slice['speech_duration'] = current_slice['end_time'] - current_slice['start_time']
                current_slice['speech_duration'] = round(current_slice['speech_duration'], 5)
                current_slice['text'] += ' ' + text # FIXME: added space for Latin. non-latin shall be fine
                current_slice['num_text_tokens'] += slice['num_text_tokens']
                # current_slice['messages'][2]['text_tokens'] += text_tokens
            else:
                # new slice
                merged_slices.append(current_slice)
                merged_slice_modes.append(slice_mode)
                current_slice = None
        if current_slice is None:
            current_slice = {
                "speech_duration": end_time - start_time,
                "num_speech_tokens": -1,
                "num_text_tokens": slice['num_text_tokens'],
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            }
            while True:
                slice_mode = get_slice_num_tokens(pt_mode, current_slice_mode_idx, current_slice)[1]
                # if shall_not_end_in_mode_c:
                #     print('ok')
                if 'c' in slice_mode and shall_not_end_in_mode_c and not is_first_slice and is_last_slice:
                    continue
                break
            current_slice_mode_idx += 1
            # current_slice['messages'][2]['text_tokens'] += text_tokens
        # print('speech_tokens: ', tokens, 'text_tokens: ', text_tokens, 'content: ', content)
    if current_slice is not None:
        merged_slices.append(current_slice)
        merged_slice_modes.append(slice_mode)
    return merged_slices, merged_slice_modes, current_slice_mode_idx


def pack_audio_preprocessor(line: str, data: list[dict]=None, split_mode: str=None) -> tuple:
    # determine mode
    # here the 'data' is a list of all segments in an audio
    pt_mode, data_type = select_audio_pt_mode(data)
    pt_mode_kwargs = AUDIO_PT_MODES[pt_mode][2] if len(AUDIO_PT_MODES[pt_mode]) > 2 else {}
    system_tokens = pt_mode_kwargs.get("system_tokens", 0)
    # print(f"Selected audio PT mode: {pt_mode}, pt_mode_kwargs: {pt_mode_kwargs}")

    for item in data:
        item["extra"]["segment_info"]["start_time"] = float(item["extra"]["segment_info"]["start_time"])
        item["extra"]["segment_info"]["end_time"] = float(item["extra"]["segment_info"]["end_time"])
    data = sorted(data, key=lambda x: x["extra"]["segment_info"]["start_time"])

    num_tokens_each_split = MAX_TOKENS - AUDIO_PADDING_TOKENS
    if split_mode == "truncate":
        raise ValueError("truncate mode is not supported for audio data")
    elif split_mode == "split":
        splits = []
        split_segments = []
        split_num_tokens = 0
        split_num_slices = 0
        audio_num_tokens = 0

        # split_num_tokens += system_tokens
        # audio_num_tokens += system_tokens
        split_num_tokens += AUDIO_PADDING_TOKENS_PER_SPLIT
        audio_num_tokens += AUDIO_PADDING_TOKENS_PER_SPLIT
        # if len(data) > 1:
        #     print(f"data: {len(data)}")
        for segment_idx, segment in enumerate(data):
            # print(f"segment_idx: {segment_idx}")
            next_segment = data[segment_idx + 1] if segment_idx < len(data) - 1 else None
            segment_end = segment["extra"]["segment_info"]["end_time"]
            if next_segment is not None:
                next_segment_start = next_segment["extra"]["segment_info"]["start_time"]
                segment_gap = next_segment_start - segment_end
            else:
                segment_gap = 1e8
            # if segment_gap < 0:
            #     print(f"WARNING: segment_gap {segment_gap} < 0, pt_mode: {pt_mode}, segment_idx: {segment_idx}")
                # continue
            extra = segment["extra"]
            extra["raw_audio_path"] = extra["segment_info"]["raw_audio_path"]
            segment_num_tokens = AUDIO_PADDING_TOKENS_SEGMENT
            truncated_slices = []
            segment_start_time = extra["segment_info"]["start_time"]
            segment_end_time = extra["segment_info"]["end_time"]
            segment_duration = segment_end_time - segment_start_time
            if segment_duration <= 0 or segment_duration > 10000:
                print(f"WARNING: segment_duration {segment_duration} is not valid, pt_mode: {pt_mode}, segment_idx: {segment_idx}")
                continue
            segment_num_text_tokens = extra["num_text_tokens"]
            segment_num_text_tokens_per_second = segment_num_text_tokens / segment_duration
            segment_text = extra["segment_info"]["postprocess"].get("text", "")
            if segment_num_text_tokens_per_second > MAX_TEXT_TOKENS_PER_SECOND and data_type != "caption":
                print(f"WARNING: segment_num_text_tokens_per_second {segment_num_text_tokens_per_second} > MAX_TEXT_TOKENS_PER_SECOND {MAX_TEXT_TOKENS_PER_SECOND}, pt_mode: {pt_mode}, segment_idx: {segment_idx}, segment_duration: {segment_duration}, text: {segment_text}")
                continue

            raw_slices = extra["slice_info"]
            if AUDIO_MERGE_SLICES:
                shall_not_end_in_mode_c = False
                if segment_gap > pt_mode_kwargs.get('max_segment_gap_to_allow_end_in_c', 1e10):
                    # print(f"WARNING: segment_gap {segment_gap} > max_segment_gap_to_allow_end_in_c {pt_mode_kwargs.get('max_segment_gap_to_allow_end_in_c', None)}, pt_mode: {pt_mode}, segment_idx: {segment_idx}")
                    shall_not_end_in_mode_c = True
                merged_slices, merged_slice_modes, merged_current_slice_mode_idx = merge_slices(
                    raw_slices, pt_mode, split_num_slices,
                    max_duration_sec=AUDIO_MERGE_SLICES_MAX_DURATION_SEC,
                    shall_not_end_in_mode_c=shall_not_end_in_mode_c,)
                    # max_repetition=pt_mode_kwargs.get('max_repetition', None))
                # if shall_not_end_in_mode_c:
                #     last_slice = merged_slices[-1]
                #     last_slice_mode = merged_slice_modes[-1]
                #     assert 'c' not in last_slice_mode, f"last_slice_mode {last_slice_mode} contains 'c', pt_mode: {pt_mode}, segment_idx: {segment_idx}"
                split_num_slices = merged_current_slice_mode_idx
                extra["merged_slices"] = merged_slices
            else:
                merged_slices = raw_slices
                merged_slice_modes = None
                merged_current_slice_mode_idx = split_num_slices

            try:
                for slice_idx, slice in enumerate(merged_slices):
                    # check slice start and end time
                    slice_rel_start_time = float(slice["start_time"])
                    slice_rel_end_time = float(slice["end_time"])
                    slice_duration = slice_rel_end_time - slice_rel_start_time
                    assert slice_duration >= 0.1 and slice_duration <= (AUDIO_MAX_SLICE_DURATION_SEC if not AUDIO_MERGE_SLICES else AUDIO_MERGE_SLICES_MAX_DURATION_SEC_MAX), f"slice_duration {slice_duration} is not valid, pt_mode: {pt_mode}, segment_idx: {segment_idx}, slice_idx: {slice_idx}, slice: {slice}"
                    assert slice_rel_start_time >= 0 and slice_rel_end_time <= segment_duration + 0.0001, f"slice_rel_start_time {slice_rel_start_time} < 0 or slice_rel_end_time {slice_rel_end_time} > segment_duration {segment_duration}, pt_mode: {pt_mode}, segment_idx: {segment_idx}, slice_idx: {slice_idx}"
                    assert slice_rel_start_time < slice_rel_end_time, f"slice_rel_start_time {slice_rel_start_time} >= slice_rel_end_time {slice_rel_end_time}, pt_mode: {pt_mode}, segment_idx: {segment_idx}, slice_idx: {slice_idx}"

                    predefined_slice_mode = None
                    if merged_slice_modes is not None:
                        predefined_slice_mode = merged_slice_modes[slice_idx]
                    slice_num_tokens, slice_mode = get_slice_num_tokens(pt_mode, split_num_slices, slice, slice_mode=predefined_slice_mode)
                    # if slice_num_tokens is None:
                    #     # bad slice, skip
                    #     continue
                    # slice.clear()
                    slice["pt_slice_mode"] = slice_mode
                    truncated_slices.append(slice)
                    if segment_num_tokens + slice_num_tokens > num_tokens_each_split:
                        print(f"WARNING: truncating segment to fit num_tokens_each_split, pt_mode: {pt_mode}, segment_idx: {segment_idx}, slice_idx: {slice_idx}")
                        break
                    if merged_slice_modes is None:
                        split_num_slices += 1
                    segment_num_tokens += slice_num_tokens + (AUDIO_PADDING_TOKENS_PER_MODAL if not isinstance(AUDIO_PADDING_TOKENS_PER_MODAL, dict) else 0)
            except Exception as e:
                print(f"ERROR: {e}, pt_mode: {pt_mode}, segment_idx: {segment_idx}")
                import traceback
                traceback.print_exc()
                continue
            extra["slice_modes"] = [s["pt_slice_mode"] for s in truncated_slices]
            del extra["slice_info"]
            postprocess = extra.get("segment_info", {}).get("postprocess", {})
            post_keys = ('speechscore', 'avg_wer')
            for key in post_keys:
                if key in postprocess:
                    value = postprocess[key]
                    if isinstance(value, list):
                        try:
                            value = np.mean(value)
                            value = round(value, 4)
                        except Exception as e:
                            value = None
                    extra[key] = value
            del extra["segment_info"]

            # filter extreme wer
            avg_wer = extra.get("avg_wer", None)
            max_avg_wer = extra.get("max_avg_wer", None)
            if max_avg_wer is None:
                max_avg_wer = AUDIO_MAX_AVG_WER
            if avg_wer is not None:
                avg_wer = float(avg_wer)
            # if pt_mode != "mapping-audio-to-caption" and (avg_wer is None or avg_wer > max_avg_wer):
            if avg_wer is not None and avg_wer > max_avg_wer:
                # print(f"WARNING: filter extreme wer, avg_wer: {avg_wer}, pt_mode: {pt_mode}, segment_idx: {segment_idx}")
                continue
            
            assert segment_num_tokens <= num_tokens_each_split, f"segment_num_tokens {segment_num_tokens} > num_tokens_each_split {num_tokens_each_split}, pt_mode: {pt_mode}"
            if segment_num_tokens + split_num_tokens > num_tokens_each_split:
                # print(f"new split, last split_num_tokens: {split_num_tokens}")
                splits.append(({
                    "segments": split_segments,
                    "num_tokens": split_num_tokens,
                    "pt_mode": pt_mode
                }, split_num_tokens))
                split_segments = []
                split_num_tokens = 0
            split_segments.append((segment, segment_num_tokens))
            split_num_tokens += segment_num_tokens
            audio_num_tokens += segment_num_tokens
        if split_segments:
            splits.append(({
                "segments": split_segments,
                "num_tokens": split_num_tokens,
                "pt_mode": pt_mode
            }, split_num_tokens))
        return splits, audio_num_tokens

        # if num_tokens > num_tokens_each_split:
        #     splits = []
        #     for i in range(0, num_tokens, num_tokens_each_split):
        #         split_tokens = tokens[i:i+num_tokens_each_split]
        #         text = text_tokenizer.decode(split_tokens)
        #         split_data = deepcopy(TEMPLATE_MESSAGE)
        #         split_data["messages"][2]["content"] = text
        #         split_data["tokens"] = split_tokens
        #         split_data["num_tokens"] = len(split_tokens)
        #         splits.append((split_data, len(split_tokens)))
        #     data = splits
    else:
        raise ValueError(f"Invalid split mode: {split_mode}")


def pack_text_handler(items: list):
    text_tokenizer = get_text_tokenizer()

    for item in items:
        data, num_tokens = item
        text = data["messages"][2]["content"]
        tokens = text_tokenizer.encode(text, add_special_tokens=False)
        # print(len(text), len(tokens), num_tokens)

    packed_text = EOD.join([item[0]["messages"][2]["content"] for item in items])
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "assistant",
                "content": packed_text
            }
        ],
        "source_len": sum([item[1] for item in items])
    }


def pack_audio_handler(items: list, pt_mode: str):
    index_items = []
    total_num_tokens = 0
    for item, num_tokens in items:
        for segment, segment_num_tokens in item["segments"]:
            # segment["extra"].update({
            #     "source_len": segment_num_tokens,
            #     "pt_mode": pt_mode
            # })
            index_items.append(segment)
        total_num_tokens += num_tokens
    index = {
        'items': index_items,
        'extra': {
            "source_len": total_num_tokens,
            "pt_mode": pt_mode
        }
    }
    return index


def pack_text_extra_fields_handler(line_no: int, offset: int, line: str, data: dict):
    return False, {
        "source_len": data["source_len"]
    }


def pack_lidong_extra_fields_handler(line_no: int, offset: int, line: str, data: dict):
    return False, {
        "speech_duration": data["end_time"] - data["start_time"],
        "num_speech_tokens": -1,
        "num_text_tokens": -1,
        "slice_info": [
            {
                "speech_duration": data["end_time"] - data["start_time"],
                "num_speech_tokens": -1,
                "num_text_tokens": -1,
                "start_time": 0.0,
                "end_time": data["end_time"] - data["start_time"],
                "text": data["postprocess"]["text"],
            },
        ],
        "segment_info": {
            "raw_audio_path": data["raw_audio_path"],
            "start_time": data["start_time"],
            "end_time": data["end_time"],
            "postprocess": {
                "text": data["postprocess"]["text"],
                "speechscore": data["postprocess"]["speechscore"],
                "cluster_ids": data["postprocess"]["cluster_ids"],
                "lid": data["postprocess"]["lid"],
                "avg_wer": data["postprocess"]["avg_wer"],
            }
        }
    }


def pack_lichun_extra_fields_handler(line_no: int, offset: int, line: str, data: dict, data_type_override: str=None, max_avg_wer: float=None):
    num_speech_tokens = -1
    num_text_tokens = -1
    text = data["postprocess"].get("text", None)
    caption = data["postprocess"].get("caption", None)
    segment_start_time = round(float(data["start_time"]), 4)
    segment_end_time = round(float(data["end_time"]), 4)
    segment_duration = round(segment_end_time - segment_start_time, 4)
    if text is None and caption is None:
        return True, None
    # if caption is not None and not isinstance(caption, str):
    #     caption = json.dumps(caption)
    if caption is not None:
        text = caption
    #     data_type = "caption"
    # else:
    #     data_type = "asr"
    # if data_type_override is not None:
    data_type = data_type_override
    postprocess = {
        "data_type": data_type,
        "text": text,
        # "caption": caption,
        "speechscore": data["postprocess"].get("speechscore", None),
        "cluster_ids": data["postprocess"].get("cluster_ids", None),
        "lid": data["postprocess"].get("lid", None),
        "avg_wer": data["postprocess"].get("avg_wer", None),
    }
    if max_avg_wer is not None:
        postprocess["max_avg_wer"] = max_avg_wer
    return False, {
        "speech_duration": segment_duration,
        "num_speech_tokens": num_speech_tokens,
        "num_text_tokens": num_text_tokens,
        "slice_info": [
            {
                "speech_duration": segment_duration,
                "num_speech_tokens": num_speech_tokens,
                "num_text_tokens": num_text_tokens,
                "start_time": 0.0,
                "end_time": segment_duration,
                "text": text,
            },
        ],
        "segment_info": {
            "raw_audio_path": data["raw_audio_path"],
            "segment_path": data.get("segment_path", None),
            "start_time": segment_start_time,
            "end_time": segment_end_time,
            "postprocess": postprocess,
        }
    }


def pack_audio_extra_fields_handler(line_no: int, offset: int, line: str, data: dict):
    # print(data['items'][0]['extra']['slice_modes'])
    if data['extra']['source_len'] < MAX_TOKENS * TOO_SHORT_THRESHOLD:
        print(f"WARNING: data['extra']['source_len'] {data['extra']['source_len']} < MAX_TOKENS * TOO_SHORT_THRESHOLD {MAX_TOKENS * TOO_SHORT_THRESHOLD}, line_no: {line_no}, offset: {offset}")
        return True, None
    seen_slice_modes = set()
    for item in data['items']:
        slice_modes = item['extra']['slice_modes']
        for slice_mode in slice_modes:
            seen_slice_modes.update(slice_mode)
    if len(seen_slice_modes) == 0 or (len(seen_slice_modes) == 1 and 'c' in seen_slice_modes):
        # drop all 'c' data since loss will be NaN
        return True, None
    return False, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=False, choices=["build_index", "pack_text_jsonl", "pack_audio_jsonl"], default="build_index")
    parser.add_argument('--pack-mode', type=str, required=False, choices=["concat", "binpacking"], default="concat")
    parser.add_argument('--input-jsonl-file', type=str, required=False)
    parser.add_argument('--input-index-files', type=str, nargs='+', required=False)
    parser.add_argument('--output-jsonl-file', type=str, required=False)
    parser.add_argument('--output-index-file', type=str, required=True)
    parser.add_argument('--data-preprocessor', type=str, required=False, default=None)
    parser.add_argument('--extra-fields-handler', type=str, required=False, default=None)
    parser.add_argument('--pack-items-handler', type=str, required=False, default=None)
    parser.add_argument('--max-tokens', type=int, required=False, default=MAX_TOKENS)
    parser.add_argument('--data-type', type=str, required=False, default=None)
    parser.add_argument('--max-avg-wer', type=float, required=False, default=None)
    args = parser.parse_args()

    MAX_TOKENS = args.max_tokens
    print(f"MAX_TOKENS: {MAX_TOKENS}")

    # seed based on hash of input jsonl and index file path
    index_files_str = ''.join(args.input_index_files) if args.input_index_files else ''
    seed = hashlib.sha256(f"{args.input_jsonl_file}{index_files_str}".encode()).hexdigest()
    # print(f"Seed: {seed}")
    random.seed(seed)

    if args.data_preprocessor == "pack_text":
        data_preprocessor = pack_text_preprocessor
    elif args.data_preprocessor == "pack_audio":
        data_preprocessor = pack_audio_preprocessor
    else:
        data_preprocessor = None

    if args.pack_items_handler == "pack_text":
        pack_items_handler = pack_text_handler
    elif args.pack_items_handler == "pack_audio":
        pack_items_handler = pack_audio_handler
    else:
        pack_items_handler = None

    if args.extra_fields_handler == "MinMo_S2S_Interleave_Pretrain":
        extra_fields_handler = MinMo_S2S_Interleave_Pretrain_handler
    elif args.extra_fields_handler == "pack_text":
        extra_fields_handler = pack_text_extra_fields_handler
    elif args.extra_fields_handler == "pack_audio":
        extra_fields_handler = pack_audio_extra_fields_handler
    elif args.extra_fields_handler == "pack_lidong":
        extra_fields_handler = pack_lidong_extra_fields_handler
    elif args.extra_fields_handler == "pack_lichun":
        extra_fields_handler = partial(pack_lichun_extra_fields_handler, data_type_override=args.data_type, max_avg_wer=args.max_avg_wer)
    else:
        extra_fields_handler = None

    if args.mode == "build_index":
        build_jsonl_index(args.input_jsonl_file, args.output_index_file, extra_fields_handler=extra_fields_handler)
    elif args.mode == "pack_text_jsonl":
        pack_text_jsonl(args.input_jsonl_file, args.output_jsonl_file, args.output_index_file, pack_mode=args.pack_mode, data_preprocessor=data_preprocessor, pack_items_handler=pack_items_handler, extra_fields_handler=extra_fields_handler)
    elif args.mode == "pack_audio_jsonl":
        pack_audio_jsonl(args.input_index_files, args.output_index_file, pack_mode=args.pack_mode, data_preprocessor=data_preprocessor, pack_items_handler=pack_items_handler, extra_fields_handler=extra_fields_handler)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")