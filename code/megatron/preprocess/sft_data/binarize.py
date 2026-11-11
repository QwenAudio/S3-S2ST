#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Binarize dataset."""

import argparse
import json
import multiprocessing
import os
import struct
import time
from pathlib import Path

import numpy as np
import regex as re
import zstandard as zstd

from data_utils.hf_qwen2_tokenizer import HFQwen2Tokenizer
from data_utils.qwen_tokenizer import QWenTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


PROCESSED_DATA_DIR = Path(__file__).absolute().parent.parent / 'data' / 'processed'
ASSET_DIR = Path(__file__).absolute().parent / 'assets'
PRESETS = {
    'qwen': (QWenTokenizer, ASSET_DIR / 'qwen.tiktoken'),
    'qwen_15w_legacy': (QWenTokenizer, ASSET_DIR / 'qwen_15w.bpe'),
    'qwen_15w': (HFQwen2Tokenizer, ASSET_DIR / 'hf_qwen2_tokenizer'),
}

REACT_SANITY_WHITELIST = [
    'en:sharegpt',
    'cn:sharegpt',
    'other:sharegpt',
    'en:sharegpt.single-turn',
    'other:sharegpt.single-turn',
    'en:sharegpt.openchat',
    'cn:system_coig',
    'en:system_niv2',
]

VISION_TAG = '✿VISION✿'
STR_MASK_START = '<|mask_start|>'
STR_MASK_END = '<|mask_end|>'
PAT_MASK_START = re.compile(r'<\|mask_start\|>')
PAT_MASK_END = re.compile(r'<\|mask_end\|>')


class Encoder(object):
    def __init__(
            self,
            preset,
            mask_prompt=False,
            mask_chatml_role=False,
            mask_vision_response=False,
            seq_length=0,
    ):
        tokenizer_type, *tokenizer_args = PRESETS[preset]
        self.tokenizer = tokenizer_type(*tokenizer_args)

        self.mask_prompt = mask_prompt
        self.mask_chatml_role = mask_chatml_role
        self.mask_vision_response = mask_vision_response

        self.vision_prefix_len = len(self.tokenizer._encode(VISION_TAG))
        self.seq_length = seq_length  # for sanity check, 0 means no check

        self.mask_special_token = True

    def encode(self, json_line):
        try:
            data = json.loads(json_line)
        except Exception as e:
            print(e)
            print(json_line)
            return [], [], 0
        source = data.get('source', '')
        mask_options = data.get('mask_options', {})
        num_last_messages_to_keep = mask_options.get('num_last_messages_to_keep', 9999999)

        if source == 'en:api.agent.opensource.code_feedback' and num_last_messages_to_keep != 9999999:
            assert data['messages'][0]['role'] == 'system'
            assert num_last_messages_to_keep == len(data['messages']) - 3, data
        elif source == 'en:api.agent.opensource.code_act' and num_last_messages_to_keep != 9999999:
            assert num_last_messages_to_keep < len(data['messages']) and not num_last_messages_to_keep % 2, data
        elif "other:" in source or "sharegpt" in source:
            num_last_messages_to_keep = 1

        keep_system = mask_options.get('keep_system', False)

        # `mask = 1` indicates participation in loss calculation
        # `mask = 0` indicates that the token's loss is not calculated.
        ids, masks = [], []

        # only support ChatML format
        im_start_token = self.tokenizer.im_start
        im_end_token = self.tokenizer.im_end
        nl_tokens = self.tokenizer._encode('\n')

        n_turns = len(data['messages'])
        for turn_idx, turn in enumerate(data['messages']):
            role, content = turn['role'], turn['content']
            content = content.lstrip('\n').rstrip()

            role_ids = self.tokenizer._encode(role) + nl_tokens
            content_ids, content_masks = self._parse_content(content)

            turn_ids = []
            turn_ids.append(im_start_token)
            turn_ids.extend(role_ids)
            turn_ids.extend(content_ids)
            turn_ids.append(im_end_token)
            turn_ids.extend(nl_tokens)

            mask_imend = False

            # Mask the whole turn (include special tokens).
            mask_content = (
                (turn_idx < n_turns - num_last_messages_to_keep and not (keep_system and role == "system")) or
                (self.mask_prompt and (role == "user" or role == "system"))
            )
            if mask_content and self.mask_special_token:
                turn_masks = [0] * len(turn_ids)

                ids.extend(turn_ids)
                masks.extend(turn_masks)
                continue

            # Common mask (always keep special tokens).
            turn_masks = []
            turn_masks.append(1)    # im_start
            turn_masks.extend([int(not self.mask_chatml_role)] * len(role_ids))     # role + '\n'

            # Modify content masks.
            if self.mask_vision_response and (role == "assistant" and content.startswith(VISION_TAG)):
                # [1] * self.vision_prefix_len indicates VISION_TAG's mask
                content_masks = [1] * self.vision_prefix_len + [0] * (len(content_masks) - self.vision_prefix_len)
                mask_imend = True

            # Mask the whole content (without special tokens).
            if mask_content and not self.mask_special_token:
                content_masks = [0] * len(content_masks)

            turn_masks.extend(content_masks)            # content

            if mask_imend:
                turn_masks.append(0)                        # im_end
                turn_masks.extend([0] * len(nl_tokens))     # '\n'
            else:
                turn_masks.append(1)                        # im_end
                turn_masks.extend([1] * len(nl_tokens))     # '\n'

            ids.extend(turn_ids)
            masks.extend(turn_masks)

        if ids:
            ids.append(self.tokenizer.eod)
            masks.append(1)

        if self.seq_length > 0:
            try:
                self._assert_mask_sanity(data, masks)
            except Exception as e:
                #print(e)
                return [], [], 0
        return ids, masks, len(json_line)

    def _parse_content(self, content: str):
        mask_starts = [match.start() for match in PAT_MASK_START.finditer(content)]
        mask_ends = [match.start() for match in PAT_MASK_END.finditer(content)]
        n_masks = len(mask_starts)

        if n_masks != len(mask_ends):
            raise ValueError(f'#mask_start ({n_masks}) != #mask_end ({len(mask_ends)}), content = {content}')

        for i in range(n_masks):
            start_i, end_i = mask_starts[i], mask_ends[i]
            if start_i >= end_i:
                raise ValueError(f'mask_start[{i}] ({start_i}) >= mask_end[{i}] ({end_i})')
            if i < n_masks - 1:
                start_i_next = mask_starts[i + 1]
                if end_i >= start_i_next:
                    raise ValueError(f'mask_end[{i}] ({end_i}) >= mask_start[{i + 1}] ({start_i_next})')

        content_ids = []
        content_masks = []

        first_part = content if n_masks == 0 else content[:mask_starts[0]]
        first_part_ids = self.tokenizer._encode(first_part)
        content_ids.extend(first_part_ids)
        content_masks.extend([1] * len(first_part_ids))

        for i in range(n_masks):
            start_i, end_i = mask_starts[i], mask_ends[i]
            start_i_next = len(content) if i == n_masks - 1 else mask_starts[i + 1]

            masked_part = content[start_i + len(STR_MASK_START):end_i]
            masked_part_ids = self.tokenizer._encode(masked_part)
            content_ids.extend(masked_part_ids)
            content_masks.extend([0] * len(masked_part_ids))

            no_masked_part = content[end_i + len(STR_MASK_END):start_i_next]
            no_masked_part_ids = self.tokenizer._encode(no_masked_part)
            content_ids.extend(no_masked_part_ids)
            content_masks.extend([1] * len(no_masked_part_ids))

        if n_masks > 0:
            raw_content = content.replace(STR_MASK_START, '').replace(STR_MASK_END, '')
            raw_content_ids = self.tokenizer._encode(raw_content)

            if raw_content_ids != content_ids:
                raise ValueError(f'Broken token in token-level mask, content = {content}')

        return content_ids, content_masks

    @staticmethod
    def _get_short_data_str(data):
        data_str = json.dumps(data, ensure_ascii=False)
        data_str = data_str[:256] + '...' + data_str[-96:]
        return data_str

    def _assert_length_sanity(self, data, tokens):
        if len(tokens) >= self.seq_length:
            print(f'WARNING: {len(tokens)} tokens exceeds the length limit ({self.seq_length}).' +
                  f' Data: {self._get_short_data_str(data)}')

    def _assert_mask_sanity(self, data, masks):
        truncated_masks = masks[1:self.seq_length+1]
        if not any(truncated_masks):
            raise RuntimeError(f"All-zero mask detected under {self.seq_length} truncation. "
                               f"Data ({len(masks)} tokens): {self._get_short_data_str(data)}")

    def _assert_react_sanity(self, data, tokens, strict=False):
        def _get_tok(s: str) -> str:
            return ',' + ','.join([str(i) for i in self.tokenizer._encode(s)]) + ','

        source = data.get('source', '')
        text = '\n'.join([turn['content'] for turn in data['messages']])
        tokens = ','.join([str(i) for i in tokens])
        keys_to_check = [
            '\nObservation:',
            '\nAction Input: {"search_query": "',  # for websearch
            '✿RESULT✿',
        ]
        if strict:
            keys_to_check.extend([
                '\nThought:',
                '\nAction:',
                '\nAction Input:',
                '\nFinal Answer:',
            ])
        for key_str in keys_to_check:
            # Do lstrip('\n'), because \n may be combined with its previous chars to form one single token.
            key_tok = _get_tok(key_str.lstrip('\n'))

            str_cnt = text.count(key_str)
            tok_cnt = tokens.count(key_tok)

            if ('code_interpreter' in source) and (key_str == '\nAction Input:'):
                tok_cnt += tokens.count(_get_tok('Action Input:\n```'))

            if (source == 'en:api.langchain') and (key_str == '\nAction:'):
                if 'Action:\n```\n' in text:
                    tok_cnt = tokens.count(_get_tok('Action:\n```\n'))

            if (source == 'en:api.glm_agent') and (key_str == '\nFinal Answer:'):
                tok_cnt -= tokens.count(_get_tok(" respond with 'Final Answer:"))

            if (source == 'cn:api.agent.online_log') and (key_str == '\nObservation:'):
                usr_texts = '\n'.join(msg['content'] for msg in data['messages'] if msg['role'] == 'user')
                str_cnt += usr_texts.count('\\nObservation:')  # noisy prompts from users

            if (source == 'en:api.agent.opensource.code_act') and (key_str == '\nObservation:'):
                bot_texts = '\n'.join(msg['content'] for msg in data['messages'] if msg['role'] != 'user')
                if key_str.lstrip('\n') in bot_texts:
                    print(f"WARNING: Abnormal data found. The assistant is fabricating the observation."
                          f" Data: {json.dumps(data, ensure_ascii=False)}")
                str_cnt = 0

            if (str_cnt > 0) and (str_cnt != tok_cnt):
                print(f"WARNING: Found {str_cnt} {repr(key_str)} but {tok_cnt} {repr(key_tok)} in {source}." +
                    f" Data: {json.dumps(data, ensure_ascii=False)} Tokens: {tokens}")


def make_builder(out_file, impl, vocab_size=None):
    def __best_fitting_dtype(vocab_size=None):
        if vocab_size is not None and vocab_size < 256:
            return np.uint8
        elif vocab_size is not None and vocab_size < 65500:
            return np.uint16
        else:
            return np.int32

    if impl == "mmap":
        return MMapIndexedDatasetBuilder(
            out_file, dtype=__best_fitting_dtype(vocab_size)
        )
    else:
        return IndexedDatasetBuilder(out_file)


def code(dtype):
    dtypes = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float32,
        7: np.float64,
        8: np.uint16,
    }

    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


class IndexWriter(object):
    def __init__(self, magic, path, dtype):
        self.magic = magic
        self.path = path
        self.dtype = dtype

    def __enter__(self):
        self._file = open(self.path, "wb")

        self._file.write(self.magic)
        self._file.write(struct.pack("<Q", 1))
        self._file.write(struct.pack("<B", code(self.dtype)))

        return self

    @staticmethod
    def exscan_from_cumsum_(arr):
        # given an array holding the result of an inclusive scan (cumsum),
        # convert to an exclusive scan (shift to the right)
        # [10, 30, 35, 50] --> [0, 10, 30, 35]
        if arr.size > 1:
            arr[1:] = arr[:-1]
        if arr.size > 0:
            arr[0] = 0

    @staticmethod
    def get_pointers_with_total(sizes, elemsize, dtype):
        """Return a numpy array of type np.dtype giving the byte offsets.
        Multiplies values in the sizes array by elemsize (bytes),
        and then computes an exclusive scan to get byte offsets.
        Returns the total number of bytes as second item in a tuple.
        """

        # scale values in sizes array by elemsize to get sizes in bytes
        pointers = np.array(sizes, dtype=dtype)
        pointers *= elemsize
        np.cumsum(pointers, axis=0, out=pointers)

        # get total number of bytes from all sizes (last element)
        bytes_last = pointers[-1] if len(sizes) > 0 else 0

        # convert to byte offsets
        IndexWriter.exscan_from_cumsum_(pointers)

        return pointers, bytes_last

    def _get_pointers(self, sizes, npdtype):
        """Return a numpy array of byte offsets given a list of sizes.
        Multiplies values in the sizes array by dtype size (bytes),
        and then computes an exclusive scan to get byte offsets.
        """

        # compute element sizes in bytes
        pointers, _ = self.get_pointers_with_total(
            sizes, self.dtype().itemsize, npdtype
        )
        return pointers

    def write(self, sizes, doc_idx):
        self._file.write(struct.pack("<Q", len(sizes)))
        self._file.write(struct.pack("<Q", len(doc_idx)))

        sizes32 = np.array(sizes, dtype=np.int32)
        self._file.write(sizes32.tobytes(order="C"))
        del sizes32

        pointers = self._get_pointers(sizes, np.int64)
        del sizes
        self._file.write(pointers.tobytes(order="C"))
        del pointers

        doc_idx = np.array(doc_idx, dtype=np.int64)
        self._file.write(doc_idx.tobytes(order="C"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()


class MMapIndexedDatasetBuilder(object):
    _HDR_MAGIC = b"MMIDIDX\x00\x00"

    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, ids):
        np_array = np.array(ids, dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file):
        self._data_file.close()

        with IndexWriter(self._HDR_MAGIC, index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float32: 4,
        np.float64: 8,
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, "wb")
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, ids):
        np_array = np.array(ids, dtype=self.dtype)
        bytes = self.out_file.write(np_array)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        self.sizes.extend(np_array.shape)
        self.dim_offsets.append(self.dim_offsets[-1] + np_array.ndim)

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    @staticmethod
    def write_longs(f, a):
        f.write(np.array(a, dtype=np.int64))

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, "wb")
        index.write(b"TNTIDX\x00\x00")
        index.write(struct.pack("<Q", 1))
        index.write(struct.pack("<QQ", code(self.dtype), self.element_size))
        index.write(struct.pack("<QQ", len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack("<Q", len(self.doc_idx)))
        self.write_longs(index, self.dim_offsets)
        self.write_longs(index, self.data_offsets)
        self.write_longs(index, self.sizes)
        self.write_longs(index, self.doc_idx)
        index.close()


def producer(io_lock, num_workers, chunk_size, inputs, queue):
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        inputs = sorted(Path(inputs[0]).glob('*.jsonl*'))
    else:
        inputs = [Path(path) for path in inputs]
    assert all(path.is_file() for path in inputs)
    chunk = []
    for input in inputs:
        with io_lock:
            print("Opening", input)
        openfunc = (
            zstd.open if input.suffix == ".zstd" or input.suffix == ".zst" else open
        )
        with openfunc(input, "r", encoding="utf8") as fin:
            for line in fin:
                chunk.append(line)
                if len(chunk) == chunk_size:
                    queue.put(tuple(chunk))
                    chunk = []
    if chunk:
        queue.put(tuple(chunk))

    for _ in range(num_workers):
        queue.put(None)


def processor(
        preset,
        mask_prompt,
        mask_chatml_role,
        mask_vision_response,
        seq_length,
        in_queue,
        out_queue):
    encoder = Encoder(
        preset,
        mask_prompt=mask_prompt,
        mask_chatml_role=mask_chatml_role,
        mask_vision_response=mask_vision_response,
        seq_length=seq_length,
    )
    while True:
        data = in_queue.get()

        if data is None:
            break

        out_queue.put(tuple(encoder.encode(item) for item in data))

    out_queue.put(None)


def consumer(
        io_lock,
        num_workers,
        output_prefix,
        dataset_impl,
        preset,
        mask_prompt,
        mask_chatml_role,
        mask_vision_response,
        vocab_size,
        log_interval,
        out_queue,
):
    bin_file = f"{output_prefix}_text_document_{preset}.bin"
    idx_file = f"{output_prefix}_text_document_{preset}.idx"
    mask_bin_file = f"{output_prefix}_mask_{preset}.bin"
    mask_idx_file = f"{output_prefix}_mask_{preset}.idx"
    builder = make_builder(bin_file, impl=dataset_impl, vocab_size=vocab_size)
    mask_builder = make_builder(mask_bin_file, impl=dataset_impl, vocab_size=2)
    with io_lock:
        print(f"Vocab size: {vocab_size}")
        print(f"Output prefix: {output_prefix}")
        print(f"Mask prompt: {mask_prompt}")
        print(f"Mask chatml role: {mask_chatml_role}")
        print(f"Mask vision response: {mask_vision_response}")
    proc_start = time.time()

    total_bytes_processed = 0
    i = 0
    worker_count = 0
    while True:
        data = out_queue.get()
        if data is None:
            worker_count += 1
            if worker_count == num_workers:
                break
            else:
                continue
        for doc, mask, bytes_processed in data:
            total_bytes_processed += bytes_processed
            i += 1
            if doc:
                builder.add_item(doc)
                builder.end_document()

                mask_builder.add_item(mask)
                mask_builder.end_document()

            if i % log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                with io_lock:
                    print(
                        f"Processed {i:,} documents ({i / elapsed:.2f} docs/s, {mbs:.2f} MB/s).",
                    )

    current = time.time()
    elapsed = current - proc_start
    mbs = total_bytes_processed / elapsed / 1024 / 1024
    with io_lock:
        print(
            f"End: Processed {i:,} documents ({i / elapsed:.2f} docs/s, {mbs:.2f} MB/s).",
        )
    builder.finalize(idx_file)
    mask_builder.finalize(mask_idx_file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('version_tag', type=str, help='Input dataset version tag')
    parser.add_argument('--root-dir', type=Path,
                        default=PROCESSED_DATA_DIR, help='Input dataset root path, default to %(default)r')
    parser.add_argument(
        "--output-prefix",
        type=str,
        default='v7',
        help="Path to binary output file without suffix",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default="qwen_15w",
    )
    parser.add_argument(
        "--dataset-impl", type=str, default="mmap", choices=["lazy", "cached", "mmap"]
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of worker processes to launch"
    )
    parser.add_argument("--chunk-size", type=int, default=512, help='size for each chunk')
    parser.add_argument("--buffer-size", type=int, default=32768, help='number of buffered documents')
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval between progress updates",
    )
    parser.add_argument("--mask-prompt", default=False, action="store_true", help="mask system and all prompts")
    parser.add_argument("--mask-chatml-role", default=False, action="store_true", help="mask role and \\n")
    parser.add_argument(
        "--mask-vision-response",
        default=False,
        action="store_true",
        help=f"mask response starting with `{VISION_TAG}`",
    )
    parser.add_argument("--seq-len", type=int, default=8192, help='')
    args = parser.parse_args()

    encoder = Encoder(
        args.preset,
        mask_prompt=args.mask_prompt,
        mask_chatml_role=args.mask_chatml_role,
        mask_vision_response=args.mask_vision_response,
        seq_length=args.seq_len,
    )
    io_lock = multiprocessing.Lock()
    in_queue = multiprocessing.Queue(args.buffer_size)
    out_queue = multiprocessing.Queue(args.buffer_size)

    output_prefix = args.output_prefix
    output_prefix = str(args.root_dir / args.version_tag / output_prefix)
    writer = multiprocessing.Process(
        target=consumer,
        args=(
            io_lock,
            args.workers,
            output_prefix,
            args.dataset_impl,
            args.preset,
            args.mask_prompt,
            args.mask_chatml_role,
            args.mask_vision_response,
            encoder.tokenizer.vocab_size,
            args.log_interval,
            out_queue,
        ),
    )
    writer.start()

    workers = []
    for _ in range(args.workers):
        process = multiprocessing.Process(
            target=processor, args=(
                args.preset,
                args.mask_prompt,
                args.mask_chatml_role,
                args.mask_vision_response,
                args.seq_len,
                in_queue,
                out_queue
            )
        )
        workers.append(process)
        process.start()

    inputs = [str(args.root_dir / args.version_tag / f'{args.version_tag}.jsonl')]
    reader = multiprocessing.Process(
        target=producer, args=(io_lock, args.workers, args.chunk_size, inputs, in_queue)
    )
    reader.start()

    try:
        reader.join()
        for p in workers:
            p.join()
        writer.join()
    except KeyboardInterrupt:
        reader.terminate()
        for p in workers:
            p.terminate()
        writer.terminate()


if __name__ == '__main__':
    main()
