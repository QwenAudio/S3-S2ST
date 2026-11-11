import argparse
import base64
import multiprocessing
import os
import re
import struct
import time
import unicodedata
from pathlib import Path
import  gzip

import numpy as np
import rapidjson as json
import tiktoken
import zstandard as zstd
 
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"

MERGE_PATH = Path(__file__).parent / "qwen_15w.tiktoken"

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def load_tiktoken_bpe(tiktoken_bpe_file: str) -> "dict[bytes, int]":
    contents = open(tiktoken_bpe_file, "rb").read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


def make_new_encoding(
    name: str = "qwen",
    pat_str: str = PAT_STR,
    merge_path: "str|Path" = MERGE_PATH,
    special_tokens: "tuple[str]" = (ENDOFTEXT, IMSTART, IMEND),
) -> tiktoken.Encoding:
    mergeable_ranks = load_tiktoken_bpe(merge_path)
    special_tokens = {
        token: index
        for index, token in enumerate(special_tokens, start=len(mergeable_ranks))
    }
    enc = tiktoken.Encoding(
        name,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    assert len(mergeable_ranks) + len(special_tokens) == enc.n_vocab, f'{len(mergeable_ranks) + len(special_tokens)} != {enc.n_vocab} in encoding'
    # return tiktoken.encoding_for_model('text-davinci-001')
    # return tiktoken.encoding_for_model('gpt-4')
    return enc


class Encoder(object):
    def __init__(
        self,
    ):
        self.enc = make_new_encoding()

    def encode(self, json_line: str) -> "tuple[list[int], int]":
        data = json.loads(json_line)
        # org_text = data.get('text', data.get('content', data.get('txt', data.get('raw_content'))))
        if "text" in data.keys():
            text = data['text'].rstrip().lstrip('\n')
        elif "content" in data.keys():
            text = data['content'].rstrip().lstrip('\n')
        elif "txt" in data.keys():
            text = data["txt"].rstrip().lstrip('\n')
        elif "raw_content" in data.keys():
            text = data["raw_content"].rstrip().lstrip('\n')
        elif "extracted_text" in data.keys():
            text = data["extracted_text"].rstrip().lstrip('\n')
        else:
            raise ValueError
        text = unicodedata.normalize('NFC', text)
        ids = self.enc.encode_ordinary(text)
        ids.append(self.enc.eot_token)

        return ids, len(json_line)
    

    def decode(self, tokens: "list[int]") -> 'str':
        return self.enc.decode(tokens, errors='replace')

    def debug(self, tokens: "list[int]") -> 'dict':
        token_bytes = [self.enc.decode_single_token_bytes(token) for token in tokens]
        token_decoded = [byte.decode('utf8', errors='replace') for byte in token_bytes]
        decoded = self.enc.decode(tokens, errors='replace')
        return {'token_id': tokens, 'token_length':len(tokens),'token_byte': token_bytes, 'token_str': token_decoded, 'text_': decoded}


    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab


def make_builder(out_file, vocab_size=None):
    def __best_fitting_dtype(vocab_size=None):
        if vocab_size is not None and vocab_size < 65500:
            return np.uint16
        else:
            return np.int32


    return MMapIndexedDatasetBuilder(
        out_file, dtype=__best_fitting_dtype(vocab_size)
    )


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
    """
    this is the index file, the format is
    // magic
    MMIDIDX\x00\x00
    // unsigned long long 1 in little endian: file format version
    \x01\x00\x00\x00\x00\x00\x00\x00
    // unsigned char of dtype in little endian: uint16 is 8 int32 is 4, e.g., 4
    \x04
    // num of documents in unsigned long long in little endian: e.g., 3
    \x03\x00\x00\x00\x00\x00\x00\x00
    // num of document index in unsigned longlong in little endian: 4
    \x04\x00\x00\x00\x00\x00\x00\x00
    // num of tokens for each document in int32 in little endian: e.g., 10, 8, 16
    \x0a\x00\x00\x00
    \x08\x00\x00\x00
    \x10\x00\x00\x00
    // byte offsets for each document in int64 in little endian: 0, 40, 72
    \x00\x00\x00\x00\x00\x00\x00\x00
    \x28\x00\x00\x00\x00\x00\x00\x00
    \x48\x00\x00\x00\x00\x00\x00\x00
    // doc idx in int64 in little endian: 0, num of documents, e.g., 0, 1, 2, 3
    \x00\x00\x00\x00\x00\x00\x00\x00
    \x01\x00\x00\x00\x00\x00\x00\x00
    \x02\x00\x00\x00\x00\x00\x00\x00
    \x03\x00\x00\x00\x00\x00\x00\x00

    // the byte offsets is of int64 type that is 9223PB

    let x be the num of documents, y be the size of the idx fiel
    9 + 8 + 1 + 8 + 8 + 4x + 8x + 8(x+1) = y
    x = (y - 42) / 20
    """


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


def producer(io_lock, num_workers, chunk_size, inputs, queue, file_match_str):
    
    # print(inputs[0].endswith((".jsonl", ".jsonl.zstd", ".jsonl.zst")), os.path.isfile(inputs[0]))
    # print(os.path.isfile(inputs[0]))
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        # inputs = sorted(Path(inputs[0]).glob("*.jsonl*"))
        # inputs = sorted(Path(inputs[0]).glob("*.tsv"))
        inputs = sorted(Path(inputs[0]).glob("*"+file_match_str+"*"))
    else:
        inputs = [Path(path) for path in inputs if os.path.isfile(path) and path.endswith((".jsonl",".jsonl.zst",".jsonl.zstd","jsonline",".tsv"))]

    print("###"*20)
    print(inputs)
    print("all files:", len(inputs))
    print("****"*20, flush=True)
    assert all(path.is_file() for path in inputs)
    chunk = []
    for input in inputs:
        with io_lock:
            print("Opening", input)
        openfunc = (
            zstd.open if input.suffix in {".zstd", ".zst"} else open
        )
        
        with (gzip.open if input.suffix in {".gz"} else openfunc)(input, 'rb' if input.suffix in {".gz"} else 'r',encoding='utf-8') as fin:
            try:
                for linei, line in enumerate(fin):
                    chunk.append(line)
                    if len(chunk) == chunk_size:
                        queue.put(tuple(chunk))
                        chunk = []
            except:
                print("********** log ********")
                print(linei)
                print("********** log ********")
    if chunk:
        queue.put(tuple(chunk))

    for _ in range(num_workers):
        queue.put(None)
    
    with io_lock:
        print('reader exits')

def processor(io_lock, worker_id, in_queue, out_queue):
    encoder = Encoder()
    while True:
        data = in_queue.get()

        if data is None:
            break

        res = []

        for item in data:
            try:
                res.append(encoder.encode(item))
            except BaseException as e:
                # sometimes the string for bpe is too long
                # regex will complain (Panic Exception: stackoverflow)
                # PanicException can only be caught with BaseException, which is not ideal
                # but currently, this is the way
                import traceback
                with io_lock:
                    print(f'worker {worker_id} encountered error: {traceback.print_exception(type(e), e, e.__traceback__)}')

        if res:
            out_queue.put(tuple(res))

    out_queue.put(None)
    with io_lock:
        print(f'worker {worker_id} exists')


def consumer(
    io_lock,
    num_workers,
    output_prefix,
    vocab_size,
    log_interval,
    out_queue,
):
    bin_file = f"{output_prefix}_qwen_15w.bin"
    idx_file = f"{output_prefix}_qwen_15w.idx"
    builder = make_builder(bin_file, vocab_size=vocab_size)
    with io_lock:
        print(f"Vocab size: {vocab_size}")
        print(f"Output prefix: {output_prefix}")

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
        for doc, bytes_processed in data:
            total_bytes_processed += bytes_processed
            i += 1
            if doc:
                builder.add_item(doc)
                builder.end_document()

            if i % log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                with io_lock:
                    print(
                        f"Processed {i:,} documents ({i/elapsed:.2f} docs/s, {mbs:.2f} MB/s).",
                    )

    current = time.time()
    elapsed = current - proc_start
    mbs = total_bytes_processed / elapsed / 1024 / 1024
    with io_lock:
        print(
            f"End: Processed {i:,} documents ({i/elapsed:.2f} docs/s, {mbs:.2f} MB/s).",
        )
    builder.finalize(idx_file)

    with io_lock:
        print('writer exits')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("inputs", type=str, nargs="+", help="Path to input JSON")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Path to binary output file without suffix",
    )
    parser.add_argument(
        "--file-match-str", type=str, default=None, help="math file string"
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of worker processes to launch"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="size for each chunk"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=0, help="number of buffered documents"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Interval between progress updates",
    )
    
    args = parser.parse_args()
    if args.output_prefix is None:
        if len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
            # if it is a dir 
            args.output_prefix = args.inputs[0].rstrip('/')
        else:
            args.output_prefix = re.sub(r'\.jsonl(\.zstd?)?$', '', args.inputs[0])
    print(args.inputs)
    encoder = Encoder()
    io_lock = multiprocessing.Lock()
    num_chunk = args.buffer_size // args.chunk_size
    if num_chunk <= 0:
        # an arbitrary but seemingly sensible choice 
        num_chunk = args.workers * 8
    in_queue = multiprocessing.Queue(num_chunk)
    out_queue = multiprocessing.Queue(num_chunk)

    writer = multiprocessing.Process(
        target=consumer,
        args=(
            io_lock,
            args.workers,
            args.output_prefix,
            encoder.vocab_size,
            args.log_interval,
            out_queue,
        ),
    )
    writer.start()

    workers = []
    for workerid in range(args.workers):
        process = multiprocessing.Process(
            target=processor,
            args=(
                io_lock,
                workerid,
                in_queue,
                out_queue,
            ),
        )
        workers.append(process)
        process.start()

    reader = multiprocessing.Process(
        target=producer,
        args=(io_lock, args.workers, args.chunk_size, args.inputs, in_queue, args.file_match_str),
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


if __name__ == "__main__":
    main()
