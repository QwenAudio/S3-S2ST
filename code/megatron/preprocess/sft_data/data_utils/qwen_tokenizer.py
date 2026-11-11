#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
from pathlib import Path
from typing import List

DEFAULT_MERGE_FILE = Path(__file__).absolute().parent.parent / 'assets' / 'qwen.tiktoken'


class QWenTokenizer:
    def __init__(self, merge_file):
        import base64
        import tiktoken

        name = 'QWen'
        ENDOFTEXT = "<|endoftext|>"
        IMSTART = "<|im_start|>"
        IMEND = "<|im_end|>"
        special_tokens = (ENDOFTEXT, IMSTART, IMEND)

        PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

        def load_tiktoken_bpe(tiktoken_bpe_file: str) -> "dict[bytes, int]":
            with open(tiktoken_bpe_file, "rb") as f:
                contents = f.read()
            return {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in contents.splitlines() if line)
            }

        mergeable_ranks = load_tiktoken_bpe(merge_file)
        special_tokens = {
            token: index
            for index, token in enumerate(special_tokens, start=len(mergeable_ranks))
        }
        enc = tiktoken.Encoding(
            name,
            pat_str=PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        assert len(mergeable_ranks) + len(
            special_tokens) == enc.n_vocab, f'{len(mergeable_ranks) + len(special_tokens)} != {enc.n_vocab} in encoding'

        self.tokenizer = enc  # type: tiktoken.Encoding
        self.eod_id = self.tokenizer.eot_token

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    @property
    def eod(self):
        return self.eod_id

    @property
    def im_start(self):
        return self.tokenizer._special_tokens['<|im_start|>']

    @property
    def im_end(self):
        return self.tokenizer._special_tokens['<|im_end|>']

    def _encode(self, x: str) -> List[int]:
        x = unicodedata.normalize('NFC', x)
        ids = self.tokenizer.encode_ordinary(x)
        return ids

    def tokenize(self, text: str) -> List[int]:
        ids = []
        if text:
            lines = text.split('\n\n')
            num_lines = len(lines)
            for i, line in enumerate(lines):
                if i < num_lines - 1:
                    line += '\n\n'
                ids.extend(self._encode(line))
        return ids
