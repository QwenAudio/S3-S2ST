#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A simple wrapper to HF Qwen2 Tokenizer."""

from typing import List

import unicodedata
from transformers import AutoTokenizer


class HFQwen2Tokenizer:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.model_max_length = int(1e30)

        self._endoftext_id = self.tokenizer.added_tokens_encoder['<|endoftext|>']
        self._imstart_id = self.tokenizer.added_tokens_encoder['<|im_start|>']
        self._imend_id = self.tokenizer.added_tokens_encoder['<|im_end|>']

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def eod(self):
        return self._endoftext_id

    @property
    def im_start(self):
        return self._imstart_id

    @property
    def im_end(self):
        return self._imend_id

    def _encode(self, x: str) -> List[int]:
        x = unicodedata.normalize('NFC', x)
        ids = self.tokenizer.encode(x)
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

