import os
import json
import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import regex as re
import string
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables
import numpy as np
from funasr.utils.load_utils import load_audio_text_image_video




@tables.register("preprocessor_classes", "TextPreprocessHasRepeatedWords")
class TextPreprocessHasRepeatedWords(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs

    
    def forward(self, tokens):
        """

        """
        ngram_length = self.kwargs.get("ngram_length", 5)
        max_occurrences = self.kwargs.get("max_occurrences", 4)
        ngram_counts = {}
        for i in range(len(tokens) - ngram_length + 1):
            ngram = tuple(tokens[i:i + ngram_length])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            if ngram_counts[ngram] >= max_occurrences:
                return True

        return False


@tables.register("preprocessor_classes", "TextPreprocessHasRepeatedWordsZK")
class TextPreprocessHasRepeatedWordsZK(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        
        self.kwargs = kwargs

    def forward(self, line: str) -> bool:
        """
        Args:
            line (str): The input text to check for repeated words.
        Returns:
            bool: True if the text contains repeated words, False otherwise.
        """
        max_ngram_length = self.kwargs.get("max_ngram_length", 8)
        max_occurrences = self.kwargs.get("max_occurrences", 12)
    
        # remove all unicode punctuation
        cleaned_line = re.sub(r'\p{P}+', '', line)
    
        # english repeat check
        word_pattern = rf'(?<!\S)(?!\d+$)(\w+)(?:\s+\1){{{max_occurrences - 1},}}(?!\S)'
        if re.search(word_pattern, cleaned_line, re.IGNORECASE):
            return True
    
        # chinese repeat check
        for length in range(1, max_ngram_length):
            pattern = rf'(?<!\d)(\S{{{length}}})\1{{{max_occurrences - 1},}}(?!\d)'
            combined_pattern = rf'(?=.*\D){pattern}'
        
            if re.search(combined_pattern, cleaned_line):
                return True
    
        return False


def rep_test():
    text_preprocess = TextPreprocessHasRepeatedWords(ngram_length=5, max_occurrences=5)
    tokens = ["我", "是", "我", "是", "我", "是", "我", "是", "我", "是", "我", "是"]
    tokens = "wo wo wo wo wo wo wo wo wo wo wo wo wo wo wo"
    tokens = 'suecia no tendrá billetes el dos mil veintitrés el país nórdico trabaja en la implementación del funcionamiento de una moneda digital que ya está en fase piloto respaldado por su banco central es el segundo país en el mundo en que un banco central patrocina esta idea por detrás de las bahamas el nombre de'
    print(text_preprocess(tokens))


    
if __name__ == "__main__":
    rep_test()
