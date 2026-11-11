#!/usr/bin/env python3

import os
os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

FUNASR_PATH = os.environ.get('FUNASR_PATH', '/path/to/FunASR.tts')
print(f"FUNASR_PATH: {FUNASR_PATH}")
import sys
sys.path.append(FUNASR_PATH)
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from funasr.tokenizer.cosyvoice_tokenizer import CosyVoiceTokenizer


pt_mode_tokens = [
    "<|pt_mode_unimodal-text|>", # unimodal-text
    "<|pt_mode_unimodal-audio|>", # unimodal-audio
    "<|pt_mode_mapping-audio-to-text|>", # mapping-audio-to-text
    "<|pt_mode_mapping-text-to-audio|>", # mapping-text-to-audio 
    "<|pt_mode_interleave-audio-to-semantic|>", # interleave-audio-to-semantic 
    "<|pt_mode_interleave-audio-to-text|>", # interleave-audio-to-text
    "<|pt_mode_interleave-audio-to-semantic-and-text|>", # interleave-audio-to-semantic-and-text
]
for ii in range(20):
    pt_mode_tokens.append(f"<|pt_mode_extra_{ii}|>")


def extend_vocab_to_cosyvoice(base_model_path,
                              output_model_path,
                              tokenizer_class=CosyVoiceTokenizer,
                              codec_unit=6761,
                              additional_special_tokens=[
                                "<|audio_blank|>",
                                "<|audio_end|>",
                                "<|audio_start|>",
                                "<|audio_token|>",
                                "<|box_end|>",
                                "<|box_start|>",
                                "<|text_token|>",
                                "<|boss|>",
                                "<|eoss|>",
                                *pt_mode_tokens,
                              ],
                              additional_special_tokens_json=None,
                              num_preserved=100):
    # ----- Load base model & tokenizer -----
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    base_model_vocab_size = base_model.vocab_size
    base_tokenizer_vocab_size = base_tokenizer.vocab_size
    base_tokenizer_len = len(base_tokenizer)
    print(f"Base model vocab size: {base_model_vocab_size}")
    print(f"Base tokenizer vocab size: {base_tokenizer_vocab_size}")
    print(f"Base tokenizer length: {base_tokenizer_len}")

    # ----- Load CosyVoice tokenizer -----
    cosyvoice_tokenizer = tokenizer_class(init_param_path=base_model_path)
    cosy_specials = set(cosyvoice_tokenizer.special_tokens_map['additional_special_tokens'])
    base_specials = set(base_tokenizer.special_tokens_map['additional_special_tokens'])

    # New special tokens from cosyvoice that base tokenizer didn't have
    new_special_tokens = [tok for tok in cosy_specials if tok not in base_specials]
    num_new_special_tokens = len(new_special_tokens)
    print(f"New special tokens from CosyVoice num {len(new_special_tokens)}: {new_special_tokens}")

    if additional_special_tokens_json is not None:
        additional_special_tokens.extend(json.load(open(additional_special_tokens_json)).keys())
        new_special_tokens_set = set(new_special_tokens)
        for tok in additional_special_tokens:
            tok_name = tok[2:-2]
            if tok_name[1:].isdigit():
                continue
            if tok not in new_special_tokens_set and tok not in base_specials:
                new_special_tokens.append(tok)
        num_new_special_tokens = len(new_special_tokens)
        print(f"New special tokens from CosyVoice and additional special tokens num {len(new_special_tokens)}: {new_special_tokens}")

    # Add codec units to new special tokens
    new_special_tokens = new_special_tokens + [f"<|c{i}|>" for i in range(codec_unit)]
    num_new_special_tokens = len(new_special_tokens)
    print(f"New special tokens from CosyVoice and codec units num {len(new_special_tokens)}: {new_special_tokens}")

    # ----- Extend tokenizer -----
    base_tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens}, replace_additional_special_tokens=False)

    # total new embeddings to add = new specials
    total_new = num_new_special_tokens + num_preserved
    print(f"Extending with {total_new} new tokens ({num_new_special_tokens} specials + {num_preserved} preserved)")

    # ----- Extend model's embeddings -----
    # Step 1: truncate base embedding to first base_tokenizer_len entries
    old_embed_tokens = base_model.get_input_embeddings()
    embed_dim = old_embed_tokens.embedding_dim

    # truncate_vocab_size = base_tokenizer_len  # use length of tokenizer
    truncate_vocab_size = base_model_vocab_size  # use length of base model vocab
    truncated_weight = old_embed_tokens.weight[:truncate_vocab_size]  # (151687, 1024)

    # Step 2: create new embeddings for special+codec tokens
    new_embed_weight = torch.nn.Embedding(total_new, embed_dim).weight
    new_embed_weight.data.normal_(mean=0.0, std=0.02)

    # Step 3: concatenate
    extended_embed_weight = torch.cat([truncated_weight, new_embed_weight], dim=0)  # shape = (151687+6771, 1024)

    def is_tied_embedding(model):
        input_emb_weights = model.get_input_embeddings().weight
        output_emb_weights = model.get_output_embeddings().weight
        return input_emb_weights is output_emb_weights
    is_tied = is_tied_embedding(base_model)

    # Replace model's input embedding
    new_embedding_layer = torch.nn.Embedding(extended_embed_weight.size(0), embed_dim, )
    new_embedding_layer.weight.data = extended_embed_weight
    base_model.set_input_embeddings(new_embedding_layer)
    # print(new_embedding_layer.weight.shape, base_model.get_input_embeddings().weight.shape)

    if is_tied:
        print("Base model's input and output embeddings are tied, not building new output lm_head.")
        base_model.tie_weights()
    else:
        print("Base model's input and output embeddings are not tied, extending output lm_head.")
        # ----- Extend output lm_head -----
        old_lm_head = base_model.get_output_embeddings()
        truncated_lm_weight = old_lm_head.weight[:truncate_vocab_size]
        new_lm_weight = torch.empty((total_new, embed_dim))
        torch.nn.init.normal_(new_lm_weight, mean=0.0, std=0.02)
        extended_lm_weight = torch.cat([truncated_lm_weight, new_lm_weight], dim=0)

        new_lm_head_layer = torch.nn.Linear(embed_dim, extended_lm_weight.size(0), bias=False)
        new_lm_head_layer.weight.data = extended_lm_weight
        base_model.set_output_embeddings(new_lm_head_layer)
        # print(new_lm_head_layer.weight.shape, base_model.get_output_embeddings().weight.shape)

    # Update config vocab_size to match
    base_model.config.vocab_size = extended_embed_weight.size(0)
    base_model.vocab_size = extended_embed_weight.size(0)

    print(f"Extended vocab size: {base_model.config.vocab_size}")

    output_model_path = os.path.abspath(output_model_path)
    base_model.save_pretrained(output_model_path, push_to_hub=False)
    base_tokenizer.save_pretrained(output_model_path, push_to_hub=False)

    return base_model, base_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, required=True)
    parser.add_argument('--output-model', type=str, required=True)
    parser.add_argument('--codec-unit', type=int, default=6761)
    parser.add_argument('--additional-special-tokens-json', type=str, required=True)
    parser.add_argument('--num-preserved', type=int, default=100)
    args = parser.parse_args()

    extended_model, extended_tokenizer = extend_vocab_to_cosyvoice(args.base_model,
            args.output_model,
            additional_special_tokens_json=args.additional_special_tokens_json,
            codec_unit=args.codec_unit,
            num_preserved=args.num_preserved)
    print(f"Extended model: {extended_model}")
    print(f"Extended tokenizer: {extended_tokenizer}")
    print(f"Saved extended model to {args.output_model}")
    print(f"Saved extended tokenizer to {args.output_model}")
