# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for generating text."""

import copy
import json
import os
import time

import torch.nn.functional as F

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.training.utils import get_ltor_masks_and_position_ids
from .generation_logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, \
    NoRepeatNGramLogitsProcessor, MinLengthLogitsProcessor, StopWordsLogitsProcessor


def get_batch(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate_samples_input_from_file(model):
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('`sample-output-file` not specified, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
                    and mpu.get_tensor_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                input_pos += 1
                if input_pos == input_count:
                    raw_text = "stop"
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor,
                                         group=mpu.get_model_parallel_group())
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if mpu.get_tensor_model_parallel_rank() == 0 \
                    and args.pipeline_model_parallel_size > 1:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                pass

            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():
                    os.system('clear')
                    print("\nContext:", raw_text, flush=True)

                    fname_out.write("\nContext:")
                    fname_out.write(raw_text)

                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[raw_text_len:]
                    print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                    fname_out.write("\n\nMegatron-LM:")
                    fname_out.write(trim_decode_tokens)
                    fname_out.write("\n")

            raw_text = None
            context_count += 1


# We added this function to support the tasks evaluation such as squad
# and drop in the https://github.com/EleutherAI/lm-evaluation-harness
# codebase. The lm-evaluation-harness code can now call this function
# similar to their current generate function call used for gpt style models.
def generate_samples_eval(model, context, max_gen_length, eos_token_id):
    # Generate samples for lm evaluation
    # NEED TO THINK ABOUT eos token

    args = get_args()
    tokenizer = get_tokenizer()

    raw_text_len = len(context)
    model.eval()

    context_tokens = tokenizer.tokenize(context)
    args.out_seq_length = max_gen_length + len(context_tokens)
    args.eos_id = eos_token_id

    with torch.no_grad():
        token_stream = get_token_stream(model, [context_tokens])
        for counter, decode_tokens in enumerate(token_stream):
            if counter == args.out_seq_length:
                break

    decode_tokens, _ = decode_tokens
    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
    trim_decode_tokens = tokenizer.detokenize(
        decode_tokens)[raw_text_len:]

    return trim_decode_tokens


def greedy_until(model, raw_text, until):
    # Generate samples for lm evaluation
    # NEED TO THINK ABOUT eos token

    args = get_args()
    tokenizer = get_tokenizer()
    args.greedy = True
    args.repetition_penalty = 1.0
    args.out_seq_length = 256
    args.seq_length = 4096
    args.no_repeat_ngram = -1
    args.min_out_seq_length = 1
    args.micro_batch_size = 1

    if not args.stop_words:
        args.stop_words = '\n\n'
    model.eval()

    if mpu.is_pipeline_first_stage() \
            and mpu.get_tensor_model_parallel_rank() == 0:

        raw_text_len = len(raw_text)
        context_tokens = tokenizer.tokenize(raw_text)
        context_length = len(context_tokens)

    else:
        context_tokens = tokenizer.tokenize("EMPTY TEXT")
        raw_text_len = 0
        context_length = 0

    input_info = [raw_text_len, context_length]
    input_info_tensor = torch.cuda.LongTensor(input_info)
    torch.distributed.all_reduce(input_info_tensor,
                                 group=mpu.get_model_parallel_group())
    raw_text_len = input_info_tensor[0].item()
    context_length = input_info_tensor[1].item()

    # For pipeline parallel we send context tokens to other stages
    # so they get the lengths correct
    if mpu.get_tensor_model_parallel_rank() == 0 \
            and args.pipeline_model_parallel_size > 1:
        if mpu.is_pipeline_first_stage():
            src = mpu.get_pipeline_model_parallel_first_rank()
            group = mpu.get_pipeline_model_parallel_group()
            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            torch.distributed.broadcast(context_tokens_tensor, src, group)
        else:
            src = mpu.get_pipeline_model_parallel_first_rank()
            group = mpu.get_pipeline_model_parallel_group()
            context_tokens_tensor = torch.empty(context_length,
                                                dtype=torch.int64,
                                                device=torch.device("cuda"))
            torch.distributed.broadcast(context_tokens_tensor, src, group)
            context_tokens = context_tokens_tensor.cpu().numpy().tolist()

    with torch.no_grad():
        token_stream = get_token_stream(model, [context_tokens])
        for counter, decode_tokens in enumerate(token_stream):
            pass

    decode_tokens, _ = decode_tokens
    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
    trim_decode_tokens = tokenizer.detokenize(
        decode_tokens)[raw_text_len:].replace('<|endoftext|>', '').split(args.stop_words)[0]

    return trim_decode_tokens


def generate_samples_interactive(model, print_frequency=24):
    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
                    and mpu.get_tensor_model_parallel_rank() == 0:
                os.system('clear')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor,
                                         group=mpu.get_model_parallel_group())
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if mpu.get_tensor_model_parallel_rank() == 0 \
                    and args.pipeline_model_parallel_size > 1:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])

            for counter, decode_tokens in enumerate(token_stream):
                if counter % print_frequency != 0 \
                        or mpu.get_tensor_model_parallel_rank() != 0 \
                        or not mpu.is_pipeline_first_stage():
                    continue

                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if mpu.is_pipeline_first_stage() \
                    and mpu.get_tensor_model_parallel_rank() == 0:
                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                if not isinstance(decode_tokens, list):
                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                input("\nPress Enter to continue >>>")

            raw_text = None
            context_count += 1


def generate_samples_unconditional(model, latencies=[], model_latencies=[], single_token_latency=[]):
    args = get_args()
    tokenizer = get_tokenizer()

    num_samples = args.num_samples
    context_tokens = [[tokenizer.eod]
                      for _ in range(args.micro_batch_size)]
    ctr = 0
    while True:
        torch.cuda.synchronize()
        start_time = time.time()
        for token_stream in get_token_stream(model,
                                             copy.deepcopy(context_tokens), model_latencies=model_latencies,
                                             single_token_latency=single_token_latency):
            pass
        torch.cuda.synchronize()
        latencies.append(time.time() - start_time)
        start_time = time.time()
        if mpu.is_pipeline_last_stage() and \
                mpu.get_tensor_model_parallel_rank() == 0:
            # if ctr % args.log_interval == 0:
            #    print('Avg s/batch:',
            #          (time.time() - start_time) / min(args.log_interval, ctr + 1))
            #    start_time = time.time()
            length = len(token_stream)
            token_batch = token_stream[0].cpu().numpy().tolist()
            length_batch = token_stream[1].cpu().numpy().tolist()
            assert len(length_batch) == args.micro_batch_size
            for tokens, length in zip(token_batch, length_batch):
                tokens = tokens[1:length - 1]
                text = tokenizer.detokenize(tokens)
                is_finished = length < args.seq_length - 1
                datum = {'text': text, 'length': length - 1, 'finished': is_finished}
                yield datum
                ctr += 1
                if ctr >= num_samples:
                    break
        else:
            for _ in range(args.micro_batch_size):
                yield None
                ctr += 1
                if ctr >= num_samples:
                    break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(model, latencies=[], single_token_latency=[], model_latencies=[]):
    args = get_args()
    assert args.genfile is not None
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    gen_file_name = f"rank_{rank}_{args.genfile}"
    with open(gen_file_name, 'w') as f:
        for datum in generate_samples_unconditional(model, latencies=latencies, model_latencies=model_latencies,
                                                    single_token_latency=single_token_latency):
            if mpu.is_pipeline_last_stage() and \
                    mpu.get_tensor_model_parallel_rank() == 0:
                f.write(json.dumps(datum) + '\n')


def pad_batch(batch, pad_id, args):
    context_lengths = []
    seq_length = -1
    # args.out_seq_length
    for tokens in batch:
        seq_length = max(seq_length, len(tokens) + args.out_seq_length)
    seq_length = min(seq_length, args.seq_length)
    for tokens in batch:
        context_length = len(tokens)
        # if context_length < args.seq_length:
        #     tokens.extend([pad_id] * (args.seq_length - context_length))
        #     context_lengths.append(context_length)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
            context_lengths.append(context_length)
        else:
            # TODO: Quick fix for long prompts (when decoding with maxlen=2048),
            #  will change into more elegant way in future.
            del tokens[args.seq_length:]
            context_lengths.append(args.seq_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens, model_latencies=[], single_token_latency=[],
                     context_tokens_tensor=None, context_length_tensor=None):
    args = get_args()
    tokenizer = get_tokenizer()

    if context_tokens is not None:
        context_tokens, context_lengths = pad_batch(context_tokens,
                                                    tokenizer.eod, args)

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)

        torch.distributed.broadcast(context_length_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        torch.distributed.broadcast(context_tokens_tensor,
                                    mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
    else:
        assert context_tokens_tensor is not None and context_length_tensor is not None, \
            "context_tokens_tensor and context_length_tensor should be set if context_tokens is None"

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids, model_latencies=model_latencies)

    count = 0

    t0 = time.time()
    for tokens, lengths in batch_token_iterator:
        if count > 1:
            torch.cuda.synchronize()
            t_elapsed = time.time() - t0
            single_token_latency.append(t_elapsed)
        torch.cuda.synchronize()
        t0 = time.time()
        count += 1
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None, model_latencies=[]):
    args = get_args()
    tokenizer = get_tokenizer()
    # added eos_id to support the function generate_samples_eval that passes
    # eos_id as an argument and needs termination when that id id found.

    batch_size = context_tokens.size(0)
    undone_id = torch.arange(batch_size).cuda()
    min_prompt_length = context_lengths.min().item()
    max_sequence_length = context_tokens.size(1)
    if hasattr(args, 'eos_id'):
        eos_id = args.eos_id
    else:
        eos_id = tokenizer.eod

    logits_processor = LogitsProcessorList()
    greedy_logits_processor = LogitsProcessorList()
    if args.repetition_penalty != 1.0:
        logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=args.repetition_penalty))
    if args.no_repeat_ngram > 0:
        logits_processor.append(NoRepeatNGramLogitsProcessor(ngram_size=args.no_repeat_ngram))
    if args.min_out_seq_length > 0:
        logits_processor.append(MinLengthLogitsProcessor(min_length=args.min_out_seq_length, eos_token_id=eos_id))
    use_fn_sampling = getattr(args, 'factual_nucleus_sampling', False)
    if use_fn_sampling:
        decay_factor = args.decay_factor
        decay_bound = args.decay_bound
    else:
        decay_factor = 1
        decay_bound = 0

    stop_words = getattr(args, "stop_words", '')
    stop_words_ids = getattr(args, "stop_words_ids", '')
    all_stop_words_ids = []
    if stop_words:
        stop_words = stop_words.split(',')
        all_stop_words_ids.extend([tokenizer.tokenize_with_special_tokens(stop_word) for stop_word in stop_words])
    if stop_words_ids:
        stop_words_ids = stop_words_ids.split(';')
        for words in stop_words_ids:
            words = [int(word) for word in words.split(',')]
            all_stop_words_ids.append(words)

    if all_stop_words_ids:
        stop_words_logits_processor = StopWordsLogitsProcessor(stop_words_ids=all_stop_words_ids, eos_token_id=eos_id)
        logits_processor.append(stop_words_logits_processor)
        greedy_logits_processor.append(stop_words_logits_processor)
    model.eval()

    forward_step = ForwardStep(model, batch_size, max_sequence_length)
    tokens = context_tokens
    org_context_length = min_prompt_length
    if maxlen is None:
        maxlen = args.seq_length - 1
        if maxlen > (org_context_length + args.out_seq_length):
            maxlen = org_context_length + args.out_seq_length

    lengths = torch.ones([batch_size]).long().cuda() * maxlen
    with torch.no_grad():
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):
            # Pick the slice that we need to pass through the network.
            # Enable dynamic batch size (remove finished samples)
            tokens2use = context_tokens[:, prev_context_length:context_length].index_select(0, undone_id)
            positions2use = position_ids[:, prev_context_length:context_length].index_select(0, undone_id)
            attention_mask2use = attention_mask[
                                 ..., prev_context_length:context_length, :context_length]
            if attention_mask2use.shape[0] == batch_size:
                attention_mask2use = attention_mask2use.index_select(0, undone_id)

            output = forward_step(tokens2use, positions2use, attention_mask2use)

            if mpu.is_pipeline_last_stage():
                logits = output[:, -1, :]
                if args.greedy:
                    logits = greedy_logits_processor(tokens[:, :context_length].index_select(0, undone_id), logits)
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= args.temperature
                    logits = logits_processor(tokens[:, :context_length].index_select(0, undone_id), logits)
                    if use_fn_sampling:
                        top_p = max(args.top_p * (decay_factor ** (context_length - min_prompt_length)), decay_bound)
                    else:
                        top_p = args.top_p
                    logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=top_p)
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)

                started = context_lengths <= context_length

                prev_full = tokens[:, context_length].clone().scatter_(0, undone_id, prev)  # batch_size
                new_tokens = switch(
                    tokens[:, context_length].view(-1), prev_full, started)
                tokens[:, context_length] = new_tokens
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                lengths.scatter_(0, undone_id, torch.ones(undone_id.shape[0], device=lengths.device,
                                                          dtype=lengths.dtype) * context_length)
                done_token = (prev == eos_id).byte() & started.byte().index_select(0, undone_id)  # left_batch_size

                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done_token, src, group)

                # prev_context_length = context_length
                yield tokens, lengths
            else:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    # prev_context_length = context_length
                    yield tokens, None
                else:
                    yield None, None

                done_token = torch.cuda.ByteTensor([0] * undone_id.shape[0])
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done_token, src, group)

            # Update the context length for the next token generation.
            prev_context_length = context_length

            if torch.all(done_token):
                break

            # select cached key value & create new undone_id
            undone_id = undone_id[~done_token.bool()]
            if undone_id.shape[0] < done_token.shape[0]:
                select_id = torch.arange(done_token.shape[0]).cuda()[~done_token.bool()]
                for layer_id, (cached_key, cached_value) in forward_step.inference_params.key_value_memory_dict.items():
                    cached_key = cached_key.index_select(1, select_id)
                    cached_value = cached_value.index_select(1, select_id)
                    forward_step.inference_params.key_value_memory_dict[layer_id] = (cached_key, cached_value)



import random
import torch


def chat_interactive(model, print_frequency=24):
    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    user_name = 'Human'
    model_name = 'Assistant'
    max_window_size = 4096
    args.stop_words = f'{user_name}:'
    history = []
    os.system('clear')
    seed = 1234
    with torch.no_grad():
        while True:
            random.seed(seed)
            torch.random.manual_seed(seed)
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
                    and mpu.get_tensor_model_parallel_rank() == 0:
                message = input("\nChat Context (stop to exit) >>> ")
                while not message:
                    print('Prompt should not be empty!')
                    message = input("\nChat Context (stop to exit) >>> ")
                raw_text = ""
                for item in history[::-1]:
                    q, a = item
                    qa_pair = f'{user_name}: {q}\n\n{model_name}: {a}\n\n'
                    if len(tokenizer.tokenize(qa_pair)) + len(tokenizer.tokenize(raw_text)) < max_window_size:
                        raw_text = qa_pair + raw_text
                    else:
                        break
                raw_text = "\n\n" + raw_text
                raw_text += f'{user_name}: {message}\n\n{model_name}:'
                raw_text_len = len(raw_text)

                if "clear" in raw_text:
                    history = []
                    continue

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor,
                                         group=mpu.get_model_parallel_group())
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if mpu.get_tensor_model_parallel_rank() == 0 \
                    and args.pipeline_model_parallel_size > 1:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])

            for counter, decode_tokens in enumerate(token_stream):
                if counter % print_frequency != 0 \
                        or mpu.get_tensor_model_parallel_rank() != 0 \
                        or not mpu.is_pipeline_first_stage():
                    continue

                os.system('clear')
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                print("\nQWen:", trim_decode_tokens, flush=True)

            if mpu.is_pipeline_first_stage() \
                    and mpu.get_tensor_model_parallel_rank() == 0:
                os.system('clear')

                if not isinstance(decode_tokens, list):
                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                trim_decode_tokens = trim_decode_tokens.replace('<|endoftext|>', '')
                trim_decode_tokens = trim_decode_tokens.split(f'{user_name}:')[0]
                trim_decode_tokens = trim_decode_tokens.strip()

                print("\nQWen:", trim_decode_tokens, flush=True)

                history.append((message, trim_decode_tokens))

            raw_text = None

            context_count += 1
