from typing import Optional, Callable, Union, Any
import warnings
import inspect
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3Model
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoePreTrainedModel, load_balancing_loss_func
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_exporting
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig, GenerateOutput, logger


import copy
import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList
from transformers.generation.utils import GenerateOutput


class DualChannelGenerationMixin(GenerationMixin):
    """
    A Mixin to add dual-head (Audio + Text) generation capabilities.
    Assumes the model returns a tuple of logits: (logits_head_a, logits_head_b).
    """

    def _cache_dependant_audio_input_preparation(
        self,
        input_ids: Optional[torch.LongTensor],
        input_ids_audio: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Generic cache-dependent input preparation
        The code is put in a separate function to allow granular unit testing
        as it needs a different implementation to be exportable.

        If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        - Exception 1: when passing input_embeds, input_ids may be missing entries
        - Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        - Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        - Exception 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
          generate the first token for each sequence. Later use the generated Input ids for continuation.

        The current implementation does not rely on ``self`` and could be
        a class method. It is left as a standard method to be easily rewritten.
        """
        if is_torchdynamo_exporting():
            return self._cache_dependant_input_preparation_exporting(input_ids, inputs_embeds, cache_position)
        if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
        elif (
            inputs_embeds is not None  # Exception 1
            or (cache_position[-1] >= input_ids.shape[1])  # Exception 3
        ):
            input_ids = input_ids[:, -cache_position.shape[0] :]
            input_ids_audio = input_ids_audio[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]
            input_ids_audio = input_ids_audio[:, cache_position]
        return inputs_embeds, input_ids, input_ids_audio


    def prepare_inputs_for_audio_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_audio: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. It includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids, input_ids_audio = self._cache_dependant_audio_input_preparation(
                input_ids, input_ids_audio, inputs_embeds, cache_position
            )
            model_inputs["input_ids_audio"] = input_ids_audio

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
        # pass)
        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            # If it's not defined, it means the model uses the new general mask API
            if causal_mask_creation_function is None:  # can't be found
                token_type_ids = model_inputs.get("token_type_ids")
                position_ids = model_inputs.get(position_ids_key)
                # Some models may overwrite the general one
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask = causal_mask_creation_function(
                    config=self.config,
                    # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs


    def _prepare_audio_generation_config(
        self,
        generation_config: GenerationConfig,
        tokenizer,
        audio_code_book_size: int,
        audio_blank_token: str,
        audio_do_sample: Optional[bool] = None,
        audio_temperature: Optional[float] = None,
        audio_top_p: Optional[float] = None,
        audio_top_k: Optional[int] = None,
        audio_repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> GenerationConfig:
        text_vocab_offset = tokenizer.vocab_size
        special_token_offset = audio_code_book_size - text_vocab_offset
        generation_config_audio = copy.deepcopy(generation_config)

        def add_offset(token_id: Union[int, list[int]]):
            if isinstance(token_id, int):
                return token_id + special_token_offset
            elif isinstance(token_id, list):
                return [token_id + special_token_offset for token_id in token_id]
            else:
                raise ValueError(f"Invalid token_id: {token_id}")
        generation_config_audio.bos_token_id = add_offset(generation_config.bos_token_id)
        generation_config_audio.eos_token_id = add_offset(generation_config.eos_token_id)
        generation_config_audio.pad_token_id = add_offset(generation_config.pad_token_id)
        generation_config_audio.ignore_token_id = add_offset(generation_config.ignore_token_id) if generation_config.ignore_token_id >= 0 else generation_config.ignore_token_id
        generation_config_audio.blank_token_id = add_offset(tokenizer.encode(audio_blank_token, add_special_tokens=False)[0])

        if audio_do_sample is not None:
            logger.info(f"Setting audio do_sample to {audio_do_sample}")
            generation_config_audio.do_sample = audio_do_sample
        if audio_temperature is not None:
            logger.info(f"Setting audio temperature to {audio_temperature}")
            generation_config_audio.temperature = audio_temperature
        if audio_top_p is not None:
            logger.info(f"Setting audio top_p to {audio_top_p}")
            generation_config_audio.top_p = audio_top_p
        if audio_top_k is not None:
            logger.info(f"Setting audio top_k to {audio_top_k}")
            generation_config_audio.top_k = audio_top_k
        if audio_repetition_penalty is not None:
            logger.info(f"Setting audio repetition_penalty to {audio_repetition_penalty}")
            generation_config_audio.repetition_penalty = audio_repetition_penalty
        return generation_config_audio


    def _prepare_special_tokens_audio(
        self,
        generation_config: GenerationConfig,
        kwargs_has_attention_mask: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        generation_config._blank_token_id_tensor = torch.tensor(generation_config.blank_token_id, device=device, dtype=torch.long)
        generation_config._ignore_token_id_tensor = torch.tensor(generation_config.ignore_token_id, device=device, dtype=torch.long)

    @torch.no_grad()
    def generate_dual_channel(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_audio: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        audio_delay: int = 0,
        if_output_text: bool = True,
        if_output_audio: bool = True,
        zero_out_non_output_embeds: bool = False,   # zero out embeddings of non-output heads
        audio_blank_token: str = "<|audio_blank|>",
        ignore_token: Union[str, int] = "<|pt_mode_extra_19|>",
        generation_config: Optional[GenerationConfig] = None,
        generation_config_audio: Optional[GenerationConfig] = None,
        audio_do_sample: Optional[bool] = None,
        audio_temperature: Optional[float] = None,
        audio_top_p: Optional[float] = None,
        audio_top_k: Optional[int] = None,
        audio_repetition_penalty: Optional[float] = None,
        use_model_defaults: Optional[bool] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_processor_audio: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        stopping_criteria_audio: Optional[StoppingCriteriaList] = None,
        **kwargs,
    ):
        if (input_ids is None or input_ids_audio is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of (input_ids & inputs_ids_audio) or inputs_embeds")
        
        if input_ids is None or input_ids_audio is None:
            inputs_tensor = inputs_embeds
            input_ids = torch.zeros(inputs_tensor.shape[0], 0, device=inputs_tensor.device, dtype=torch.long)
            input_ids_audio = torch.zeros(inputs_tensor.shape[0], 0, device=inputs_tensor.device, dtype=torch.long)
            use_inputs_embeds = True
        else:
            inputs_tensor = input_ids
            use_inputs_embeds = False
        input_ids_length = inputs_tensor.shape[1]
        device = inputs_tensor.device

        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            raise ValueError("tokenizer is required for getting vocab size")
        audio_code_book_size = kwargs.pop("audio_code_book_size", None)
        if audio_code_book_size is None:
            raise ValueError("audio_code_book_size is required")

        # 1. Prepare Generation Config
        # We assume the default config applies to text head. 
        # You might want to separate configs, but for simplicity, we share max_length etc.
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        
        do_sample = generation_config.do_sample
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # Add ignore token idd
        if isinstance(ignore_token, int):
            generation_config.ignore_token_id = ignore_token
        elif isinstance(ignore_token, str):
            generation_config.ignore_token_id = tokenizer.encode(ignore_token, add_special_tokens=False)[0]
        else:
            raise ValueError(f"Invalid ignore_token: {ignore_token}")
        generation_config._ignore_token_id_tensor = torch.tensor(generation_config.ignore_token_id, device=device, dtype=torch.long)
        
        # 2. Setup Caching (Standard Dynamic Cache)
        # We use the standard cache setup because the transformer body is "vanilla"
        # The cache stores the fused hidden states.
        
        # We treat text input as the "main" input for length calculations
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=False,
            has_default_min_length=False,
            model_input_name="input_ids",
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if use_inputs_embeds:
            generation_config.max_length = generation_config.max_new_tokens

        # Audio generation config
        if generation_config_audio is None:
            generation_config_audio = self._prepare_audio_generation_config(
                generation_config,
                tokenizer,
                audio_code_book_size,
                audio_blank_token,
                audio_do_sample=audio_do_sample,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                **kwargs
            )
        self._prepare_special_tokens_audio(generation_config_audio, kwargs_has_attention_mask, device=device)

        # Prepare standard cache
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model=None, 
            batch_size=inputs_tensor.shape[0], max_cache_length=generation_config.max_length
        )

        # 3. Prepare Logits Processors (The "brains" of sampling)
        # We need TWO sets of processors because Audio and Text have different distributions.
        # e.g., Text needs repetition penalty, Audio might not.
        
        # Processor for text head
        if logits_processor is None:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=None,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=device,
            )
            
        # Processor for audio head - We might want to customize config here
        # For now, we clone the config logic
        # generation_config_audio.sequence_bias=[[[24082], -1e2],[[24073], -1e2]]
        if logits_processor_audio is None:
            logits_processor_audio = self._get_logits_processor(
                generation_config=generation_config_audio, # Pass a different config here if text needs diff temp
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=None,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=device,
            )
        
        # Stopping criteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        stopping_criteria_audio = stopping_criteria_audio if stopping_criteria_audio is not None else StoppingCriteriaList()
        stopping_criteria_audio = self._get_stopping_criteria(
            generation_config=generation_config_audio, stopping_criteria=stopping_criteria_audio, tokenizer=tokenizer, **kwargs
        )

        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        # 4. The Dual-Head Loop
        # We cannot use self._sample because it assumes 1 input -> 1 output.
        # We write a custom synchronous sampling loop.
        
        batch_size, cur_len = inputs_tensor.shape[:2]
        cur_output_pos = 0
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device) * (1 if if_output_text else 0)
        unfinished_sequences_audio = torch.ones(batch_size, dtype=torch.long, device=device) * (1 if if_output_audio else 0)
        this_peer_finished = False
        model_kwargs = self._get_initial_cache_position(cur_len, device, model_kwargs)
        
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus=False, device=device):
            # A. Prepare Inputs
            # We must pass BOTH inputs to prepare_inputs so the cache position is updated correctly
            model_inputs = self.prepare_inputs_for_audio_generation(
                input_ids=input_ids, 
                input_ids_audio=input_ids_audio, # Specific to your model signature
                inputs_embeds=inputs_embeds,
                ignore_token_id=generation_config._ignore_token_id_tensor,
                ignore_token_id_audio=generation_config_audio._ignore_token_id_tensor,
                **model_kwargs
            )

            # if cur_output_pos > 0 and not if_output_text:
            #     input_ids[:] = -100

            # if cur_output_pos > 0 and not if_output_audio:
            #     input_ids_audio[:] = -100
            
            # B. Forward Pass
            # Expecting output.logits to be a tuple (logits_a, logits_b)
            outputs = self.forward_dual_channel(**model_inputs, return_dict=True)

            # Update model kwargs (Cache position etc)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )
            
            # C. Extract Logits
            # The Kimi code returns logits=(text_logits, audio_logits)
            # Adjust index based on your specific model output order
            next_token_logits_text = outputs.logits[0][:, -1, :]
            next_token_logits_audio = outputs.logits[1][:, -1, :]

            # D. Process Logits
            next_token_scores_text = logits_processor(input_ids, next_token_logits_text)
            next_token_scores_audio = logits_processor_audio(input_ids_audio, next_token_logits_audio)

            # E. Sample 
            if do_sample:
                probs_text = nn.functional.softmax(next_token_scores_text, dim=-1)
                next_tokens_text = torch.multinomial(probs_text, num_samples=1).squeeze(1)
                probs_audio = nn.functional.softmax(next_token_scores_audio, dim=-1)
                next_tokens_audio = torch.multinomial(probs_audio, num_samples=1).squeeze(1)
            else:
                next_tokens_text = torch.argmax(next_token_scores_text, dim=-1)
                next_tokens_audio = torch.argmax(next_token_scores_audio, dim=-1)

            # G. Update Inputs for next step
            if cur_output_pos < audio_delay:
                # Set the audio token to audio_blank
                next_tokens_audio = next_tokens_audio * 0 + generation_config_audio._blank_token_id_tensor

            # Pad finished sequences if necessary
            if has_eos_stopping_criteria:
                text_pad_tensor = generation_config._ignore_token_id_tensor if zero_out_non_output_embeds else generation_config._pad_token_tensor
                audio_pad_tensor = generation_config_audio._ignore_token_id_tensor if zero_out_non_output_embeds else generation_config_audio._pad_token_tensor
                next_tokens_text = next_tokens_text * unfinished_sequences + text_pad_tensor * (1 - unfinished_sequences)
                next_tokens_audio = next_tokens_audio * unfinished_sequences_audio + audio_pad_tensor * (1 - unfinished_sequences_audio)

            # Append to history
            input_ids = torch.cat([input_ids, next_tokens_text[:, None]], dim=-1)
            input_ids_audio = torch.cat([input_ids_audio, next_tokens_audio[:, None]], dim=-1)

            # F. Check Stopping Criteria (EOS)
            # Usually we stop if Text hits EOS, or Audio hits EOS, or both?
            # Let's assume we stop if Head B (Text) hits EOS for now.
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            unfinished_sequences_audio = unfinished_sequences_audio & ~stopping_criteria_audio(input_ids_audio, None)
            this_peer_finished = unfinished_sequences.max() == 0 and unfinished_sequences_audio.max() == 0
            if this_peer_finished:
                logger.info(f"Sequence finished")
            cur_len += 1
            cur_output_pos += 1

        return input_ids, input_ids_audio


class FunAudioForDualChannelModel(Qwen3PreTrainedModel, DualChannelGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config,
            audio_code_book_size: int = None,
            audio_embedding_path: str = None,
            audio_output_layer_path: str = None,
            debug_mode=False
        ):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.special_tokens_audio_count = 463
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.audio_code_book_size = audio_code_book_size
        self.audio_embedding_path = audio_embedding_path
        self.audio_output_layer_path = audio_output_layer_path

        self.debug_mode = debug_mode

        # for debugging
        if self.debug_mode:
            self.embed_tokens_audio = self.model.embed_tokens
            self.lm_head_audio = self.lm_head
        else:
            if audio_code_book_size is not None:
                self.embed_tokens_audio = nn.Embedding(self.audio_code_book_size + self.special_tokens_audio_count, config.hidden_size)
                self.lm_head_audio = nn.Linear(config.hidden_size, self.audio_code_book_size + self.special_tokens_audio_count)

        # Initialize weights and apply final processing
        self.post_init()
    
    def load_audio_modules(self, audio_embedding_path: str=None, audio_output_layer_path: str=None):

        audio_embed_ckpt_path = audio_embedding_path if audio_embedding_path is not None else self.audio_embedding_path
        if audio_embed_ckpt_path is not None:
            audio_embed_weight = torch.load(audio_embed_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, self.config.hidden_size)
            if audio_embed_weight.shape == expected_shape:
                self.embed_tokens_audio.weight.data = audio_embed_weight.to(self.embed_tokens_audio.weight.data.dtype)
                logger.info(f"Loading audio_embed from {audio_embed_ckpt_path}, shape: {audio_embed_weight.shape}")
            else:
                logger.warning(f"audio_embed shape mismatch: expected {expected_shape}, got {audio_embed_weight.shape}")
        else:
            logger.info(f"No audio_embed_ckpt_path provided, using default initialization")
        
        audio_head_ckpt_path = audio_output_layer_path if audio_output_layer_path is not None else self.audio_output_layer_path
        if audio_head_ckpt_path is not None:
            audio_head_weight = torch.load(audio_head_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, self.config.hidden_size)
            if audio_head_weight.shape == expected_shape:
                self.lm_head_audio.weight.data = audio_head_weight.to(self.lm_head_audio.weight.data.dtype)
                logger.info(f"Loading audio_head weight from {audio_head_ckpt_path}, shape: {audio_head_weight.shape}")
            else:
                logger.warning(f"audio_head weight shape mismatch: expected {expected_shape}, got {audio_head_weight.shape}")
        else:
            logger.info(f"No audio_head_ckpt_path provided, using default initialization")


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def forward_dual_channel(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_audio: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_token_id: Optional[torch.Tensor] = None,
        ignore_token_id_audio: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_audio: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        if (input_ids is None or input_ids_audio is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of (input_ids & inputs_ids_audio) or inputs_embeds")

        if inputs_embeds is None:
            text_mask = input_ids >= 0
            if ignore_token_id is not None:
                text_mask = text_mask & (input_ids != ignore_token_id)
            text_mask = text_mask.to(torch.int32)
            input_ids = input_ids * text_mask
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * text_mask.unsqueeze(-1)

            audio_mask = input_ids_audio >= 0
            if ignore_token_id_audio is not None:
                audio_mask = audio_mask & (input_ids_audio != ignore_token_id_audio)
            audio_mask = audio_mask.to(torch.int32)
            input_ids_audio = input_ids_audio * audio_mask
            audio_embeds = self.embed_tokens_audio(input_ids_audio)
            audio_embeds = audio_embeds * audio_mask.unsqueeze(-1)
            inputs_embeds += audio_embeds
            if self.debug_mode:
                inputs_embeds *= 0.5

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits_audio = self.lm_head_audio(hidden_states[:, slice_indices, :])

        loss = None
        loss_audio = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            loss_audio = self.loss_function(logits=logits_audio, labels=labels_audio, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=[loss, loss_audio],
            logits=[logits, logits_audio],
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FunAudioMoeFusedForDualChannelModel(Qwen3MoePreTrainedModel, DualChannelGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config,
            audio_code_book_size: int = None,
            audio_embedding_path: str = None,
            audio_output_layer_path: str = None,
            debug_mode=False
        ):
        super().__init__(config)
        from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedModel, Qwen3MoeFusedSparseMoeBlock, Qwen3MoeFusedDecoderLayer
        from transformers.utils.generic import OutputRecorder
        self.model = Qwen3MoeFusedModel(config)
        self._can_record_outputs["router_logits"] = OutputRecorder(Qwen3MoeFusedSparseMoeBlock, index=1)
        self._can_record_outputs["hidden_states"] = Qwen3MoeFusedDecoderLayer

        self.vocab_size = config.vocab_size
        self.special_tokens_audio_count = 463
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.audio_code_book_size = audio_code_book_size
        self.audio_embedding_path = audio_embedding_path
        self.audio_output_layer_path = audio_output_layer_path

        self.debug_mode = debug_mode

        # for debugging
        if self.debug_mode:
            self.embed_tokens_audio = self.model.embed_tokens
            self.lm_head_audio = self.lm_head
        else:
            if audio_code_book_size is not None:
                self.embed_tokens_audio = nn.Embedding(self.audio_code_book_size + self.special_tokens_audio_count, config.hidden_size)
                self.lm_head_audio = nn.Linear(config.hidden_size, self.audio_code_book_size + self.special_tokens_audio_count)

        # Initialize weights and apply final processing
        self.post_init()
    
    def load_audio_modules(self, audio_embedding_path: str=None, audio_output_layer_path: str=None):

        audio_embed_ckpt_path = audio_embedding_path if audio_embedding_path is not None else self.audio_embedding_path
        if audio_embed_ckpt_path is not None:
            audio_embed_weight = torch.load(audio_embed_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, self.config.hidden_size)
            if audio_embed_weight.shape == expected_shape:
                self.embed_tokens_audio.weight.data = audio_embed_weight.to(self.embed_tokens_audio.weight.data.dtype)
                logger.info(f"Loading audio_embed from {audio_embed_ckpt_path}, shape: {audio_embed_weight.shape}")
            else:
                logger.warning(f"audio_embed shape mismatch: expected {expected_shape}, got {audio_embed_weight.shape}")
        else:
            logger.info(f"No audio_embed_ckpt_path provided, using default initialization")
        
        audio_head_ckpt_path = audio_output_layer_path if audio_output_layer_path is not None else self.audio_output_layer_path
        if audio_head_ckpt_path is not None:
            audio_head_weight = torch.load(audio_head_ckpt_path, map_location="cpu")
            expected_shape = (self.audio_code_book_size + self.special_tokens_audio_count, self.config.hidden_size)
            if audio_head_weight.shape == expected_shape:
                self.lm_head_audio.weight.data = audio_head_weight.to(self.lm_head_audio.weight.data.dtype)
                logger.info(f"Loading audio_head weight from {audio_head_ckpt_path}, shape: {audio_head_weight.shape}")
            else:
                logger.warning(f"audio_head weight shape mismatch: expected {expected_shape}, got {audio_head_weight.shape}")
        else:
            logger.info(f"No audio_head_ckpt_path provided, using default initialization")


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


    def forward_dual_channel(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_audio: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_token_id: Optional[torch.Tensor] = None,
        ignore_token_id_audio: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_audio: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        if (input_ids is None or input_ids_audio is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of (input_ids & inputs_ids_audio) or inputs_embeds")

        if inputs_embeds is None:
            text_mask = input_ids >= 0
            if ignore_token_id is not None:
                text_mask = text_mask & (input_ids != ignore_token_id)
            text_mask = text_mask.to(torch.int32)
            input_ids = input_ids * text_mask
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * text_mask.unsqueeze(-1)

            audio_mask = input_ids_audio >= 0
            if ignore_token_id_audio is not None:
                audio_mask = audio_mask & (input_ids_audio != ignore_token_id_audio)
            audio_mask = audio_mask.to(torch.int32)
            input_ids_audio = input_ids_audio * audio_mask
            audio_embeds = self.embed_tokens_audio(input_ids_audio)
            audio_embeds = audio_embeds * audio_mask.unsqueeze(-1)
            inputs_embeds += audio_embeds
            if self.debug_mode:
                inputs_embeds *= 0.5
        
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits_audio = self.lm_head_audio(hidden_states[:, slice_indices, :])

        loss = None
        loss_audio = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            loss_audio = self.loss_function(logits=logits_audio, labels=labels_audio, vocab_size=self.config.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=[loss, loss_audio],
            aux_loss=aux_loss,
            logits=[logits, logits_audio],
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


if __name__ == "__main__":
    # debug_mode = True

    CKPT = "/path/to/iter_0105000"
    DEBUG_MODE = False
    EMBEDS_IN = True
    # EMBEDS_IN = False

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    fun_llm = FunAudioForDualChannelModel.from_pretrained(
        CKPT,
        audio_code_book_size=59049,
        audio_embedding_path=os.path.join(CKPT, "embedding_audio.pt"),
        audio_output_layer_path=os.path.join(CKPT, "output_layer_audio.pt"),
        debug_mode=DEBUG_MODE
    )
    if DEBUG_MODE:
        audio_code_book_size = tokenizer.vocab_size
    else:
        audio_code_book_size = 59049
        fun_llm.load_audio_modules()
    fun_llm.eval()
    fun_llm = fun_llm.to("cuda")

    # text = "1 + 1 ="
    text = "<|pt_mode_unimodal-text|>1 + 1 ="

    # text = tokenizer.apply_chat_template(
    #     [
    #         {
    #             "role": "user",
    #             "content": "tell me a joke"
    #         }
    #     ],
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=False
    # )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    input_ids_text = inputs["input_ids"]

    input_ids_audio = torch.ones_like(inputs["input_ids"]) * -100
    input_ids_audio[0, -1] = 0


    if EMBEDS_IN:
        # make input embeds
        inputs_embeds = fun_llm.model.embed_tokens(input_ids_text)
        audio_mask = (input_ids_audio >= 0).to(torch.int32)
        input_ids_audio = input_ids_audio * audio_mask
        audio_embeds = fun_llm.embed_tokens_audio(input_ids_audio)
        audio_embeds = audio_embeds * audio_mask.unsqueeze(-1)
        inputs_embeds += audio_embeds
    else:
        inputs_embeds = None


    output_tokens = fun_llm.generate_dual_channel(
        input_ids=None if EMBEDS_IN else input_ids_text,
        input_ids_audio=None if EMBEDS_IN else input_ids_audio,
        inputs_embeds=inputs_embeds,
        attention_mask=inputs["attention_mask"],
        tokenizer=tokenizer,
        audio_code_book_size=audio_code_book_size,
        max_new_tokens=100,
        do_sample=False,
        # do_sample=True,
        temperature=0.1,
        audio_delay=5
    )
    output_tokens_text = output_tokens[0]
    output_tokens_audio = output_tokens[1]
    if EMBEDS_IN:
        output_tokens_text = torch.cat([input_ids_text, output_tokens_text], dim=-1)
        output_tokens_audio = torch.cat([input_ids_audio, output_tokens_audio], dim=-1)
    output_tokens_audio[output_tokens_audio < 0] = 0
    output_text = tokenizer.batch_decode(output_tokens_text)[0]
    output_audio = tokenizer.batch_decode(output_tokens_audio)[0]
    print("\n===== text :\n", output_text)
    print("\n===== audio:\n", output_audio)
