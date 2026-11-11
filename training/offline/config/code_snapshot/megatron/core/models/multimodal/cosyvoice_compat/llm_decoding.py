from contextlib import nullcontext
import torch
import torch.nn as nn
from typing import Union
import logging
from funasr.utils.hinter import hint_once
import numpy as np
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class LLMDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(LLMDecoder, self).__init__()
        self.eos_token = kwargs["eos"]
        if isinstance(self.eos_token, int):
            self.eos_token = [self.eos_token]
        self.token_embeder = kwargs["token_embeder"]
        self.ras_conf = kwargs.get("ras_conf", {})
        # self.silent_tokens = [2193, 2220, 2922, 2445, 2949, 4380, 4381, 4389, 4390, 4416, 4623, 4624, 4632, 5352, 11193]

    def nucleus_sampling(self, weighted_scores, top_p=0.8, top_k=25, beam_size=1):
        prob, indices = [], []
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        for i in range(len(sorted_idx)):
            # sampling both top-p and numbers.
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sorted_value[i]
                prob.append(sorted_value[i])
                indices.append(sorted_idx[i])
            else:
                break
        prob = torch.tensor(prob).to(weighted_scores)
        indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
        sampling_ids = prob.multinomial(beam_size, replacement=True)
        top_ids = indices[sampling_ids]
        return top_ids

    def random_sampling(self, weighted_scores, beam_size=1):
        top_ids = weighted_scores.softmax(dim=0).multinomial(beam_size, replacement=True)
        return top_ids

    # Repetition Aware Sampling in VALL-E 2
    def ras_sampling(
            self, weighted_scores, decoded_tokens, *,
            top_p=0.8, top_k=25, win_size=10, tau_r=0.1
    ):
        if self.ras_conf is not None:
            top_p = self.ras_conf.get("top_p", top_p)
            top_k = self.ras_conf.get("top_k", top_k)
            win_size = self.ras_conf.get("win_size", win_size)
            tau_r = self.ras_conf.get("tau_r", tau_r)

        hint_once(f"using Repetition Aware Sampling: top_p: {top_p}, top_k: {top_k},win_size: {win_size}, tau_r: {tau_r}", "ras_sampling")
        top_ids = self.nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(top_ids) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            top_ids = self.random_sampling(weighted_scores)

        return top_ids

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            sampling: Union[bool, int, float] = True,
            decoded_tokens: list = None,
    ):
        if isinstance(sampling, bool):
            if sampling:
                top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
            else:
                top_ids = weighted_scores.topk(1)[1]
        elif isinstance(sampling, int):
            prob, indices = weighted_scores.softmax(dim=0).topk(sampling)
            sampling_ids = prob.multinomial(1, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, float):
            prob, indices = [], []
            cum_prob = 0.0
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            for i in range(len(sorted_idx)):
                # sampling both top-p and numbers.
                if cum_prob < sampling and len(prob) < 25:
                    cum_prob += sorted_value[i]
                    prob.append(sorted_value[i])
                    indices.append(sorted_idx[i])
                else:
                    break
            prob = torch.tensor(prob).to(weighted_scores)
            indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
            sampling_ids = prob.multinomial(1, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, str) and sampling.lower() == "ras":
            top_ids = self.ras_sampling(weighted_scores, decoded_tokens=decoded_tokens)
        else:
            raise NotImplementedError(f"Not implemented for {type(sampling)} sampling")

        return top_ids

    def __call__(self, input_embeddings, llm, states, quantize=False, **kwargs):
        max_length = kwargs.get("max_length", 30 * 25)
        min_length = kwargs.get("min_length", None)
        sampling = kwargs.get("sampling", True)
        device = kwargs.get("device", "cuda")
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        use_llm_cache = kwargs.get("use_llm_cache", True)
        include_eos = kwargs.get("include_eos", False)
        custom_eos_token = kwargs.get("custom_eos_token", self.eos_token)
        # Tokens to suppress while below min_length. Defaults to every custom_eos
        # token, but streaming passes only the *real* eos here so the chunk-switch
        # fill_token stays available during the min-length window.
        min_length_token = kwargs.get("min_length_token", None)
        if min_length_token is None:
            min_length_token = custom_eos_token
        avoid_token = kwargs.get("avoid_token", None)
        return_hidden = kwargs.get("return_hidden", False)

        llm_cache = states.get("llm_cache", None)
        out_tokens, hit_eos = [], False
        out_hiddens = []
        # GDN / linear-attention models compute a multi-token step (chunked
        # prefill) differently from sequential single-token decode. When we
        # CONTINUE from a non-empty cache with several new tokens at once (e.g.
        # injecting a text chunk during streaming), that chunked path diverges
        # from the single-token decode used for codec, corrupting generation
        # after the injection. Feed those leading tokens ONE AT A TIME so every
        # post-cache step uses the (correct) single-token path; the last token is
        # left for the normal loop below. The initial prefill (empty cache) is
        # untouched, so it stays a single multi-token call. out_hiddens total
        # width is unchanged ((N-1)x[1,1,H] here + the loop's hiddens), so the
        # unified hidden/text_emb_flag bookkeeping still aligns.
        # Only needed for linear-attention (GDN) models WITH written recurrent
        # state: `has_previous_state` (Qwen3_5DynamicCache) is True iff the GDN
        # conv/recurrent state was already populated — exactly the case a
        # multi-token continuation forward would corrupt, and the same signal the
        # model itself uses (use_precomputed_states). A pure full-attention cache
        # (standard DynamicCache) has no such attribute -> getattr returns False ->
        # keep the fast single multi-token forward (it's exact there).
        if (
            use_llm_cache
            and input_embeddings.shape[1] > 1
            and getattr(llm_cache, "has_previous_state", False)
        ):
            with torch.amp.autocast(
                device_type="cuda",
                enabled=True if llm_dtype != "fp32" else False,
                dtype=dtype_map[llm_dtype],
            ) if quantize is False else nullcontext():
                _pre = input_embeddings[:, :-1]
                for _t in range(_pre.shape[1]):
                    _step = _pre[:, _t:_t + 1]
                    if (kwargs.get("use_qlora", False) or kwargs.get("infer_use_lora", False)) and (not kwargs.get("infer_lora_merged", False)):
                        _out = llm.base_model.model(
                            inputs_embeds=_step.to(torch.bfloat16) if quantize is True else _step,
                            output_hidden_states=True, return_dict=True,
                            use_cache=True, past_key_values=llm_cache,
                        )
                    else:
                        _out = llm(
                            inputs_embeds=_step.to(torch.bfloat16) if quantize is True else _step,
                            output_hidden_states=True, return_dict=True,
                            use_cache=True, past_key_values=llm_cache,
                        )
                    llm_cache = _out.past_key_values
                    if return_hidden:
                        out_hiddens.append(_out.hidden_states[-1])
            input_embeddings = input_embeddings[:, -1:]
        # num_silence = 0
        for i in range(max_length):
            with torch.amp.autocast(
                device_type="cuda",
                enabled=True if llm_dtype != "fp32" else False, 
                dtype=dtype_map[llm_dtype]
            ) if quantize is False else nullcontext():
                # default attention_mask is causal, no longer need manually construct
                # input_masks = torch.ones((1, input_embeddings.shape[1]), device=input_embeddings.device).to(torch.bool)
                
                if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
                    outputs = llm.base_model.model(
                        inputs_embeds=input_embeddings.to(torch.bfloat16) if quantize is True else input_embeddings,
                        # attention_mask=input_masks,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=use_llm_cache,
                        past_key_values=llm_cache,
                    )
                else:
                    outputs = llm(
                        inputs_embeds=input_embeddings.to(torch.bfloat16) if quantize is True else input_embeddings,
                        # attention_mask=input_masks,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=use_llm_cache,
                        past_key_values=llm_cache,
                    )
                lm_hidden_states = outputs.hidden_states[-1]
                h = llm.lm_head(lm_hidden_states[:, -1])
                # logp = h.log_softmax(dim=-1).squeeze(0)
                logp = h.squeeze(0)
                if use_llm_cache:
                    llm_cache = outputs.past_key_values

                pred = torch.log_softmax(logp, dim=-1)
            if min_length is not None and i < min_length:
                for x in min_length_token:
                    if pred.dtype == torch.bfloat16:
                        pred[x] = float(np.finfo(np.float16).min)
                    else:
                        pred[x] = float(np.finfo(np.float32).min)
            # if num_silence >= 3:
            #     avoid_token = self.silent_tokens
            # else:
            #     avoid_token = None
            if avoid_token is not None and len(avoid_token) > 0:
                for x in avoid_token:
                    if pred.dtype == torch.bfloat16:
                        pred[x] = float(np.finfo(np.float16).min)
                    else:
                        pred[x] = float(np.finfo(np.float32).min)
            top_id = self.sampling_ids(pred, sampling, out_tokens)[0].item()
            # if top_id in self.silent_tokens:
            #     num_silence += 1
            # else:
            #     num_silence = 0

            if top_id in custom_eos_token:
                if include_eos:
                    out_tokens.append(top_id)
                hit_eos = True
                break

            out_tokens.append(top_id)
            if return_hidden:
                out_hiddens.append(lm_hidden_states)
            if use_llm_cache:
                input_embeddings = self.token_embeder(torch.tensor([[top_id]], dtype=torch.int64, device=device))
            else:
                input_embeddings = torch.cat([
                    input_embeddings,
                    self.token_embeder(torch.tensor([[top_id]], dtype=torch.int64, device=device))
                ], dim=1)

        out_tokens = torch.tensor([out_tokens], dtype=torch.int64, device=device)

        states = {"llm_cache": llm_cache}
        
        if return_hidden:
            return out_tokens, hit_eos, states, out_hiddens

        return out_tokens, hit_eos, states


class GroupLLMDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(GroupLLMDecoder, self).__init__()
        self.sos_token = kwargs["sos"]
        self.eos_token = kwargs["eos"]
        self.turn_of_speech = kwargs["turn_of_speech"]
        self.ras_conf = kwargs.get("ras_conf", {})
        self.group_size = kwargs.get("group_size", 1)

    def nucleus_sampling(self, weighted_scores, top_p=0.8, top_k=25, beam_size=1):
        prob, indices = [], []
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        for i in range(len(sorted_idx)):
            # sampling both top-p and numbers.
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sorted_value[i]
                prob.append(sorted_value[i])
                indices.append(sorted_idx[i])
            else:
                break
        prob = torch.tensor(prob).to(weighted_scores)
        indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
        sampling_ids = prob.multinomial(beam_size, replacement=True)
        top_ids = indices[sampling_ids]
        return top_ids

    def random_sampling(self, weighted_scores, beam_size=1):
        top_ids = weighted_scores.softmax(dim=0).multinomial(beam_size, replacement=True)
        return top_ids

    # Repetition Aware Sampling in VALL-E 2
    def ras_sampling(
            self, weighted_scores, decoded_tokens, *,
            top_p=0.8, top_k=25, win_size=10, tau_r=0.1
    ):
        if self.ras_conf is not None:
            top_p = self.ras_conf.get("top_p", top_p)
            top_k = self.ras_conf.get("top_k", top_k)
            win_size = self.ras_conf.get("win_size", win_size)
            tau_r = self.ras_conf.get("tau_r", tau_r)

        hint_once(f"using Repetition Aware Sampling: top_p: {top_p}, top_k: {top_k},win_size: {win_size}, tau_r: {tau_r}", "ras_sampling")
        top_ids = self.nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(top_ids) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            top_ids = self.random_sampling(weighted_scores)

        return top_ids

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            sampling: Union[bool, int, float] = True,
            decoded_tokens: list = None,
    ):
        if isinstance(sampling, bool):
            if sampling:
                top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
            else:
                top_ids = weighted_scores.topk(1)[1]
        elif isinstance(sampling, int):
            prob, indices = weighted_scores.softmax(dim=0).topk(sampling)
            sampling_ids = prob.multinomial(1, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, float):
            prob, indices = [], []
            cum_prob = 0.0
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            for i in range(len(sorted_idx)):
                # sampling both top-p and numbers.
                if cum_prob < sampling and len(prob) < 25:
                    cum_prob += sorted_value[i]
                    prob.append(sorted_value[i])
                    indices.append(sorted_idx[i])
                else:
                    break
            prob = torch.tensor(prob).to(weighted_scores)
            indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
            sampling_ids = prob.multinomial(1, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, str) and sampling.lower() == "ras":
            top_ids = self.ras_sampling(weighted_scores, decoded_tokens=decoded_tokens)
        else:
            raise NotImplementedError(f"Not implemented for {type(sampling)} sampling")

        return top_ids

    def __call__(
            self, 
            all_text_ids,
            prompt_codec, 
            llm,
            codec_embedder,
            group_encoder,
            group_decoder,
            group_decoder_lm_head,
            group_encoder_downcast,
            hidden_states_downcast,
            hidden_states_upcast,
            states,
            **kwargs
        ):
        max_length = kwargs.get("max_length", 30 * 25)
        min_length = kwargs.get("min_length", None)
        sampling = kwargs.get("sampling", True)
        device = kwargs.get("device", "cuda")
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        use_llm_cache = kwargs.get("use_llm_cache", True)
        include_eos = kwargs.get("include_eos", False)
        custom_eos_token = kwargs.get("custom_eos_token", [self.eos_token])
        avoid_token = kwargs.get("avoid_token", None)

        llm_cache = states.get("llm_cache", None)
        out_tokens, hit_eos = [], False

        all_text_embed = llm.model.get_input_embeddings()(all_text_ids)
        sos_token = torch.tensor([self.sos_token], dtype=torch.int64, device=device)
        sos_embed = hidden_states_upcast(codec_embedder(sos_token)).unsqueeze(0)
        turn_of_speech = torch.tensor([self.turn_of_speech], dtype=torch.int64, device=device)
        turn_of_speech_embed = hidden_states_upcast(codec_embedder(turn_of_speech)).unsqueeze(0)

        codec_len = prompt_codec.shape[1]
        group_remain_len = codec_len % self.group_size
        group_len = codec_len // self.group_size

        input_codec = prompt_codec[:, :codec_len - group_remain_len]

        for i in range(max_length // self.group_size):
            input_codec_embed = codec_embedder(input_codec)
            input_codec_embed = input_codec_embed.reshape(group_len, self.group_size, -1)

            output = group_encoder(
                inputs_embeds=input_codec_embed,
                return_dict=True)
            input_codec_embed = output.last_hidden_state
            input_codec_embed = input_codec_embed.reshape(1, group_len, -1)
            input_codec_embed = group_encoder_downcast(input_codec_embed)

            if i == 0:
                llm_input_embed = torch.cat([sos_embed, all_text_embed, turn_of_speech_embed, input_codec_embed], dim=1)
            else:
                llm_input_embed = input_codec_embed

            output = llm(
                inputs_embeds=llm_input_embed,
                # attention_mask=input_masks,
                output_hidden_states=True,
                return_dict=True,
                use_cache=use_llm_cache,
                past_key_values=llm_cache,
            )
            lm_hidden_states = output.hidden_states[-1]
            if use_llm_cache:
                llm_cache = output.past_key_values

            local_input_embed = lm_hidden_states[:, -1:]
            local_input_embed = hidden_states_downcast(local_input_embed)
            if i == 0:
                group_remain_embed = codec_embedder(prompt_codec[:, codec_len - group_remain_len:])
                local_input_embed = torch.cat([local_input_embed, group_remain_embed], dim=1)
            else:
                group_remain_len = 0
            local_llm_cache = None
            for _ in range(self.group_size - group_remain_len):
                local_output = group_decoder(
                    inputs_embeds=local_input_embed,
                    return_dict=True,
                    use_cache=use_llm_cache,
                    past_key_values=local_llm_cache,)
                local_lm_hidden = local_output.last_hidden_state[:, -1]
                h = group_decoder_lm_head(local_lm_hidden)
                logp = h.squeeze(0)
                if use_llm_cache:
                    local_llm_cache = local_output.past_key_values

                pred = torch.log_softmax(logp, dim=-1)

                if min_length is not None and len(out_tokens) < min_length:
                    for x in custom_eos_token:
                        if pred.dtype == torch.bfloat16:
                            pred[x] = float(np.finfo(np.float16).min)
                        else:
                            pred[x] = float(np.finfo(np.float32).min)
                if avoid_token is not None and len(avoid_token) > 0:
                    for x in avoid_token:
                        if pred.dtype == torch.bfloat16:
                            pred[x] = float(np.finfo(np.float16).min)
                        else:
                            pred[x] = float(np.finfo(np.float32).min)
                top_id = self.sampling_ids(pred, sampling, out_tokens)[0].item()

                if top_id in custom_eos_token:
                    if include_eos:
                        out_tokens.append(top_id)
                    hit_eos = True
                    break

                out_tokens.append(top_id)
                if use_llm_cache:
                    local_input_embed = codec_embedder(torch.tensor([[top_id]], dtype=torch.int64, device=device))
                else:
                    local_input_embed = torch.cat([
                        local_input_embed,
                        codec_embedder(torch.tensor([[top_id]], dtype=torch.int64, device=device))
                    ], dim=1)

            if hit_eos:
                break

            input_codec = torch.tensor(out_tokens[-self.group_size:], dtype=torch.int64, device=device).unsqueeze(0)
            if input_codec.shape[1] < self.group_size:
                input_codec = torch.cat([prompt_codec[:, input_codec.shape[1]-self.group_size:], input_codec], dim=1)
            group_len = 1

        out_tokens = torch.tensor([out_tokens], dtype=torch.int64, device=device)

        states = {"llm_cache": llm_cache}

        return out_tokens, hit_eos, states
