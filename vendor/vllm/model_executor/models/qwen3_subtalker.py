import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeDecoderLayer,
    Qwen3OmniMoePreTrainedModel,
    Qwen3OmniMoeRMSNorm,
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerCodePredictorOutputWithPast,
)
from transformers.utils import can_return_tuple

from vllm.v1.sample.metadata import SubtalkerSamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler


class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3OmniMoePreTrainedModel):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"

    def __init__(
        self, config: Qwen3OmniMoeTalkerCodePredictorConfig, embedding_dim: int = None
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3OmniMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # Use embedding_dim if provided,
        # otherwise use config.hidden_size (backward compatible)
        codec_embed_dim = (
            embedding_dim if embedding_dim is not None else config.hidden_size
        )
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, codec_embed_dim)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding

    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if past_key_values is not None:
            raise ValueError("The `past_key_values` should be `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class SubtalkerSampler(Sampler):

    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits: torch.Tensor,
        sampling_tensors: SubtalkerSamplingMetadata,
        rand_buf: torch.Tensor,
    ) -> torch.Tensor:

        logits = logits.to(torch.float32)
        # Sample the next token.
        sampled = self.sample(logits, sampling_tensors, rand_buf)

        sampled = sampled.long()
        # These are GPU tensors.
        sampled_token_ids=sampled.unsqueeze(-1)

        return sampled_token_ids
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_tensors: SubtalkerSamplingMetadata,
        rand_buf: torch.Tensor,
    ) -> torch.Tensor:

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_tensors.temperature, True)

        # Apply top_k and/or top_p.
        logits = apply_top_k_top_p(
            logits, sampling_tensors.top_k, sampling_tensors.top_p
        )
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        random_sampled = probs.div_(rand_buf).argmax(dim=-1).view(-1)

        return random_sampled

class Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(
    Qwen3OmniMoePreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(
        self, config: Qwen3OmniMoeTalkerCodePredictorConfig, talker_config=None
    ):
        super().__init__(config)
        # Determine embedding_dim from talker_config if provided
        self.num_code_groups = config.num_code_groups
        embedding_dim = talker_config.hidden_size if talker_config is not None else None
        self.model = Qwen3OmniMoeTalkerCodePredictorModel(
            config, embedding_dim=embedding_dim
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Projection layer when talker hidden_size != code_predictor hidden_size
        if (
            talker_config is not None
            and config.hidden_size != talker_config.hidden_size
        ):
            self.small_to_mtp_projection = nn.Linear(
                talker_config.hidden_size, config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
        self.position_ids = torch.arange(16).reshape(1, -1)
        self.cache_position = torch.arange(16)
        self.attention_mask = torch.ones(1, 16)
        self.sampler = SubtalkerSampler()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        else:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](
                input_ids
            )

        # Apply projection if hidden sizes differ
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_steps](hidden_states)

        return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

    @torch.inference_mode
    @torch.compile
    def generate_fix_times(
        self,
        inputs_embeds=None,
        sampling_tensors=None,
        rand_buf=None,
    ):
        output_ids = []
        all_hidden_states = []

        for generation_steps in range(self.num_code_groups - 1):
            position_ids = self.position_ids[:, : generation_steps + 2]

            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_hidden_states=True,
                position_ids=position_ids,
            )

            hidden_states = outputs.last_hidden_state
            logits = self.lm_head[generation_steps](hidden_states)

            if sampling_tensors is not None:
                input_ids = self.sampler(logits[:, -1], sampling_tensors, 
                                         rand_buf[:,generation_steps])
            else:
                input_ids = torch.argmax(logits[:, -1:], dim=-1)

            new_inputs_embeds = self.model.get_input_embeddings()[generation_steps](
                input_ids
            )
            # Apply projection if hidden sizes differ
            new_inputs_embeds = self.small_to_mtp_projection(new_inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, new_inputs_embeds], dim=1)

            output_ids.append(input_ids)
            all_hidden_states.append(outputs.hidden_states[0][:, -1:])
        output_ids = torch.cat(output_ids, dim=-1)
        all_hidden_states = torch.cat(all_hidden_states[1:], dim=1)
        return output_ids, all_hidden_states


class Qwen3TTSMoeTalkerCodePredictorModelForConditionalGeneration(
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration
):
    def __init__(
        self, config: Qwen3OmniMoeTalkerCodePredictorConfig, talker_config=None
    ):
        super().__init__(config, talker_config=talker_config)
