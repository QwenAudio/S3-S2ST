from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch import Tensor, nn
import torch.nn.functional as F

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class QwenAudioEncoder(nn.Module):
    def __init__(
        self,
        audio_encoder,
        llm_dtype,
        input_size: int = 128,
        linear_units: int = 1280,
        output_size: int = 1536,
        fix_layer: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.fix_layer = fix_layer
        self.conv1 = Conv1d(input_size, linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(linear_units, output_size, kernel_size=3, stride=2, padding=1)

        if audio_encoder.endswith("json"):
            config = AutoConfig.from_pretrained(audio_encoder)
            self.encoder = AutoModelForCausalLM.from_config(
                config,
                attn_implementation="flash_attention_2",
            ).model
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                audio_encoder,
                attn_implementation="flash_attention_2",
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
            ).model
        del self.encoder.embed_tokens

        for block in self.encoder.layers:
            block.self_attn.is_causal = False

        self.downsample_rate = kwargs.get("downsample_rate", 4)
        self.llm_dtype = llm_dtype
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        **kwargs,
    ):
        # print(self.kwargs.get("ctc_intermiddle_layer", -1))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        inputs_embeds = x.permute(0, 2, 1)

        n_frames = inputs_embeds.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.encoder.rotary_emb(hidden_states, position_ids)
        with torch.amp.autocast('cuda',
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            ctc_intermiddle_layer = None
            for i, decoder_layer in enumerate(self.encoder.layers):
                layer_outputs = decoder_layer(
                    hidden_states,
                     attention_mask=None,
                     position_ids=position_ids,
                     past_key_value=None,
                     output_attentions=False,
                     use_cache=False,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                )

                hidden_states = layer_outputs
                if i == self.fix_layer:
                    hidden_states = hidden_states.detach()
                if i == kwargs.get("ctc_intermiddle_layer", -1):
                    ctc_intermiddle_layer = hidden_states

        hidden_states = self.encoder.norm(hidden_states)
        
        if ctc_intermiddle_layer is None:
            return hidden_states, olens
        else:
            return (hidden_states, ctc_intermiddle_layer), olens
