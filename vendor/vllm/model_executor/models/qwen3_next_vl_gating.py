from typing import Optional

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.models.qwen3_moe_gating import GatingMixin
from vllm.model_executor.models.qwen3_next_vl import Qwen3NextVLForConditionalGeneration


class Qwen3NextVLForConditionalGenerationGating(GatingMixin,
                                                Qwen3NextVLForConditionalGeneration):
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.language_model is None:
            return hidden_states
        
        logits =  self.language_model.compute_logits(hidden_states)

        if not envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            return logits

        hidden_lang = self.code_switch_pre(hidden_states)
        hidden_lang = nn.functional.relu(hidden_lang)
        batch_lang_logits = self.code_switch_head(hidden_lang)

        return [logits, batch_lang_logits]
