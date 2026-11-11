import unicodedata
from collections.abc import Callable, Iterable
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoTokenizer

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader

logger = init_logger(__name__)


class GatingMixin:
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config)
        print("[CodeSwitchGate] after init")
        self.packed_modules_mapping = {
            "qkv_proj": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
            "gate_up_proj": [
                "gate_proj",
                "up_proj",
            ],
        }
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        hidden_size = getattr(config, "hidden_size", None) 
        if hidden_size is None:
            hidden_size = config.text_config.hidden_size

        intermediate_dim = hidden_size
        if envs.VLLM_CODE_SWITCH_INTERMEDIATE_DIM != 0:
            intermediate_dim = envs.VLLM_CODE_SWITCH_INTERMEDIATE_DIM

        logger.info_once(
            f"code_switch intermediate_dim = {intermediate_dim}, "
            f"config.hidden_size = {hidden_size}"
        )

        self.code_switch_pre = nn.Linear(hidden_size, 
                                         intermediate_dim, bias=False)
        self.code_switch_head = nn.Linear(intermediate_dim, 4, bias=False)
        self._lang_head = nn.Sequential(
            self.code_switch_pre,
            nn.ReLU(),
            self.code_switch_head,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)

        if not envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            return logits

        hidden_lang = self.code_switch_pre(hidden_states)
        hidden_lang = nn.functional.relu(hidden_lang)
        batch_lang_logits = self.code_switch_head(hidden_lang)

        return [logits, batch_lang_logits]

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loaded_params = super().load_weights(weights)
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["lm_head."])

        params_to_ignore = {'code_switch_head.bias', 'code_switch_pre.bias'}
        return loader.load_weights(weights).union(loaded_params).union(params_to_ignore)

class Qwen3MoeGating(GatingMixin, Qwen3MoeForCausalLM):
    pass
