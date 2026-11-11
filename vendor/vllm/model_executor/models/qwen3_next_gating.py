from vllm.model_executor.models.qwen3_moe_gating import GatingMixin
from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM


class Qwen3NextGating(GatingMixin, Qwen3NextForCausalLM):
    pass
