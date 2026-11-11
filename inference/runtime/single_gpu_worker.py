"""Worker RPC and model hooks for one shared Thinker, entirely experiment-local."""
import os
class SingleGPUWorkerExtension:
    def set_single_gpu_mode(self, mode):
        assert mode in ("gate","text")
        model=self.model_runner.model
        model._single_gpu_mode=mode
        return {"mode":mode,"has_gate":model.stream_gate is not None}
def install_model_hooks():
    from vllm.model_executor.models.omnidrca2_next_thinker import OmniDRCA2NextForConditionalGeneration as Model
    if getattr(Model,"_single_gpu_hooked",False):return
    original=Model.compute_logits
    def compute_logits(self,hidden_states):
        if getattr(self,"_single_gpu_mode","text")=="gate":
            assert self.stream_gate is not None
            return original(self,hidden_states)
        return self.language_model.compute_logits(hidden_states)
    Model.compute_logits=compute_logits
    Model._single_gpu_hooked=True
