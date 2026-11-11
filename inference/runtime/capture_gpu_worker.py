"""Capture projected Thinker states on-device and export CUDA IPC metadata."""
from single_gpu_worker import SingleGPUWorkerExtension,install_model_hooks
class CaptureGPUWorkerExtension(SingleGPUWorkerExtension):
    def begin_hidden_capture(self):
        model=self.model_runner.model
        model._capture_chunks=[]
        model._capture_enabled=True
        return True
    def export_hidden_capture(self):
        import torch,pickle
        from torch.multiprocessing.reductions import reduce_tensor
        model=self.model_runner.model
        model._capture_enabled=False
        assert model._capture_chunks
        model._capture_export=torch.cat(model._capture_chunks,dim=0).contiguous()
        torch.cuda.synchronize()
        payload=pickle.dumps(reduce_tensor(model._capture_export))
        return {"ipc":payload,"shape":list(model._capture_export.shape),"chunk_lengths":[int(x.shape[0]) for x in model._capture_chunks]}
    def release_hidden_capture(self):
        model=self.model_runner.model
        model._capture_export=None;model._capture_chunks=[]
        return True
def install_capture_hooks():
    install_model_hooks()
    from vllm.model_executor.models.omnidrca2_next_thinker import OmniDRCA2NextForConditionalGeneration as Model
    if getattr(Model,"_capture_hooked",False):return
    original=Model.forward
    def forward(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if getattr(self,"_capture_enabled",False):
            assert isinstance(result,tuple) and len(result)==2
            self._capture_chunks.append(result[1].detach().clone())
        return result
    Model.forward=forward;Model._capture_hooked=True
