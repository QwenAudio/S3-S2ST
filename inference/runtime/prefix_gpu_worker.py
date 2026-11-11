"""GPU-only hidden capture for cached canonical continuation."""
from capture_gpu_worker import CaptureGPUWorkerExtension
class PrefixGPUWorkerExtension(CaptureGPUWorkerExtension):
    def set_single_gpu_mode(self,mode):
        result=super().set_single_gpu_mode(mode)
        if not getattr(self,"_prefix_handler_installed",False):
            import types
            worker=self
            def handle(runner,hidden,scheduler_output,num_scheduled_tokens_np,skip_decode_hidden_states=False):
                if not getattr(worker,"_prefix_align_active",False):return
                assert runner.input_batch.num_reqs==1
                start=int(runner.input_batch.num_computed_tokens_cpu[0])
                count=int(num_scheduled_tokens_np.sum())
                prompt=int(runner.input_batch.num_prompt_tokens[0])
                worker._prefix_align_chunks.append((start,prompt,hidden[:count].detach().clone()))
            self.model_runner._handle_accept_hidden_states=types.MethodType(handle,self.model_runner)
            self._prefix_handler_installed=True
        return result
    def begin_cached_alignment(self):
        self._prefix_align_active=True;self._prefix_align_chunks=[]
        return True
    def export_cached_alignment(self):
        import torch,pickle
        from torch.multiprocessing.reductions import reduce_tensor
        self._prefix_align_active=False
        chunks=self._prefix_align_chunks;assert chunks
        start=chunks[0][0];prompt=chunks[0][1];cursor=start
        for pos,n,h in chunks:
            assert pos==cursor and n==prompt
            cursor+=len(h)
        assert cursor==prompt,(start,cursor,prompt)
        self._prefix_align_export=torch.cat([x[2] for x in chunks],dim=0).contiguous()
        torch.cuda.synchronize()
        return {"start":start,"prompt":prompt,"ipc":pickle.dumps(reduce_tensor(self._prefix_align_export))}
    def release_cached_alignment(self):
        self._prefix_align_export=None;self._prefix_align_chunks=[]
        return True
