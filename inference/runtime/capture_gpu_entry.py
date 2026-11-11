"""Phase B instrumentation: original alignment is an oracle for captured GPU states."""
import os,json,pickle,asyncio,uuid
from pathlib import Path
os.environ.setdefault("SINGLE_GPU_EXTENSION","capture_gpu_worker.CaptureGPUWorkerExtension")
import shared_gpu_entry as shared
from capture_gpu_worker import install_capture_hooks
install_capture_hooks()
m=shared.m
import torch
def rpc(e,method,*args):
    return asyncio.run_coroutine_threadsafe(e.thinker_engine.collective_rpc(method,args=args),e._loop).result(timeout=60)[0]
original_request=shared.request
def request(view,prompt,params,request_id=None,**kwargs):
    if view.mode!="text":return original_request(view,prompt,params,request_id,**kwargs)
    e=view.engine;rpc(e,"begin_hidden_capture")
    outputs=original_request(view,prompt,params,request_id,**kwargs)
    exported=rpc(e,"export_hidden_capture")
    rebuild,args=pickle.loads(exported["ipc"])
    args=list(args);args[6]=int(os.environ.get("SINGLE_GPU_PHYSICAL_ID","0"))
    remote=rebuild(*args)
    states=remote.clone()
    torch.cuda.synchronize(states.device)
    del remote
    rpc(e,"release_hidden_capture")
    last=outputs[-1]
    e._captured_text={"states":states,"output":last,"chunks":exported["chunk_lengths"],"prompt":prompt}
    return outputs
m.StreamingS2STBackend._request=staticmethod(request)
original_align=m.CodecHistoryReplayEngine._run_global_alignment
def alignment(self,request_id,thinker_output):
    result=original_align(self,request_id,thinker_output)
    captured=getattr(self,"_captured_text",None)
    if captured and not self._global_next_write_finalize_only:
        a=captured["states"];b=result[0][:len(a)].to(a.device)
        raw_ids=list(captured["output"].outputs[0].token_ids)
        end=int(self.thinker_tokenizer.convert_tokens_to_ids("<|im_end|>"))
        while raw_ids and raw_ids[-1]==end:raw_ids.pop()
        decoded=str(captured["output"].outputs[0].text or "").replace("<|im_end|>","").strip()
        canonical=list(self.thinker_tokenizer.encode(decoded,add_special_tokens=False))
        same_shape=a.shape==b.shape
        row={"request":request_id,"chunks":captured["chunks"],"captured_shape":list(a.shape),"aligned_shape":list(result[0].shape),"raw_equals_canonical":raw_ids==canonical,"raw_token_ids":raw_ids,"canonical_token_ids":canonical,"raw_text":str(captured["output"].outputs[0].text or ""),"shape_matches_prefix":same_shape}
        if same_shape:
            diff=(a.float()-b.float()).abs()
            row.update(max_abs=float(diff.max()),mean_abs=float(diff.mean()),relative_l2=float(torch.linalg.vector_norm(a.float()-b.float())/torch.linalg.vector_norm(b.float()).clamp_min(1e-12)),exact=bool(torch.equal(a,b)))
        path=Path(os.environ["SINGLE_CAPTURE_AUDIT"]);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a") as f:f.write(json.dumps(row)+"\n")
    return result
m.CodecHistoryReplayEngine._run_global_alignment=alignment
if __name__=="__main__":m.main()
