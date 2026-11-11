"""Same vocoder weights; cache a CUDA graph for the fixed first FM8 packet."""
import time
import s3_prefix_refresh_entry as refresh
common=refresh.common;torch=common.torch
init0=common.m.StreamingS2STBackend.__init__
def init(self,*a,**kw):
    init0(self,*a,**kw)
    pcm=self.pcm;gen=pcm.voc.generator
    timed=gen.inference_cache
    original=timed.__kwdefaults__['_fn']
    gen.remove_weight_norm()
    graph_state={}
    def clone_tree(x):
        if torch.is_tensor(x):return x.clone()
        if isinstance(x,dict):return {k:clone_tree(v) for k,v in x.items()}
        if isinstance(x,list):return [clone_tree(v) for v in x]
        if isinstance(x,tuple):return tuple(clone_tree(v) for v in x)
        return x
    def graph_call(feat,finalize=True,cache=None):
        if finalize or cache:return original(feat,finalize=finalize,cache=cache or {})
        key=(tuple(feat.shape),feat.dtype)
        if key not in graph_state:
            with torch.cuda.device(pcm.device),torch.inference_mode():
                buf=feat.clone();stream=torch.cuda.Stream(device=pcm.device)
                stream.wait_stream(torch.cuda.current_stream(pcm.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):original(buf,finalize=False,cache={})
                torch.cuda.current_stream(pcm.device).wait_stream(stream)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):result=original(buf,finalize=False,cache={})
                torch.cuda.current_stream(pcm.device).wait_stream(stream)
                ref=original(feat,finalize=False,cache={})[0]
                g.replay();torch.cuda.synchronize(pcm.device)
                delta=float((result[0]-ref).abs().max())
                assert delta<=1e-5,('graph numerical mismatch',delta)
                graph_state[key]=(buf,g,result)
                print('VOCODER_GRAPH_READY',key,'max_abs',delta,flush=True)
        buf,g,result=graph_state[key];buf.copy_(feat);g.replay()
        return clone_tree(result)
    def wrapped(feat,finalize=True,cache=None):
        torch.cuda.synchronize(pcm.device);t=time.monotonic()
        r=graph_call(feat,finalize=finalize,cache=cache)
        torch.cuda.synchronize(pcm.device)
        common.record(pcm,'vocoder',seconds=time.monotonic()-t,finalize=bool(finalize),cuda_graph=not finalize and not bool(cache))
        return r
    gen.inference_cache=wrapped
common.m.StreamingS2STBackend.__init__=init
if __name__=='__main__':common.m.main()
