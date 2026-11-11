"""Capture the unchanged ten-step FM solver for the first FM8 packet."""
import s3_vocgraph_entry
common=s3_vocgraph_entry.common;torch=common.torch
init0=common.m.StreamingS2STBackend.__init__
def init(self,*a,**kw):
    init0(self,*a,**kw)
    fm=self.pcm.fm;original=fm.solve_euler;graphs={}
    def solve(x,t_span,mu,mask,spks=None,cond=None,**kwargs):
        if mu.shape[1]!=32 or not torch.is_tensor(cond):return original(x,t_span,mu,mask,spks=spks,cond=cond,**kwargs)
        values=(x,t_span,mu,mask,spks,cond)
        key=tuple((tuple(v.shape),v.dtype) for v in values)+(str(sorted(kwargs.items())),)
        if key not in graphs:
            with torch.cuda.device(mu.device),torch.inference_mode():
                buffers=tuple(v.clone() for v in values)
                def call():return original(*buffers[:4],spks=buffers[4],cond=buffers[5],**kwargs)
                stream=torch.cuda.Stream(device=mu.device);stream.wait_stream(torch.cuda.current_stream(mu.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):call()
                torch.cuda.current_stream(mu.device).wait_stream(stream)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):out=call()
                torch.cuda.current_stream(mu.device).wait_stream(stream)
                ref=original(x,t_span,mu,mask,spks=spks,cond=cond,**kwargs)
                g.replay();torch.cuda.synchronize(mu.device)
                delta=float((out-ref).abs().max());assert delta<=1e-4,('FM graph numerical mismatch',delta)
                print('FM_SOLVER_GRAPH_READY','max_abs',delta,flush=True)
                graphs[key]=(buffers,g,out)
        buffers,g,out=graphs[key]
        for b,v in zip(buffers,values):b.copy_(v)
        g.replay();return out.clone()
    fm.solve_euler=solve
common.m.StreamingS2STBackend.__init__=init
if __name__=='__main__':common.m.main()
