"""Conservative deployment candidate: shared TP1 model, no text replay, GPU hidden I/O.
Canonical alignment is retained because text normalization changes token IDs.
The experimental Mamba prefix cache is disabled.
"""
import os,gc,time,uuid
from pathlib import Path
import prefix_gpu_entry as prefix
os.environ["SINGLE_PREFIX_CACHE"]="0"
m=prefix.m
def request(view,prompt,params,request_id=None,**kwargs):
    prompt=dict(prompt);prompt.setdefault("cache_salt",uuid.uuid4().hex)
    outputs=prefix.capture.shared.request(view,prompt,params,request_id,**kwargs)
    if view.mode=="text":
        view.engine._captured_text={"output":outputs[-1],"prompt":prompt,"states":None}
    return outputs
m.StreamingS2STBackend._request=staticmethod(request)
original_close=m.StreamingS2STBackend.close
def close(self):
    parents={};commands={}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        try:
            pid=int(path.name)
            parents[pid]=int((path/"stat").read_text().split(") ",1)[1].split()[1])
            commands[pid]=(path/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        except (OSError,ValueError):pass
    descendants={os.getpid()}
    while True:
        new=descendants|{p for p,q in parents.items() if q in descendants}
        if new==descendants:break
        descendants=new
    workers={p for p in descendants if "multiprocessing.spawn" in commands.get(p,"") or "VLLM::" in commands.get(p,"")}
    self.s2st_engine._captured_text=None
    gc.collect()
    import torch
    with torch.cuda.device(int(os.environ.get("SINGLE_GPU_PHYSICAL_ID","0"))):
        torch.cuda.synchronize();torch.cuda.ipc_collect()
    original_close(self)
    deadline=time.monotonic()+20
    while workers and time.monotonic()<deadline:
        remaining=set()
        for pid in workers:
            try:
                state=Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[0]
                if state!="Z":remaining.add(pid)
            except OSError:pass
        workers=remaining
        if workers:time.sleep(.1)
    self.s2st_engine._cuda_ipc_payloads=[]
    gc.collect()
m.StreamingS2STBackend.close=close
if __name__=="__main__":m.main()
