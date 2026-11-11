"""Reuse only exact token prefixes from the current WRITE, with GPU-resident states."""
import os,uuid,copy,inspect,textwrap,pickle,json
os.environ["SINGLE_PREFIX_CACHE"]="1"
os.environ["SINGLE_GPU_EXTENSION"]="prefix_gpu_worker.PrefixGPUWorkerExtension"
import gpu_transport_entry as transport
m=transport.m
capture=transport.direct.capture
import torch
original_request=m.StreamingS2STBackend._request
def request(view,prompt,params,request_id=None,**kwargs):
    prompt=dict(prompt)
    if view.mode=="text":prompt["cache_salt"]=uuid.uuid4().hex
    params=copy.copy(params);params.skip_reading_prefix_cache=True
    return original_request(view,prompt,params,request_id,**kwargs)
m.StreamingS2STBackend._request=staticmethod(request)
def get_cached_hidden(engine,align_prompt,generator):
    captured=getattr(engine,"_captured_text",None)
    capture.rpc(engine,"begin_cached_alignment")
    actual_prompt_ids=None
    for output in m.SynchronizedGenerator(generator,engine._loop):
        actual_prompt_ids=list(output.prompt_token_ids)
        if output.finished:break
    meta=capture.rpc(engine,"export_cached_alignment")
    rebuild,args=pickle.loads(meta["ipc"]);args=list(args)
    args[6]=int(os.environ.get("SINGLE_GPU_PHYSICAL_ID","0"))
    shared=rebuild(*args);tail=shared.clone();torch.cuda.synchronize(tail.device)
    del shared
    capture.rpc(engine,"release_cached_alignment")
    start=meta["start"]
    common=0
    if start:
        assert captured is not None
        old=list(captured["output"].prompt_token_ids)+list(captured["output"].outputs[0].token_ids)
        new=actual_prompt_ids
        for x,y in zip(old,new):
            if x!=y:break
            common+=1
        assert start<=common and start<=len(captured["states"]),(start,common,len(captured["states"]))
        hidden=torch.cat([captured["states"][:start],tail],dim=0)
    else:hidden=tail
    assert len(hidden)==meta["prompt"]==len(actual_prompt_ids),(len(hidden),meta["prompt"],len(actual_prompt_ids))
    if not hasattr(engine,"_gpu_codec_embedding"):
        engine._gpu_codec_embedding=engine._replay_codec_embedding.to(hidden.device)
    path=os.environ["SINGLE_CAPTURE_AUDIT"]
    with open(path,"a") as f:f.write(json.dumps({"cached_prefix":start,"recomputed_positions":len(tail),"prompt_positions":len(hidden),"verified_common_prefix":common})+"\n")
    return hidden
original=capture.original_align
source=inspect.getsource(original)
anchor='        align_request_id ='
source=source.replace(anchor,'''        captured = getattr(self, "_captured_text", None)
        if captured is not None and not finalize_only:
            align_prompt["cache_salt"] = captured["prompt"]["cache_salt"]
        else:
            align_prompt["cache_salt"] = uuid.uuid4().hex
'''+anchor)
start=source.index('        hidden = None')
end=source.index('        target_len =',start)
source=source[:start]+'''        hidden = get_cached_hidden(self, align_prompt, generator)
'''+source[end:]
scope=original.__globals__.copy();scope.update(get_cached_hidden=get_cached_hidden,uuid=uuid)
ns={};exec(compile(textwrap.dedent(source),__file__,"exec"),scope,ns)
m.CodecHistoryReplayEngine._run_global_alignment=ns["_run_global_alignment"]
if __name__=="__main__":m.main()
