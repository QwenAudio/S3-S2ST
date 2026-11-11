"""Phase A: a single shared Thinker for gate, text and alignment; original codec schedule."""
import os,sys,inspect,textwrap,asyncio,uuid
from pathlib import Path
import infer_sampling as adapter
from single_gpu_worker import install_model_hooks
install_model_hooks()
m=adapter.entry.s2st
class RequestView:
    def __init__(self,engine,mode):self.engine=engine;self.mode=mode
    def shutdown(self):pass
def request(engine,prompt,params,request_id=None,**kwargs):
    assert isinstance(engine,RequestView)
    e=engine.engine
    result=asyncio.run_coroutine_threadsafe(e.thinker_engine.collective_rpc("set_single_gpu_mode",args=(engine.mode,)),e._loop).result(timeout=60)
    assert result[0]["has_gate"]
    generator=e.thinker_engine.generate(request_id=request_id or "shared_"+uuid.uuid4().hex,prompt=prompt,sampling_params=params,using_external_data=False)
    outputs=list(m.SynchronizedGenerator(generator,e._loop))
    assert outputs and outputs[-1].finished
    return outputs
original_init=m.StreamingS2STBackend.__init__
source=inspect.getsource(original_init)
start=source.index("        # Gate classification")
end=source.index("        talker_args =")
source=source[:start]+'''        os.environ["VLLM_STREAM_GATE_PATH"] = str(Path(args.gate).resolve())
        os.environ["VLLM_STREAM_GATE_THRESHOLD"] = str(args.threshold)
'''+source[end:]
source=source.replace("enable_prefix_caching=False", "enable_prefix_caching=os.environ.get(\"SINGLE_PREFIX_CACHE\")==\"1\"",1)
source=source.replace("enable_chunked_prefill=False", "enable_chunked_prefill=os.environ.get(\"SINGLE_PREFIX_CACHE\")==\"1\"",1)
needle="            enable_hidden_state_caching=False,\n"
assert source.count(needle)==1
source=source.replace(needle,needle+'            worker_extension_cls=os.environ.get("SINGLE_GPU_EXTENSION", "single_gpu_worker.SingleGPUWorkerExtension"),\n')
needle='        # The installed aligned-Talker engine'
assert source.count(needle)==1
source=source.replace(needle,'''        self.gate_engine = RequestView(self.s2st_engine, "gate")
        self.text_engine = RequestView(self.s2st_engine, "text")
        os.environ.pop("VLLM_STREAM_GATE_PATH", None)
        os.environ.pop("VLLM_STREAM_GATE_THRESHOLD", None)
'''+needle)
namespace={}
context=original_init.__globals__.copy();context["RequestView"]=RequestView
exec(compile(textwrap.dedent(source),__file__,"exec"),context,namespace)
m.StreamingS2STBackend.__init__=namespace["__init__"]
m.StreamingS2STBackend._request=staticmethod(request)
original_speech=m.StreamingS2STBackend.generate_forced_speech
def speech(self,*args,**kwargs):
    e=self.s2st_engine
    asyncio.run_coroutine_threadsafe(e.thinker_engine.collective_rpc("set_single_gpu_mode",args=("text",)),e._loop).result(timeout=60)
    return original_speech(self,*args,**kwargs)
m.StreamingS2STBackend.generate_forced_speech=speech
if __name__=="__main__":
    print("SHARED_THINKER_PHASE_A: one thinker+gate, one talker; original alignment retained as control",flush=True)
    m.main()
