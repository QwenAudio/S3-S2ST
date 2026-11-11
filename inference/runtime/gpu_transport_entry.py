"""Phase D: CUDA IPC for per-position Talker inputs, original timeline preserved."""
import base64,pickle,asyncio,inspect,textwrap,os
import direct_text_entry as direct
m=direct.m
import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.engine.shared_hidden_buffer import SharedHiddenBuffer
from vllm.v1.engine import ExternalData
PREFIX="single_gpu_cuda_ipc:"
original_read=SharedHiddenBuffer.read_global_by_request_id
def read(request_id,*args,**kwargs):
    if request_id.startswith(PREFIX):
        rebuild,metadata=pickle.loads(base64.b64decode(request_id[len(PREFIX):]))
        metadata=list(metadata);metadata[6]=torch.cuda.current_device()
        return rebuild(*metadata)
    return original_read(request_id,*args,**kwargs)
SharedHiddenBuffer.read_global_by_request_id=staticmethod(read)
original_align=m.CodecHistoryReplayEngine._run_global_alignment
def alignment(self,*args,**kwargs):
    result=original_align(self,*args,**kwargs)
    device=torch.device("cuda",int(os.environ.get("SINGLE_GPU_PHYSICAL_ID","0")))
    hidden=result[0].to(device)
    if not hasattr(self,"_gpu_codec_embedding"):
        self._gpu_codec_embedding=self._replay_codec_embedding.to(device)
    return (hidden,*result[1:])
m.CodecHistoryReplayEngine._run_global_alignment=alignment
original_send=m.CodecHistoryReplayEngine._send_global_position
source=inspect.getsource(original_send)
assert source.count(".cpu()")==1
source=source.replace("            .cpu()\n","").replace("self._replay_codec_embedding","self._gpu_codec_embedding")
ns={};exec(compile(textwrap.dedent(source),__file__,"exec"),original_send.__globals__,ns)
m.CodecHistoryReplayEngine._send_global_position=ns["_send_global_position"]
def add(self,request_id,prompt_embeds,is_padding,add_token_embedding):
    assert prompt_embeds.is_cuda and not is_padding and not add_token_embedding
    payload=prompt_embeds.detach().contiguous()
    if not hasattr(self,"_cuda_ipc_payloads"):self._cuda_ipc_payloads=[]
    self._cuda_ipc_payloads.append(payload)
    key=PREFIX+base64.b64encode(pickle.dumps(reduce_tensor(payload))).decode("ascii")
    data=ExternalData(request_id=request_id,external_embeds=None,external_embeds_shm_key=key,is_padding=False,add_token_embedding=False)
    asyncio.run_coroutine_threadsafe(self.talker_engine.update_request_with_data(data),self._loop).result(timeout=60)
m.CodecHistoryReplayEngine._add_talker_external_data=add
original_reset=m.CodecHistoryReplayEngine.reset_global_timeline
def reset(self,*args,**kwargs):
    result=original_reset(self,*args,**kwargs)
    self._cuda_ipc_payloads=[]
    return result
m.CodecHistoryReplayEngine.reset_global_timeline=reset
if __name__=="__main__":m.main()
