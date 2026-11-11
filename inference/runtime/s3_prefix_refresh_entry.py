"""Refresh source speaker at every nonempty WRITE; no prefix truncation."""
import inspect, textwrap, threading, time, json
from pathlib import Path
import s3_grid_curve_entry as curve
from pcm_runtime import SourceSpeaker
common=curve.common
PCM=common.PCM
# Keep exactly the existing feature/ONNX path, removing only the 10-second crop.
original=SourceSpeaker.extract
src=textwrap.dedent(inspect.getsource(original))
assert src.count('[:,:160000]')==1
src=src.replace('[:,:160000]','')
ns={};exec(compile(src,__file__,'exec'),original.__globals__,ns)
SourceSpeaker.extract=ns['extract']
old_begin=PCM.begin
def begin(self,*a,**kw):
    old_begin(self,*a,**kw)
    self.speaker_refreshes=[]
    self._speaker_refresh_lock=threading.RLock()
PCM.begin=begin

def fix_speaker(self):
    with self._speaker_refresh_lock:
        available=int(self.available)
        started=time.monotonic()
        emb,info=self.speaker.extract(self.wave[:available])
        assert info['samples']==available, (info['samples'],available)
        info.update(available_source_ms=available/16.,refresh_index=len(self.speaker_refreshes),wall_s=time.monotonic()-self.play.start,extraction_s=time.monotonic()-started)
        self.emb=emb.to(self.device,dtype=common.torch.float16)
        self.speaker_info=info
        self.speaker_refreshes.append(dict(info))
PCM.fix_speaker=fix_speaker
old_decode=PCM.decode
def decode(self,*a,**kw):
    with self._speaker_refresh_lock:
        return old_decode(self,*a,**kw)
PCM.decode=decode
old_finish=PCM.finish
def finish(self,*a,**kw):
    result=old_finish(self,*a,**kw)
    result['speaker_refreshes']=self.speaker_refreshes
    result['speaker_protocol']='all_consumed_prefix_each_nonempty_write_no_crop'
    assert len(self.speaker_refreshes)==len(self.text_events)
    for refresh,text in zip(self.speaker_refreshes,self.text_events):
        assert abs(refresh['available_source_ms']-text['source_ms'])<.001
    if result.get('wav'):
        (Path(result['wav']).parent/'trace.json').write_text(json.dumps(result,indent=2))
    return result
PCM.finish=finish
if __name__=='__main__':common.m.main()
