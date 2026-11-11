"""Bounded no-EOS output is a sample failure, not a process failure."""
import numpy as np

def generate_sample(backend,prompt,segments,text):
 try:
  backend.generate_forced_speech(prompt,segments,text)
  codec,annotations=backend.finish_sample()
  return codec,annotations,dict(status='ok',eos_observed=True,truncated=False,error=None)
 except RuntimeError as exc:
  if 'Talker request ended before the final WRITE produced codec EOS' not in str(exc):raise
  control=np.asarray(backend.s2st_engine._global_codec_tokens,dtype=np.int64).reshape(-1).copy()
  if len(control)<2 or control[0]!=backend.codec_bos_id:raise
  codec=control[1:]
  if np.any(codec<0) or np.any(codec>=59049):raise
  # Do not fabricate EOS or mark the truncated sample successful.
  backend._global_codec_full=control
  backend._close_consumed_writer_shm_handles()
  annotations=[dict(text=text,decoded_text=text,global_final_write=True,codec_len=len(codec),truncated=True,eos_observed=False)]
  return codec,annotations,dict(status='no_eos',eos_observed=False,truncated=True,error=str(exc))

def summarize(rows):
 failed=sum(row.get('generation_status','ok')!='ok' for row in rows)
 return dict(total=len(rows),failed=failed,no_eos=sum(row.get('generation_status')=='no_eos' for row in rows),failure_rate=failed/len(rows) if rows else 0.0,metrics_include_truncated_audio=True)
