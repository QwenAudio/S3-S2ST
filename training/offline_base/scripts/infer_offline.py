#!/usr/bin/env python3
"""Offline full-source S2ST using the audited explicit codec-feedback engine."""
import argparse, json, os, sys, hashlib, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'vendor/streaming_eval'
sys.path.insert(0,str(CODE))
# Resolve before importing model overrides, including inside spawned workers.
_model=os.environ.get('OMNIDRCA2_MODEL_DIR')
if '--model' in sys.argv:_model=sys.argv[sys.argv.index('--model')+1]
_fusion=None
if _model:
 os.environ['OMNIDRCA2_MODEL_DIR']=_model
 from profiles.layerfusion.layer_fusion_assets import resolve
 _fusion=resolve(_model,mode='auto',profile='continuous')
 os.environ['OFFLINE_LAYER_FUSION']='1' if _fusion['enabled'] else '0'
# Spawned vLLM workers re-import this module: register overrides there too.
import infer_deferred_s2st as audited
from terminal_delivery import install as _install_terminal_delivery
_install_terminal_delivery()

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
 p.add_argument('--text-gpu',type=int,default=0);p.add_argument('--thinker-gpu',type=int,default=1);p.add_argument('--talker-gpu',type=int,default=2)
 p.add_argument('--encoder-device',default='cuda:0');p.add_argument('--max-new-tokens',type=int,default=512);p.add_argument('--max-codec-tokens',type=int,default=1500)
 p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=0)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 os.environ['OMNIDRCA2_MODEL_DIR']=a.model
 os.environ['CODEX_AUDIO_BLANK_TOKEN_COUNT']='5';os.environ['CODEX_ALIGNMENT_MASK_TOKEN_ID']='-60'
 os.environ['CODEX_TALKER_MAX_CODEC_TEXT_RATIO']='0'
 os.environ.pop('VLLM_STREAM_GATE_PATH',None)
 import numpy as np
 import torch
 import infer_deferred_s2st as audited
 from transformers import AutoTokenizer
 from vllm.v1.engine.async_llm import AsyncEngineArgs
 from forced_sequence_logits import ForcedSequenceLogitsProcessor
 # Full-record offline training uses utterance-max Whisper normalization,
 # NOT the fixed-max streaming frontend. Both encoder and adaptor are noncausal.
 encoder=audited.s2tt.FullRecordEncoder(Path(a.model),a.encoder_device,true_streaming=False)
 encoder.processor._frontend=audited.s2tt.VllmWhisperFrontend(fs=16000,n_mels=128,do_pad_trim=False,filters_path=audited.s2tt.MEL_FILTERS_PATH)
 class OfflineBackend(audited.StreamingS2STBackend):
  def __init__(self):
   self.args=a;self.model_dir=Path(a.model)
   self.tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
   self.im_end_id=int(self.tokenizer.convert_tokens_to_ids('<|im_end|>'))
   common=dict(model=a.model,trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
      distributed_executor_backend='mp',enforce_eager=True,enable_prefix_caching=False,
      enable_chunked_prefill=False,max_model_len=8192,max_num_seqs=1,block_size=16)
   mm=dict(limit_mm_per_prompt={'audio':1},enable_mm_embeds=True,skip_mm_profiling=True,enable_hidden_state_caching=False)
   self.text_engine=audited.OmniDRCA2Engine(AsyncEngineArgs(**common,**mm,gpu_memory_utilization=.55),thinker_visible_devices=[a.text_gpu])
   self.s2st_engine=audited.CodecHistoryReplayEngine(
     AsyncEngineArgs(**common,**mm,gpu_memory_utilization=.55,logits_processors=[ForcedSequenceLogitsProcessor]),
     AsyncEngineArgs(**common,gpu_memory_utilization=.25,enable_prompt_embeds=True),
     thinker_visible_devices=[a.thinker_gpu],talker_visible_devices=[a.talker_gpu],codec_embed_path=self.model_dir/'cosyvoice_codec_embed.pt')
   self.codec_eos_id=59050;self.codec_bos_id=59049;self.codec_final_bos_id=59049;self.codec_pause_id=59051
   self.codec_boundary_policy='unified_eos';self.codec_envelope=True;self.final_eos_only=False;self.codec_history_replay=False
   self.codec_blank_count=5;self.codec_alignment_mask_id=-60
   self.audio_pad_id=int(self.tokenizer.convert_tokens_to_ids('<|audio_pad|>'))
   self.s2st_engine.enable_global_timeline()
   self.s2st_engine.global_codec_final_bos_id=59049
   self.s2st_engine.global_boundary_policy='unified_eos'
   self.start_sample()
  def close(self):
   self._close_consumed_writer_shm_handles();self.text_engine.shutdown();self.s2st_engine.shutdown()
 a.codec_temperature=.8;a.codec_top_p=.8;a.codec_top_k=5
 rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
 if a.limit:rows=rows[:a.limit]
 manifest={'layer_fusion':_fusion,'mode':'offline_full_source','frontend':'utterance_max','streaming_encoder':False,'gate':False,
  'model':str(Path(a.model).resolve()),'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
  'expected':len(rows),'sampling':{'thinker':'greedy','temperature':.8,'top_p':.8,'top_k':5,'seed':a.seed},
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/terminal_delivery.py',ROOT/'scripts/generation_policy.py',ROOT/'scripts/offline_audio.py',ROOT/'scripts/offline_io.py',ROOT/'scripts/audio_io_checks.py',*CODE.glob('*.py'),*CODE.glob('vllm_overrides/*.py'),*CODE.glob('profiles/layerfusion/*.py')]}}
 if (out/'RUN_MANIFEST.json').exists():
  prior=json.loads((out/'RUN_MANIFEST.json').read_text())
  assert prior['model']==manifest['model'] and prior['input_sha256']==manifest['input_sha256'], 'Cannot resume different model/input'
  with (out/'PRIOR_MANIFESTS.jsonl').open('a') as f:f.write(json.dumps(prior)+'\n')
 manifest['generation_failure_policy']='no-EOS: record failure and decode bounded truncated audio; include in metrics'
 (out/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 (out/'input.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 (out/'codec').mkdir(exist_ok=True);backend=None
 from generation_policy import generate_sample,summarize
 completed=[json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines() if x.strip()] if (out/'raw.jsonl').exists() else []
 assert [x['index'] for x in completed]==list(range(len(completed))) and len(completed)<=len(rows)
 assert all(Path(x['codec_path']).is_file() and Path(x['codec_control_path']).is_file() for x in completed)

 try:
  backend=OfflineBackend()
  with (out/'raw.jsonl').open('a') as f:
   for i,obj in enumerate(rows):
    if i<len(completed):continue
    wav_path,task,source,reference=audited.s2tt.load_row(obj)
    from offline_audio import load_source
    user=next(m for m in obj['messages'] if m['role']=='user')
    wav=load_source(wav_path,duration=(user.get('start_time',0),user.get('end_time')))
    embeds=encoder(wav)
    prompt,segments=audited.s2tt.build_prompt(audited.s2tt.SYSTEM,task,embeds,[],open_for_write=True)
    backend.start_sample();backend.set_write_context(final_write=True)
    text=backend.generate_text(prompt,segments)
    if not text:raise RuntimeError('Empty Thinker output')
    codec,annotations,generation=generate_sample(backend,prompt,segments,text)
    if not len(codec) or np.any(codec<0) or np.any(codec>=59049):raise RuntimeError('Invalid acoustic codec')
    key=audited.s2tt.make_key(wav_path,i);cp=out/'codec'/f'{key}.npy';ctrl=out/'codec'/f'{key}.control.npy'
    np.save(cp,codec);np.save(ctrl,backend._global_codec_full)
    row=dict(generation_status=generation['status'],eos_observed=generation['eos_observed'],truncated=generation['truncated'],generation_error=generation['error'],index=i,key=key,wav=wav_path,source=source,reference=reference,prediction=text,
      codec_path=str(cp.resolve()),codec_control_path=str(ctrl.resolve()),codec_len=len(codec),
      codec_segments=annotations,talker_write_count=1,first_write_ratio=1.0,decisions=[],audio_seconds=len(wav)/16000)
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(i,key,len(codec),generation['status'],flush=True)
  all_rows=[json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines() if x.strip()]
  assert len(all_rows)==len(rows)
  (out/'GENERATION_SUMMARY.json').write_text(json.dumps(summarize(all_rows),indent=2))
  (out/'FAILED').unlink(missing_ok=True)
  (out/'INFERENCE_COMPLETE').write_text(str(len(rows))+'\n')
 except BaseException:
  (out/'FAILED').write_text(traceback.format_exc());raise
 finally:
  if backend:backend.close()
if __name__=='__main__':main()
