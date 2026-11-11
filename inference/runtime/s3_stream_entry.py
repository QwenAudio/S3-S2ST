"""Final evaluation entry: native Thinker/Talker + prefix voice + live PCM buffer."""
import os,json,time,inspect,textwrap,math
from pathlib import Path
import gpu_native_entry as native
m=native.m;s=m.s2tt
from pcm_runtime import PCMRuntime
cfg=json.loads(Path(os.environ['S3_POLICY_CONFIG']).read_text())
original_init=m.StreamingS2STBackend.__init__
def init(self,args):
    original_init(self,args)
    self.pcm=PCMRuntime(Path(args.output).parent/'pcm',args.encoder_device,cfg)
    self.s2st_engine._s3_pcm=self.pcm;s._s3_pcm=self.pcm
m.StreamingS2STBackend.__init__=init
# Callback on true acoustic tokens only, before the next Talker position.
original=m.CodecHistoryReplayEngine._listen_talker
src=inspect.getsource(original);needle='                                self._global_codec_tokens.append(token)'
assert src.count(needle)==1
src=src.replace(needle,needle+'\n                                self._s3_pcm.on_codec(token)')
ns={};exec(compile(textwrap.dedent(src),__file__,'exec'),original.__globals__,ns)
m.CodecHistoryReplayEngine._listen_talker=ns['_listen_talker']
# Pace each required prefix before any computation can observe it.
original=inspect.unwrap(s.FullRecordEncoder.stream_prefixes)
src=inspect.getsource(original).replace('@torch.inference_mode()\n','')
needle='        prefix_embeds = self(prefix)';assert src.count(needle)==1
src=src.replace(needle,'''        pcm = s2tt._s3_pcm
        if pcm.config.get('paced', True):
            wait = pcm.play.start + available_prefix_samples / 16000. - time.monotonic()
            if wait > 0: time.sleep(wait)
        pcm.prefix(wav, available_prefix_samples)
'''+needle)
gl=original.__globals__.copy();gl['time']=time
ns={};exec(compile(textwrap.dedent(src),__file__,'exec'),gl,ns)
s.FullRecordEncoder.stream_prefixes=m.torch.inference_mode()(ns['_stream_prefixes'])
# Freeze source speaker only after the first nonempty target text.
original_text=m.StreamingS2STBackend.generate_text
def text(self,*args,**kwargs):
    out=original_text(self,*args,**kwargs)
    if out:
        self.pcm.text_events.append(dict(source_ms=self.pcm.source_ms,wall_s=time.monotonic()-self.pcm.play.start,text=out))
        self.pcm.fix_speaker()
    return out
m.StreamingS2STBackend.generate_text=text
# Gate model remains at 0.5. Per-request odds bias implements arbitrary tau.
def gate(self,prompt_text,segments):
    tau=self.pcm.gate()
    prompt=m.TextPrompt(prompt=prompt_text,multi_modal_data={'audio':self._mm(segments)},mm_processor_kwargs={})
    params=m.SamplingParams(temperature=0.,top_k=1,max_tokens=1,stop_token_ids=[self.im_end_id],detokenize=False,logit_bias={1:-math.log(tau/(1.-tau))},logprobs=2 if cfg.get('audit_gate_probabilities',False) else None)
    outputs=self._request(self.gate_engine,prompt,params)
    last=next((x for x in reversed(outputs) if isinstance(x,m.RequestOutput)),None)
    if last is None:raise RuntimeError('no gate output')
    ids=list(last.outputs[0].token_ids or []);assert ids and ids[0] in (0,1)
    self.pcm.gates[-1]['write']=ids[0]==1
    if cfg.get('audit_gate_probabilities',False):
        lp=last.outputs[0].logprobs[0]
        assert 0 in lp and 1 in lp
        adjusted=float(lp[1].logprob)-float(lp[0].logprob)
        raw_odds=adjusted+math.log(tau/(1.-tau))
        p_write=1./(1.+math.exp(-max(-80.,min(80.,raw_odds))))
        self.pcm.gates[-1].update(p_write=p_write,adjusted_log_odds=adjusted,read_due_to_raised_threshold=(ids[0]==0 and p_write>.5 and tau>.5))
    return ids[0]==1
m.StreamingS2STBackend.gate_decision=gate
original_eval=s.evaluate_one_streaming_thinker_deferred_talker
def evaluate(index,obj,encoder,backend,args):
    if cfg.get('warmup') and not getattr(backend,'_s3_warmed',False):
        warmup_rows=[json.loads(x) for x in Path(cfg['warmup']).read_text().splitlines()][:2]
        original_root=backend.pcm.root;backend.pcm.root=original_root.parent/'warmup_pcm';backend.pcm.root.mkdir(exist_ok=True)
        for j,row in enumerate(warmup_rows):
            backend.start_sample();backend.pcm.begin(-j-1)
            result=original_eval(-j-1,row,encoder,backend,args)
            backend.finish_sample();backend.pcm.finish(result['key'])
        backend.pcm.root=original_root;backend._s3_warmed=True;backend.start_sample()
        print('WARMUP_COMPLETE',flush=True)
    backend.pcm.begin(index)
    result=original_eval(index,obj,encoder,backend,args)
    # finish_sample is called by the upstream driver; a finalized EOS is required here too.
    if backend.s2st_engine._global_failure:raise RuntimeError(backend.s2st_engine._global_failure)
    result['live_pcm']=backend.pcm.finish(result['key'])
    return result
s.evaluate_one_streaming_thinker_deferred_talker=evaluate
if __name__=='__main__':m.main()
