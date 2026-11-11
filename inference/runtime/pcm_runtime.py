"""Causal PCM emission and speaker conditioning, isolated from research sources."""
_S3_ROOT = next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir())
import time, math, hashlib, json, threading
from pathlib import Path
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from torchaudio.compliance import kaldi

class SourceSpeaker:
    def __init__(self):
        opts=ort.SessionOptions();opts.intra_op_num_threads=1
        self.session=ort.InferenceSession(('__S3_ROOT__/models/speaker/campplus.onnx'.replace('__S3_ROOT__', str(_S3_ROOT))),sess_options=opts,providers=['CPUExecutionProvider'])
    def extract(self,wave):
        wave=torch.as_tensor(np.asarray(wave,dtype=np.float32).copy()).reshape(1,-1)[:,:160000]
        if wave.shape[1]<400:raise ValueError('insufficient source prefix for speaker embedding')
        feat=kaldi.fbank(wave,num_mel_bins=80,dither=0,sample_frequency=16000)
        feat=feat-feat.mean(dim=0,keepdim=True)
        x=self.session.run(None,{self.session.get_inputs()[0].name:feat.unsqueeze(0).numpy()})[0]
        x=np.asarray(x,dtype=np.float32).reshape(1,-1)
        assert np.isfinite(x).all() and np.linalg.norm(x)>0
        return torch.from_numpy(x),dict(samples=int(wave.shape[1]),source_prefix_sha256=hashlib.sha256(wave.numpy().tobytes()).hexdigest(),embedding_sha256=hashlib.sha256(x.tobytes()).hexdigest(),norm=float(np.linalg.norm(x)))

class Playback:
    def __init__(self,clock=time.monotonic):self.clock=clock;self.lock=threading.RLock();self.reset()
    def reset(self):
        self.start=self.last=self.clock();self.buffer=0.;self.started=False;self.underrun=0.;self.stalls=0;self.in_stall=False
    def advance(self):
        now=self.clock();elapsed=max(0.,now-self.last)
        if self.started and elapsed>self.buffer:
            self.underrun+=elapsed-self.buffer
            if not self.in_stall:self.stalls+=1;self.in_stall=True
        self.buffer=max(0.,self.buffer-elapsed);self.last=now
        return now-self.start
    def emit(self,seconds):
        with self.lock:
            t=self.advance();self.buffer+=seconds;self.started=True;self.in_stall=False
            return t,self.buffer
    def threshold(self,base,cap,saturation,power=1.,reserve=0.):
        with self.lock:
            t=self.advance();usable=max(0.,self.buffer-reserve)
            return base+(cap-base)*min(usable/saturation,1.)**power,dict(wall_s=t,playable_s=self.buffer,usable_s=usable)

class PCMRuntime:
    def __init__(self,root,device,config):
        import streaming_codex_to_wav as dec
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.device=torch.device(device);self.config=config
        self.speaker=SourceSpeaker()
        self.fm=dec._load_model_from_config(Path(config['fm_config']),((_S3_ROOT / 'models/flow_matching/model.pt') if (_S3_ROOT / 'models/flow_matching/model.pt').is_file() else Path(('__S3_ROOT__/models/flow_matching/epoch_2_step_780000.pt'.replace('__S3_ROOT__', str(_S3_ROOT))))),self.device).to(self.device,dtype=torch.float16).eval()
        vroot=Path(('__S3_ROOT__/models/vocoder'.replace('__S3_ROOT__', str(_S3_ROOT))));vpath=Path(config['vocoder_checkpoint']) if config.get('vocoder_checkpoint') else dec._resolve_checkpoint_path(vroot)
        vconfig=Path(config['vocoder_config']) if config.get('vocoder_config') else vpath.parent/'config.yaml'
        self.voc=dec._load_model_from_config(vconfig,vpath,self.device).to(self.device,dtype=torch.float32).eval()
        self.sr=int(getattr(self.voc,'sample_rate',24000));self.play=Playback();self.active=False
    def begin(self,index):
        self.index=index;self.play.reset();self.events=[];self.gates=[];self.waves=[];self.speaker_info=None;self.emb=None
        self.source_ms=0.;self.tokens=[];self.cursor=0;self.cache={};self.codec_count=0;self.envelope=0;self.active=True;self.pending_decode=None
        self.text_events=[];self.first_codec=None;self.compute_calls=[]
    def prefix(self,wave,available):
        self.source_ms=available/16.;self.available=available;self.wave=wave
    def fix_speaker(self):
        if self.emb is None:
            self.emb,self.speaker_info=self.speaker.extract(self.wave[:self.available])
            self.speaker_info['available_source_ms']=self.source_ms
            self.emb=self.emb.to(self.device,dtype=torch.float16)
    @torch.inference_mode()
    def on_codec(self,token):
        token=int(token)
        if 0<=token<59049:
            if self.first_codec is None:self.first_codec=dict(source_ms=self.source_ms,wall_s=time.monotonic()-self.play.start)
            self.tokens.append(token);self.codec_count+=1
            # A one-token lookahead distinguishes a complete interior chunk
            # from an utterance-final chunk. It prevents a non-final FM call
            # from committing samples whose boundary padding changes at EOS.
            if self.pending_decode is not None:
                self.decode(False,prefix_n=self.pending_decode);self.pending_decode=None
            n=len(self.tokens);c=int(self.fm.context_size)
            if n%15==0 or (n>=c+15 and (n-c)%15==0):self.pending_decode=n
        elif token==59050:
            if self.tokens:self.decode(True)
            self.tokens=[];self.cursor=0;self.cache={};self.envelope+=1;self.pending_decode=None
        else:raise ValueError(f'unexpected emitted codec {token}')
    @torch.inference_mode()
    def decode(self,final,prefix_n=None):
        assert self.emb is not None,'speaker must be frozen before codec emission'
        n=prefix_n if prefix_n is not None else len(self.tokens);ratio=int(self.fm.feat_token_ratio);block=15*ratio
        look=int(getattr(self.voc.generator.h,'conv_pre_look_right',0))
        if not final:
            if n<=int(self.fm.context_size):return
            complete=((n-int(self.fm.context_size))//15)*block
            if complete-look<=self.cursor:return
        started=time.monotonic()
        codes=torch.tensor([self.tokens[:n]],dtype=torch.long,device=self.device)
        feat=self.fm._inference(codec=codes,codec_lengths=torch.tensor([n],device=self.device),chunk_size=15,finalize=final,xvec=self.emb,xvec_lengths=torch.tensor([1],device=self.device),fm_dtype='fp16',infer_causal_mask_type=0).float()
        end=feat.shape[1] if final else feat.shape[1]//block*block
        feat=feat[:,:end,:];stable=end if final else max(self.cursor,end-look)
        if stable<=self.cursor:return
        wave,self.cache=self.voc.generator.inference_cache(feat[:,self.cursor:,:].transpose(1,2),finalize=final,cache=self.cache)
        self.cursor=stable
        pcm=wave.detach().float().cpu().reshape(-1).numpy()
        assert np.isfinite(pcm).all()
        if not len(pcm):return
        elapsed,buffer=self.play.emit(len(pcm)/self.sr)
        self.compute_calls.append(dict(envelope=self.envelope,codec_prefix=n,final=bool(final),seconds=time.monotonic()-started))
        self.events.append(dict(event='audio',source_ms=self.source_ms,wall_s=elapsed,samples=len(pcm),sample_rate=self.sr,duration_s=len(pcm)/self.sr,buffer_s=buffer,codec_tokens_available=self.codec_count,envelope=self.envelope))
        self.waves.append(pcm.copy())
    def gate(self):
        tau,rec=self.play.threshold(self.config['base'],self.config['cap'],self.config['saturation'],self.config.get('power',1.))
        rec.update(source_ms=self.source_ms,threshold=tau);self.gates.append(rec)
        return tau
    def finish(self,key,truncated=False,allow_empty=False):
        if self.tokens:
            if not truncated:raise RuntimeError('unflushed codec envelope')
            self.decode(True);self.tokens=[];self.pending_decode=None
        if not self.waves:
            if not allow_empty:raise RuntimeError('empty PCM output')
            out=self.root/key;out.mkdir(exist_ok=True)
            info=dict(events=[],gates=self.gates,text_events=self.text_events,first_codec=self.first_codec,speaker=self.speaker_info,startoffset_source_ms=None,startoffset_wall_s=None,wav=None,samples=0,sample_rate=self.sr,config=self.config,status='empty_audio')
            (out/'trace.json').write_text(json.dumps(info,indent=2));return info
        wav=np.concatenate(self.waves);out=self.root/key;out.mkdir(exist_ok=True)
        torchaudio.save(str(out/'output.wav'),torch.from_numpy(wav).unsqueeze(0),self.sr,encoding='PCM_S',bits_per_sample=16)
        np.save(out/'speaker.npy',self.emb.detach().float().cpu().numpy())
        last=self.events[-1];play_end=last['wall_s']+last['buffer_s']
        info=dict(events=self.events,gates=self.gates,text_events=self.text_events,first_codec=self.first_codec,compute_calls=self.compute_calls,speaker=self.speaker_info,startoffset_source_ms=self.events[0]['source_ms'],startoffset_wall_s=self.events[0]['wall_s'],underrun_s=self.play.underrun,stall_count=self.play.stalls,playback_end_s=play_end,source_duration_s=len(self.wave)/16000.,tail_s=play_end-len(self.wave)/16000.,rtf=last['wall_s']/(len(self.wave)/16000.),wav=str(out/'output.wav'),samples=len(wav),sample_rate=self.sr,config=self.config,truncated=truncated)
        (out/'trace.json').write_text(json.dumps(info,indent=2));return info
