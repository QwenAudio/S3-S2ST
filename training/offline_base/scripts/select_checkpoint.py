#!/usr/bin/env python3
"""Select only complete bidirectional dev evaluations; no train-loss early stop."""
import argparse,json,math
from pathlib import Path

def summarize(path):
 values=[]
 for lang in ['en','zh']:
  d=Path(path)/lang
  if not (d/'COMPLETE').is_file():raise RuntimeError(f'Incomplete evaluation: {d}')
  s=json.loads((d/'summary.json').read_text());w=json.loads((d/'metrics/wer_corpus.json').read_text())
  values.append(dict(wer=w['score'],mos=s['utmos']['score'],asr_comet=s['asr_comet']['system_score'],text_comet=s['comet']['system_score'],failure_rate=s.get('generation',{}).get('failure_rate',0.0)))
 result={k:sum(v[k] for v in values)/2 for k in values[0]}
 if not all(math.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metric')
 return result

def decide(records,stage):
 best=records[0];stale=0
 for r in records[1:]:
  b=best['metrics'];m=r['metrics']
  if stage=='talker':
   better=m['wer']<=b['wer']-.2 and m['mos']>=b['mos']-.05 and m['asr_comet']>=b['asr_comet']-.005
  else:
   better=m['asr_comet']>=b['asr_comet']+.002 and m['wer']<=b['wer']+.3 and m['mos']>=b['mos']-.05 and m['text_comet']>=records[0]['metrics']['text_comet']-.005
  # Never improve selection by dropping or truncating more utterances.
  if m.get('failure_rate',0)>b.get('failure_rate',0):better=False
  if m.get('failure_rate',0)<b.get('failure_rate',0) and m['mos']>=b['mos']-.05 and m['asr_comet']>=b['asr_comet']-.005:better=True
  if better:best=r;stale=0
  else:stale+=1
 step=records[-1]['step']
 cap=10000 if stage=='talker' else 5000
 minimum=5000 if stage=='talker' else 2000
 return dict(best=best,stale_evaluations=stale,stop=(step>=minimum and stale>=3) or step>=cap,
  reason=(f'{cap}-update cap' if step>=cap else f'dev plateau after minimum {minimum} updates' if step>=minimum and stale>=3 else 'continue'),
  status='measured_dev_selection_not_global_upper_bound')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('records');p.add_argument('--stage',choices=['talker','e2e'],required=True);a=p.parse_args()
 print(json.dumps(decide(json.loads(Path(a.records).read_text()),a.stage),indent=2))
