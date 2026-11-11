from pathlib import Path
import json,re,sys,collections
import numpy as np
from rapidfuzz.distance import Levenshtein
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tn_wer import clean
CODE=Path(__file__).resolve().parents[1]
ROOTS={'ours9':Path(sys.argv[1]),'oursf':Path(sys.argv[2])}
OUT=CODE/'analysis_0905_quality'
OUT.mkdir(exist_ok=True)
def cid(k): return re.sub(r'_[a-f0-9]{16}$','',Path(k).stem if str(k).endswith('.wav') else Path(k).name)
def stats(rows):
 if not rows:return {}
 return {'n':len(rows),'macro':round(np.mean([r['wer'] for r in rows])*100,4),'micro':round(sum(r['errors'] for r in rows)/sum(r['nref'] for r in rows)*100,4),'zero_pct':round(np.mean([r['wer']==0 for r in rows])*100,2),'gt20':sum(r['wer']>.2 for r in rows),'sub':round(np.mean([r['sub'] for r in rows])*100,4),'del':round(np.mean([r['dele'] for r in rows])*100,4),'ins':round(np.mean([r['ins'] for r in rows])*100,4),'utmos':round(np.mean([r['utmos'] for r in rows]),4),'mean_length':round(np.mean([r['nref'] for r in rows]),2)}
ALL={}
for dataset,root in [('9pair',ROOTS['ours9']),('fleurs',ROOTS['oursf'])]:
 for d in ['zh2en','en2zh']:
  lang=d[-2:];name=f'test_{dataset}_{d}.out'
  raw={o['key']:o for o in map(json.loads,(root/'raw'/f'{d}.jsonl').read_text().splitlines())}
  for system,p in [('ours',root/name)]+[(x.name,x/name) for x in (CODE/'retest_20260902').iterdir() if (x/name/'metrics/asr_details.tsv').exists()]:
   m=p/'metrics'
   ut={x['key']:x['utmos'] for x in json.loads((m/'utmos.json').read_text())['segment_scores']}
   audit={Path(x['wav']).stem:x for x in map(json.loads,(m/'wer_tn_audit.jsonl').read_text().splitlines())}
   rows=[];edits=collections.Counter();bad=[]
   for line in (m/'asr_details.tsv').read_text().splitlines():
    c=line.split('\t'); key=Path(c[0]).stem; ref=c[2].split(); hyp=c[3].split()
    ops=list(Levenshtein.editops(ref,hyp)); a=audit[key]
    r=dict(key=key,id=cid(key),wer=float(c[1]),ins=float(c[4]),dele=float(c[5]),sub=float(c[6]),nref=len(ref),errors=len(ops),utmos=ut[key],ref=a['reference_raw'],hyp=a['asr_raw'],ref_tn=c[2],hyp_tn=c[3])
    if system=='ours':
     x=raw[key];r.update(writes=x['talker_write_count'],pieces=x['pieces'],codec_len=x['codec_len'],segment_lens=[s['codec_len'] for s in x['codec_segments']])
     for tag,i,j in ops:edits[(tag,ref[i] if tag!='insert' else '',hyp[j] if tag!='delete' else '')]+=1
    rows.append(r)
   assert len(set(r['id'] for r in rows))==len(rows)
   ALL[(dataset,d,system)]=rows
   info={'dataset':dataset,'direction':d,'system':system,**stats(rows),'tn_profiles':sorted(set(a['tn_profile'] for a in audit.values()))}
   print(json.dumps(info,ensure_ascii=False))
   if system=='ours':
    group={str(w):stats([r for r in rows if r['writes']==w]) for w in sorted(set(r['writes'] for r in rows))}
    print('GROUPS',json.dumps(group,ensure_ascii=False))
    top=sorted(rows,key=lambda r:r['wer'],reverse=True)
    print('TOP_ERROR_SHARE',round(sum(r['wer'] for r in top[:int(len(rows)*.05)])/sum(r['wer'] for r in rows)*100,2))
    print('EDITS',json.dumps([(list(k),v) for k,v in edits.most_common(20)],ensure_ascii=False))
    print('EXAMPLES',json.dumps([{k:r[k] for k in ['id','wer','ref','hyp','writes','pieces','segment_lens']} for r in top[:6]],ensure_ascii=False))
    print('ZERO_UTMOS',stats([r for r in rows if r['wer']==0]),'NONZERO_UTMOS',stats([r for r in rows if r['wer']>0]))
    (OUT/f'{dataset}_{d}_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
  ours={r['id']:r for r in ALL[(dataset,d,'ours')]}
  for system in ['qwen35_livetranslate','seed_liveinterpret_2','uniss_quality']:
   baseline={r['id']:r for r in ALL[(dataset,d,system)]}
   matched=set(ours)&set(baseline)
   same=[k for k in matched if ours[k]['ref_tn']==baseline[k]['ref_tn']]
   print('PAIRED',json.dumps({'dataset':dataset,'direction':d,'system':system,'matched':len(matched),'same_text_n':len(same),'ours_same_text':stats([ours[k] for k in same]),'baseline_same_text':stats([baseline[k] for k in same])},ensure_ascii=False))
(OUT/'all_system_stats.json').write_text(json.dumps([{'dataset':k[0],'direction':k[1],'system':k[2],**stats(v)} for k,v in ALL.items()],ensure_ascii=False,indent=2))
