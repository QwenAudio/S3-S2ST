import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=json.loads((a.run/'quality/summary.json').read_text());groups={}
for r in rows:
 d=r['dataset'].rsplit('_',1)[0];groups.setdefault(d,[]).append(r)
macro={}
for d,rr in groups.items():
 if len(rr)!=2:raise SystemExit(f'{d}: expected both directions before direction macro')
 macro[d]={k:sum(r[k] for r in rr)/2 if all(isinstance(r[k],(int,float)) for r in rr) else None for k in rr[0] if k!='dataset'}
raw=[json.loads(x) for x in (a.run/'raw.jsonl').read_text().splitlines()];onsets=[x.get('live_pcm',{}).get('startoffset_wall_s') for x in raw];onsets=[x for x in onsets if x is not None]
report=dict(direction_macro=macro,per_direction=rows,coverage=dict(rows=len(raw),audio=sum(bool(x.get('live_pcm',{}).get('wav')) for x in raw),onsets=len(onsets)),latency_note='Inspect per-direction traces; pooled onset here is descriptive only',pooled_onset_seconds=statistics.mean(onsets) if onsets else None)
(a.run/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
