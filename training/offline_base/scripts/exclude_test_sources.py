#!/usr/bin/env python3
"""Exclude benchmark source recordings from new training, with full audit."""
from pathlib import Path
import concurrent.futures as cf,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
PAT=re.compile(r'<\|startofspeech\|>!?(.*?)<\|endofspeech\|>')
def canonical(s):
 return s.replace('oss://example-storage-4/','').replace('/nls_spider_sh_cache/','').replace('oss://example-storage-1/','').replace('/fun_resource_sh_cache/','')
def keys(item):
 return {canonical(m.group(1)) for msg in item['messages'] if msg['role']=='user' for m in PAT.finditer(msg.get('content',msg.get('text','')))}
def clean(job):
 path,blocked=job;p=Path(path);kept=0;removed=0;bins=0;tmp=p.with_suffix('.exclude.tmp')
 with p.open() as f,tmp.open('w') as out:
  for line in f:
   items=json.loads(line);good=[x for x in items if not keys(x)&blocked]
   removed+=len(items)-len(good);kept+=len(good)
   if good:
    out.write(line if len(items)==len(good) else json.dumps(good,ensure_ascii=False)+'\n');bins+=1
 if removed:
  # Only replace new experiment-owned packed data, never source clean-v2.
  tmp.replace(p)
 else:tmp.unlink()
 return dict(path=str(p),kept=kept,removed_test_source_records=removed,bins=bins)
def main():
 if (ROOT/'audit/test_exclusion.json').exists():return
 source=Path(json.loads((ROOT/'audit/origin.json').read_text())['streaming_eval'])/'datasets'
 blocked=set()
 manifests=[]
 for ds in ['9pair','fleurs']:
  for direction in ['zh2en','en2zh']:
   p=source/ds/f'test_{ds}_{direction}.jsonl';manifests.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   for line in p.open():blocked.update(keys(json.loads(line)))
 lists={n:(ROOT/'data/full_v2'/n).read_text().splitlines() for n in ['train.list','valid.list']}
 paths=lists['train.list']+lists['valid.list']
 with cf.ProcessPoolExecutor(max_workers=16) as pool:report=list(pool.map(clean,[(p,blocked) for p in paths]))
 nonempty={r['path'] for r in report if r['bins']}
 for name,original in lists.items():
  (ROOT/'data/full_v2'/name).write_text(''.join(p+'\n' for p in original if p in nonempty))
 (ROOT/'audit/test_exclusion.json').write_text(json.dumps(dict(test_manifests=manifests,unique_test_sources=len(blocked),shards=report,total_removed=sum(r['removed_test_source_records'] for r in report),remaining_packed_bins=sum(r['bins'] for r in report)),indent=2))
if __name__=='__main__':
 import fcntl
 with (ROOT/'audit/test_exclusion.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  main()
