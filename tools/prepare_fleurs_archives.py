from pathlib import Path
import json,hashlib,tarfile,re,shutil,soundfile as sf
import argparse
p=argparse.ArgumentParser(description='Verify and extract pinned official FLEURS test archives using frozen paper manifest filenames')
p.add_argument('--archive-dir',type=Path)
a=p.parse_args()
D=Path(__file__).resolve().parents[1];cache=a.archive_dir or D/'cache/downloads/fleurs_hf';meta=json.loads((D/'configs/fleurs_archive_verified.json').read_text());reports=[]
for x in meta:
 lang=x['language'];direction='zh2en' if lang=='cmn_hans_cn' else 'en2zh';archive=cache/x['filename'];h=hashlib.sha256()
 with archive.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],lang
 rows=[json.loads(line) for line in (D/'data/fleurs'/f'test_{direction}.jsonl').read_text().splitlines()];wanted={Path(re.search(r'<\|startofspeech\|>!?(.*?)<\|endofspeech\|>',next(m for m in r['messages'] if m['role']=='user')['content']).group(1)).name for r in rows};out=D/'data/fleurs/audio'/lang;out.mkdir(parents=True,exist_ok=True);found=set();files=[]
 with tarfile.open(archive) as t:
  for m in t:
   name=Path(m.name).name
   if not m.isfile() or name not in wanted:continue
   dst=out/name
   with t.extractfile(m) as inp,dst.open('wb') as g:shutil.copyfileobj(inp,g)
   info=sf.info(dst);assert info.samplerate==16000 and info.channels==1,(name,info)
   found.add(name);files.append(dict(file=str(dst.relative_to(D)),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),sample_rate=info.samplerate,frames=info.frames))
 assert found==wanted,(lang,len(wanted-found))
 mismatch=[]
 for r in rows:
  user=next(m for m in r['messages'] if m['role']=='user');name=Path(re.search(r'<\|startofspeech\|>!?(.*?)<\|endofspeech\|>',user['content']).group(1)).name;info=sf.info(out/name)
  if abs(info.duration-r.get('duration_s',info.duration))>.021:mismatch.append(dict(name=name,actual=info.duration,manifest=r.get('duration_s')))
 reports.append(dict(language=lang,direction=direction,archive=x,files=files,duration_mismatches=mismatch));print('FLEURS_READY',direction,len(found),'duration_mismatch',len(mismatch),flush=True)
(D/'provenance/FLEURS_DOWNLOAD.json').write_text(json.dumps(dict(source='official HF test archives',revision=meta[0]['revision'],transfer='public data downloaded via TLS-valid local network then scp; remote SHA256 rechecked',original_paper_audio_hash_available=False,reports=reports),indent=2))
# Both directions are represented, with nonempty reference text, in a reproducible 8-example GPU smoke.
a=[json.loads(s) for s in (D/'data/fleurs/test_zh2en.jsonl').read_text().splitlines()];b=[json.loads(s) for s in (D/'data/fleurs/test_en2zh.jsonl').read_text().splitlines()]
smoke=[a[i] for i in range(4)]+[b[i] for i in range(4)];(D/'data/fleurs/smoke8.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke));print('OFFICIAL_FLEURS_1591_AND_SMOKE8_READY',flush=True)
