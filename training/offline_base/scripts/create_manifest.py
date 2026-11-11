from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parents[1]
files={}
for d in ['scripts','config','vendor/streaming_eval','vendor/funasr_snapshot']:
 for p in (R/d).rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in {'.py','.sh','.yaml','.json','.md'} and '__pycache__' not in p.parts:
   files[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'audit/source_manifest.json').write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2))
print(len(files),'source files fingerprinted')
