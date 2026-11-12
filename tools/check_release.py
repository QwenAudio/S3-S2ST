#!/usr/bin/env python3
"""Check release files for secrets, generated assets, syntax and broken README links."""
import argparse,ast,json,re,subprocess,sys,warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--staged',action='store_true');a=p.parse_args()
 cmd=['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'] if a.staged else ['git','ls-files','--cached','--others','--exclude-standard','-z']
 files=sorted(set(x for x in subprocess.check_output(cmd,cwd=ROOT).decode().split('\0') if x));errors=[];total=0
 patterns={
 'Aliyun access key':r'\bLTAI[A-Za-z0-9]{12,}\b',
 'AWS access key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'HuggingFace token':r'\bhf_[A-Za-z0-9]{25,}\b',
 'GitHub token':r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}\b',
 'private key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 'environment credential':r'''(?im)os\.environ\[\s*["'][^"']*(?:KEY|TOKEN|SECRET|PASSWORD)[^"']*["']\s*\]\s*=\s*["'][A-Za-z0-9+/=_-]{16,}["']''',
 'literal credential':r'''(?im)(?:access_?key(?:_?id|_?secret)?|secret_?key|api_?key|auth_?token|password|passwd)\s*[:=]\s*["']([A-Za-z0-9+/=_-]{16,})["']'''}
 for n in files:
  f=ROOT/n
  if not f.is_file():continue
  total+=f.stat().st_size
  if n.split('/')[0] in {'data','models','checkpoints','envs','cache','runs','provenance'}:errors.append((n,'local-only directory'))
  if f.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.onnx','.bin','.wav','.mp3','.flac','.pem','.key'}:errors.append((n,'weight/data/secret file'))
  if f.stat().st_size>20*1024*1024:errors.append((n,'unexpected large file'))
  try:s=f.read_text()
  except UnicodeError:continue
  for label,pattern in patterns.items():
   if re.search(pattern,s):errors.append((n,label))
  if f.suffix=='.py':
   try:ast.parse(s,filename=n)
   except SyntaxError as exc:errors.append((n,f'Python syntax at line {exc.lineno}'))
  if f.suffix=='.sh':
   r=subprocess.run(['bash','-n',str(f)],capture_output=True)
   if r.returncode:errors.append((n,'shell syntax'))
 for name in ['README.md','README.zh-CN.md']:
  for link in re.findall(r'\]\(([^)]+)\)',(ROOT/name).read_text()):
   if '://' not in link and not (ROOT/link.split('#')[0]).is_file():errors.append((name,'missing link '+link))
 print(json.dumps(dict(status='FAIL' if errors else 'PASS',files=len(files),bytes=total,errors=errors,secret_values_printed=False),indent=2))
 return bool(errors)

if __name__=='__main__':sys.exit(main())
