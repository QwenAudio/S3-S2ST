"""CPU-only command check; generates its own synthetic input, with no data in Git."""
from pathlib import Path
import subprocess,sys,json,time,wave,math,struct
root=Path(__file__).resolve().parents[1]
tag=str(time.time_ns());fixtures=root/'runs'/('command_fixtures_'+tag);fixtures.mkdir(parents=True)
audio=fixtures/'tone.wav'
with wave.open(str(audio),'wb') as f:
 f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
 f.writeframes(b''.join(struct.pack('<h',int(1000*math.sin(2*math.pi*220*i/16000))) for i in range(16000)))
input_path=fixtures/'input.jsonl';input_path.write_text(json.dumps(dict(messages=[dict(role='user',content='Translate into English:<|startofspeech|>!'+str(audio)+'<|endofspeech|>'),dict(role='assistant',content='')],transcript='',dataset='synthetic',direction='zh2en'))+'\n')
results=[]
for mode in ['streaming','offline']:
 cmd=['bash',str(root/'scripts/infer.sh'),'--mode',mode,'--input',str(input_path),'--output',str(root/'runs'/('check_'+mode+'_'+tag)),'--no-warmup','--dry-run']
 results.append(dict(mode=mode,returncode=subprocess.run(cmd,cwd=root,stdout=subprocess.DEVNULL).returncode))
print(json.dumps(results));raise SystemExit(any(x['returncode'] for x in results))
