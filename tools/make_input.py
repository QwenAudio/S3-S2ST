import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--audio',type=Path,required=True);p.add_argument('--direction',choices=['zh2en','en2zh'],required=True);p.add_argument('--source',default='');p.add_argument('--reference',default='');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];audio=root/'data/user_audio'/a.audio.name;audio.parent.mkdir(parents=True,exist_ok=True)
if a.audio.resolve()!=audio.resolve():shutil.copy2(a.audio,audio)
row=dict(messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content='Translate into '+('English' if a.direction=='zh2en' else 'Chinese')+':<|startofspeech|>!'+str(audio)+'<|endofspeech|>'),dict(role='assistant',content=a.reference)],transcript=a.source,dataset='custom',direction=a.direction)
a.output.resolve().relative_to(root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(row,ensure_ascii=False)+'\n')
