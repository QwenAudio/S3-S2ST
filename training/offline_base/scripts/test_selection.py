import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from select_checkpoint import decide
m=dict(wer=10.,mos=4.,asr_comet=.8,text_comet=.85)
def r(step,**k):return dict(step=step,metrics=m|k,checkpoint=str(step))
a=[r(i) for i in range(5000,6501,500)]
assert not decide([r(4500),r(4600),r(4700),r(4900)],'talker')['stop']
assert decide(a,'talker')['stop'] and decide(a,'talker')['best']['step']==5000
assert decide(a+[r(7000,wer=9.7)],'talker')['best']['step']==7000
assert decide(a+[r(7000,wer=9.7,mos=3.8)],'talker')['best']['step']==5000
assert decide(a+[r(7000,asr_comet=.81)],'e2e')['best']['step']==7000
assert decide(a+[r(7000,asr_comet=.81,text_comet=.83)],'e2e')['best']['step']==5000
assert decide([r(0),r(10000,wer=9.)],'talker')['reason']=='10000-update cap'
print('PASS: minimum budget, plateau, regression constraints, improving candidate, hard cap')
