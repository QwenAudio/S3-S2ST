"""Complete ASR rows only for explicitly audited missing-waveform examples."""
import sys,json
from pathlib import Path
from metric_inputs import read_keyed,read_asr_tsv
ref,raw,recog,dest=map(Path,sys.argv[1:])
refs=read_keyed(ref);_,hyp=read_asr_tsv(raw)
known=set(json.loads((recog/'empty_audio_keys.json').read_text()))
missing=set(refs)-set(hyp)
assert not(set(hyp)-set(refs)),'Unexpected ASR IDs'
assert missing<=known,('Unexplained missing ASR',sorted(missing-known))
lines=raw.read_text().splitlines()
for key in sorted(missing):
 lines.append('\t'.join([str(recog/'s2st_wav/wav'/(key+'.wav')),'nan',refs[key],'','nan','nan','nan']))
dest.write_text('\n'.join(lines)+'\n')
(dest.parent/'asr_coverage_audit.json').write_text(json.dumps({'expected':len(refs),'asr_returned':len(hyp),'known_empty_audio':sorted(known),'inserted_empty_hypotheses':sorted(missing),'raw_source':str(raw)},indent=2))
assert set(read_asr_tsv(dest)[1])==set(refs)
