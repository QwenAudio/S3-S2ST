"""Sampling-only adapter. Imported inference/READ-WRITE/FM sources remain unchanged."""
import hashlib,inspect,json,os,sys
from pathlib import Path
ROOT=Path(os.environ["TUNE_ROOT"])
cfg=json.loads(Path(os.environ["TUNE_CONFIG"]).read_text())
settings=json.loads((ROOT/"settings.json").read_text())
for p,h in json.loads((ROOT/"evidence/source_hashes.json").read_text()).items():
    if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError("External source changed: "+p)
sys.path.insert(0,settings["code"])
import run_block640_lookahead160 as entry
from terminal_delivery import install as install_terminal_delivery
install_terminal_delivery()
original=entry.s2st.SamplingParams
seen_branches=set()
def sampling(*args,**kwargs):
    caller=inspect.currentframe().f_back.f_code.co_name
    branch=None
    if caller=="generate_text":branch="thinker"
    elif caller=="generate_forced_speech" and "logit_bias" in kwargs:branch="talker"
    if branch:
        overrides=cfg[branch]
        assert set(overrides)<={"temperature","top_p","top_k","repetition_penalty","presence_penalty","frequency_penalty","seed"}
        kwargs.update(overrides)
        if branch not in seen_branches:
            print("APPLIED_SAMPLING_OVERRIDE",branch,json.dumps(overrides),flush=True)
            seen_branches.add(branch)
    return original(*args,**kwargs)
entry.s2st.SamplingParams=sampling
if __name__=="__main__":
    print("SAMPLING_ONLY_CONFIG",json.dumps(cfg),flush=True)
    entry.s2st.main()
