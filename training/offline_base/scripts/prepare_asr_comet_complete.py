"""Compatibility entry for the canonical, full-coverage ASR-COMET preparer."""
import os,json,runpy,sys
from pathlib import Path
r=Path(os.environ['TUNE_ROOT']);settings=json.loads((r/'settings.json').read_text())
p=Path(settings['code'])/'metrics/prepare_asr_comet.py'
sys.path.insert(0,str(p.parent))
runpy.run_path(str(p),run_name='__main__')
