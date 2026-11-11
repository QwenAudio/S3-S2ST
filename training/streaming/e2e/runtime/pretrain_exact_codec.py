"""Use as TRAIN_SCRIPT; EXACT_CODEC_BASE_ENTRY selects the existing entrypoint."""
import os,runpy,sys
from pathlib import Path
from exact_codec_loader import install
entry=Path(os.environ['EXACT_CODEC_BASE_ENTRY']).resolve()
if not entry.is_file() or entry==Path(__file__).resolve():raise ValueError('Invalid EXACT_CODEC_BASE_ENTRY')
if os.environ.get('STREAMING_S2ST_CODEC_ENVELOPE','0')!='1':raise ValueError('Precomputed streaming codec requires CODEC_ENVELOPE=1')
os.environ['STREAMING_S2ST_CODEC_RESERVATION_MARGIN']='0'
install()
print("[V7 EXACT CODEC] adapter installed; reservation_margin=0; base_entry="+str(entry),flush=True)
sys.path.insert(0,str(entry.parent))
sys.argv[0]=str(entry)
runpy.run_path(str(entry),run_name='__main__')
