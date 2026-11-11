"""Offline adaptation of the latest checked FFmpeg reader.

Keep the repaired offline loader's retained-item alignment. Never install the
streaming full-record loader or random batch replacement for this experiment.
"""
def install():
 import os,time
 from pathlib import Path
 os.environ.setdefault("EXP_DIR",str(Path(__file__).resolve().parents[1]))
 import funasr.utils.load_utils as lu
 import audio_io_checks as guard
 if getattr(lu,'_offline_checked_io',False):return
 class StrictSources:
  def __contains__(self,path):return True
 guard._strict_sources=StrictSources()
 def checked(path,*args,**kwargs):
  # Offline source clips require complete PCM. Whole target files have no
  # rounded forced-alignment duration, so the streaming EOF relaxation is unused.
  for attempt in range(3):
   try:
    audio,sr=guard.decode_checked(path,*args,**kwargs)
    break
   except RuntimeError as exc:
    # Retry only rejected reads; never return partial PCM or substitute samples.
    if not str(exc).startswith('AUDIO_IO ') or attempt==2:raise
    guard.event('offline_read_retry',attempt=attempt+1,source=guard.source_key(path),error=str(exc))
    time.sleep(2*(attempt+1))
  if not guard.np.isfinite(audio).all():raise RuntimeError('Nonfinite decoded PCM')
  return audio.copy(),sr
 guard.install_io()
 lu.load_audio_ffmpeg=checked
 lu._offline_checked_io=True
 print('[OFFLINE IO] checked FFmpeg: timeout, exit status, truncation, transport and finite PCM',flush=True)
