"""Offline adaptation of the latest checked FFmpeg reader.

Keep the repaired offline loader's retained-item alignment. Never install the
streaming full-record loader or random batch replacement for this experiment.
"""
def install():
 import funasr.utils.load_utils as lu
 import audio_io_checks as guard
 if getattr(lu,'_offline_checked_io',False):return
 class StrictSources:
  def __contains__(self,path):return True
 guard._strict_sources=StrictSources()
 def checked(path,*args,**kwargs):
  # Offline source clips require complete PCM. Whole target files have no
  # rounded forced-alignment duration, so the streaming EOF relaxation is unused.
  audio,sr=guard.decode_checked(path,*args,**kwargs)
  if not guard.np.isfinite(audio).all():raise RuntimeError('Nonfinite decoded PCM')
  return audio.copy(),sr
 guard.install_io()
 lu.load_audio_ffmpeg=checked
 lu._offline_checked_io=True
 print('[OFFLINE IO] checked FFmpeg: timeout, exit status, truncation, transport and finite PCM',flush=True)
