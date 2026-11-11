"""Bounded, checked source decoding shared by inference and speaker conditioning."""
import numpy as np
from offline_io import install

def load_source(path, fs=16000, duration=(0,None), **kwargs):
 start=float(duration[0]); end=None if duration[1] is None else float(duration[1])
 if not np.isfinite(start) or start<0 or (end is not None and (not np.isfinite(end) or end<=start)):
  raise ValueError('Invalid source interval')
 install()
 from funasr.utils.load_utils import load_audio_text_image_video
 x=np.asarray(load_audio_text_image_video(path,fs=fs,duration=(start,end),use_ffmpeg=True))
 if x.ndim!=1 or not x.size or not np.isfinite(x).all():raise ValueError('Invalid source PCM')
 if end is not None and abs(len(x)-round((end-start)*fs))>32:raise ValueError('Incomplete source interval')
 return x
