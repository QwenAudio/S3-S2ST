import unittest
from types import SimpleNamespace
import numpy as np
from generation_policy import generate_sample,summarize
class Fake:
 codec_bos_id=59049
 def __init__(self,mode):self.mode=mode;self.s2st_engine=SimpleNamespace(_global_codec_tokens=[59049,1,2,3]);self.closed=False
 def generate_forced_speech(self,*args):
  if self.mode=='no_eos':raise RuntimeError("global Talker failed: RuntimeError('Talker request ended before the final WRITE produced codec EOS')")
  if self.mode=='engine':raise RuntimeError('CUDA engine failure')
  self._global_codec_full=np.array([59049,1,2,3,59050])
 def finish_sample(self):return np.array([1,2,3]),[dict(codec_len=3)]
 def _close_consumed_writer_shm_handles(self):self.closed=True
class PolicyTest(unittest.TestCase):
 def test_failure_then_next_sample(self):
  b=Fake('no_eos');codec,ann,g=generate_sample(b,None,None,'text')
  self.assertEqual(codec.tolist(),[1,2,3]);self.assertEqual(g['status'],'no_eos');self.assertTrue(g['truncated']);self.assertFalse(g['eos_observed']);self.assertNotIn(59050,b._global_codec_full);self.assertTrue(b.closed)
  b.mode='ok';_,_,g=generate_sample(b,None,None,'next');self.assertEqual(g['status'],'ok')
 def test_engine_error_not_masked(self):
  with self.assertRaisesRegex(RuntimeError,'CUDA'):generate_sample(Fake('engine'),None,None,'x')
 def test_no_audio_not_fabricated(self):
  b=Fake('no_eos');b.s2st_engine._global_codec_tokens=[59049]
  with self.assertRaises(RuntimeError):generate_sample(b,None,None,'x')
 def test_denominator_includes_failure(self):
  s=summarize([dict(generation_status='no_eos'),dict(generation_status='ok')]);self.assertEqual(s['failure_rate'],.5);self.assertEqual(s['total'],2)
if __name__=='__main__':unittest.main()
