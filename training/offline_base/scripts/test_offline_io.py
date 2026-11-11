import sys,types,unittest
from unittest.mock import patch
import numpy as np
import audio_io_checks as d
fake=types.ModuleType('funasr.utils.load_utils');fake.PATH_TRANSFORMER=types.SimpleNamespace(transform=lambda p:p)
sys.modules['funasr']=types.ModuleType('funasr');sys.modules['funasr.utils']=types.ModuleType('funasr.utils');sys.modules['funasr.utils.load_utils']=fake
class Proc:
 def __init__(self,raw,err=b'',code=0):self.raw=raw;self.err=err;self.returncode=code
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def communicate(self,timeout=None):return self.raw,self.err
 def kill(self):self.returncode=-9
class Tests(unittest.TestCase):
 def decode(self,raw,err=b'',code=0):
  with patch.object(d.subprocess,'Popen',return_value=Proc(raw,err,code)),patch.object(d,'event'),patch.object(d,'retain_pcm'):
   return d.decode_checked('source.wav',sr=16000,duration=1)
 def test_complete(self):self.assertEqual(len(self.decode(np.ones(16000,dtype=np.float32).tobytes())[0]),16000)
 def test_truncated(self):
  with self.assertRaisesRegex(RuntimeError,'short_pcm'):self.decode(np.ones(15800,dtype=np.float32).tobytes())
 def test_transport_exit_zero(self):
  with self.assertRaisesRegex(RuntimeError,'transport_error'):self.decode(np.ones(16000,dtype=np.float32).tobytes(),b'HTTP error 403')
 def test_nonzero(self):
  with self.assertRaisesRegex(RuntimeError,'nonzero_exit'):self.decode(np.ones(16000,dtype=np.float32).tobytes(),code=1)
 def test_bad_float(self):
  with self.assertRaisesRegex(RuntimeError,'incomplete_float'):self.decode(b'123')
if __name__=='__main__':unittest.main()
