"""Allocate distinct internal vLLM ports across all engines of one GPU worker.
A file lock and monotonic counter avoid the ephemeral-port check/close/rebind race.
Each GPU has a separate 512-port range below Linux's ephemeral client port range.
"""
import fcntl,os,socket
from pathlib import Path

def install():
 state=os.environ.get('OFFLINE_PORT_STATE');base=os.environ.get('OFFLINE_PORT_BASE')
 if not state or not base:return
 base=int(base);assert 1024<=base and base+511<65536
 path=Path(state);path.parent.mkdir(parents=True,exist_ok=True)
 import vllm.utils.network_utils as network
 def allocate():
  with path.open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);n=int(f.read().strip() or 0)
   for _ in range(512):
    port=base+(n%512);n+=1
    try:
     with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:s.bind(('',port))
    except OSError:continue
    f.seek(0);f.truncate();f.write(str(n));f.flush();return port
  raise RuntimeError(f'No free vLLM port in worker range {base}..{base+511}')
 network._get_open_port=allocate
