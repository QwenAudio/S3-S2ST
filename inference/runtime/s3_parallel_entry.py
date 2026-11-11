"""Bound KV memory without changing paper decoding or model code."""
import os
from vllm.engine.arg_utils import AsyncEngineArgs
_original_post_init = AsyncEngineArgs.__post_init__
def _bounded_kv(self):
    self.kv_cache_memory_bytes = int(os.environ['FLEURS_KV_CACHE_BYTES'])
    _original_post_init(self)
AsyncEngineArgs.__post_init__ = _bounded_kv
import s3_syncgraphs_entry as frozen
if __name__ == '__main__':
    frozen.common.m.main()
