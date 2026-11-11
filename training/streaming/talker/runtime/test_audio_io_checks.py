"""CPU-only fault injection and optional real OSS read checks."""
import os
import sys
import types
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import audio_io_checks as guard


class FakeProcess:
    def __init__(self, raw, rc=0, err=b''):
        self.raw, self.returncode, self.err = raw, rc, err
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def communicate(self, **kwargs): return self.raw, self.err


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        env = patch.dict(os.environ, {"EXP_DIR": temporary.name})
        env.start()
        self.addCleanup(env.stop)
        # Unit tests isolate decoder contract; avoid FunASR's eager CUDA JIT imports.
        lu = types.ModuleType('funasr.utils.load_utils')
        lu.PATH_TRANSFORMER = types.SimpleNamespace(transform=lambda p: p)
        utils = types.ModuleType('funasr.utils')
        utils.load_utils = lu
        funasr = types.ModuleType('funasr')
        funasr.utils = utils
        stub = patch.dict(sys.modules, {'funasr': funasr, 'funasr.utils': utils,
                                       'funasr.utils.load_utils': lu})
        stub.start()
        self.addCleanup(stub.stop)
        guard._strict_sources.clear()
    def decode(self, samples, rc=0, err=b''):
        raw = np.zeros(samples, dtype=np.float32).tobytes()
        with patch.object(guard.subprocess, 'Popen', return_value=FakeProcess(raw, rc, err)):
            return guard.decode_checked('https://example.invalid/a.wav?Signature=SECRET', duration=1)
    def test_valid(self):
        a, sr = self.decode(32000)
        self.assertEqual(a.size, 16000)
        self.assertEqual(sr, 16000)
    def test_short_even_exit_zero(self):
        with self.assertRaisesRegex(RuntimeError, 'short_pcm'): self.decode(1000)
    def test_nonempty_nonzero(self):
        with self.assertRaisesRegex(RuntimeError, 'nonzero_exit'): self.decode(16000, 1)
    def test_transport_even_full(self):
        with self.assertRaisesRegex(RuntimeError, 'transport_error'):
            self.decode(16000, err=b'HTTP error 503 Service Unavailable')
    def test_roundoff_not_padded(self):
        self.assertEqual(self.decode(15999)[0].size, 15999)
    def test_target_grid_tail(self):
        self.assertEqual(self.decode(15520, err=b'Duration: 00:00:00.97')[0].size, 15520)
    def test_source_grid_tail(self):
        guard._strict_sources.add('oss://example/a.wav')
        self.assertEqual(self.decode(15520, err=b'Duration: 00:00:00.97')[0].size, 15520)
    def test_midfile_short_not_relaxed(self):
        with self.assertRaisesRegex(RuntimeError, 'short_pcm'):
            self.decode(15520, err=b'Duration: 00:00:02.97')
    def test_redaction(self):
        self.assertNotIn('SECRET', guard.redact('https://x/a?Signature=SECRET'))
    def test_collator_recovery(self):
        class Dataset:
            _prepare_single_data = staticmethod(lambda packed_items: None)
            def __len__(self): return 5
            def __getitem__(self, index): return {'input_ids': torch.ones(8, dtype=torch.long)}
            def collator(self, samples): return {'input_ids': torch.stack([s['input_ids'] for s in samples])}
        guard.install_dataset_guards(Dataset)
        self.assertEqual(tuple(Dataset().collator([None])['input_ids'].shape), (1, 8))
    def test_persistent_failure(self):
        class Dataset:
            _prepare_single_data = staticmethod(lambda packed_items: None)
            def __len__(self): return 5
            def __getitem__(self, index): return None
            def collator(self, samples): return {}
        guard.install_dataset_guards(Dataset)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            Dataset().collator([None])


if __name__ == '__main__':
    unittest.main()
