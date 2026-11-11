import copy
import io
import types
import unittest
import torch
from torch import nn
from frozen_layer_fusion import HadamardLayerFusion, install_thinker_layer_fusion


class Block(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer_number = n
        self.weight = nn.Parameter(torch.eye(3) * (1 + n / 100))
    def forward(self, x):
        return x @ self.weight + 0.01 * self.layer_number, None


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=3, recompute_granularity="selective")
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([Block(i) for i in range(1, 33)])
        self.decoder.final_layernorm = nn.LayerNorm(3)
        self.decoder.requires_grad_(False)
        self.cosyvoice_llm_input_proj = nn.Linear(3, 2)
    def forward(self, x):
        for block in self.decoder.layers:
            x, _ = block(x)
        x = self.decoder.final_layernorm(x)
        return self.cosyvoice_llm_input_proj(x)


class Tests(unittest.TestCase):
    def test_off_no_change(self):
        m = Model(); x = torch.randn(5, 1, 3)
        keys = set(m.state_dict()); y = m(x)
        self.assertIsNone(install_thinker_layer_fusion(m, enabled=False))
        self.assertEqual(keys, set(m.state_dict()))
        self.assertTrue(torch.equal(y, m(x)))

    def test_identity_gradients_and_repeated_forward(self):
        torch.manual_seed(1024)
        m = Model(); original = copy.deepcopy(m); x = torch.randn(5, 1, 3)
        f = install_thinker_layer_fusion(m, enabled=True)
        self.assertEqual(f.layer_numbers, (4, 8, 12, 16, 20, 24, 28, 32))
        y = m(x); self.assertTrue(torch.equal(y, original(x)))
        # Reproduce the existing post-projection mapping, with zero drain slots.
        mapped = y[[0, 1, 1, 2, 3]] * torch.tensor([1, 1, 0, 1, 1]).view(5, 1, 1)
        self.assertTrue(torch.equal(mapped[2], torch.zeros_like(mapped[2])))
        mapped.square().mean().backward()
        self.assertTrue(torch.isfinite(f.weights.grad).all())
        self.assertTrue((f.weights.grad.abs().sum(dim=1) > 0).all())
        self.assertIsNotNone(m.cosyvoice_llm_input_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in m.decoder.parameters()))
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        before = f.weights.detach().clone(); opt.step()
        self.assertFalse(torch.equal(before, f.weights))
        self.assertTrue(torch.isfinite(m(x)).all())
        # State-dict restore carries learned vectors exactly.
        buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
        restored = Model(); install_thinker_layer_fusion(restored, enabled=True)
        restored.load_state_dict(torch.load(buf, weights_only=True))
        self.assertTrue(torch.equal(m(x), restored(x)))

    def test_hadamard_sum(self):
        f = HadamardLayerFusion([4, 8], 3)
        with torch.no_grad():
            f.weights.copy_(torch.tensor([[1., 2., 3.], [-1., 0.5, 2.]]))
        a = torch.randn(4, 2, 3); b = torch.randn_like(a)
        self.assertTrue(torch.equal(f([a, b]), a * f.weights[0] + b * f.weights[1]))

    def test_backbone_trainability_guard(self):
        m = Model(); m.decoder.requires_grad_(True)
        with self.assertRaises(ValueError):
            install_thinker_layer_fusion(m, enabled=True)

    def test_dist_checkpoint(self):
        # Exercise the same Megatron replicated-module serialization as the real model.
        from megatron.core.transformer.module import MegatronModule
        from megatron.core import dist_checkpointing
        from megatron.core import parallel_state
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            torch.distributed.init_process_group('gloo', init_method='file://' + td + '/init', rank=0, world_size=1)
            parallel_state.initialize_model_parallel()
            try:
                class Tiny(MegatronModule):
                    def __init__(self):
                        super().__init__(types.SimpleNamespace())
                        self.thinker_layer_fusion = HadamardLayerFusion(range(4, 33, 4), 3)
                m = Tiny()
                with torch.no_grad(): m.thinker_layer_fusion.weights.uniform_(-1, 1)
                import os
                path = td + '/ckpt'; os.mkdir(path)
                dist_checkpointing.save(m.sharded_state_dict(), path)
                n = Tiny(); state = dist_checkpointing.load(n.sharded_state_dict(), path)
                n.load_state_dict(state)
                self.assertTrue(torch.equal(m.thinker_layer_fusion.weights, n.thinker_layer_fusion.weights))
            finally:
                parallel_state.destroy_model_parallel()
                torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
