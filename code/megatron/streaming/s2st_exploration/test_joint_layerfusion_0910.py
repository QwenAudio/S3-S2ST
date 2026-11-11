import copy,types,unittest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from joint_layer_fusion import install_thinker_layer_fusion

class Layer(nn.Module):
    def __init__(self,n):
        super().__init__();self.layer_number=n;self.proj=nn.Linear(4,4)
    def forward(self,hidden_states,**kwargs):return hidden_states+torch.tanh(self.proj(hidden_states))/32,None
class Decoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.config=cfg;self.layers=nn.ModuleList([Layer(i) for i in range(1,33)])
        self.num_layers_per_pipeline_rank=32;self.norm=nn.LayerNorm(4)
    def _get_layer(self,i):return self.layers[i]
    def forward(self,x):
        if self.config.recompute_granularity=='full' and self.training:
            x=self._checkpointed_forward(x,None,None,None,None,None)
        else:
            for layer in self.layers:x,_=layer(x)
        return self.norm(x)
class Model(nn.Module):
    def __init__(self,full=False):
        super().__init__()
        self.config=types.SimpleNamespace(hidden_size=4,recompute_granularity='full' if full else 'selective',
            recompute_method='uniform',recompute_num_layers=1,distribute_saved_activations=False,fp8=False)
        self.audio_encoder=nn.Linear(4,4);self.decoder=Decoder(self.config);self.cosyvoice_llm_input_proj=nn.Linear(4,3)
    def forward(self,x):
        h=self.decoder(self.audio_encoder(x));return h,self.cosyvoice_llm_input_proj(h)
def ckpt(f,distribute,*args):return checkpoint(f,*args,use_reentrant=True)

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123)
    def test_identity_matches_original_text_and_codec(self):
        a=Model();b=copy.deepcopy(a);install_thinker_layer_fusion(b)
        x=torch.randn(5,4)
        for y,z in zip(a(x),b(x)):self.assertTrue(torch.equal(y,z))
    def test_full_recompute_matches_all_gradients(self):
        a=Model();b=copy.deepcopy(a);b.config.recompute_granularity='full'
        fa=install_thinker_layer_fusion(a);fb=install_thinker_layer_fusion(b,checkpoint_fn=ckpt)
        with torch.no_grad():fa.weights.normal_(0,.2);fb.weights.copy_(fa.weights)
        for _ in range(2):
            x=torch.randn(3,4)
            ya=a(x);yb=b(x)
            for y,z in zip(ya,yb):torch.testing.assert_close(y,z,rtol=0,atol=0)
            (ya[1].square().mean()+ya[0].square().mean()).backward()
            (yb[1].square().mean()+yb[0].square().mean()).backward()
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m);self.assertIsNotNone(p.grad,n);self.assertIsNotNone(q.grad,m)
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)
        self.assertTrue((fb.weights.grad.abs().sum(1)>0).all())
    def test_early_tap_codec_gradient_reaches_encoder(self):
        m=Model(True);f=install_thinker_layer_fusion(m,checkpoint_fn=ckpt)
        with torch.no_grad():f.weights.zero_();f.weights[0].fill_(1)
        m(torch.randn(3,4))[1].square().mean().backward()
        self.assertGreater(m.audio_encoder.weight.grad.abs().sum(),0)
        self.assertGreater(m.decoder.layers[3].proj.weight.grad.abs().sum(),0)
        self.assertTrue(m.decoder.layers[4].proj.weight.grad is None or m.decoder.layers[4].proj.weight.grad.abs().sum()==0)
    def test_full_model_eval_uses_same_taps(self):
        a=Model();b=copy.deepcopy(a);b.config.recompute_granularity='full'
        fa=install_thinker_layer_fusion(a);fb=install_thinker_layer_fusion(b,checkpoint_fn=ckpt)
        with torch.no_grad():fa.weights.normal_();fb.weights.copy_(fa.weights)
        a.eval();b.eval();x=torch.randn(2,4)
        with torch.no_grad():torch.testing.assert_close(a(x)[1],b(x)[1],rtol=0,atol=0)
    def test_parent_state_dict_key_and_loaded_weights(self):
        m=Model(True);f=install_thinker_layer_fusion(m,checkpoint_fn=ckpt)
        self.assertIn('thinker_layer_fusion.weights',m.state_dict())
        state=copy.deepcopy(m.state_dict());state['thinker_layer_fusion.weights'].normal_();m.load_state_dict(state)
        torch.testing.assert_close(f.weights,state['thinker_layer_fusion.weights'])

if __name__=='__main__':unittest.main()
