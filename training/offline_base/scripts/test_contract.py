import torch
from transformers import Qwen2Config,Qwen2Model
from offline_contract import talker_attention_mask

def main():
 torch.manual_seed(7)
 cfg=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.)
 cfg._attn_implementation='sdpa'
 model=Qwen2Model(cfg).eval()
 x=torch.randn(1,9,32,requires_grad=True)
 ids=torch.tensor([[1,1,1,1,2,2,2,2,2]])
 positions=torch.tensor([[0,1,2,3,0,1,2,3,4]])
 y=model(inputs_embeds=x,position_ids=positions,attention_mask=talker_attention_mask(ids,x.dtype),use_cache=False).last_hidden_state
 single=model(inputs_embeds=x[:,4:],position_ids=positions[:,4:],use_cache=False).last_hidden_state
 torch.testing.assert_close(y[:,4:],single,atol=1e-5,rtol=1e-5)
 y[:,4:].sum().backward();assert x.grad[:,:4].abs().max()==0
 assert x.grad[:,4:].abs().sum()>0
 changed=x.detach().clone();changed[:,:4]*=100
 z=model(inputs_embeds=changed,position_ids=positions,attention_mask=talker_attention_mask(ids,x.dtype),use_cache=False).last_hidden_state
 torch.testing.assert_close(y[:,4:],z[:,4:])
 print('PASS: packed==single; zero cross-sample gradients; earlier sample perturbation isolated')
if __name__=='__main__':main()
