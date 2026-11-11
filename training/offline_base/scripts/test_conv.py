import torch
from packed_causal_conv import packed_safe_causal_conv1d as conv
x=torch.randn(1,3,11,requires_grad=True);w=torch.randn(3,4)
idx=torch.tensor([[0]*5+[1]*6])
a=conv(x,w,seq_idx=idx,activation='silu')
b=conv(x[:,:,5:],w,activation='silu')
torch.testing.assert_close(a[:,:,5:],b)
a[:,:,5:].sum().backward();assert x.grad[:,:,:5].abs().sum()==0
print('PASS: causal convolution pack reset, values and gradients')
