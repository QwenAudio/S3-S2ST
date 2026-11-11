"""Shared, tested offline packed-attention contract."""
def talker_attention_mask(sample_ids, dtype):
    import torch
    n=sample_ids.shape[1]
    same=sample_ids[:,:,None].eq(sample_ids[:,None,:])
    causal=torch.ones((n,n),device=sample_ids.device,dtype=torch.bool).tril()
    allowed=same & causal
    return torch.zeros((sample_ids.shape[0],1,n,n),device=sample_ids.device,dtype=dtype).masked_fill_(~allowed[:,None],torch.finfo(dtype).min)
