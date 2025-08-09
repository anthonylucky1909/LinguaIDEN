import torch

def create_src_mask(src, pad_token=0):
    return (src != pad_token).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_token=0):
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    return pad_mask & causal_mask