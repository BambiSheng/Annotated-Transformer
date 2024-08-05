import torch.nn as nn
from transformer import make_model, subsequent_mask

class T4T(nn.Module): # short for Transformer for Translation
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(T4T, self).__init__()
        self.base_model = make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)
        
    def forward(self, src, tgt, pad=2): # pad=2 is the default padding token
        # upper triangle mask for target
        tgt_mask = self.make_std_mask(tgt)
        src_mask = (src != pad).unsqueeze(-2)
        out = self.base_model(src, tgt, src_mask, tgt_mask)
        out = self.base_model.generator(out)
        return  out

    @staticmethod
    def make_std_mask(tgt, pad=2):
        "Create a mask to hide padding and future words."
        "tgt: (batch_size, tgt_len)"
        tgt_mask = (tgt != pad).unsqueeze(-2) # tgt_mask: (batch_size, 1, tgt_len)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data) # tgt_mask: (batch_size, tgt_len, tgt_len)
        return tgt_mask