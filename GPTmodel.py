import torch.nn as nn
from transformer import *
class GPT(nn.Module):
    def __init__(self, vocab_size, max_len=1024, N=12, d_model=768, d_ff=768*4, h=12, dropout=0.1, pad=2):
        super(GPT, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.generator = Generator(d_model, vocab_size)
        self.pad = pad
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens):
        sum_emb = self.token_emb(tokens)
        positions = torch.arange(tokens.size(1)).unsqueeze(0).expand_as(tokens).to(tokens.device)
        sum_emb += self.pos_emb(positions)
        attention_masks = self.make_std_mask(tokens, pad=self.pad)
        encoded_X = self.encoder(sum_emb, attention_masks)
        return self.generator(encoded_X)

    @staticmethod
    def make_std_mask(tgt, pad=2):
        "Create a mask to hide padding and future words."
        "tgt: (batch_size, tgt_len)"
        tgt_mask = (tgt != pad).unsqueeze(-2) # tgt_mask: (batch_size, 1, tgt_len)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data) # tgt_mask: (batch_size, tgt_len, tgt_len)
        return tgt_mask