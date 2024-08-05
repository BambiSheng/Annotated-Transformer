import torch.nn as nn
from transformer import *
class BertEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=1000, N=12, d_model=768, d_ff=768*4, h=12, dropout=0.1):
        super(BertEncoder, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.segment_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, segments, valid_lens):
        sum_emb = self.token_emb(tokens) + self.segment_emb(segments)
        positions = torch.arange(tokens.size(1)).unsqueeze(0).expand_as(tokens).to(tokens.device)
        sum_emb += self.pos_emb(positions)
        mask = torch.zeros_like(tokens).to(tokens.device)
        for i, valid_len in enumerate(valid_lens):
            mask[i, :valid_len] = 1
        mask = mask.unsqueeze(-2)
        return self.encoder(sum_emb, mask)

class MaskLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, d_ff=768*4):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.LayerNorm(d_ff),
            nn.Linear(d_ff, vocab_size),
        )
    
    def forward(self, X, pred_positions):
        # X: output of encoder，[batch_size, seq_length, d_model]
        # pred_positions: positions to predict，[batch_size, num_pred_positions]    
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
        
class NSP(nn.Module):
    def __init__(self, dim):
        super(NSP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.output = nn.Linear(dim, 2)

    def forward(self, x):
        return self.output(self.mlp(x))
    
class BERT(nn.Module):
    def __init__(self, vocab_size, max_len=1000, N=12, d_model=768, d_ff=768*4, h=12, dropout=0.1):
        super(BERT, self).__init__()
        self.encoder = BertEncoder(vocab_size, max_len, N, d_model, d_ff, h, dropout)
        self.masklm = MaskLM(vocab_size, d_model, d_ff)
        self.nsp = NSP(d_model)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.masklm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(encoded_X[:, 0, :])
        return encoded_X, mlm_Y_hat, nsp_Y_hat