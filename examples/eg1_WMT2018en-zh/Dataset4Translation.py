from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn.functional import pad

def getTranslationCF(max_len, pad_id):
    def collate_fn(batch):
        bos_id = torch.tensor([0])
        eos_id = torch.tensor([1])
        src_list = []
        tgt_list = []
        for (batch_src, batch_tgt) in batch:
            real_src = torch.cat([bos_id, torch.tensor(batch_src, dtype=torch.int64), eos_id], dim=0)
            real_tft = torch.cat([bos_id, torch.tensor(batch_tgt, dtype=torch.int64), eos_id], dim=0)
            # Truncate or pad
            if len(real_src) > max_len:
                real_src = real_src[:max_len]
            else:
                real_src = pad(real_src, (0, max_len - len(real_src)), value=pad_id)
            if len(real_tft) > max_len:
                real_tft = real_tft[:max_len]
            else:
                real_tft = pad(real_tft, (0, max_len - len(real_tft)), value=pad_id)
            src_list.append(real_src)
            tgt_list.append(real_tft)
        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        # tgt_y is used for calculating loss
        # tgt is used for decoding
        tgt_y = tgt[:, 1:]
        tgt = tgt[:, :-1]
        token_num = (tgt_y != pad_id).sum()
        return src, tgt, tgt_y, token_num
    return collate_fn


class TranslationDataset(Dataset):
    def __init__(self, src_tokens, tgt_tokens):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        assert len(self.src_tokens) == len(self.tgt_tokens)
        
    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        return self.src_tokens[idx], self.tgt_tokens[idx]

class TranslationLoss(nn.Module):  
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(TranslationLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())