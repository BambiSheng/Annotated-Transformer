import torch.nn as nn
import sys
sys.path.append('../..')    
sys.path.append('../eg2_miniBert')
from BERTmodel import BERT
from transformers import BertModel
import torch

class BertForQA(nn.Module):
    def __init__(self, bert, d_model=768):
        super(BertForQA, self).__init__()
        self.bert_encoder = bert.encoder
        self.qa_nn = nn.Linear(d_model, 2)
        
    def forward(self, tokens, segments, attention_mask=None):
        encoded_X = self.bert_encoder(tokens, segments, attention_mask)
        encoded_X = self.qa_nn(encoded_X)
        # return the start and end positions
        return encoded_X[:, :, 0], encoded_X[:, :, 1]
