from squadDataset import SquadDataset
from squadModel import BertForQA

import sys
sys.path.append('../..')
sys.path.append('../eg2_miniBert')
from BERTmodel import BERT
import torch
import datasets
from tqdm import tqdm
import os
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import random_split

MODEL_TYPE = 'mini'  # 'base' or 'mini'

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 384
batch_size = 64
dataset_cache_dir = './dataset'
model_dir = './model'
use_cache = True
use_DDP = True # DistributedDataParallel
pretrain_model_path = '../eg2_miniBert/model/miniBERT.pth'

# Data Preprocessing
if use_cache and os.path.exists(dataset_cache_dir):
    print("loading dataset from cache.")
    train_set = torch.load(os.path.join(dataset_cache_dir, 'train_set.pth'))
    valid_set = torch.load(os.path.join(dataset_cache_dir, 'valid_set.pth'))
else: 
    print("loading dataset from scratch.")
    ds = datasets.load_dataset("rajpurkar/squad")
    dataset_dir = './dataset'
    vocab_path = '../eg2_miniBert/model/vocab.pth'
    vocab_txt_path = '../eg2_miniBert/model/vocab.txt'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(vocab_txt_path):
        vocab = torch.load(vocab_path)
        with open(vocab_txt_path, 'w') as f:
            for word in vocab.get_itos():
                f.write(word + '\n')

    tokenizer = BertTokenizerFast(vocab_file=vocab_txt_path, pad_token='<pad>', cls_token='<cls>', sep_token='<sep>', unk_token='<unk>')
    vocab = torch.load(vocab_path)
    train_set = SquadDataset(ds['train'], tokenizer, vocab, 384)
    valid_set = SquadDataset(ds['validation'], tokenizer, vocab, 384)
    # save the dataset
    import os
    torch.save(train_set, os.path.join(dataset_dir, 'train_set.pth'))
    torch.save(valid_set, os.path.join(dataset_dir, 'valid_set.pth'))
    
    

# divide the dataset into train_dataloader and valid_dataloader
valid_size = 1000
train_size = len(train_set) - valid_size
train_dataset, valid_dataset = random_split(train_set, [train_size, valid_size])

# sample 5000 data for training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)

# Model
bert_dict = torch.load(pretrain_model_path)
vocab_size = len(train_set.vocab)
if MODEL_TYPE == 'base':
    bert = BERT(vocab_size).to(device)    
    bert.load_state_dict(bert_dict)
    model = BertForQA(bert)

elif MODEL_TYPE == 'mini':
    bert = BERT(vocab_size, N=2, d_model=128, d_ff=256, h=2).to(device)
    bert.load_state_dict(bert_dict)
    model = BertForQA(bert, d_model=128)

# Use DataParallel for multi-GPU
if use_DDP and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# Loss
def _get_batch_loss_bert(model, loss, input_ids, token_type_ids, start_positions, end_positions, attn_mask):
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    start_positions = start_positions.to(device)
    end_positions = end_positions.to(device)

    start_positions_hat, end_positions_hat = model(input_ids, token_type_ids, attention_mask = attn_mask)
    l1 = loss(start_positions_hat, start_positions)
    l2 = loss(end_positions_hat, end_positions)
    l = l1 + l2
    return l1, l2, l

def _eval_correct(model, input_ids, token_type_ids, start_positions, end_positions, attn_mask):
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    start_positions = start_positions.to(device)
    end_positions = end_positions.to(device)

    start_positions_hat, end_positions_hat = model(input_ids, token_type_ids, attention_mask = attn_mask)
    start_pred = start_positions_hat.argmax(dim=-1)
    end_pred = end_positions_hat.argmax(dim=-1)
    start_real = start_positions
    end_real = end_positions
    start_correct = (start_pred == start_real).sum().item()
    end_correct = (end_pred == end_real).sum().item()
    return start_correct + end_correct

# Training
lr, num_epochs, weight_decay = 2e-4, 5, 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
train_acc = []
valid_acc = []
loss_list = []
loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
model.train()
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (input_ids, token_type_ids, attention_mask, start_positions, end_positions) in loop:
        model.train()
        optimizer.zero_grad()
        l1, l2, l = _get_batch_loss_bert(model, loss_func, input_ids, token_type_ids, start_positions, end_positions, attention_mask)
        l.backward()
        optimizer.step()
        loop.set_description(f'epoch {epoch + 1}')
        loop.set_postfix(loss=l.item()/2)
        loop.update(1)

        if i % 100 == 0:
            model.eval()
            train_total = 0
            train_correct = 0
            valid_total = 0
            valid_correct = 0
            # evaluate train_set
            for (input_ids, token_type_ids, attention_mask, start_positions, end_positions) in train_loader:
                train_total += input_ids.shape[0]
                train_correct += _eval_correct(model, input_ids, token_type_ids, start_positions, end_positions, attention_mask)
                break
            train_acc.append(train_correct / train_total / 2)
            # evaluate valid_set
            for (input_ids, token_type_ids, attention_mask, start_positions, end_positions) in valid_loader:
                valid_total += input_ids.shape[0]
                valid_correct += _eval_correct(model, input_ids, token_type_ids, start_positions, end_positions, attention_mask)
                
            valid_acc.append(valid_correct / valid_total / 2)
            loss_list.append(l.item() / 2)
            print(f"train_acc: {train_acc[-1]}, valid_acc: {valid_acc[-1]}, loss: {loss_list[-1]}")
            
# Save the model and vocab
if not os.path.exists(model_dir):
     os.makedirs(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, f'{MODEL_TYPE}BERTForQA.pth'))

# save log as csv
import matplotlib.pyplot as plt
plt.plot(train_acc, label='train_acc')
plt.plot(valid_acc, label='valid_acc')
plt.legend()
plt.savefig(os.path.join(model_dir, f'{MODEL_TYPE}_train_acc.png'))
plt.cla()
plt.plot(loss_list, label='loss')
plt.legend()
plt.savefig(os.path.join(model_dir, f'{MODEL_TYPE}_train_loss.png'))
import pandas as pd
log = pd.DataFrame({'train_acc': train_acc, 'valid_acc': valid_acc, 'loss': loss_list})

log.to_csv(os.path.join(model_dir, f'{MODEL_TYPE}_train_log.csv'), index=False)