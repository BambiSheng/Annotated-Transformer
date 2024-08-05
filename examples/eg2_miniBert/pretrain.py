import sys
sys.path.append('../..')
from BERTmodel import BERT
import torch
import datasets
from tqdm import tqdm
import random
from WikiTextDataset import WikiTextDataset
import os
import torch.nn as nn

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 64
batch_size = 128
dataset_cache_dir = 'dataset/pretrain/train_set.pth'
model_dir = 'model/pretrain/BERT.pth'
use_cache = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data Preprocessing
if use_cache and os.path.exists(dataset_cache_dir):
    print("loading dataset from cache.")
    train_set = torch.load(dataset_cache_dir)
else: 
    print("loading dataset from scratch.")
    ds = datasets.load_dataset("carlosejimenez/wikitext__wikitext-2-raw-v1")
    ds_train = ds["train"]
    paras = []
    for example in tqdm(ds_train):
            if(len(example["text"].split(' . ')) < 2):
                    continue
            para = example["text"].strip().lower().split(" . ")
            paras.append(para)
    random.shuffle(paras)
    train_set = WikiTextDataset(paras, max_len)
    if not os.path.exists('dataset/pretrain'):
        os.makedirs('dataset/pretrain')
    torch.save(train_set, dataset_cache_dir)

train_loader = torch.utils.data.DataLoader(train_set, batch_size,shuffle=True)

# Model
vocab_size = len(train_set.vocab)
model = BERT(vocab_size).to(device)
loss = nn.CrossEntropyLoss(reduce='none')

# Loss
def _get_batch_loss_bert(model, loss, vocab_size, tokens_X,segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,mlm_Y, nsp_y):
    # transfer Tensor to List of integer
    valid_lens_x = valid_lens_x.int()
    _, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X, valid_lens_x, pred_positions_X)
    # calculate the loss of MLM
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # calculate the loss of NSP
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

# Training
lr, num_epochs = 1e-4, 10
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in train_loader:
        tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y = tokens_X.to(device), segments_X.to(device), valid_lens_x.to(device), pred_positions_X.to(device), mlm_weights_X.to(device), mlm_Y.to(device), nsp_y.to(device)
        optimizer.zero_grad()
        mlm_l, nsp_l, l = _get_batch_loss_bert(model, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
        l.backward()
        optimizer.step()
        loop.set_description(f'epoch {epoch + 1}')
        loop.set_postfix(mlm_loss=mlm_l.item(), nsp_loss=nsp_l.item())
        loop.update(1)

# Save the model
if not os.path.exists(model_dir):
     os.makedirs(model_dir)
torch.save(model.state_dict(), model_dir)

