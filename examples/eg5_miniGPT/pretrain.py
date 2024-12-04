import torch
import torch.nn as nn
import os
import sys
sys.path.append('../..')
from GPTmodel import GPT
import tiktoken
import numpy as np
from tqdm import tqdm

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 512
block_size = 256
batch_size = 32
save_steps = 5000
eval_steps = 500
model_dir = './model'
use_cache = True
use_DP = True # DataParallel
trainset_path = 'dataset/train.npy'
valset_path = 'dataset/val.npy'

# tokenizer
enc = tiktoken.get_encoding("gpt2")
n_vocab = enc.n_vocab

# Data Preprocessing
if use_cache and os.path.exists(trainset_path):
    print("loading dataset from cache.")
    train_set = np.load(trainset_path)
    eval_set = np.load(valset_path)
else: 
    raise Exception("No cache file found.")

# Model
model = GPT(n_vocab, pad=-1).to(device)

# Use DataParallel for multi-GPU
if use_DP and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
lr, num_steps, weight_decay = 0.0001, 10000, 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training
print("Training...")
loss_list = []
eval_loss_list = []
loop = tqdm(range(num_steps))
for i in loop:
    model.train()
    data = train_set
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]).to(device)
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out.view(-1, n_vocab), y.view(-1))
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    loop.set_description(f"Epoch {i}/{num_steps}")
    loop.set_postfix(loss=loss.item())
    loop.update(1)
    if i % save_steps == 0 and i > 0:
        torch.save(model, os.path.join(model_dir, f'model_step_{i}.pt'))
    if i % eval_steps == 0:
        model.eval()
        data = eval_set
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]).to(device)
        out = model(x)
        loss = criterion(out.view(-1, n_vocab), y.view(-1))
        eval_loss_list.append(loss.item())
        print(f"Eval loss: {loss.item()}")


# save model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.save(model, os.path.join(model_dir, 'model.pt'))

# plot loss
import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig(os.path.join(model_dir, 'loss.png'))
print("Training finished.")