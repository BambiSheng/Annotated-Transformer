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
MODEL_TYPE = 'base' # 'base' or 'mini'
# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 64
batch_size = 128
dataset_cache_dir = 'dataset/train_set.pth'
model_dir = './model'
use_cache = True
use_DP = True # DataParallel

# Data Preprocessing
if use_cache and os.path.exists(dataset_cache_dir):
    print("loading dataset from cache.")
    train_set = torch.load(dataset_cache_dir)
else: 
    raise Exception("No cache file found.")

train_loader = torch.utils.data.DataLoader(train_set, batch_size,shuffle=True)

# Model
vocab_size = len(train_set.vocab)

if MODEL_TYPE == 'base':
     model = BERT(vocab_size).to(device)
elif MODEL_TYPE == 'mini':
    model = BERT(vocab_size, N=2, d_model=128, d_ff=256, h=2).to(device)


# Use DataParallel for multi-GPU
if use_DP and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

loss = nn.CrossEntropyLoss(reduction='none')

# Loss
def _get_batch_loss_bert(model, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y, debug=False):
    # transfer Tensor to List of integer
    valid_lens_x = valid_lens_x.int()
    _, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X, valid_lens_x, pred_positions_X)
    # calculate the loss of MLM
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # calculate the loss of NSP
    nsp_l = loss(nsp_Y_hat, nsp_y).sum() / nsp_y.shape[0]
    l = mlm_l + nsp_l
    if debug:
            # print mlm sample
            pred_positions = pred_positions_X[0].nonzero().squeeze()
            for pos in pred_positions:
                print(train_set.vocab.get_itos()[mlm_Y[0][pos].item()], end=' ')
                print('->', end=' ')
                print(train_set.vocab.get_itos()[mlm_Y_hat[0][pos].argmax(dim=-1).item()])
            print()

    return mlm_l, nsp_l, l

# Training
lr, num_epochs, weight_decay = 1e-4, 10, 0.01
mlm_l_list = []
nsp_l_list = []
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.train()
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in loop:
        tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y = tokens_X.to(device), segments_X.to(device), valid_lens_x.to(device), pred_positions_X.to(device), mlm_weights_X.to(device), mlm_Y.to(device), nsp_y.to(device)
        optimizer.zero_grad()
        mlm_l, nsp_l, l = _get_batch_loss_bert(model, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
        l.backward()
        optimizer.step()
        loop.set_description(f'epoch {epoch + 1}')
        loop.set_postfix(mlm_loss=mlm_l.item(), nsp_loss=nsp_l.item())
        loop.update(1)
        mlm_l_list.append(mlm_l.item())
        nsp_l_list.append(nsp_l.item())
    # save the model every epoch
    torch.save(model.state_dict(), os.path.join(model_dir, MODEL_TYPE + 'BERT_checkpoint.pth'))
        

# Save the model and vocab
if not os.path.exists(model_dir):
     os.makedirs(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, MODEL_TYPE + 'BERT.pth'))

# Plot the loss and save
import matplotlib.pyplot as plt
plt.plot(mlm_l_list, label='mlm')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('MLM Loss' + ' ' + MODEL_TYPE)
plt.legend()
plt.savefig(f'mlm_loss_{MODEL_TYPE}.png')

plt.cla()

plt.plot(nsp_l_list, label='nsp')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('NSP Loss' + ' ' + MODEL_TYPE)
plt.legend()
plt.savefig(f'nsp_loss_{MODEL_TYPE}.png')