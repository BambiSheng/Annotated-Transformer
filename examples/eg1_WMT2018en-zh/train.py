import os
from os.path import exists
import sys
sys.path.append('../..')
from datasets import load_dataset
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from Dataset4Translation import getTranslationCF, TranslationDataset, TranslationLoss
from tqdm import tqdm
from T4Tmodel import T4T
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator

# Traing Parameters
max_len = 64
batch_size = 128
epoch = 1
save_interval = 10000
use_cache = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

dataset_dir = "./dataset"
model_dir = "./model"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

model_checkpoint = None
model_checkpoint_dir = model_dir
model_checkpoint_path = os.path.join(model_checkpoint_dir, "model_0.pt")
en_vocab_path = os.path.join(dataset_dir, "vocab/en_vocab.pt")
zh_vocab_path = os.path.join(dataset_dir, "vocab/zh_vocab.pt")
show_sample = False # Show sample translation result during training

# Vocab
print("Loading dataset...")
ds = load_dataset("wmt/wmt18", "zh-en")
print("Dataset loaded.")

if(use_cache and exists(en_vocab_path)):
    print("Loading cached en-vocab.")
    en_vocab = torch.load(en_vocab_path)
    print("Loading cached en-vocab finished.")
else:
    print("building en-vocab")
    en_vocab = build_vocab_from_iterator(map(lambda x: en_tokenizer(x['en']), tqdm(ds['train']['translation'])), min_freq=2, specials=["<s>", "</s>", "<pad>", "<unk>"])
    en_vocab.set_default_index(en_vocab['<unk>'])
    torch.save(en_vocab, en_vocab_path)
    print("Building en-vocab finished.")

if(use_cache and exists(zh_vocab_path)):
    print("Loading cached zh-vocab.")
    zh_vocab = torch.load(zh_vocab_path)
    print("Loading cached zh-vocab finished.")
else:
    print("building zh-vocab")
    zh_vocab = build_vocab_from_iterator(map(lambda x: zh_tokenizer(x['zh']), tqdm(ds['train']['translation'])), min_freq=2, specials=["<s>", "</s>", "<pad>", "<unk>"])
    zh_vocab.set_default_index(zh_vocab['<unk>'])
    torch.save(zh_vocab, zh_vocab_path)
    print("Building zh-vocab finished.")

print("en_vocab size: ", len(en_vocab))
print("en_vacab_example:", dict((i, en_vocab.lookup_token(i)) for i in range(10))) 
print("zh_vocab size: ", len(zh_vocab))
print("zh_vacab_example:", dict((i, zh_vocab.lookup_token(i)) for i in range(10)))

# Tokenizer
print("Loading tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def en_tokenizer(line):
    return bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.encode(line,  add_special_tokens=False))
def zh_tokenizer(line):
    return list(line.strip().replace(" ", ""))
print("Tokenizer loaded.")

# Building tokens
token_cache_dir = os.path.join(dataset_dir, "tokens")
if not os.path.exists(token_cache_dir):
    os.makedirs(token_cache_dir)
src_cache_path = os.path.join(token_cache_dir, "src_tokens.pt")
tgt_cache_path = os.path.join(token_cache_dir, "tgt_tokens.pt")
src_tokens = []
tgt_tokens = []
if use_cache and os.path.exists(src_cache_path) and os.path.exists(tgt_cache_path):
    print("Loading cached tokens.")
    src_tokens = torch.load(src_cache_path)
    tgt_tokens = torch.load(tgt_cache_path)
    print("Loading cached tokens finished.")
else:        
    print("Building tokens.")
    for d in tqdm(ds['train']['translation']):
        tgt_tokens.append(en_vocab(en_tokenizer(d['en'])))
        src_tokens.append(zh_vocab(zh_tokenizer(d['zh'])))
        
    if use_cache:
        print("Saving tokens.")
        torch.save(src_tokens, src_cache_path)
        torch.save(tgt_tokens, tgt_cache_path)
        print("Saving tokens finished.")

# dataset
print("Building dataset.")
dataset = TranslationDataset(src_tokens, tgt_tokens)
print("Dataset built.")

# dataloader
print("Building dataloader.")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=getTranslationCF(max_len, 2))
print("Dataloader built.")

# model
step = 0
if model_checkpoint:
    print("Loading model checkpoint.")
    model = torch.load(model_checkpoint_path)
    print("Model checkpoint loaded.")
    cp_name = model_checkpoint.split("/")[-1]
    step = int(cp_name.replace("model_", "").replace(".pt", ""))
else:
    print("Building model.")
    model = T4T(len(zh_vocab), len(en_vocab))
    print("Model built.")
model = model.to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print("Training...")
model.train()
criterion = TranslationLoss(len(en_vocab), 2, 0.1)
for e in range(epoch):
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, data in enumerate(dataloader):
        src, tgt, tgt_y, token_num = data
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)
        optimizer.zero_grad()
        out = model(src, tgt)
        loss = criterion(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1))
        loss /= token_num
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch {e}/{epoch}")
        loop.set_postfix(loss=loss.item())
        step += 1
        loop.update(1)
                    
        if show_sample and step % 100 == 0:
            src_sample = src[0]
            tgt_sample = tgt[0]
            tgt_y_sample = tgt_y[0]
            src_sample = ''.join(zh_vocab.lookup_tokens(src_sample.tolist()))
            tgt_sample = ''.join(en_vocab.lookup_tokens(tgt_sample.tolist()))
            tgt_y_sample = ''.join(en_vocab.lookup_tokens(tgt_y_sample.tolist()))
            print("src:", src_sample)
            print("tgt:", tgt_sample)
            print("tgt_y:", tgt_y_sample)
            out_sample = out[0]
            out_sample = out_sample.argmax(dim=-1)
            print(out_sample.size())
            out_sample = ' '.join(en_vocab.lookup_tokens(out_sample.tolist()))
            print("out:", out_sample)

        del src
        del tgt
        del tgt_y

        if step % save_interval == 0 and step != 0:
            print("Saving model checkpoint at step", step)
            torch.save(model, os.path.join(model_checkpoint_dir, f"model_{step}.pt"))
            print("Model checkpoint saved.")



print("Training finished.")
print("Saving model.")
torch.save(model, os.path.join(model_checkpoint_dir, f"model_{step}.pt"))
print("Model saved.")