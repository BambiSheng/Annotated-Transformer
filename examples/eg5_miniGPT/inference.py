import torch
import torch.nn as nn
import os
import sys
sys.path.append('../..')
from GPTmodel import GPT
import tiktoken
import numpy as np
from tqdm import tqdm

model_dir = './model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inference
model = torch.load(os.path.join(model_dir, 'model.pt')).to(device)
model.eval()
print("Inference...")


# tokenizer
gpt2 = tiktoken.get_encoding("gpt2")
n_vocab = gpt2.n_vocab

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="gpt2",
    pat_str=gpt2._pat_str,
    mergeable_ranks=gpt2._mergeable_ranks,
    special_tokens={
        **gpt2._special_tokens,
        # "<|im_start|>": n_vocab,
        # "<|im_end|>": n_vocab + 1,
        # "<|pad|>": n_vocab + 2
    }
)

def generate_text(text, max_len=512):
    model.eval()
    text = enc.encode(text)
    for i in range(max_len):
        x = torch.tensor(text).unsqueeze(0).to(device)
        out = model(x)
        out = out[:, -1, :]
        next_token = torch.argmax(out, dim=-1).item()
        text.append(next_token)
    return enc.decode(text)

def generate_once(text):
    model.eval()
    text = enc.encode(text)
    x = torch.tensor(text).unsqueeze(0).to(device)
    out = model(x)
    out = out.argmax(dim=-1)
    return enc.decode(out.tolist()[0])
text = "冰岛克朗的出现最早是在第一次世界大战时,"
print("Input: ", text)
print("Output: ", generate_once(text))