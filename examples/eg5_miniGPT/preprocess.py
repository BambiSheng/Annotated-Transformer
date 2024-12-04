import numpy as np
import tiktoken
import json
from tqdm import tqdm
import os

# get tokenizer
enc = tiktoken.get_encoding("gpt2")

dataset_list = ['wiki-zh-subset-AA.jsonl', 'wiki-zh-subset-AB.jsonl', 'wiki-zh-subset-AC.jsonl', 'wiki-zh-subset-AD.jsonl']
encoded_data = []
for i, dataset in enumerate(dataset_list):
    with open(os.path.join("./dataset/ignore", dataset), 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    for d in tqdm(data):
        encoding = enc.encode(d['text'])
        encoded_data += encoding

# split dataset
train_size = int(len(encoded_data) * 0.99)
print("train size: ", train_size)
print("val size: ", len(encoded_data) - train_size)
train_data = encoded_data[:train_size]
val_data = encoded_data[train_size:]
# save dataset
np.save("./dataset/train", train_data)
np.save("./dataset/val", val_data)

