import torch
from transformers import BertTokenizer
import sys
sys.path.append('../..')
from T4Tmodel import T4T
max_len = 64

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model...")
model_path = "./model/model_196566.pt"
model = torch.load(model_path)
model.eval()
print("Model loaded.")

# Tokenizer
print("Loading tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def en_tokenizer(line):
    return bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.encode(line,  add_special_tokens=False))
def zh_tokenizer(line):
    return list(line.strip().replace(" ", ""))
print("Tokenizer loaded.")

# Vocab
print("Loading vocab...")
en_vocab = torch.load("./dataset/vocab/en_vocab.pt")
zh_vocab = torch.load("./dataset/vocab/zh_vocab.pt")
print("Vocab loaded.")

def translate(src):
    src = torch.tensor([0] + zh_vocab(zh_tokenizer(src)) + [1])
    assert src.max() < len(zh_vocab), "src contains index out of range"
    src = src.unsqueeze(0).to(device)
    tgt = torch.tensor([0]).unsqueeze(0).to(device)
    for _ in range(64):
        with torch.no_grad():
            output = model(src, tgt)
        pred = output.argmax(dim=-1)[:, -1]
        if pred.item() == 1:
            break
        pred = pred.unsqueeze(0)
        tgt = torch.cat([tgt, pred], dim=-1).to(device)
    tgt = ' '.join(en_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
    return tgt

while True:
    src = input("Input sentence: ")
    if src == "exit":
        break
    tgt = translate(src)
    print("Output sentence:", tgt)
