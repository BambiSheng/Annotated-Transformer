import sys
sys.path.append('../..')
from BERTmodel import BERT, MaskLM
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
vocab_size = 10000
model = BERT(vocab_size).to(device)
print("Model loaded.")
tokens = torch.randint(0, vocab_size, (2, 8)).to(device)
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]]).to(device)
encoded_X = model(tokens, segments, None)
print("encoded_X:", encoded_X.shape)

mlm = MaskLM(vocab_size, 768).to(device)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]]).to(device)
mlm_Y_hat = mlm(encoded_X, mlm_positions)
print("mlm_Y_hat:", mlm_Y_hat.shape)