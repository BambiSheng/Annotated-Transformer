'''
Using the pretrained miniBert to predict the <mask> token.
'''

import torch
import sys
sys.path.append('../..')
from BERTmodel import BERT
from transformers import BertTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = torch.load('./model/vocab.pth')
tokenizer = BertTokenizerFast(vocab_file='./model/vocab.txt', mask_token='<mask>')

model = BERT(len(vocab)).to(device)
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./model/baseBERT.pth').items()})
model.eval()

def predict_masked_text(masked_text, mask_token='<mask>'):
    tokens = tokenizer.tokenize(masked_text)
    tokens = ['<cls>'] + tokens + ['<sep>']
    input_ids = [vocab[token] for token in tokens]
    token_type_ids = [0] * len(input_ids)
    valid_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)
    valid_len = torch.tensor(valid_len).unsqueeze(0).to(device)
    pred_positions = [i for i, token in enumerate(tokens) if token == mask_token]
    pred_positions = torch.tensor(pred_positions).unsqueeze(0).to(device)
    print(pred_positions)
    
    _, mlm_Y_hat, _ = model(input_ids, token_type_ids, valid_len, pred_positions)

    mlm_Y_hat = torch.functional.F.softmax(mlm_Y_hat, dim=-1).squeeze(0).squeeze(0)
    # Get the top 5 predictions
    topk_values, topk_indices = torch.topk(mlm_Y_hat, 5)
    # Convert the predictions to tokens
    topk_tokens = [vocab.get_itos()[index] for index in topk_indices]

    # Convert the probabilities to a Python list
    topk_probs = topk_values.tolist()

    return list(zip(topk_tokens, topk_probs))


    

if __name__ == '__main__':
    masked_text = 'I went on vacation to beijing this summer. <mask> visited the Great Wall.'
    print(predict_masked_text(masked_text))

