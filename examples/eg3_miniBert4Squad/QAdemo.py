# Use trained model to predict the answer of a question

from squadModel import BertForQA
import sys
sys.path.append('../..')
sys.path.append('../eg2_miniBert')
from BERTmodel import BERT
from transformers import BertTokenizerFast
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = torch.load('../eg2_miniBert/model/vocab.pth')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = BERT(len(vocab))
model = BertForQA(bert).to(device)

model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./model/BERTForQA.pth').items()})
model.eval()

def predict_answer(question, context, max_len=384):
    question_tokens = tokenizer.tokenize(question)
    context_tokens = tokenizer.tokenize(context)
    max_context_length = max_len - len(question_tokens) - 3
    if len(context_tokens) > max_context_length:
        context_tokens = context_tokens[:max_context_length]
    valid_len = len(question_tokens) + len(context_tokens) + 3
    tokens = ['<cls>'] + question_tokens + ['<sep>'] + context_tokens + ['<sep>'] + ['<pad>'] * (max_len - len(question_tokens) - len(context_tokens) - 3)
    input_ids = [vocab[token] for token in tokens]
    token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1) + [0] * (max_len - len(question_tokens) - len(context_tokens) - 3)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor([1] * valid_len, dtype=torch.long).unsqueeze(0).to(device)
    start_positions_hat, end_positions_hat = model(input_ids, token_type_ids, attention_mask = attention_mask)
    start_pred = start_positions_hat.argmax(dim=-1)
    end_pred = end_positions_hat.argmax(dim=-1)
    answer_start = start_pred.item()
    answer_end = end_pred.item()
    answer_ids = input_ids[0][answer_start:answer_end+1]
    answer_list = [vocab.get_itos()[idx] for idx in answer_ids]
    answer = tokenizer.convert_tokens_to_string(answer_list)
    return answer

question = "Who went on vacation to Spain?"
context = "Some time ago, Jack Ma was photographed on vacation in Spain, wearing a white ball cap and a golf suit, looking very comfortable, and the yacht he had previously purchased for $200 million was moored on the shore. Many netizens lamented that this is really the standard leisure life of the rich."

answer = predict_answer(question, context)
print(answer)  
# Correct: Jack Ma
# Output: jack ma was photographed on vacation

    

