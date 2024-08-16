import torch
from tqdm import tqdm
from transformers import BertTokenizerFast
import datasets
import os
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, vocab, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab = vocab
        print("Preprocessing data...")
        self.input_ids, self.token_type_ids, self.attention_mask, self.start_positions, self.end_positions = self.preprocess_function(data)
        print(self.input_ids[0])
            

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.start_positions[idx], self.end_positions[idx])

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=self.max_seq_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors='pt'
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in tqdm(enumerate(offset_mapping)):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

            
        return inputs['input_ids'], inputs["token_type_ids"], inputs["attention_mask"], start_positions, end_positions

    def valid_start_end_positions(self, start_positions, end_positions, context_len):
        if start_positions is None or start_positions < 0 or start_positions >= context_len:
            return False
        if end_positions is None or end_positions < 0 or end_positions >= context_len:
            return False
        if start_positions > end_positions:
            return False
        return True

    

if __name__ == "__main__":
    ds = datasets.load_dataset("rajpurkar/squad")
    dataset_dir = './dataset'
    vocab_path = '../eg2_miniBert/model/vocab.pth'
    vocab_txt_path = '../eg2_miniBert/model/vocab.txt'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(vocab_txt_path):
        vocab = torch.load(vocab_path)
        with open(vocab_txt_path, 'w') as f:
            for word in vocab.get_itos():
                f.write(word + '\n')

    tokenizer = BertTokenizerFast(vocab_file=vocab_txt_path, pad_token='<pad>', cls_token='<cls>', sep_token='<sep>', unk_token='<unk>')
    vocab = torch.load(vocab_path)
    train_set = SquadDataset(ds['train'], tokenizer, vocab, 384)
    valid_set = SquadDataset(ds['validation'], tokenizer, vocab, 384)
    # save the dataset
    import os
    torch.save(train_set, os.path.join(dataset_dir, 'train_set.pth'))
    torch.save(valid_set, os.path.join(dataset_dir, 'valid_set.pth'))
    