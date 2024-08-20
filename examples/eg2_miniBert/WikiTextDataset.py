from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchtext.vocab import build_vocab_from_iterator
import random
import torch
import datasets
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def _get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

def _get_next_sentence(sentence, next_sentence, paragraphs):
    '''With probability 0.5, next_sentence is the actual next sentence'''
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = _get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    # do the replacement
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # with 80% probability, replace the word with <mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # with 10% probability, keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # with 10% probability, replace the word with a random word
            else:
                masked_token = random.choice(vocab.get_itos())
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        # skip cls and sep tokens
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # with probability 0.15, set the token as a prediction token
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    mlm_input_tokens = [vocab[tk] for tk in mlm_input_tokens]
    mlm_pred_labels = [vocab[tk] for tk in mlm_pred_labels]
    return mlm_input_tokens, pred_positions, mlm_pred_labels

def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

class WikiTextDataset(Dataset):
    def __init__(self, paras, max_len):
        # Tokenize paragraphs
        print("tokenizing paragraphs.")
        counter = 0
        for para in tqdm(paras):
            for i, sentence in enumerate(para):
                para[i] = tokenizer.tokenize(sentence)
        
        # Build vocabulary
        def yield_tokens(data_iter):
            for para in tqdm(data_iter, total=len(data_iter)):
                for sentence in para:
                    yield sentence
        print("buiding vocab.")
        self.vocab = build_vocab_from_iterator(yield_tokens(paras), min_freq=2, specials=['<pad>', '<mask>', '<cls>', '<sep>', "<unk>"])
        self.vocab.set_default_index(self.vocab['<unk>'])
        print("vocab size:", len(self.vocab))

        # Generate nsp examples
        # examples = [[(tokens, segments, is_next)]]
        examples = []
        for para in paras:
            examples.extend(_get_nsp_data_from_paragraph(
                para, paras, max_len))
        # Generate mlm examples
        # examples = [(tokens, pred_positions, mlm_pred_label_ids, segments, is_next)]
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

        
    def __len__(self):
        return len(self.all_token_ids)


    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])
if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    ds_train = ds["train"]
    paras = []
    for example in tqdm(ds_train):
            if(len(example["text"].split(' . ')) < 2):
                continue
            para = example["text"].strip().lower().split(" . ")
            paras.append(para)
    random.shuffle(paras)
    batch_size, max_len = 512, 64
    train_set = WikiTextDataset(paras, max_len)
    torch.save(train_set, 'train_set.pth')
    # save vocab
    torch.save(train_set.vocab, 'vocab.pth')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size,shuffle=True)

    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
        mlm_Y, nsp_y) in train_loader:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
            pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
            nsp_y.shape)
        break