# Example 2. Mini BERT Pretrain
> [BERT](https://arxiv.org/abs/1810.04805) unveiled the high potential that transformer owns in a great range of NLP tasks.
> Inspired by [d2l](https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html), we use the dataset [WikiText-2](https://huggingface.co/datasets/carlosejimenez/wikitext__wikitext-2-raw-v1) to pretrain a miniBERT.
> Then we finetune the miniBERT to apply to both token-level tasks ([SQuAD](https://hf-mirror.com/datasets/rajpurkar/squad)) and sequence-level tasks ([IMDb](https://hf-mirror.com/datasets/stanfordnlp/imdb)).

## File Structure
- `./model`: Save model files.
- `./dataset`: Cache directory for datasets.
- `pretrain.py`: Pretrain a miniBERT.
- `WikiTextDataset.py`: Dataset Class for WikiText-2.
- TODO: Finetune MiniBERT on SQuAD & IMDb.
## Run
- `python pretrain.py`: Start the pretraining.