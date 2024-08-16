# Example 2. Mini BERT Pretrain
> [BERT](https://arxiv.org/abs/1810.04805) unveiled the high potential that transformer owns in a great range of NLP tasks.
> Inspired by [d2l](https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html), we use the dataset [WikiText-2](https://huggingface.co/datasets/carlosejimenez/wikitext__wikitext-2-raw-v1) to pretrain a miniBERT.
> Then we finetune the miniBERT to apply to both token-level tasks ([SQuAD](https://hf-mirror.com/datasets/rajpurkar/squad)) in eg.3 and sequence-level tasks ([IMDb](https://hf-mirror.com/datasets/stanfordnlp/imdb)) in eg.4.

## File Structure
- `./model`: Save model files.
- `./dataset`: Cache directory for datasets.
- `pretrain.py`: Pretrain a miniBERT.
- `WikiTextDataset.py`: Dataset Class for WikiText-2.
- `maskDemo.py`: Demo for MLM task.
## Run
- `python pretrain.py`: Start the pretraining.
- `python maskDemo.py`: Show MLM demo.
## Result
- `miniBERT`
  - parameters: $d_{model}=128, d_{ff}=256, Layer = 2, Head = 2$
  - MLM loss: [MLM](./mlm_loss_mini.png)
  - NSP loss: [NSP](./nsp_loss_mini.png)
- `baseBERT`
  - parameters: $d_{model}=768, d_{ff}=3072, Layer = 12, Head = 12$
  - MLM loss: [MLM](./mlm_loss_base.png)
  - NSP loss: [NSP](./nsp_loss_base.png)
- Note that our BERT model can't do as well as the official BERT model in MLM tasks, due to the insufficient pretraining data we use (only about 10MB wiki-text).