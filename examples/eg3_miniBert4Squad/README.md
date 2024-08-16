# Example 3. Mini BERT's token-level task Fine-Tuning
> This example is based on the miniBERT model pretrained in eg.2.
> In eg.3 we finetune the miniBERT to apply to both token-level tasks - QA. We use this dataset: ([SQuAD](https://hf-mirror.com/datasets/rajpurkar/squad)).

## File Structure
- `./model`: Save model files.
- `./dataset`: Cache directory for datasets.
- `squadTune.py`: Finetune miniBERT on squad dataset.
- `SquadDataset.py`: Dataset Class for QA task.
- `SquadModel.py`: Implementation of the Model Class `BertForQA`.
- `QAdemo.py`: Demo for QA task.
- `squadTuneOfficial.py`: Finetune the official Google BERT on squad, using huggingface🤗 style.
## Run
- `python squadTune.py`: Start the fine-tuning.
- `python QAdemo.py`: Show QA demo.
## Result (Accuracy Plot)
- `miniBERT`
[mini](./model/mini_train_acc.png)
- `baseBERT`
[base](./model/train_acc.png)
- Note that the performance of our miniBERT/baseBERT is much poorer than the official well-pretrained BERT, due to the insufficient pretraining data we use (only about 10MB wiki-text).