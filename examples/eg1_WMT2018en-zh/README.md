# Example 1. Chinese to English Translation
> We use the dataset [WMT 2018 zh-en](https://huggingface.co/datasets/wmt/wmt18/viewer/zh-en) to train a **Chinese to English** Machine Translation model, as a first example of Transformers.

## File Structure
- `./dataset` : Cache for preprocessed data, like vocabulary or tokenized texts.
- `./model` : Directory for trained models.
- `train.py` : Code for training.
- `translate.py` : Code for translation using trained model.
- `Dataset4Translation.py` : Includes Dataset class, collate_fn and Loss class for this task.

## Run
- For Training: `python train.py`
- For Translation Demos: `python translate.py`
- **Remember to check and modify (if necessary) the parameters (especially those about file paths)** in `train.py`/`translate.py` before running the code. Maybe in later updates the parameters can be set via CLI.