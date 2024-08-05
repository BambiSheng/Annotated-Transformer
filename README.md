# Annotated Transformer
> One usable torch-based implementation of transformer for learning purpose. Copied from [harvard nlp](http://nlp.seas.harvard.edu/annotated-transformer).

## NoteBooks
-  `transformer.ipynb`: Notebook version of the original Encoder-Decoder structure described in [`Attention is All you Need`](https://arxiv.org/abs/1706.03762). 

## Models
- `transformer.py`: The original Encoder-Decoder structure described in [`Attention is All you Need`](https://arxiv.org/abs/1706.03762). 
- `T4Tmodel.py`: Transformer for Translation Model. Used in [`Example 1`](./examples/eg1_WMT2018en-zh/).
- `BERTmodel.py`: Implementation of BERT. Used in [`Example 2`](./examples/eg2_miniBert/).

## Examples
Check out at `./examples`.

- [`Example 1`](./examples/eg1_WMT2018en-zh/): One Chinese to English translation model, trained on WMT 2018 en-zh dataset.
- [`Example 2`](./examples/eg2_miniBert/): Pretrain a miniBert, and finetune it to apply to some downstream tasks.

## Reference
- Attention is all you need: https://arxiv.org/abs/1706.03762
- Annotated Transformer, Havard NLP: http://nlp.seas.harvard.edu/annotated-transformer
- Pytorch Doc: https://pytorch.org/docs
- BERT: https://arxiv.org/abs/1810.04805
- d2l: https://d2l.ai/