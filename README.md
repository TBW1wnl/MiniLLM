# MiniLLM

A small GPT (decoder-only Transformer) trained from scratch on Tiny Shakespeare, in a few hundred lines of commented PyTorch. Educational project: every file is meant to be read.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy
```

## Usage

```bash
python data/prepare.py      # download the corpus and encode it to data/*.bin
python train.py             # ~2-3 minutes on an RTX 4090, saves out/ckpt.pt
python sample.py --prompt "ROMEO:" --num_tokens 500
```

Any training setting can be overridden from the command line, e.g. `python train.py --n_layer 8 --n_embd 512 --max_iters 8000`. Run `python train.py --help` for the full list.

## Files

| File | What it does |
|---|---|
| `data/prepare.py` | Downloads the text, builds the 65-character vocabulary, writes `train.bin` / `val.bin` |
| `tokenizer.py` | Character-level tokenizer: text <-> list of integer ids |
| `model.py` | The GPT itself: embeddings, causal self-attention, MLP, Transformer blocks, sampling |
| `train.py` | Training loop: batching, AdamW, LR warmup + cosine decay, bf16, evaluation, checkpoints |
| `sample.py` | Loads a checkpoint and streams generated text |

## What to expect

- **Step 0**: loss ~4.17 = ln(65), i.e. a uniform guess over the 65 characters. Output is random characters.
- **~300 steps**: the model has learned the play format (`NAME:` + line breaks) and common short words.
- **~1500 steps**: best val loss, around 1.47. Mostly real English words, plausible character names and verse layout, but no real meaning.
- **After that**: train loss keeps falling (down to ~0.6) while val loss climbs back up (~1.72 at step 5000). The model is memorizing the training text instead of learning general patterns: this is overfitting. Only the checkpoint with the best val loss is saved, so `out/ckpt.pt` is the step-1500 model.

`out/metrics.csv` logs train/val loss at every evaluation. Note that the loss printed every 50 steps is higher than the "train loss" at evaluations: the former is measured with dropout on, the latter with dropout off.

## Ideas to explore

- **Size**: change `n_layer`, `n_embd`, `block_size` and watch speed and val loss.
- **Overfitting**: train with `--dropout 0.0` and compare the train/val gap.
- **Sampling**: compare `--temperature 0.5` and `--temperature 1.2`, or `--top_k 0`.
- **Attention**: set `flash=False` in `GPTConfig` to run the explicit attention math (slower, same result), then plot an attention matrix.
- **Speed**: `pip install triton-windows`, then `python train.py --compile true`.
- **Tokenizer**: replace the character tokenizer with BPE (e.g. `tiktoken`'s GPT-2 encoding) and train on a bigger corpus.
