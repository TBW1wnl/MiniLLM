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

## Mixture of Experts

```bash
python train.py --n_experts 8 --experts_per_token 2 --out_dir out_moe
python sample.py --checkpoint out_moe/ckpt.pt
```

With `--n_experts N`, the MLP of every block becomes a `MoE` layer (see `model.py`): N independent MLPs plus a router that sends each token to its `experts_per_token` best experts and mixes their outputs. With 8 experts and 2 active, the model holds 60.2M parameters but each token only uses 17.75M, so compute grows much less than size.

A load-balancing loss (`--aux_loss_coef`, Switch Transformer style) stops the router from sending everything to a few experts. Training prints its value (`aux`, 1.0 = perfectly balanced) and, at each eval, the share of tokens each expert receives per layer.

On Tiny Shakespeare this mostly shows how routing works, not why MoE helps: the dense model already overfits 1M characters, and 6x more parameters only memorize them faster. MoE pays off when data is not the bottleneck.

## TinyStories

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is ~2.1M short stories with a small child's vocabulary, designed so that models with tens of millions of parameters can write coherent English. Download the parquet files into `TinyStories/` at the repo root, then:

```bash
pip install pyarrow tokenizers
python data/prepare_tinystories.py     # trains an 8192-token BPE, encodes ~470M tokens into data/tinystories/
python train.py --dataset tinystories --block_size 512 --n_layer 8 --n_head 8 --n_embd 512 --dropout 0 --out_dir out_ts
python sample.py --checkpoint out_ts/ckpt.pt --prompt "Once upon a time, a dragon"
```

The tokenizer is byte-level BPE trained on the corpus itself: frequent words become single tokens, so a story is ~4x fewer tokens than characters. With ~470M training tokens, one pass over the data takes many thousands of steps and overfitting is no longer the problem, so dropout can be turned off. This is the setting where a Mixture of Experts can use its extra parameters.

Dense vs MoE on the same 197M tokens (6000 steps x 32k tokens, lr 6e-4, RTX 4090):

| | Dense | MoE, 8 experts / 2 active |
|---|---|---|
| Parameters (total / active) | 29.4M / 29.4M | 146.9M / 46.2M |
| Val loss at step 6000 | 1.487 | **1.423** |
| Speed | ~330k tok/s (~10 min) | ~165k tok/s (~20 min) |

Per token seen, the MoE is clearly better (it reaches the dense model's final loss around step 4300). Per second of GPU, the dense model still wins here: our MoE layer loops over experts in Python, and its active parameter count is 1.6x the dense one. Both write coherent short stories.

## Ideas to explore

- **Size**: change `n_layer`, `n_embd`, `block_size` and watch speed and val loss.
- **Overfitting**: train with `--dropout 0.0` and compare the train/val gap.
- **Sampling**: compare `--temperature 0.5` and `--temperature 1.2`, or `--top_k 0`.
- **Attention**: set `flash=False` in `GPTConfig` to run the explicit attention math (slower, same result), then plot an attention matrix.
- **Speed**: `pip install triton-windows`, then `python train.py --compile true`.
- **Tokenizer**: replace the character tokenizer with BPE (e.g. `tiktoken`'s GPT-2 encoding) and train on a bigger corpus.
