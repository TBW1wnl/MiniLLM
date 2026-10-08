"""
Train the GPT on an encoded dataset.

    python train.py                                # Tiny Shakespeare, default config (~10.7M params)
    python train.py --n_layer 4 --max_iters 2000   # any TrainConfig field can be overridden
    python train.py --dataset tinystories ...      # after python data/prepare_tinystories.py

Every eval_interval steps it reports train/val loss, prints a short sample so
you can watch the model learn to write, and saves the best checkpoint to
<out_dir>/ckpt.pt.
"""

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer

DATA_DIR = Path(__file__).parent / "data"

# name -> (directory holding train.bin / val.bin, tokenizer loader, default sample prompt)
DATASETS = {
    "shakespeare": (DATA_DIR, lambda: CharTokenizer.from_meta(), "\n"),
    "tinystories": (
        DATA_DIR / "tinystories",
        lambda: BPETokenizer.from_file(DATA_DIR / "tinystories" / "tokenizer.json"),
        "Once upon a time",
    ),
}


@dataclass
class TrainConfig:
    dataset: str = "shakespeare"  # a key of DATASETS

    # Model shape (see GPTConfig)
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2

    # Mixture of Experts (n_experts = 0 trains the plain dense model)
    n_experts: int = 0
    experts_per_token: int = 2
    aux_loss_coef: float = 0.01  # weight of the load-balancing loss

    # Optimization
    batch_size: int = 64  # sequences per step -> 64 * 256 = 16k tokens per step
    max_iters: int = 5000
    learning_rate: float = 1e-3  # peak LR, reached after warmup
    min_lr: float = 1e-4  # LR at the end of the cosine decay
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Evaluation and logging
    eval_interval: int = 250
    eval_iters: int = 200  # batches averaged per loss estimate
    log_interval: int = 50
    sample_tokens: int = 200  # length of the sample printed at each eval
    sample_prompt: str = ""  # empty = the dataset's default prompt

    # System
    out_dir: str = "out"
    seed: int = 1337
    compile: bool = False  # torch.compile; needs Triton (pip install triton-windows on Windows)


def parse_args():
    """Expose every TrainConfig field as a --flag with the same name."""
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if f.type is bool:
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    return TrainConfig(**vars(parser.parse_args()))


def load_split(data_dir, name, device):
    # The whole split is copied to the GPU so batches can be sliced there
    # directly. Stored as int32 (4 bytes per token) that is 4 MB for Tiny
    # Shakespeare and ~1.9 GB for TinyStories: fine on a 24 GB card. For a
    # dataset bigger than VRAM you would keep it on disk with np.memmap and
    # copy each batch over instead.
    data = np.fromfile(data_dir / f"{name}.bin", dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int32)).to(device)


def get_batch(data, batch_size, block_size):
    """
    Pick batch_size random windows of block_size+1 tokens. The input is the
    window minus its last token, the target is the window shifted by one:

        text:    T  o     b  e
        x:       T  o     b
        y:          o     b  e      (y[t] is the token that follows x[..t])
    """
    starts = torch.randint(len(data) - block_size, (batch_size,), device=data.device)
    offsets = torch.arange(block_size + 1, device=data.device)
    windows = data[starts[:, None] + offsets].long()  # (B, block_size + 1); embeddings need int64
    return windows[:, :-1], windows[:, 1:]


def get_lr(it, cfg):
    """Linear warmup, then cosine decay from learning_rate down to min_lr."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def make_optimizer(model, cfg):
    # Weight decay pulls weights towards zero to limit overfitting. It is
    # applied to matrices (linear layers, embeddings) but not to 1-D
    # parameters like LayerNorm gains and biases, where it only hurts.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.99), fused=True)


@torch.no_grad()
def estimate_loss(model, splits, cfg):
    """Average the loss over many batches: a single batch is too noisy."""
    model.eval()  # disables dropout
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, cfg.block_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def format_expert_load(model):
    """One line per block: share of routing slots each expert received in the
    last forward pass. Perfect balance is 100% / n_experts for every expert."""
    lines = []
    for i, block in enumerate(model.blocks):
        load = block.mlp.last_load.tolist()
        lines.append(f"    layer {i}: " + " ".join(f"{p:5.1%}" for p in load))
    return "\n".join(lines)


@torch.no_grad()
def sample_text(model, tok, prompt, n_tokens):
    model.eval()
    start = torch.tensor([tok.encode(prompt)], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        new = torch.cat(list(model.generate(start, n_tokens, temperature=0.8, top_k=40)), dim=1)
    model.train()
    return prompt + tok.decode(new[0].tolist())


def main():
    cfg = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: check that the CUDA build of PyTorch is installed.")

    torch.manual_seed(cfg.seed)
    # Let float32 matmuls use TF32 tensor cores (much faster on RTX 30/40).
    torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(exist_ok=True)

    data_dir, load_tokenizer, default_prompt = DATASETS[cfg.dataset]
    tok = load_tokenizer()
    prompt = cfg.sample_prompt or default_prompt
    splits = {name: load_split(data_dir, name, "cuda") for name in ("train", "val")}
    print(f"Dataset {cfg.dataset}: {len(splits['train']):,} train tokens, {len(splits['val']):,} val tokens")

    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        n_experts=cfg.n_experts,
        experts_per_token=cfg.experts_per_token,
    )
    use_moe = cfg.n_experts > 0
    model = GPT(model_cfg).cuda()
    print(f"Model: {model.num_params() / 1e6:.2f}M parameters", end="")
    if use_moe:
        print(f", {model.num_active_params() / 1e6:.2f}M active per token "
              f"({cfg.n_experts} experts, {cfg.experts_per_token} per token)", end="")
    print()
    print(f"Untrained loss should be about ln({tok.vocab_size}) = {math.log(tok.vocab_size):.2f}")

    optimizer = make_optimizer(model, cfg)

    # Keep a handle on the plain module: torch.compile wraps it, and we want
    # to save clean state_dict keys.
    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss", "val_loss"])

    best_val = float("inf")
    tokens_per_step = cfg.batch_size * cfg.block_size
    t0, steps_timed = time.perf_counter(), 0

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, splits, cfg)
            print(f"\n=== step {it}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if use_moe:
                # last_load comes from the last val batch of estimate_loss.
                print("    expert load (share of tokens per expert):\n" + format_expert_load(raw_model))
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}"])

            # Only keep the checkpoint that generalizes best. When train loss
            # keeps falling but val loss rises, the model is memorizing.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "train_config": asdict(cfg),
                        "tokenizer": tok.to_dict(),
                        "iter": it,
                        "val_loss": best_val,
                    },
                    out_dir / "ckpt.pt",
                )
                print(f"    saved checkpoint (best val loss so far)")

            print("--- sample ---\n" + sample_text(raw_model, tok, prompt, cfg.sample_tokens).strip("\n") + "\n--------------")
            t0, steps_timed = time.perf_counter(), 0  # don't count eval time in the step timing

        if it == cfg.max_iters:
            break

        # One optimization step: forward, backward, update.
        x, y = get_batch(splits["train"], cfg.batch_size, cfg.block_size)
        # bfloat16 autocast: matmuls run in 16-bit on the tensor cores, with
        # the same exponent range as float32 so no loss scaling is needed.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, aux_loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        # The aux loss only matters for MoE (it is 0 for a dense model). Its
        # small weight keeps it from competing with the real objective.
        (loss + cfg.aux_loss_coef * aux_loss).backward()
        # Rescale the gradient if its norm is too large: protects against the
        # occasional huge update that can derail training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        steps_timed += 1
        if (it + 1) % cfg.log_interval == 0:
            loss_val = loss.item()  # .item() waits for the GPU, so timing is accurate
            dt = (time.perf_counter() - t0) / steps_timed
            aux_str = f" | aux {aux_loss.item():.3f}" if use_moe else ""
            print(f"step {it + 1:5d} | loss {loss_val:.4f}{aux_str} | lr {lr:.2e} | {dt * 1000:.1f} ms/step | {tokens_per_step / dt / 1e3:.0f}k tok/s")
            t0, steps_timed = time.perf_counter(), 0

    print(f"\nDone. Best val loss {best_val:.4f}, checkpoint in {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
