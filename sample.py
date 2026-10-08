"""
Generate text with a trained checkpoint, streamed one token at a time.

    python sample.py
    python sample.py --prompt "ROMEO:" --num_tokens 1000 --temperature 0.7
    python sample.py --checkpoint out_ts/ckpt.pt --prompt "Once upon a time, a dragon"
"""

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out/ckpt.pt")
    parser.add_argument("--prompt", default=None, help="text the model continues from")
    parser.add_argument("--num_tokens", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40, help="0 disables top-k filtering")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    ckpt = torch.load(args.checkpoint, map_location="cuda")
    # Checkpoints from before BPE support only stored the character list.
    tok = tokenizer_from_dict(ckpt["tokenizer"]) if "tokenizer" in ckpt else CharTokenizer(ckpt["chars"])
    model = GPT(GPTConfig(**ckpt["model_config"])).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt['iter']}, val loss {ckpt['val_loss']:.4f})")

    is_bpe = isinstance(tok, BPETokenizer)
    prompt = args.prompt if args.prompt is not None else ("Once upon a time" if is_bpe else "\n")
    start = torch.tensor([tok.encode(prompt)], device="cuda")
    for i in range(args.num_samples):
        print(f"\n===== sample {i + 1} =====")
        print(prompt, end="", flush=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for next_id in model.generate(start, args.num_tokens, args.temperature, args.top_k or None):
                # The model writes <|endoftext|> when it considers the story finished.
                if is_bpe and next_id.item() == tok.eot_id:
                    break
                print(tok.decode(next_id[0].tolist()), end="", flush=True)
        print()


if __name__ == "__main__":
    main()
