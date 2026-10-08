"""
Prepare TinyStories: train a BPE tokenizer on it, then encode the corpus.

TinyStories (Eldan & Li, 2023) is ~2.1M short stories written by GPT-3.5/4
with the vocabulary of a 3-4 year old. It was designed so that small models
(a few million to a few tens of millions of parameters) can learn to write
coherent English: exactly our setting.

Expects the Hugging Face parquet files in TinyStories/ at the repo root and
writes to data/tinystories/:

    tokenizer.json   the trained BPE tokenizer
    train.bin        ~470M token ids (uint16), stories separated by <|endoftext|>
    val.bin          ~4.7M token ids

Run once:  python data/prepare_tinystories.py   (a few minutes)
"""

import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "TinyStories"
OUT_DIR = Path(__file__).parent / "tinystories"

VOCAB_SIZE = 8192
EOT = "<|endoftext|>"  # marks the boundary between two stories
TOKENIZER_TRAIN_STORIES = 300_000  # a subset is enough to learn the merges
BATCH = 50_000  # stories encoded per call


def read_stories(pattern):
    """Yield the stories of every parquet file matching pattern, in order."""
    for path in sorted(glob.glob(str(SRC_DIR / pattern))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=["text"]):
            yield batch.column("text").to_pylist()


def train_tokenizer():
    """
    Byte-level BPE (the GPT-2 recipe): start from the 256 possible bytes, so
    any text can be encoded, then repeatedly merge the most frequent adjacent
    pair into a new token until the vocabulary has VOCAB_SIZE entries.
    Frequent words ("the", " once", " upon") end up as single tokens, rare
    ones are spelled with several sub-word pieces.
    """
    tok = Tokenizer(models.BPE())
    # Split on spaces/punctuation first so merges never cross word boundaries;
    # a leading space stays attached to the word (" upon" != "upon").
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],  # gets id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def subset():
        seen = 0
        for stories in read_stories("train-*.parquet"):
            yield stories
            seen += len(stories)
            if seen >= TOKENIZER_TRAIN_STORIES:
                return

    tok.train_from_iterator(subset(), trainer=trainer)
    return tok


def encode_split(tok, pattern, out_path):
    """Encode every story, prefixed by <|endoftext|>, and append to out_path."""
    eot_id = tok.token_to_id(EOT)
    n_tokens = n_stories = 0
    with open(out_path, "wb") as f:
        for stories in read_stories(pattern):
            # encode_batch runs on all CPU cores.
            encodings = tok.encode_batch(stories)
            ids = [eot_id]
            for enc in encodings:
                ids.extend(enc.ids)
                ids.append(eot_id)
            ids.pop()  # the next batch starts with its own EOT
            arr = np.array(ids, dtype=np.uint16)  # 8192 < 65536, so 2 bytes per token
            arr.tofile(f)
            n_tokens += len(arr)
            n_stories += len(stories)
            print(f"  {out_path.name}: {n_stories:,} stories, {n_tokens:,} tokens", end="\r")
    print()
    return n_tokens, n_stories


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tok_path = OUT_DIR / "tokenizer.json"

    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
        print(f"Reusing {tok_path}")
    else:
        print(f"Training a {VOCAB_SIZE}-token BPE on {TOKENIZER_TRAIN_STORIES:,} stories...")
        tok = train_tokenizer()
        tok.save(str(tok_path))

    example = "Once upon a time, there was a little girl named Lily."
    pieces = tok.encode(example).tokens
    print(f"Example: {len(example)} chars -> {len(pieces)} tokens: {pieces}")

    for split, pattern in (("val", "validation-*.parquet"), ("train", "train-*.parquet")):
        n_tokens, n_stories = encode_split(tok, pattern, OUT_DIR / f"{split}.bin")
        print(f"{split}: {n_stories:,} stories -> {n_tokens:,} tokens ({n_tokens / n_stories:.0f} per story)")


if __name__ == "__main__":
    main()
