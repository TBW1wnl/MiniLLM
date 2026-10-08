"""
Download Tiny Shakespeare and turn it into training data.

The model never sees text: it only sees integers. This script builds the
simplest possible tokenizer -- one token per character -- and uses it to
convert the whole corpus into two arrays of integers saved on disk:

    data/train.bin   first 90% of the text
    data/val.bin     last 10%, never trained on, used to detect overfitting
    data/meta.json   the vocabulary, needed later to decode generated tokens

Run once:  python data/prepare.py
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    input_path = DATA_DIR / "input.txt"
    if not input_path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, input_path)

    text = input_path.read_text(encoding="utf-8")
    print(f"Corpus length: {len(text):,} characters")

    # The vocabulary is every distinct character in the corpus (65 here).
    # Each character gets an integer id: its index in the sorted list.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"Vocabulary size: {len(chars)}")
    print("Vocabulary:", "".join(chars).encode("unicode_escape").decode())

    # Encode the full text. uint16 is plenty for 65 ids and halves the file
    # size compared to int32.
    ids = np.array([stoi[ch] for ch in text], dtype=np.uint16)

    # Split by position, not randomly: the validation text is a contiguous
    # chunk the model has never seen, which is a more honest test.
    n = int(0.9 * len(ids))
    ids[:n].tofile(DATA_DIR / "train.bin")
    ids[n:].tofile(DATA_DIR / "val.bin")
    print(f"train: {n:,} tokens | val: {len(ids) - n:,} tokens")

    (DATA_DIR / "meta.json").write_text(json.dumps({"chars": chars}), encoding="utf-8")


if __name__ == "__main__":
    main()
