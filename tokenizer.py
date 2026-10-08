"""
Character-level tokenizer: one token = one character.

This is the simplest tokenizer that works. Real LLMs use sub-word tokenizers
(BPE) where one token is roughly 3/4 of a word, which makes sequences much
shorter -- but the model code does not care: it only ever sees integer ids.
"""

import json
from pathlib import Path

META_PATH = Path(__file__).parent / "data" / "meta.json"


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_meta(cls, path=META_PATH):
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(meta["chars"])

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"Characters not in the vocabulary: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
