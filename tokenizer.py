"""
Tokenizers: turn text into integer ids and back.

CharTokenizer -- one token per character (Tiny Shakespeare). The simplest
    tokenizer that works, but sequences are long: one token per letter.
BPETokenizer -- byte-level BPE trained by data/prepare_tinystories.py. One
    token is a frequent word or word piece, so the same context window holds
    ~4x more text.

The model code does not care which one is used: it only ever sees ids.
Both can be saved into a checkpoint (to_dict) and rebuilt from it
(tokenizer_from_dict), so a checkpoint is enough to generate text.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

DATA_DIR = Path(__file__).parent / "data"


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_meta(cls, path=DATA_DIR / "meta.json"):
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

    def to_dict(self):
        return {"type": "char", "chars": self.chars}


class BPETokenizer:
    EOT = "<|endoftext|>"

    def __init__(self, tok):
        self.tok = tok

    @classmethod
    def from_file(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    @property
    def eot_id(self):
        return self.tok.token_to_id(self.EOT)

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        # Keep <|endoftext|> visible so story boundaries show up in samples.
        return self.tok.decode(ids, skip_special_tokens=False)

    def to_dict(self):
        return {"type": "bpe", "json": self.tok.to_str()}


def tokenizer_from_dict(d):
    if d["type"] == "char":
        return CharTokenizer(d["chars"])
    return BPETokenizer(Tokenizer.from_str(d["json"]))
