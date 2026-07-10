import re
import json
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

PAD, UNK = "<pad>", "<unk>"
MAX_LEN = 64


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    return text.split()


def build_vocab(texts: List[str], min_freq: int = 1, max_size: int = 30000) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int = MAX_LEN) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


def load_dataset(csv_path: str, vocab: dict = None, max_len: int = MAX_LEN):
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), "Нужны колонки text,label"

    texts = df["text"].astype(str).tolist()
    labels = (df["label"].astype(int) - 1).to_numpy(dtype=np.int32)

    if vocab is None:
        vocab = build_vocab(texts)

    X = np.array([encode(t, vocab, max_len) for t in texts], dtype=np.int32)
    return X, labels, vocab


def save_vocab(vocab: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
