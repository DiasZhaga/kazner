"""Reference oracle: functions copied VERBATIM from the pilot repository.

Source: ner-project @ b6ee209 — src/train_ner_mbert.py (read_iob2, make_hf_dataset,
build_label_maps) and src/run_peft_comparison.py (sample_train_subset). Do not edit:
tests compare kazner against this code to prove that behaviour is unchanged.
"""

# ruff: noqa
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset


def read_iob2(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    s_tokens, s_labels = [], []
    tokens, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    s_tokens.append(tokens)
                    s_labels.append(labels)
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            if label == "0":
                label = "O"
            tokens.append(token)
            labels.append(label)
    if tokens:
        s_tokens.append(tokens)
        s_labels.append(labels)
    return s_tokens, s_labels


def make_hf_dataset(tokens: List[List[str]], labels: List[List[str]]) -> Dataset:
    return Dataset.from_dict({"tokens": tokens, "labels_str": labels})


def build_label_maps(train_labels: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted({l for sent in train_labels for l in sent})
    label2id = {l: i for i, l in enumerate(uniq)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def sample_train_subset(
    train_tokens: List[List[str]],
    train_labels: List[List[str]],
    fraction: float,
    seed: int,
) -> Dataset:
    total_sentences = len(train_tokens)
    sample_size = max(1, math.ceil(total_sentences * fraction))

    rng = random.Random(seed)
    indices = list(range(total_sentences))
    rng.shuffle(indices)
    selected = sorted(indices[:sample_size])

    subset_tokens = [train_tokens[i] for i in selected]
    subset_labels = [train_labels[i] for i in selected]
    return make_hf_dataset(subset_tokens, subset_labels)
