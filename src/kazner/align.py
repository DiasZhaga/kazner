"""Word -> subword label alignment, as in the pilot (``train_ner_mbert.tokenize_and_align``).

Only the first subword of every word carries the word's label id; continuation subwords
and special tokens get :data:`IGNORE_INDEX`, so they contribute to neither loss nor metrics.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any

#: Label id ignored by the cross-entropy loss of Hugging Face token-classification models.
IGNORE_INDEX = -100

WordIds = Sequence[int | None]


def align_labels(
    word_ids: WordIds, word_labels: Sequence[str], label2id: Mapping[str, int]
) -> list[int]:
    """Label ids per subword position for one tokenized sentence."""
    labels: list[int] = []
    previous: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id == previous:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(label2id[word_labels[word_id]])
        previous = word_id
    return labels


def tokenize_and_align(
    examples: Mapping[str, Sequence[Sequence[str]]],
    tokenizer: Any,
    label2id: Mapping[str, int],
    max_len: int,
) -> Any:
    """Tokenize a batch of pre-split sentences and attach aligned ``labels``.

    ``examples`` has the pilot's keys ``tokens`` and ``labels_str`` (word-level), so the
    function can be passed to ``datasets.Dataset.map(..., batched=True)``.
    """
    tokenized = tokenizer(
        list(examples["tokens"]),
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
    )
    tokenized["labels"] = [
        align_labels(tokenized.word_ids(batch_index=i), word_labels, label2id)
        for i, word_labels in enumerate(examples["labels_str"])
    ]
    return tokenized


def first_subword_positions(word_ids: WordIds) -> dict[int, int]:
    """Map each word index that survived tokenization to the position of its first subword.

    Words cut off by truncation at ``max_len`` are absent from the result.
    """
    positions: dict[int, int] = {}
    for position, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in positions:
            positions[word_id] = position
    return positions


def truncated_words(word_ids: WordIds, n_words: int) -> list[int]:
    """Indices of the words of a sentence that have no subword in ``word_ids``."""
    seen = first_subword_positions(word_ids)
    return [i for i in range(n_words) if i not in seen]


def split_mismatches(tokenizer: Any, sentences: Sequence[Sequence[str]]) -> list[tuple[int, int]]:
    """``(sentence, word)`` indices of words whose subwords differ between pre-split input and
    the same sentence as running text.

    Training feeds pre-split words; the model was pre-trained on running text. Byte-level BPE
    tokenizers (RoBERTa family) encode a pre-split word without its leading-space marker unless
    they were created with ``add_prefix_space=True``, so every word but the first differs.
    Running-text tokens are assigned to words by character offsets; a token that ends in the
    space between two words belongs to the next word.
    """
    sentences = [list(words) for words in sentences]
    pre = tokenizer(sentences, is_split_into_words=True, add_special_tokens=False)
    running = tokenizer(
        [" ".join(words) for words in sentences],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    mismatches: list[tuple[int, int]] = []
    for i, words in enumerate(sentences):
        by_pre: list[list[int]] = [[] for _ in words]
        for word_id, token in zip(pre.word_ids(batch_index=i), pre["input_ids"][i], strict=True):
            by_pre[word_id].append(token)
        ends, position = [], 0
        for word in words:
            position += len(word)
            ends.append(position)
            position += 1
        by_running: list[list[int]] = [[] for _ in words]
        offsets = running["offset_mapping"][i]
        for (start, end), token in zip(offsets, running["input_ids"][i], strict=True):
            word_id = bisect.bisect_right(ends, max(end - 1, start))
            by_running[min(word_id, len(words) - 1)].append(token)
        mismatches.extend((i, j) for j in range(len(words)) if by_pre[j] != by_running[j])
    return mismatches
