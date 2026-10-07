"""Word -> subword label alignment, as in the pilot (``train_ner_mbert.tokenize_and_align``).

Only the first subword of every word carries the word's label id; continuation subwords
and special tokens get :data:`IGNORE_INDEX`, so they contribute to neither loss nor metrics.
"""

from __future__ import annotations

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
