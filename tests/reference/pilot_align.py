"""Reference oracle: ``tokenize_and_align`` copied VERBATIM from the pilot.

Source: ner-project @ b6ee209 — src/train_ner_mbert.py. The pilot read ``MAX_LEN`` from a
module constant (128); tests override it with ``monkeypatch``. Do not edit.
"""

# ruff: noqa
MAX_LEN = 128  # src/train_ner_mbert.py


def tokenize_and_align(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    all_labels = []
    for i, word_labels in enumerate(examples["labels_str"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(label2id[word_labels[word_id]])
            else:
                # Subword continuation: ставим -100 (не обучаем на кусках слова)
                labels.append(-100)
            prev_word_id = word_id
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized
