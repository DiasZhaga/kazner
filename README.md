# kazner

[![CI](https://github.com/DiasZhaga/kazner/actions/workflows/ci.yml/badge.svg)](https://github.com/DiasZhaga/kazner/actions/workflows/ci.yml)

Kazakh named entity recognition toolkit: a small, tested Python package refactored from the
pilot experiments of an MSc thesis on adapting multilingual language models (mBERT) to
low-resource Kazakh NER on [KazNERD](https://github.com/IS2AI/KazNERD).

> Status: under development towards v0.1.0. Work is tracked in the
> [issues](https://github.com/DiasZhaga/kazner/issues) and planned in [ROADMAP.md](ROADMAP.md).

## Installation (development)

Requires Python 3.11.

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Data

kazner reads any whitespace-separated IOB2 corpus (token in the first column, label in the
last, blank line between sentences). The reference corpus is **KazNERD** [1]: 112,702
sentences, 25 entity classes, splits `IOB2_train.txt` (90,228 sentences),
`IOB2_valid.txt` (11,167) and `IOB2_test.txt` (11,307).

```bash
python scripts/download_kaznerd.py            # -> data/kaznerd/
python scripts/download_kaznerd.py --dest D   # any other directory
```

The script downloads the splits from
[IS2AI/KazNERD](https://github.com/IS2AI/KazNERD/tree/bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f/KazNERD)
at a pinned commit (`bd4333d`) and verifies their SHA-256 checksums, so every run uses the
same data version. Files that are already valid are not downloaded again.

```python
from kazner.data import build_label2id, read_splits, sample_fraction

splits = read_splits("data/kaznerd")  # {"train": (tokens, labels), ...}
tokens, labels = splits["train"]
label2id, id2label = build_label2id(labels)  # always from the FULL train split
sub_tokens, sub_labels = sample_fraction(tokens, labels, 0.10, seed=42)  # 9,023 sentences
```

The reader behaves like the pilot code, with one deliberate change. Label `0` becomes `O`,
and both LF and CRLF files are accepted. A row with fewer than two columns raises
`IOB2FormatError` with its file and line number; the pilot skipped such rows silently.

**Licence.** KazNERD is © ISSAI / IS2AI and distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This repository does not
redistribute it: `data/` is git-ignored, and the test fixtures in `tests/data` are
synthetic sentences written for the tests. If you use the data, cite [1].

[1] R. Yeshpanov, Y. Khassanov and H. A. Varol, "KazNERD: Kazakh Named Entity Recognition
Dataset," in *Proc. 13th Language Resources and Evaluation Conference (LREC)*, Marseille,
France, 2022, pp. 417–426. <https://aclanthology.org/2022.lrec-1.44>

## Label alignment

Words are labelled, but models see subwords. `kazner.align` follows the pilot's rule: the
**first subword of every word carries the word's label**, while continuation subwords and
special tokens get `-100` and are ignored by the loss and the metrics.

```text
words      Алматыға            барды
subwords   [CLS] Алма ##ты ##ға  барды [SEP]
labels      -100   B  -100 -100    O   -100
```

```python
from kazner.align import first_subword_positions, tokenize_and_align

batch = {"tokens": tokens, "labels_str": labels}  # word level, as read by kazner.data
encoded = tokenize_and_align(batch, tokenizer, label2id, max_len=128)
positions = first_subword_positions(encoded.word_ids(0))  # word index -> subword position
```

`tokenize_and_align` can be passed to `datasets.Dataset.map(..., batched=True)`. Words cut off
by `max_len` are missing from `first_subword_positions` (see `truncated_words`). The test
suite checks that the output matches the pilot's function exactly.

## Prediction contract and evaluation

Every model, encoder now and generative later, writes its test predictions in one
word-level CSV schema. One evaluator scores every model in exactly the same way.

| Column | Meaning |
|---|---|
| `sentence_id` | sentence index in the split |
| `word_idx` | word index within the sentence, `0..n-1` |
| `word` | the word |
| `gold` / `pred` | gold and predicted IOB2 label |
| `truncated` *(optional)* | `1` if the model never saw the word (cut off at `max_len`); then `pred = O` |

Every gold word gets a row. The reader validates the columns, contiguous `word_idx`, and
known labels. All values are read as strings, so a word such as `NA` is not turned into a
missing value.

```bash
kazner evaluate --predictions predictions.csv --out metrics.json   --labels-from data/kaznerd/IOB2_train.txt
```

`metrics.json` holds entity-level precision, recall and F1 from seqeval, called directly in
its default (conlleval) mode as in the pilot, plus token accuracy, per-type scores and
counts (`n_words`, `n_truncated_words`, ...).

**Pilot files.** Both pilot formats map onto the contract:

| Pilot file | Mapping |
|---|---|
| `test_predictions.csv` of `run_cross_lingual_twnertc_kaznerd.py` | `sentence_id` → `sentence_id`, running index → `word_idx`, `token` → `word`, `true_label` → `gold`, `pred_label` → `pred` |
| `test_predictions_mbert.csv` of `train_ner_mbert.py` | `sent_id` → `sentence_id`, `token_id` → `word_idx`, `token` → `word`, `gold` → `gold`, `pred` → `pred` |

```bash
python scripts/convert_pilot_predictions.py PILOT.csv contract.csv --gold IOB2_test.txt
```

Re-scoring the pilot's files reproduces its stored metrics exactly; see
[docs/reproduction.md](docs/reproduction.md).

## Usage

```bash
kazner --version
kazner evaluate --help
```

## Development

```bash
pytest -m "not network and not smoke"
ruff check .
ruff format --check .
```

Tests that download from the Hugging Face Hub are marked `network`; end-to-end training
tests are marked `smoke`. Run them with `pytest -m "network or smoke"`.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull request and on
pushes to `main`:

| Job | Runner | What it does |
|---|---|---|
| `lint` | ubuntu | `ruff check .`, `ruff format --check .` |
| `test (ubuntu-latest)`, `test (windows-latest)` | ubuntu, windows | CPU-only PyTorch, offline tests with coverage; `coverage.xml` uploaded as an artifact |
| `smoke` | ubuntu | `network` and `smoke` tests, Hugging Face cache kept between runs |

`main` is protected: merging requires a pull request with passing `lint` and both `test` jobs.

## License

[MIT](LICENSE).
