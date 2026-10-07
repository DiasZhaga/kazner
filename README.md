# kazner

[![CI](https://github.com/DiasZhaga/kazner/actions/workflows/ci.yml/badge.svg)](https://github.com/DiasZhaga/kazner/actions/workflows/ci.yml)
[![Release](https://github.com/DiasZhaga/kazner/actions/workflows/release.yml/badge.svg)](https://github.com/DiasZhaga/kazner/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**kazner** is a small, tested Python toolkit for Kazakh named entity recognition (NER). It
provides word-level IOB2 data loading, word → subword label alignment, one prediction format
with one evaluator for every model, and tokenizer fertility analysis.

## Purpose and thesis context

kazner is the codebase of the MSc thesis *"Adaptation Methods for Large Language Models in
Low-Resource Languages: Kazakh as a Case Study"* (Astana IT University, defence planned for
2027). The thesis builds on a pilot study, which adapted multilingual BERT to Kazakh NER on
KazNERD and was accepted for publication by IEEE [2]. The pilot code (`ner-project`) was a
set of experiment scripts. Its configuration was spread across constants and command-line
flags, and some reported numbers could not be traced to a stored result.

kazner extracts the reusable parts of the pilot into a package with tests, CI/CD and
documentation, **keeping the pilot's behaviour bit for bit** where it matters. It implements
part of the target architecture from the course's Assignment 2: one word-level prediction
contract scored by one evaluator (FR6), reproducible data access (NFR1) and code that runs
identically on Windows and Linux (NFR4). It is also the practical part of Assignment 3
(*Software Development and Integration*).

## Features (v0.1.0)

| Module | What it does | Pilot equivalence |
|---|---|---|
| `kazner.data` | Reads IOB2 files (label `0` → `O`, LF/CRLF), builds `label2id` from the full train split, deterministic subsampling, checksum-verified KazNERD download | identical output on full KazNERD |
| `kazner.align` | First subword carries the label, continuations and special tokens get `-100`; maps predictions back to words | identical to the pilot's `tokenize_and_align` |
| `kazner.contract` | One CSV schema `sentence_id, word_idx, word, gold, pred` for every model; validating reader; converter for the pilot's prediction files | — |
| `kazner.evaluate` | Entity-level P/R/F1 (seqeval, default mode), token accuracy, per-type scores, `metrics.json`; offline | stored pilot metrics reproduced with difference 0 |
| `kazner.fertility` | Subwords per word, split-word and UNK rates for any tokenizer and IOB2 corpus | identical to the pilot (overall 2.325) |
| `kazner` CLI | `kazner evaluate`, `kazner fertility` | — |

**Planned for v0.2.0** ([milestone](https://github.com/DiasZhaga/kazner/milestone/2)): YAML
configuration with deterministic run IDs (#7), an adaptation-method registry
(full / frozen / LoRA, #8), `kazner train` (#9), aggregation across seeds as mean ± sd (#10),
and the training-based reproduction of the pilot's F1 = 0.9020 (#11).

## Installation

Requires **Python 3.11**. PyTorch is installed first, so that you choose the CPU or CUDA build
yourself; `pip` then keeps it.

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu    # CPU
# NVIDIA GPU: use the cuXXX index that https://pytorch.org/get-started/locally/ lists for your driver
python -m pip install -e ".[dev]"
```

**Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu   # or a cuXXX index
python -m pip install -e ".[dev]"
```

To install a released version instead, download the wheel from
[Releases](https://github.com/DiasZhaga/kazner/releases) and run `pip install kazner-*.whl`.
v0.1.0 needs no GPU: evaluation and fertility run on CPU.

## Quick start

```bash
python scripts/download_kaznerd.py                       # KazNERD -> data/kaznerd/
kazner fertility --data data/kaznerd --out fertility.json
kazner evaluate --predictions predictions.csv --labels-from data/kaznerd/IOB2_train.txt --out metrics.json
```

```python
from kazner.data import build_label2id, read_splits, sample_fraction

splits = read_splits("data/kaznerd")  # {"train": (tokens, labels), "validation": ..., "test": ...}
tokens, labels = splits["train"]
label2id, id2label = build_label2id(labels)  # always from the FULL train split
sub_tokens, sub_labels = sample_fraction(tokens, labels, 0.10, seed=42)  # 9,023 sentences
```

## Data and licence

kazner reads any whitespace-separated IOB2 corpus: token in the first column, label in the
last, a blank line between sentences. The reference corpus is **KazNERD** [1]: 112,702
sentences in 25 entity classes, with splits `IOB2_train.txt` (90,228 sentences),
`IOB2_valid.txt` (11,167) and `IOB2_test.txt` (11,307).

`scripts/download_kaznerd.py [--dest DIR] [--force]` downloads the splits from
[IS2AI/KazNERD](https://github.com/IS2AI/KazNERD/tree/bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f/KazNERD)
at a pinned commit (`bd4333d`) and verifies their SHA-256 checksums. Every run therefore uses
the same data version, and valid files are not downloaded again.

The reader behaves like the pilot, with one deliberate change: a row with fewer than two
columns raises `IOB2FormatError` with its file and line number. The pilot skipped such rows
silently.

**Licence.** KazNERD is © ISSAI / IS2AI and is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This repository does **not**
redistribute it: `data/` is git-ignored, and the fixtures in `tests/data` are synthetic
sentences written for the tests. If you use the data, cite [1].

## Label alignment

Words are labelled, but models see subwords. `kazner.align` follows the pilot's rule: the
**first subword of every word carries the word's label**. Continuation subwords and special
tokens get `-100`, so the loss and the metrics ignore them.

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
by `max_len` are missing from `first_subword_positions` (see `truncated_words`).

## Prediction contract and evaluation

Every model, encoder now and generative later, writes its test predictions in one
word-level CSV schema, and one evaluator scores all of them in exactly the same way.

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
kazner evaluate --predictions predictions.csv --labels-from data/kaznerd/IOB2_train.txt --out metrics.json
```

`metrics.json` holds entity-level precision, recall and F1 from seqeval, called directly in
its default (conlleval) mode as in the pilot. It also holds token accuracy, per-type scores
and counts (`n_words`, `n_truncated_words`, ...).

**Pilot files.** Both pilot formats map onto the contract:

| Pilot file | Mapping |
|---|---|
| `test_predictions.csv` of `run_cross_lingual_twnertc_kaznerd.py` | `sentence_id` → `sentence_id`, running index → `word_idx`, `token` → `word`, `true_label` → `gold`, `pred_label` → `pred` |
| `test_predictions_mbert.csv` of `train_ner_mbert.py` | `sent_id` → `sentence_id`, `token_id` → `word_idx`, `token` → `word`, `gold` → `gold`, `pred` → `pred` |

```bash
python scripts/convert_pilot_predictions.py PILOT.csv contract.csv --gold IOB2_test.txt
```

## Tokenizer fertility

Fertility is the average number of subword tokens that a tokenizer produces per word. When
fertility is high, the tokenizer fragments the language: less text fits into `max_len`, and
the model has to compose words from pieces.

```bash
kazner fertility --data data/kaznerd --tokenizer bert-base-multilingual-cased --out fertility.json
```

The command works with any Hugging Face tokenizer name or path (`--lowercase` lowercases
words first). For each split and overall it reports:

- words and subword tokens;
- fertility;
- the shares of single-token and split words;
- the mean number of subwords of split words;
- the median, standard deviation and maximum subwords per word;
- the UNK rate.

Each word is tokenized on its own, and an empty tokenization counts as one unknown token, as
in the pilot.

## Reproduction results

Full details and commands: [docs/reproduction.md](docs/reproduction.md).

| Check | Result |
|---|---|
| Pilot's 4 stored cross-lingual test metrics, re-scored from their prediction files | **identical**: difference 0 in P, R, F1 and accuracy (e.g. F1 0.7400769175) |
| Same files as complete contracts (274 words truncated at `max_len = 64`, scored as `O`) | F1 lower by 0.00012–0.00085, as expected |
| Pilot paper, Table I (fine-tuned mBERT 0.939 / 0.951 / 0.945 / 0.988) | traced to `test_predictions_mbert.csv` (a metric the pilot never saved) |
| mBERT fertility on full KazNERD | **identical** to the pilot in all 11 metrics × 4 splits; overall **2.325**, 60.6% words split |

## Testing

```bash
pytest -m "not network and not smoke"   # offline unit tests (what CI runs on both OSs)
pytest -m "network or smoke"            # tests that download from the Hugging Face Hub
pytest --cov=kazner                     # with coverage
ruff check . && ruff format --check .
```

Tests use synthetic fixtures and an offline WordPiece vocabulary (`tests/data`). Behaviour
that must stay identical to the pilot is checked against **verbatim copies of the pilot
functions** in `tests/reference/` (oracle tests). Coverage is about 99%.

## CI/CD

| Workflow | Trigger | Jobs |
|---|---|---|
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | every pull request, pushes to `main` | `lint` (ruff); `test (ubuntu-latest)`, `test (windows-latest)` (CPU PyTorch, coverage artifact); `smoke` (network tests, cached HF hub) |
| [`.github/workflows/release.yml`](.github/workflows/release.yml) | tags `v*` | tag ↔ version check (PEP 440) → lint + tests → `python -m build` → GitHub Release with wheel, sdist and CHANGELOG notes |

`main` is protected: merging requires a pull request with passing `lint` and both `test`
jobs, and administrators cannot bypass it. Releases are **continuous delivery**: everything is
automated except the decision to release, which is pushing a tag:

```bash
git tag -a v0.1.0 -m "kazner 0.1.0"
git push origin v0.1.0
```

Pre-release versions (`0.1.0rc1`) are published as GitHub pre-releases. Contribution rules
(branches, commits, pull requests) are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Project structure

```text
src/kazner/
  data.py          IOB2 reader, label maps, sampling, KazNERD download
  align.py         word -> subword label alignment
  contract.py      word-level prediction schema, validation, pilot converter
  evaluate.py      seqeval-based evaluator, metrics.json
  fertility.py     tokenizer fertility analysis
  cli.py           kazner evaluate | fertility
scripts/           download_kaznerd.py, convert_pilot_predictions.py,
                   reproduce_pilot_metrics.py, release_tools.py
tests/             pytest suite; data/ = synthetic fixtures; reference/ = pilot oracles
docs/              technology-choices.md, traceability.md, reproduction.md, challenges.md
.github/workflows/ ci.yml, release.yml
```

Documentation:

- [technology choices](docs/technology-choices.md): every tool against the Lecture 5 criteria;
- [requirements traceability](docs/traceability.md);
- [reproduction](docs/reproduction.md);
- [decisions and problems log](docs/challenges.md);
- [changelog](CHANGELOG.md).

## How to cite

Citation metadata is in [CITATION.cff](CITATION.cff); GitHub shows it under
"Cite this repository".

```text
D. Zhagaparov, "kazner: a Kazakh named entity recognition toolkit," version 0.1.0, 2026.
https://github.com/DiasZhaga/kazner
```

## References

[1] R. Yeshpanov, Y. Khassanov and H. A. Varol, "KazNERD: Kazakh Named Entity Recognition
Dataset," in *Proc. 13th Language Resources and Evaluation Conference (LREC)*, Marseille,
France, 2022, pp. 417–426. <https://aclanthology.org/2022.lrec-1.44>

[2] D. Zhagaparov and M. Zhartybayeva, "Adapting Multilingual BERT for Low-Resource Kazakh NER:
Data Efficiency, Parameter-Efficient Fine-Tuning, Tokenization, and Cross-Lingual Transfer,"
accepted for publication, IEEE, 2026.

## License

Code: [MIT](LICENSE). Data: see [Data and licence](#data-and-licence).
