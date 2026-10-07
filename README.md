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

## Usage

```bash
kazner --version
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
