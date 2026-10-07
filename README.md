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
