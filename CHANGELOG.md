# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Package skeleton: `pyproject.toml` (src layout, pinned runtime dependencies, `dev` extra),
  `kazner` console script stub with `--version`, ruff and pytest configuration, MIT licence,
  README skeleton and documentation stubs (#1).
- GitHub Actions CI: ruff lint and format check, offline tests with coverage on Ubuntu and
  Windows with CPU-only PyTorch, network/smoke job with a cached Hugging Face hub; CI badge (#2).
- Data module `kazner.data`: `read_iob2` / `read_splits` (label `0` → `O`, LF and CRLF,
  `IOB2FormatError` on malformed rows), `build_label2id` from the full train split, deterministic
  `sample_fraction` identical to the pilot, checksum-verified `download_kaznerd` pinned to
  IS2AI/KazNERD `bd4333d`; `scripts/download_kaznerd.py`; requirements traceability matrix (#3).
- Label alignment `kazner.align`: `tokenize_and_align` (first subword labelled, continuations
  and special tokens `-100`, identical to the pilot), `first_subword_positions` and
  `truncated_words` for mapping predictions back to words; offline tiny WordPiece tokenizer
  for tests (#4).
