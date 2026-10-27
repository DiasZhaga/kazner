# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Configuration layer `kazner.config`: YAML → frozen dataclasses with validation (unknown keys
  and invalid values name the key), pilot defaults, seed lists expanded into runs, deterministic
  run IDs from the resolved config and the git tree hash of `src/kazner`, `config.json` and
  `environment.json` per run; `configs/pilot_full_10pct.yaml` (#7).
- Adaptation-method registry `kazner.methods` (Factory Method): `full`, `frozen`, `lora`, built as
  in the pilot; correct trainable-parameter counts plus the pilot's counting as
  `trainable_params_pilot` (#8).
- Training entry point `kazner train` (`kazner.train`): pilot Trainer arguments, seeds set before
  the model is built, contract predictions, `metrics.json` with contract- and Trainer-based test
  metrics plus validation history, idempotent runs (`--force`, `--max-train-samples`), CPU smoke
  test in CI; configs for the multi-seed PEFT comparison and the LoRA screen (#9).
- Multi-seed aggregation `kazner aggregate` (`kazner.aggregate`): mean, sample sd and n per group
  of config fields, CSV output, warnings for incomplete runs and mixed code versions (#10).

- Reproduction of the pilot's F1 = 0.9020 by `kazner train` (bit-identical test metrics), the
  three-seed PEFT comparison and the LoRA learning-rate × rank screen, recorded in
  `docs/reproduction.md` §3–4 (#11).
- Fragmentation analysis `kazner fragmentation` (`kazner.fragmentation`): errors and exact-span
  recall vs pieces per word under a fixed reference tokenizer, controlled for training
  frequency; logistic regression (Newton's method in NumPy) with a sentence bootstrap;
  `entities.csv` with the fixed entity subsets and per-run hits. Reproduces experiment B2;
  `docs/fragmentation.md` (#33).
- Unlabeled Kazakh corpus for continued pre-training: `kazner.corpus` (NFC, length and
  Kazakh-script rules, exact dedup, removal of lines that contain a KazNERD sentence, seeded
  row-group sample of parquet files, per-rule counts) and `scripts/download_kk_corpus.py`
  (FineWeb-2 `kaz_Cyrl` train split at a pinned commit, `--dry-run`, manifest with SHA-256) (#34).
- Vocabulary extension `kazner extend-vocab` (`kazner.vocab`): WordPiece candidates learned on
  the corpus with mBERT's normalizer and pre-tokenizer (new, multi-character, spellable without
  `[UNK]`, used; ranked by use), appended to the WordPiece vocabulary with every old id kept,
  FVT initialisation of embeddings and masked-LM bias, a per-size report on KazNERD validation
  and on the corpus, `--pieces` to rebuild a saved extension exactly (#35).
- Continued masked-LM pre-training `kazner pretrain` (`kazner.pretrain`): corpus chunks cut by
  a fixed reference tokenizer, whole-word masks seeded per chunk and shared by every vocabulary,
  run IDs from config, code, corpus SHA-256 and model fingerprint, resumable runs that write a
  model directory usable as `model.name`; configs for the two arms of the causal test in
  `configs/pretrain/` (#36).
- Baselines next to mBERT: learning-rate screen configs for mBERT, XLM-R base and KazRoBERTa
  with pinned model commits (`configs/baselines/`), protocol in `docs/baselines.md`; a test
  that every shipped config loads and validates (#32).

### Changed
- Run IDs hash a fingerprint of the data content (CRLF-normalised SHA-256 of the split files)
  instead of the data path, so the same experiment gets the same ID from any config (#9).
- Model identity: `model.revision` pins a Hub branch, tag or commit for the tokenizer and the
  model and is hashed; the resolved commit is recorded in `environment.json`. `model.name` may
  be a local directory (`./`, `../` or absolute), resolved against the YAML file and hashed by
  a content fingerprint (`model_fingerprint` in `config.json`) instead of its path (#31).
- `kazner.data.training_subsample` is the one definition of a run's training sentences (used
  by training and by the fragmentation analysis); `kazner.aggregate.config_field` is public (#33).

### Fixed
- Byte-level BPE tokenizers (RoBERTa family) silently received pre-split words without their
  leading-space marker (KazRoBERTa: 1.569 instead of 1.205 subwords per word on KazNERD
  validation). New `model.add_prefix_space` config key and `kazner fertility
  --add-prefix-space`; training now stops before it starts if pre-split words are tokenized
  differently from running text (#30).
- The corpus download opened the Hub parquet files with fsspec's default 5 MiB read-ahead, which
  would have fetched about three times the ~1.85 MiB of text per row group. The files are now
  opened with `cache_type="none"`; `kazner.corpus.CountingReader` records the bytes read, and the
  manifest stores them as `bytes_read` (#42).

## [0.1.0] - 2026-10-07

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
- Prediction contract `kazner.contract` (`sentence_id, word_idx, word, gold, pred` + optional
  `truncated`; validating reader; converter for both pilot prediction formats) and evaluator
  `kazner.evaluate` (seqeval default mode, token accuracy, per-type scores, `metrics.json`);
  `kazner evaluate` command; `scripts/convert_pilot_predictions.py`,
  `scripts/reproduce_pilot_metrics.py`; `docs/reproduction.md` (#5).
- Tokenizer fertility `kazner.fertility` and `kazner fertility` command: fertility, single-token
  and split-word rates, mean subwords of split words, median, sd, maximum and UNK rate per split
  and overall; reproduces the pilot's KazNERD numbers exactly (overall 2.325) (#6).
- Release workflow (continuous delivery): on `v*` tags, PEP 440 tag/version check, lint and
  tests, `python -m build`, GitHub Release with wheel, sdist and CHANGELOG notes; pre-releases
  marked automatically (#12).
- Documentation: full README (purpose, installation with CPU/CUDA PyTorch, quick start,
  reproduction results, CI/CD, structure, citation), `CITATION.cff`, `CONTRIBUTING.md`,
  technology choices assessed against the Lecture 5 criteria, completed traceability
  matrix (#13).

### Release notes
- `v0.1.0rc1` was published first as a pre-release to verify the release pipeline (#12).

[Unreleased]: https://github.com/DiasZhaga/kazner/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiasZhaga/kazner/releases/tag/v0.1.0
