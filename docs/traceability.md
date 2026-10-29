# Requirements traceability

Requirement → issue → module → tests (Lecture 3, requirements tracing).
Requirement IDs and wording come from Assignment 2, Table 1.

Status: **Met** = implemented and tested; **Partial** = part of the requirement implemented;
**Planned** = scheduled in a later issue; **Out of scope** = not planned for v0.1.0 / v0.2.0.

| ID | Requirement (Assignment 2) | Priority | Issue | Module | Tests | Status |
|---|---|---|---|---|---|---|
| FR1 | Load KazNERD, TWNERTC and an out-of-domain set into one word-level IOB2 format | Must | #3, #51 | `kazner.data` | `tests/test_data.py` | Partial: KazNERD (any IOB2 corpus) and WikiANN `kk` as an out-of-domain set (#51); TWNERTC converter not ported |
| FR2 | Swap the backbone without duplicating code | Must | #9, #30 | `kazner.config` (`model.name`, `model.add_prefix_space`), `kazner.train`, `kazner.align` | `tests/test_train.py` (smoke run with another backbone), `tests/test_align.py`, `tests/test_network.py` | Partial: any Hugging Face token-classification encoder by name, including byte-level BPE tokenizers (checked at start: pre-split words tokenized as in running text); generative 3–8B backbones (WP3) not yet |
| FR3 | Swap the adaptation method (full, frozen, LoRA, …) | Must | #8, #9 | `kazner.methods`, `kazner.train` | `tests/test_methods.py`, `tests/test_train.py` | Partial: full / frozen / LoRA selectable by name and trainable end to end; QLoRA and few-shot prompting (WP3) not yet |
| FR4 | Replace or extend the tokenizer | Must | #4, #6, #33, #35 | `kazner.align`, `kazner.fertility`, `kazner.fragmentation`, `kazner.vocab` | `tests/test_align.py`, `tests/test_fertility.py`, `tests/test_fragmentation.py`, `tests/test_vocab.py`; `docs/reproduction.md` §2, `docs/fragmentation.md` | Met for WordPiece models: alignment and fertility analysis work with any Hugging Face tokenizer (mBERT fertility 2.325); `kazner extend-vocab` extends mBERT's WordPiece vocabulary with corpus-learned pieces, old ids kept, FVT initialisation (#35) |
| FR5 | Seeds and sweeps declared in configuration, not code | Must | #7 | `kazner.config`, `configs/` | `tests/test_config.py` | Partial: seeds and all hyper-parameters in YAML, seed lists expand into runs; grid sweeps are separate YAML files; used by training in #9 |
| FR6 | One identical evaluator for token-classification and generative models | Must | #5 | `kazner.contract`, `kazner.evaluate`, `kazner.cli` | `tests/test_contract.py`, `tests/test_evaluate.py`; `docs/reproduction.md` §1 | Met for the evaluator side: any model that writes the word-level contract is scored identically; the generative adapter (WP3) is out of scope |
| FR7 | Aggregate results across seeds as mean ± standard deviation | Should | #10 | `kazner.aggregate`, `kazner.cli` | `tests/test_aggregate.py` | Met: `kazner aggregate` → mean, sample sd and n per group |
| FR8 | Regenerate every thesis table and figure from stored results | Should | — | — | — | Out of scope |
| NFR1 | Reproducibility: pinned environment, data version, config snapshot per run | Must | #1, #3, #7, #12, #31 | `pyproject.toml`, `kazner.data`, `kazner.config`, `.github/workflows/release.yml` | `tests/test_data.py`, `tests/test_config.py`, `tests/test_release_tools.py` | Partial: pinned dependencies; data pinned to IS2AI/KazNERD `bd4333d` with SHA-256; models pinned by Hub revision or identified by content fingerprint (#31); versioned releases with wheel + sdist; per-run `config.json` + `environment.json` (#7) |
| NFR2 | Traceability: every number maps to one run ID, configuration and commit | Must | #7, #9, #31 | `kazner.config`, `kazner.train` | `tests/test_config.py`, `tests/test_train.py` | Met: every training run writes `config.json` (with data and, for local models, model fingerprints), `environment.json` (commit, tree hash, model revision and resolved Hub commit) and `metrics.json` under its run ID |
| NFR3 | Fit 6 GB VRAM through FP16 / 4-bit quantisation | Must | #9 | `kazner.train` | — | Partial: FP16 on CUDA (`fp16: auto`), small batches with gradient accumulation; 4-bit (QLoRA) in WP3 |
| NFR4 | Portability: identical code on Windows and Linux | Should | #2 | `.github/workflows/ci.yml` | CI matrix `ubuntu-latest` + `windows-latest` | Met |
| NFR5 | Fault tolerance: resume a series after interruption | Should | #9 | `kazner.train` | `tests/test_train.py` | Met for local series: completed run IDs are skipped, incomplete ones restart cleanly; `--force` re-runs |
| NFR6 | Modifiability: a new model or method = one module + one config | Should | #8 | `kazner.methods` | `tests/test_methods.py` (a custom method is registered in a test) | Met for adaptation methods: one `@register_method` function + one config value |
| NFR7 | No credentials in the repository; data licences respected | Could | #1, #3 | `.gitignore`, `scripts/download_kaznerd.py` | — | Met: `.env`/data ignored; KazNERD (CC BY 4.0) downloaded, never redistributed |

## Issue → requirement → verification (v0.1.0)

| Issue | Requirement(s) | Verified by |
|---|---|---|
| #1 Bootstrap | NFR1, NFR7 | `tests/test_package.py`; `pip install -e ".[dev]"` on Windows |
| #2 Continuous integration | NFR4 | CI matrix on Ubuntu + Windows; merge blocked by a failing test (screenshot in the report) |
| #3 IOB2 data module | FR1, NFR1, NFR7 | `tests/test_data.py` (incl. oracle vs pilot); full-KazNERD parity check (`docs/challenges.md`) |
| #4 Label alignment | FR4 | `tests/test_align.py` (incl. oracle vs pilot, mBERT network test) |
| #5 Prediction contract and evaluator | FR6 | `tests/test_contract.py`, `tests/test_evaluate.py`; `docs/reproduction.md` §1 |
| #6 Tokenizer fertility | FR4 (goal G3) | `tests/test_fertility.py` (incl. oracle vs pilot); `docs/reproduction.md` §2 |
| #12 Release pipeline | NFR1 | `tests/test_release_tools.py`; release `v0.1.0rc1` and `v0.1.0` |
| #13 Documentation | — | this file, README, CONTRIBUTING, technology choices |
| #11 Reproduction check | G1, NFR1, NFR2 | `docs/reproduction.md` §3: run `53a89e2dbc4c` reproduces test F1 0.9020369285 bit for bit; §4: three-seed PEFT comparison |
| #30 Byte-level BPE prefix space | FR2, FR4 | `tests/test_align.py`, `tests/test_train.py`, `tests/test_fertility.py` (offline byte-level BPE fixture); `tests/test_network.py` (mBERT, XLM-R, KazRoBERTa) |
| #31 Model identity | NFR1, NFR2 | `tests/test_config.py` (path rule, resolution, fingerprint vs location/CRLF/content), `tests/test_methods.py`, `tests/test_train.py` (smoke: training from a local directory; a moved copy is skipped as the same run) |
| #32 Baselines (XLM-R base, KazRoBERTa) | FR2, NFR1, NFR2 | `tests/test_config.py` (every shipped config loads; baseline configs pin a model commit); `docs/baselines.md`: screen, grid extension and three seeds on one code tree (`44db383`), mBERT seed 42 bit-identical to the pilot, per-model fragmentation analysis |
| #33 Fragmentation analysis | FR4 (goal G3) | `tests/test_fragmentation.py` (hand-computed tables, cross-product odds ratio, CLI on synthetic runs); `docs/fragmentation.md` (reproduces experiment B2) |
| #34 Unlabeled Kazakh corpus | FR4 (goal G3), NFR1, NFR7 | `tests/test_corpus.py` (line rules, KazNERD overlap, seeded row groups, the script offline on synthetic parquet) |
| #35 Vocabulary extension | FR4 (goal G3) | `tests/test_vocab.py` (WordPiece spelling, candidate rules and ranking, old ids kept, identical tokenization for an empty extension, `[UNK]` unchanged, FVT rows and bias, save/load, rebuild from `new_pieces.txt`; mBERT network test) |
| #51 Transfer to WikiANN | FR1, FR6, NFR1, NFR2 | `tests/test_data.py` (WikiANN parquet -> IOB2 on synthetic parquet, class names checked, pinned download offline), `tests/test_config.py` (extra test sets: validation, content in the run ID), `tests/test_train.py` (an offline run predicts an extra set), `tests/test_transfer.py` (mapping validation, hand-computed scores, bootstrap, sentence kinds and subset scores, CLI); `docs/transfer.md` |
| #36 Continued masked-LM pre-training | FR4 (goal G3), NFR1, NFR2, NFR5 | `tests/test_pretrain.py` (chunking by hand, seeded word masks and their shares, identical masked words for two vocabularies, config and run ID, an offline run on a tiny local model with skip and force, the arm configs differ only in the model; smoke: pre-training then NER training on the result) |
