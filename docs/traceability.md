# Requirements traceability

Requirement → issue → module → tests (Lecture 3, requirements tracing).
Requirement IDs and wording come from Assignment 2, Table 1.

Status: **Met** = implemented and tested; **Partial** = part of the requirement implemented;
**Planned** = scheduled in a later issue; **Out of scope** = not planned for v0.1.0 / v0.2.0.

| ID | Requirement (Assignment 2) | Priority | Issue | Module | Tests | Status |
|---|---|---|---|---|---|---|
| FR1 | Load KazNERD, TWNERTC and an out-of-domain set into one word-level IOB2 format | Must | #3 | `kazner.data` | `tests/test_data.py` | Partial: KazNERD (any IOB2 corpus); TWNERTC converter and out-of-domain set not ported |
| FR2 | Swap the backbone without duplicating code | Must | #9 | — | — | Planned (v0.2.0) |
| FR3 | Swap the adaptation method (full, frozen, LoRA, …) | Must | #8, #9 | — | — | Planned (v0.2.0) |
| FR4 | Replace or extend the tokenizer | Must | #4, #6 | `kazner.align`, `kazner.fertility` | `tests/test_align.py`, `tests/test_fertility.py`; `docs/reproduction.md` §2 | Partial: alignment and fertility analysis work with any Hugging Face tokenizer (goal G3 baseline measured: mBERT fertility 2.325); vocabulary extension not in v0.1.0 |
| FR5 | Seeds and sweeps declared in configuration, not code | Must | #7 | — | — | Planned (v0.2.0) |
| FR6 | One identical evaluator for token-classification and generative models | Must | #5 | `kazner.contract`, `kazner.evaluate`, `kazner.cli` | `tests/test_contract.py`, `tests/test_evaluate.py`; `docs/reproduction.md` §1 | Met for the evaluator side: any model that writes the word-level contract is scored identically; the generative adapter (WP3) is out of scope |
| FR7 | Aggregate across seeds as mean ± sd | Should | #10 | — | — | Planned (v0.2.0) |
| FR8 | Regenerate every thesis table and figure from stored results | Should | — | — | — | Out of scope |
| NFR1 | Reproducibility: pinned environment, data version, config snapshot per run | Must | #1, #3, #12, #7 | `pyproject.toml`, `kazner.data`, `.github/workflows/release.yml` | `tests/test_data.py`, `tests/test_release_tools.py` | Partial: pinned dependencies; data pinned to IS2AI/KazNERD `bd4333d` with SHA-256; versioned releases with wheel + sdist; per-run config snapshot in #7 (v0.2.0) |
| NFR2 | Traceability: every number maps to one run ID, configuration and commit | Must | #7 | — | — | Planned (v0.2.0) |
| NFR3 | Fit 6 GB VRAM through FP16 / 4-bit quantisation | Must | #9 | — | — | Planned (v0.2.0) |
| NFR4 | Portability: identical code on Windows and Linux | Should | #2 | `.github/workflows/ci.yml` | CI matrix `ubuntu-latest` + `windows-latest` | Met |
| NFR5 | Fault tolerance: resume a series after interruption | Should | #9 | — | — | Planned (v0.2.0) |
| NFR6 | Modifiability: a new model or method = one module + one config | Should | #8 | — | — | Planned (v0.2.0) |
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
