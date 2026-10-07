# ROADMAP — kazner v0.1.0 (Assignment 3)

How to use: create each item below as a GitHub issue (title, body, labels) and work strictly in
order. CI (#2) comes early on purpose, so that every later PR shows green checks.

Labels: priority `must` / `should` / `could` (MoSCoW, Lecture 3) and type `feature` / `test` /
`ci` / `docs` / `chore`. Requirement IDs (FR*, NFR*) refer to Assignment 2, Table 1.
Stories follow "As a …, I want …, so that …" and should satisfy INVEST (Lecture 3).

---

## Manual setup (user, once)
1. Create a public GitHub repo `kazner` without an auto-generated README.
2. Install GitHub CLI and run `gh auth login`.
3. Keep the repos side by side: `projects/ner-project` and `projects/kazner`.
4. Create a GitHub Project board (Kanban: Todo / In progress / Review / Done) and add the
   issues to it. WIP limit: one open PR at a time.
5. After #2 is merged: Settings → Branches → protect `main`: require a PR and passing CI.

---

## #1 Bootstrap the repository — `must` `chore`
As the developer, I want a standard Python package skeleton, so that code, tests and tooling
share one agreed structure.

Acceptance criteria
- `pyproject.toml`: src layout, package `kazner`, pinned runtime deps (see CLAUDE.md),
  `[project.optional-dependencies] dev = [...]`, console script `kazner` (stub).
- `.gitignore`: Python, `.venv/`, `data/`, `runs/`, `results/`, `checkpoint-*`,
  `*.safetensors`, `.env`.
- `LICENSE` (MIT), README skeleton, `CHANGELOG.md` (Keep a Changelog),
  `docs/challenges.md`, `docs/technology-choices.md` (stub).
- ruff configured; one trivial passing test; `pip install -e ".[dev]"` works on Windows.

## #2 Continuous integration — `must` `ci` (NFR4)
As the developer, I want every push and pull request checked automatically, so that
regressions are caught before they reach `main`.

Acceptance criteria
- `.github/workflows/ci.yml` as specified in CLAUDE.md (lint; test matrix ubuntu + windows;
  CPU torch; pip cache; coverage artifact).
- CI badge in README.
- Demonstrate once that a failing test blocks the merge (keep a screenshot for the report),
  then fix it.

## #3 IOB2 data module — `must` `feature` (FR1)
As a researcher, I want every KazNERD split loaded into one word-level format, so that all
experiments read data identically.

Acceptance criteria
- `read_iob2`: sentences separated by blank lines; label `0` → `O`; clear error on malformed
  lines; tolerant of trailing blank lines.
- `build_label2id` from the FULL train split, same ordering as the pilot (read the pilot code);
  a test proves that subsampling does not change the mapping.
- `scripts/download_kaznerd.py`: fetches the official IS2AI release into `data/`. Check the
  dataset licence and state it in README. Not run in CI.
- Tests on synthetic fixtures only.

## #4 Label alignment — `must` `feature`
As a researcher, I want word labels aligned to subwords exactly as in the pilot, so that
results stay comparable after refactoring.

Acceptance criteria
- First subword gets the label id; continuations and special tokens get -100; works with
  `is_split_into_words=True`.
- Tests include a word split into three pieces and an equivalence test against the pilot's
  function (copied into the tests as a reference oracle).

## #5 Prediction contract and evaluator — `must` `feature` (FR6)
As a researcher, I want every model to write word-level predictions in one schema, so that a
single evaluator scores encoder models now and generative models later in exactly the same way.

Acceptance criteria
- `contract.py`: schema `sentence_id, word_idx, word, gold, pred`; writer and validating reader
  (matching lengths, known labels).
- `evaluate.py`: entity-level precision / recall / F1 via seqeval plus token accuracy; returns a
  dict and writes `metrics.json`.
- Tests: a hand-computed example with known F1; an invalid file is rejected.
- README documents how the pilot's `test_predictions.csv` maps onto the contract.

## #6 Tokenizer fertility analysis — `must` `feature` (FR4, G3)
As a researcher, I want a CLI that measures tokenizer fertility on any IOB2 corpus, so that I
can compare tokenizers in WP2.

Acceptance criteria
- `kazner fertility --data <dir> --tokenizer bert-base-multilingual-cased` → JSON per split:
  fertility, % single-token words, % split words, mean subwords of split words, maximum,
  UNK rate.
- Offline unit tests with a tiny tokenizer; one `network` test with the mBERT tokenizer on
  fixtures.
- `docs/reproduction.md`: run locally on full KazNERD; expected overall fertility 2.325.

## #7 Configuration layer and run IDs — `must` `feature` (FR5, NFR1, NFR2)
As a researcher, I want every experiment defined by one YAML file and identified by a
deterministic run ID, so that every reported number traces back to exactly one configuration
and commit.

Acceptance criteria
- Dataclasses: data, model, method, training (lr, epochs, batch size, grad accumulation,
  max_len, fp16), list of seeds, output root.
- Loading with validation (unknown keys are an error); the resolved config is saved as JSON
  in every run directory.
- `run_id` = first 12 hex chars of sha256(canonical JSON of resolved config + git commit);
  a dirty working tree is recorded in the snapshot.
- `configs/pilot_full_10pct.yaml` reproduces the pilot settings: full FT, 10% KazNERD,
  lr 5e-5, 3 epochs, batch 4, grad accumulation 2, max_len 128, seed 42.
- Tests: same config → same ID; any field change → different ID; a seed list expands into
  N runs.

## #8 Adaptation-method registry — `should` `feature` (FR3, NFR6)
As a researcher, I want adaptation methods selected by name, so that adding a method costs one
function and one config value instead of a new script.

Acceptance criteria
- Registry (Factory Method): `full`, `frozen` (encoder frozen, head trainable), `lora`
  (r, alpha, dropout and target modules from config; defaults r=8, alpha=16, dropout 0.1,
  `query`,`value`).
- Each builder returns the model and its trainable-parameter count.
- Tests on `tiny-random-bert`: frozen → only the classifier is trainable; LoRA ratio is small
  but larger than frozen; unknown name → clear error.

## #9 Training entry point — `must` `feature` (FR2, FR3, NFR5)
As a researcher, I want `kazner train --config X.yaml` to run every seed and write metrics and
contract predictions, so that an experiment is reproducible from a single command.

Acceptance criteria
- HF Trainer wiring with the pilot's arguments; best model by validation F1 where enabled;
  test-set evaluation through the contract and the evaluator.
- Per run directory: resolved config, `metrics.json`, `predictions.csv`, trainer state;
  checkpoints are deleted unless the config asks to keep them.
- Idempotent: a completed run ID is skipped; `--force` re-runs it.
- `--max-train-samples` for quick runs; a `smoke` test (tiny model, synthetic fixtures, CPU)
  runs in the CI `smoke` job.

## #10 Multi-seed aggregation — `should` `feature` (FR7)
As a researcher, I want results aggregated across seeds, so that every headline number is
reported as mean ± standard deviation.

Acceptance criteria
- `kazner aggregate --runs runs/ --by method,fraction` → CSV with mean, sd and n.
- Tests on fabricated metrics files.

## #11 Reproduction check — `must` `test` (local GPU)
As a researcher, I want the refactored pipeline to reproduce a pilot result, so that I know the
refactoring changed structure and not behaviour.

Acceptance criteria
- Run `configs/pilot_full_10pct.yaml` on the laptop (ask before starting).
- Test F1 within ±0.01 of the pilot's 0.9020.
- Record run ID, commit, environment and result in `docs/reproduction.md`.
- If outside tolerance: log it in `docs/challenges.md` and investigate (library versions,
  hardware). Do not tune hyper-parameters to force a match.

## #12 Release pipeline (continuous delivery) — `must` `ci`
As the developer, I want a tagged version to be built and published automatically, so that
every release is reproducible and downloadable.

Acceptance criteria
- `.github/workflows/release.yml` on tags `v*`: tests → `python -m build` → GitHub Release
  with wheel, sdist and CHANGELOG notes.
- Tag `v0.1.0` once everything above is merged.

## #13 Documentation — `must` `docs`
As a reader or examiner, I want complete documentation, so that I can install, run, verify and
cite the project without asking the author.

Acceptance criteria
- README: purpose and thesis context; features; installation (Windows and Linux, CPU vs CUDA
  torch); quick start; config example; data download and licence note; testing; CI/CD with
  links to the workflow files; reproduction result; project structure; how to cite.
- `CITATION.cff`, `CONTRIBUTING.md` (branching, commit and PR rules).
- `docs/technology-choices.md`: one table per decision against the Lecture 5 criteria
  (performance, scalability, community support, compatibility), including rejected
  alternatives (Hydra, `evaluate`, Poetry, Jenkins, GitLab CI).
- `docs/traceability.md`: Assignment 2 requirement → module → test (Lecture 3 requirements
  tracing).

## #14 Docker image — `could` `ci`
CPU Dockerfile; image built in CI on tags; publishing to GHCR optional.

## #15 Developer conveniences — `could` `chore`
pre-commit hooks running ruff; coverage badge.

---

## Out of scope for v0.1.0 (Won't have — now)
TWNERTC cross-lingual pipeline, CWEA augmentation, generative-model adapter (WP3 — the
contract in #5 is designed so it can plug in later), XLM-R experiments, W&B / MLflow
experiment tracking (planned for the thesis phase).
