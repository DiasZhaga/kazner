# kazner — refactored Kazakh NER toolkit (Assignment 3 + thesis codebase)

## Purpose
A small, tested, documented Python package extracted from the pilot research repo
(`../ner-project` — read-only reference; adjust the path if it differs). It has two roles:

1. Course Assignment 3 "Software Development and Integration". Graded on: use of Git/GitHub,
   CI/CD, justified technology choices, automated tests, code quality and documentation.
2. The future codebase of the MSc thesis. It implements part of the target architecture from
   Assignment 2: a configuration layer with run IDs, an adaptation-method registry, a
   word-level prediction contract with one evaluator, and a captured environment.

The work plan is `ROADMAP.md`. Work through it one issue at a time, in order.

## Hard rules
- Never push to `main` directly. One issue → one branch (`feat/3-iob2-reader` style) →
  PR with "Closes #N" → CI green → merge with a MERGE COMMIT. Do not squash: the commit
  history is part of the grade.
- Small commits with Conventional Commit messages in English
  (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, `chore:`).
- Never rewrite published history: no force-push, no amending pushed commits, no backdating.
- Never commit datasets, model checkpoints, `.venv/`, caches or tokens. KazNERD is
  downloaded by a script.
- Before creating issues, PRs, releases or changing repo settings via `gh`, show the plan
  and wait for confirmation.
- Never start a training run longer than ~10 minutes on the user's laptop without
  confirmation.
- Read the pilot code before porting it. Keep behaviour identical unless the issue says
  otherwise (alignment rule, label normalisation, label2id built from the full train split).
- When you make a non-obvious decision or hit a real problem, append a short entry to
  `docs/challenges.md` (date, problem, resolution). The report's self-analysis is written
  from this file, so it must be honest and specific.

## Stack (keep `docs/technology-choices.md` in sync with this list)
- Python 3.11, src layout, `pyproject.toml` (setuptools). Package `kazner`, console script
  `kazner`.
- Runtime deps pinned to the pilot: `torch>=2.5,<3`, `transformers==5.0.0`,
  `datasets==4.5.0`, `peft==0.19.1`, `accelerate==1.12.0`, `seqeval==1.2.2`, `numpy`,
  `pandas`, `PyYAML`.
- Use `seqeval` directly instead of `evaluate.load("seqeval")`: no network access needed
  at evaluation time.
- Configuration: YAML → frozen dataclasses with validation. No Hydra (too heavy for a
  small package).
- Dev tools: `pytest`, `pytest-cov`, `ruff` (lint + format), `build`.
- Design patterns: the method registry is a Factory Method. Do NOT use a Singleton for
  configuration or logging — pass config explicitly so that everything stays testable.

## Target layout
```
src/kazner/
  data.py        # read_iob2, label normalisation, build_label2id, KazNERD download
  align.py       # word -> subword label alignment (-100 on continuations/specials)
  contract.py    # word-level prediction schema: sentence_id, word_idx, word, gold, pred
  evaluate.py    # entity-level P/R/F1 (seqeval) + token accuracy on contract files
  fertility.py   # tokenizer fertility analysis
  config.py      # dataclasses, YAML loading, validation, run_id
  methods.py     # registry: full / frozen / lora
  train.py       # HF Trainer wiring, contract predictions, idempotent runs
  aggregate.py   # mean ± sd across seeds
  cli.py         # kazner fertility | train | evaluate | aggregate
configs/         # example YAML experiment configs
tests/           # pytest; tests/data = tiny SYNTHETIC IOB2 fixtures (no real KazNERD text)
docs/            # technology-choices.md, traceability.md, reproduction.md, challenges.md
.github/workflows/ci.yml, release.yml
```

## Testing
- Unit tests run offline on CPU, under ~2 minutes in total.
- Markers: `@pytest.mark.network` for anything that downloads a model or tokenizer;
  `@pytest.mark.smoke` for end-to-end training on `hf-internal-testing/tiny-random-bert`.
- Commands:
  - `pytest -m "not network and not smoke"`
  - `pytest -m smoke`
  - `ruff check .` and `ruff format --check .`
- Every module ships with its tests in the same PR. Target ≥ 80% coverage for all modules
  except `train.py`.

## CI/CD
- `ci.yml` — on push and pull_request:
  - `lint` job: ruff check + format check;
  - `test` job: matrix `ubuntu-latest` + `windows-latest`, Python 3.11, CPU-only torch from
    `https://download.pytorch.org/whl/cpu`, pip cache, coverage report uploaded as artifact;
  - `smoke` job (ubuntu only): network + tiny-model end-to-end training.
- `release.yml` — on tags `v*`: run tests, `python -m build`, create a GitHub Release with
  the wheel and sdist and the CHANGELOG section. This is continuous DELIVERY: deployment is
  a deliberate, manual tag decision.
- README shows the CI badge. Remind the user to protect `main` (require PR + passing CI)
  once CI exists.

## Definition of done (every issue)
Code + tests + README/docs updated + CHANGELOG entry under "Unreleased" + ruff clean +
CI green + a row in `docs/traceability.md` if the issue implements a requirement.

## Environment notes
- The user works on Windows + PowerShell; CI runs Linux and Windows. Use `pathlib`; never
  assume the current working directory — paths come from config or CLI arguments.
- Local GPU: RTX 4050, 6 GB — used only for the reproduction check (ROADMAP #11).

## Communication
Talk to the user in Russian. Code, comments, docs, commit messages, issue and PR texts in
English.
