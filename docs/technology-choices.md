# Technology choices

Every decision below is assessed against the four selection criteria of Lecture 5
(Applied Software Development Project, *Development and Integration*):

- **Performance**: does the tool meet the computational needs of the project?
- **Scalability**: does it keep working as data, experiments and code grow?
- **Community support**: are there active maintainers, documentation and answers?
- **Compatibility**: does it fit the existing stack, the pilot, Windows + Linux, and CI?

The project context sets the weights. kazner has one developer, who is also the
researcher. It runs on a Windows laptop with a 6 GB GPU and on rented Linux GPUs. It must
reproduce the pilot (`ner-project`) exactly, and the code must stay re-runnable by
examiners for years (Assignment 2, §1.3). **Compatibility** with the pilot therefore
dominates. Where it does not separate the options, the simpler and lighter option wins.

Ratings: ++ strong, + adequate, − weak, −− unsuitable.

---

## 1. Language: Python 3.11

| Criterion | **Python 3.11** | R | Julia |
|---|---|---|---|
| Performance | + heavy computation runs in PyTorch's C++/CUDA kernels; Python is only glue | − no first-class transformer training stack | ++ fast native code |
| Scalability | ++ same code on laptop, cloud GPUs and clusters | − | + |
| Community support | ++ default language of NLP research; the Hugging Face ecosystem | + strong in statistics, weak in NLP | − small NLP ecosystem |
| Compatibility | ++ the pilot is Python 3.11; every dependency is Python-first | −− would mean rewriting the pilot | −− same |

**Decision: Python 3.11**, the version verified with the pilot (3.11.9). Newer Pythons are
not used, because the pinned ML stack was validated on 3.11 only.

## 2. ML stack: PyTorch + Hugging Face `transformers` / `datasets` / `peft` / `accelerate`

| Criterion | **PyTorch + HF (pinned to the pilot)** | TensorFlow / Keras | JAX / Flax |
|---|---|---|---|
| Performance | ++ FP16, gradient accumulation; fits 6 GB VRAM for mBERT | + | ++ |
| Scalability | ++ the same Trainer code goes from CPU tests to multi-GPU (`accelerate`) | + | + |
| Community support | ++ most NER and PEFT work (LoRA, QLoRA) is published for PyTorch + HF | − HF support for TF is shrinking | − smaller PEFT ecosystem |
| Compatibility | ++ identical to the pilot, so results stay comparable | −− not the pilot | −− not the pilot |

**Decision:** keep the pilot's stack and pin it (`transformers==5.0.0`, `datasets==4.5.0`,
`peft==0.19.1`, `accelerate==1.12.0`, `torch>=2.5,<3`). Pinning is what makes "same code,
same numbers" possible (NFR1). Upgrades are deliberate, versioned changes. The PyTorch
build (CPU or CUDA) is chosen at install time, because one wheel cannot serve both. CI
uses the CPU index.

## 3. NER metrics: `seqeval` called directly

| Criterion | **`seqeval` directly** | `evaluate.load("seqeval")` (pilot) | own implementation |
|---|---|---|---|
| Performance | ++ | + | + |
| Scalability | + | + | − every new scheme is new code |
| Community support | + reference library for CoNLL-style entity metrics | + | −− |
| Compatibility | ++ same numbers as the pilot (verified: difference 0, `docs/reproduction.md`) | − downloads the metric script from the Hub at evaluation time and needs network or cache | − risk of subtle differences |

**Decision: `seqeval` 1.2.2 directly.** `kazner.evaluate` reproduces exactly what
`evaluate.load("seqeval")` computes (`classification_report` micro average in default
mode, plus `accuracy_score`). It works offline, so CI and examiners need no network
access. **Rejected: `evaluate`**: an extra dependency and a network fetch for a five-line
computation.

## 4. Packaging and dependency management: `pyproject.toml` + setuptools, pip

| Criterion | **setuptools + pip** | Poetry | conda |
|---|---|---|---|
| Performance | + | + | − slow solver |
| Scalability | + standard PEP 517/621 metadata, works with any build frontend | + | + |
| Community support | ++ the reference implementation; every tutorial and CI template | + | + strong in science |
| Compatibility | ++ `pip install -e ".[dev]"` works the same on Windows and Linux; torch's CPU/CUDA index works with plain pip | − its own resolver and lockfile; installing torch from a custom index is awkward; another tool for examiners to install | − mixes conda and pip packages; CI images get heavier |

**Decision: setuptools with a src layout**, a `dev` extra, and `python -m build` for
releases. **Rejected: Poetry**: its benefits (lockfile, virtualenv management) do not
outweigh an extra tool and the friction with PyTorch's per-platform index for a
one-person project. A lockfile for the thesis machines is planned (Assignment 2, captured
environment) and does not require switching the build backend.

## 5. Configuration: YAML → frozen dataclasses with validation (v0.2.0, #7)

| Criterion | **YAML + frozen dataclasses** | Hydra | argparse flags only (pilot) |
|---|---|---|---|
| Performance | ++ | + | ++ |
| Scalability | + sweeps are expanded explicitly (seed lists) | ++ composition, multirun sweeps | −− the pilot's configuration drift (Assignment 2, §3.2) |
| Community support | ++ standard library + PyYAML | + | ++ |
| Compatibility | ++ plain Python objects, trivially testable, hashable for run IDs | − creates its own per-run output directory and configures logging by default, its own CLI override grammar, OmegaConf objects inside the code | + |

**Decision: YAML files loaded into frozen dataclasses**, with unknown keys rejected.
Assignment 2 proposed "a tool *such as* Hydra". For a small package, Hydra's composition
and multirun features cost more than they give. Its per-run output directories and logging set-up overlap with kazner's own run
directories, and its OmegaConf objects complicate deterministic hashing for run IDs. **Rejected: Hydra** (too heavy for this
package; can be revisited if sweeps outgrow seed lists).

## 6. Testing: pytest + pytest-cov

| Criterion | **pytest** | unittest |
|---|---|---|
| Performance | + markers select fast offline tests (`not network and not smoke`) | + |
| Scalability | ++ fixtures, parametrisation (e.g. 4 `max_len` × 3 splits in one test) | − boilerplate grows with cases |
| Community support | ++ de facto standard | + standard library |
| Compatibility | ++ runs unittest tests too; coverage via `pytest-cov` | + |

**Decision: pytest** with the markers `network` (downloads from the Hugging Face Hub) and
`smoke` (end-to-end training, v0.2.0). Pilot behaviour is pinned by **oracle tests**:
verbatim copies of pilot functions in `tests/reference/`, compared with the new code.

## 7. Code quality: ruff (lint + format)

| Criterion | **ruff** | flake8 + black + isort | pylint |
|---|---|---|---|
| Performance | ++ milliseconds for the whole repository | − three tools, several seconds | −− slow |
| Scalability | ++ | + | + |
| Community support | ++ very actively developed | + | + |
| Compatibility | ++ one tool, configured in `pyproject.toml`, same result on every OS | − three configs that must agree | − noisy defaults |

**Decision: ruff**, pinned to an exact version (0.16.10), because formatter output changes
between releases. A version mismatch between a laptop and CI would otherwise fail
`ruff format --check` on unchanged code. Rules: `E, W, F, I, B, UP, SIM`, line length 100.

## 8. Version control and hosting: Git + GitHub

| Criterion | **Git + GitHub** | Git + GitLab | Git + Bitbucket |
|---|---|---|---|
| Performance | ++ | ++ | ++ |
| Scalability | ++ | ++ | + |
| Community support | ++ the largest open-source and research community; issues, Projects, Releases | + | − |
| Compatibility | ++ public repo with a shareable link, as the assignment requires; free CI minutes for public repos; the pilot dataset (IS2AI/KazNERD) is on GitHub | + self-hosting possible, not needed | − oriented to Jira teams |

**Decision: Git on GitHub.** Workflow: one issue → one short-lived branch
(`feat/3-iob2-reader`) → pull request with "Closes #N" → green CI → **merge commit**, with
no squashing, so the history shows every step. Commit messages follow Conventional
Commits. `main` is protected (PR required, required checks, no administrator bypass).
This is trunk-based development with short-lived branches, which suits one developer
better than Git Flow's long-lived `develop` and `release` branches (Lecture 5).

## 9. CI/CD: GitHub Actions

| Criterion | **GitHub Actions** | GitLab CI | Jenkins | Travis CI |
|---|---|---|---|---|
| Performance | + hosted Ubuntu and Windows runners; pip and Hugging Face caches | + | + depends on own hardware | + |
| Scalability | + matrix builds (OS × Python) in a few lines | + | ++ | + |
| Community support | ++ marketplace actions (`setup-python`, `upload-artifact`, `action-gh-release`) | + | + many plugins, ageing | − free open-source plan replaced by limited credits |
| Compatibility | ++ built into the GitHub repository and branch protection; **Windows runners** match the user's laptop (NFR4) | − requires moving the repository | −− a server to install, secure and maintain | − |

**Decision: GitHub Actions.**
- `ci.yml` runs on every PR and push to `main`: lint, tests on Ubuntu and Windows with
  CPU-only PyTorch and coverage artifacts, and a network/smoke job.
- `release.yml` runs on tags. This is continuous **delivery**: building and publishing
  are automated, but releasing is a deliberate tag.

**Rejected:** Jenkins (server administration for one person is pure overhead), GitLab CI
(it would mean leaving GitHub, where the repository, issues and branch protection live),
Travis CI (open-source builds now depend on limited credits).

## 10. Data distribution: download script with checksums

| Criterion | **script + SHA-256 (pinned upstream commit)** | commit the data | Git LFS | DVC |
|---|---|---|---|---|
| Performance | + 20 MB once | − bloats every clone | + | + |
| Scalability | + one entry per dataset | −− | + | ++ |
| Community support | ++ standard library only | | + | + |
| Compatibility | ++ respects the CC BY 4.0 licence (no redistribution), works offline afterwards | −− redistributes a licensed corpus | − LFS quota, and still redistributes | − another tool and remote storage for 3 files |

**Decision:** `scripts/download_kaznerd.py` pins IS2AI/KazNERD to commit `bd4333d` and
verifies SHA-256 (NFR1, NFR7). The test fixtures are synthetic. DVC remains an option for
WP2 and WP3, when derived datasets appear.

## 11. Design patterns: Factory Method registry; no Singleton

| Option | Fit |
|---|---|
| **Factory Method registry** (`@register_method("lora")`, v0.2.0 #8) | ++ a new adaptation method costs one function and one config value (NFR6). Each builder returns an interchangeable model, i.e. a *strategy* in the sense of Assignment 2. The registry is the factory that creates it. |
| `if method == ...` chains (pilot) | − every new method edits a central function (Assignment 2, limitation 2–3) |
| **Singleton** for configuration or logging | −− rejected. Lecture 4 lists it for "loggers and configs", but global state makes tests depend on each other and hides inputs. Configuration is passed explicitly, so every function is testable in isolation, with no reset hooks. |

## 12. Deferred: experiment tracking (W&B / MLflow)

Assignment 2 proposed W&B or MLflow. v0.1.0 does not train, so it has nothing to track.
v0.2.0 writes per-run `config.json`, `environment.json` and `metrics.json` under a
deterministic run ID, which covers traceability (NFR2) offline. A tracking server is
planned for the thesis phase (ROADMAP: *Won't have — now*).
