# Technology choices

> Stub. Each decision gets a full comparison table against the Lecture 5 criteria
> (performance, scalability, community support, compatibility), including the rejected
> alternatives, in #13.

| Area | Choice | Rejected alternatives (to be justified in #13) |
|---|---|---|
| Language / runtime | Python 3.11 | — |
| Packaging | `pyproject.toml` + setuptools, src layout | Poetry |
| ML stack | PyTorch, Hugging Face `transformers` 5.0.0, `datasets` 4.5.0, `peft` 0.19.1, `accelerate` 1.12.0 (pinned to the pilot) | — |
| NER metrics | `seqeval` 1.2.2 called directly | `evaluate.load("seqeval")` (needs network access at evaluation time) |
| Configuration | YAML → frozen dataclasses with validation | Hydra |
| Tests | `pytest`, `pytest-cov` | — |
| Lint / format | `ruff` (pinned version) | — |
| CI/CD | GitHub Actions | Jenkins, GitLab CI |
| Design patterns | Factory Method for the adaptation-method registry; configuration passed explicitly | Singleton for configuration or logging |
