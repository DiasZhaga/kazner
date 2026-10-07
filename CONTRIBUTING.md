# Contributing

kazner is developed by one researcher, but it follows a team workflow so that every change
is reviewed by CI and traceable to an issue.

## Set-up

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## Workflow: one issue, one branch, one pull request

1. **Issue first.** Every change starts as a GitHub issue written as a user story ("As a …,
   I want …, so that …") with acceptance criteria, a MoSCoW label (`must` / `should` /
   `could`), a type label (`feature` / `test` / `ci` / `docs` / `chore`) and a milestone.
2. **Branch** from an up-to-date `main`, named `<type>/<issue>-<slug>`, e.g.
   `feat/3-iob2-reader`, `ci/12-release`, `docs/13-documentation`.
3. **Commit** small, focused changes with [Conventional Commit](https://www.conventionalcommits.org/)
   messages in English: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, `chore:`.
4. **Pull request** with `Closes #N` in the description. Only one pull request is open at a
   time (Kanban WIP limit).
5. **Merge** only when CI is green, with a **merge commit**. No squash and no rebase-merge:
   the history of small commits is kept.

`main` is protected: direct pushes are impossible, the `lint`, `test (ubuntu-latest)` and
`test (windows-latest)` checks are required, and administrators cannot bypass the rules.
Published history is never rewritten: no force-push, and no amending of pushed commits.

## Definition of done

- [ ] code and tests in the same pull request; at least 80% coverage per module
- [ ] README / docs updated
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] CI green
- [ ] a row in `docs/traceability.md` if a requirement is implemented
- [ ] non-obvious decisions and real problems logged in `docs/challenges.md`

## Tests

- Offline unit tests: `pytest -m "not network and not smoke"`. They must pass on Windows and
  Linux, on CPU, in under two minutes.
- `@pytest.mark.network` marks anything that downloads a model or tokenizer.
  `@pytest.mark.smoke` marks end-to-end training on a tiny model.
- Use synthetic fixtures in `tests/data`. **Never commit real KazNERD text**, datasets, model
  checkpoints, `.venv/`, caches or tokens.
- When porting pilot code, copy the pilot function verbatim into `tests/reference/` and add
  a test that compares the new code with it.
- Use `pathlib`, and take paths from arguments or configuration. Never assume the current
  working directory.

## Releases

1. Move the `[Unreleased]` entries in `CHANGELOG.md` to a new `## [X.Y.Z] - YYYY-MM-DD`
   section, then set `version` in `pyproject.toml` and `CITATION.cff`, in a pull request.
2. After the merge, tag `main` and push the tag:
   `git tag -a vX.Y.Z -m "kazner X.Y.Z" && git push origin vX.Y.Z`.
3. `release.yml` checks the tag against the version, tests, builds and publishes the GitHub
   Release.
