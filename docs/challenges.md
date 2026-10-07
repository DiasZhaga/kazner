# Challenges and decisions log

Short, dated entries for non-obvious decisions and real problems met during development.
The self-analysis section of the Assignment 3 report is written from this file.

Format: **date — problem** · resolution · consequences.

---

## 2026-10-07 — Submission deadline forced a scope split

**Problem.** The roadmap planned one release (v0.1.0) with thirteen issues, including the
configuration layer, the method registry, training and a training-based reproduction of the
pilot's F1 = 0.9020. With the deadline one day away, the training part could not be finished
and verified properly.

**Resolution.** The work was split into two milestones:

- **v0.1.0 (submission):** #1 bootstrap, #2 CI, #3 IOB2 data, #4 label alignment,
  #5 prediction contract and evaluator, #6 tokenizer fertility, #12 release pipeline,
  #13 documentation.
- **v0.2.0 (after submission):** #7 configuration and run IDs, #8 method registry,
  #9 training, #10 aggregation, #11 training-based reproduction.

Behaviour is still checked against the pilot in v0.1.0, but without training: the pilot's
saved prediction files are re-scored by the new evaluator (#5), and tokenizer fertility is
recomputed on full KazNERD (#6).

**Consequences.** v0.1.0 cannot train models. The requirements that depend on training stay
open in `docs/traceability.md` and point to their v0.2.0 issues.

## 2026-10-07 — Required checks cannot be chosen before CI has run once

**Problem.** Branch protection on `main` can only require status checks that GitHub has
already seen. Before #2 the repository had no workflow, so the checks did not exist yet, and
the check names of a matrix job (`test (ubuntu-latest)`) depend on how the job is named.

**Resolution.** The CI pull request was opened first, so that one run registered the checks.
Job names were set explicitly (`name: test (${{ matrix.os }})`) to keep them stable. Then
`main` was protected: pull request required, `lint` and both `test` jobs required,
no bypass for administrators. `smoke` is deliberately not required, because it depends
on the Hugging Face Hub being reachable. A deliberately failing test was then pushed to
the same pull request. Both `test` jobs turned red and the merge button was disabled
(screenshot kept for the report). The test was removed in a follow-up commit, without
rewriting history.

**Consequences.** Renaming a CI job now also requires updating the branch-protection rule.
Otherwise the old required check never reports and every pull request stays blocked.
