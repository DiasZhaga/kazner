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
