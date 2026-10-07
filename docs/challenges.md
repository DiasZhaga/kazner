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

## 2026-10-07 — The pilot's KazNERD copy has different checksums from the official release

**Problem.** The download script (#3) was supposed to pin the data version with SHA-256
checksums. The checksums of the pilot's `data/IOB2_*.txt` matched neither the upstream
files nor their git blob hashes, and the files were about 7% larger. It was not clear
whether the pilot had been trained on a different version of the corpus.

**Resolution.** Each file was downloaded from IS2AI/KazNERD at commit `bd4333d` and compared
byte by byte. The only difference is the line endings. The pilot's copy uses CRLF: there is
exactly one extra byte per line (1,133,533 lines in train), most likely added by a Windows
checkout. After converting CRLF to LF, all three files are identical to upstream. The
script therefore pins the commit and the SHA-256 of the files as published (LF) and writes
the downloaded bytes unchanged. The reader accepts both line endings, and a test checks
that LF and CRLF files give the same result. On the full corpus, `kazner.data.read_splits`
returns exactly the sentences and the `label2id` of the pilot code for both copies.

**Consequences.** The pilot results were obtained on the official KazNERD release. A
checksum is only a stable data version if line endings are fixed too, which is why the
repository also enforces LF through `.gitattributes`.

## 2026-10-07 — Malformed IOB2 rows: fail loudly instead of skipping

**Problem.** The pilot's `read_iob2` silently skipped rows with fewer than two columns. In a
corrupted file this would shift nothing visibly but drop tokens from sentences, and every
downstream number would quietly change.

**Resolution.** `kazner.data.read_iob2` raises `IOB2FormatError` with file name and line
number. This is a deliberate deviation from the pilot (issue #3). It changes nothing on
KazNERD, which has no malformed rows (the pilot's own fertility run counted 0, and the new
reader parses all three splits). A test documents both behaviours side by side. The
sampling functions in `run_data_efficiency.py` and `run_peft_comparison.py` were also
compared: they are identical (same AST, both called with seed 42), so the data-efficiency
and PEFT experiments used the same subsets for equal fractions.

## 2026-10-07 — transformers 5 silently ignores `vocab_file` when building a tokenizer

**Problem.** The alignment tests (#4) needed a tokenizer that works offline, so they build a
tiny WordPiece tokenizer from a hand-made vocabulary in `tests/data`. In
transformers 5.0.0, `BertTokenizer(vocab_file=...)` raised no error but ignored the file.
The tokenizer contained only the five special tokens, and every word became `[UNK]`. A test
that only checked the labels would still have passed. Passing the path as `vocab=` worked
but produced a `tokenizers` deprecation warning.

**Resolution.** The test fixture reads the vocabulary into a dict and passes it as `vocab=`.
One test asserts the exact pieces (`Алматыға` → `Алма ##ты ##ға`), so a tokenizer that
falls back to `[UNK]` makes the suite fail. Suites are also run with
`-W error::DeprecationWarning` locally.

**Also checked.** A word that tokenizes to zero subwords would make position-based mapping
of predictions back to words unreliable. The pilot's `train_ner_mbert.py` export relied on
such positional counting. With the mBERT tokenizer no KazNERD word in any split produces
zero subwords, so the pilot's exported predictions are aligned correctly. `kazner` maps by
`word_id` anyway (`first_subword_positions`).
