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

## 2026-10-07 — Words lost to truncation: score them, but measure the effect first

**Problem.** The pilot scored only the words that fitted into `max_len` subwords. Words
beyond that were never predicted and silently left out of P/R/F1. A word-level contract
that "contains every gold word" has to decide what such words get. While planning, these
words were first described as containing "no entity starts", and the user asked for a
stricter check. It turned out that this check alone was misleading. At `max_len = 128`, the
5 lost test words carry `I-LAW O O O O`, so one `LAW` entity is cut. At `max_len = 64`
(cross-lingual runs), 274 test words are lost.

**Resolution.** Truncated words are scored with `pred = O` and flagged by an optional
`truncated` column, and `metrics.json` reports `n_truncated_words`. A model is thus
penalised for what it cannot see, rather than having its F1 quietly computed on fewer
words. The effect was measured on the pilot's own prediction files
(`docs/reproduction.md`): F1 changes by −0.00012 to −0.00085 at `max_len = 64` and by 0 at
`max_len = 128`. The pilot-compatible number is still available by scoring the file as the
pilot wrote it.

**Consequences.** From v0.2.0 on, thesis numbers are contract-based. They are slightly
lower than Trainer-based numbers whenever sentences are truncated. Both values are stored
(#9), and the reproduction check (#11) compares the Trainer-based one with the pilot.

## 2026-10-07 — The pilot paper's Table I could not be traced to a stored metric

**Problem.** Table I of the pilot paper reports fine-tuned mBERT at
P/R/F1/acc = 0.939 / 0.951 / 0.945 / 0.988. No metrics file in the pilot repository has
these values. The closest one, the 100% data-efficiency run, has recall 0.9515, which
rounds to 0.952.

**Resolution.** The new evaluator re-scored `results/test_predictions_mbert.csv`, written by
`train_ner_mbert.py`, which printed its metrics but never saved them. The result was
0.93945 / 0.95107 / 0.94522 / 0.98804, i.e. exactly Table I. The table is therefore
traceable to that run (full train split, 1 epoch, batch 8). This is the kind of drift that
run IDs (NFR2, #7) are meant to prevent: a published number whose only record was console
output.

## 2026-10-13 — Run IDs keyed on the code tree, not on the commit

**Problem.** Assignment 2 defined the run ID as a hash of the configuration and the git
*commit*. With that definition, every documentation-only commit would change every run ID.
Already completed runs would then look new, and the idempotent runner (#9) would retrain
them, which costs hours of GPU time for nothing.

**Resolution.** The run ID is the hash of the resolved single-seed configuration plus the
git tree hash of `src/kazner` (`git rev-parse HEAD:src/kazner`). This refines the
Assignment 2 architecture. The full commit hash is still stored in `environment.json`, so
traceability to a commit is not lost. Tests on a temporary repository check that a
docs-only commit keeps the ID and that a code commit changes it. `name` and `output_root`
are excluded from the hash, because they label or place a run without changing what it
computes.

**Limitations.** Changes outside `src/kazner`, such as dependency pins in `pyproject.toml`,
do not change the ID; they are recorded in `environment.json`. Uncommitted changes under
`src/kazner` are flagged with a warning and `package_dirty: true`, because the tree hash
then does not describe the code that actually ran. Outside a git checkout (an installed
wheel), the package version is used as the code reference.

## 2026-10-13 — PyYAML reads `5e-5` as a string

**Problem.** YAML 1.1, which PyYAML implements, only recognises a float when it contains a
dot. `learning_rate: 5e-5` is therefore loaded as the string `"5e-5"`, while `5.0e-5` is
loaded as a number. A naive loader would either crash late inside the Trainer or, worse,
silently compare strings.

**Resolution.** Float fields accept numeric strings explicitly and convert them; any other
string is a `ConfigError` naming the key. A test covers `5e-5`. The example configs use
`5.0e-5` so that they read the same in any YAML parser.

## 2026-10-13 — The paper's LoRA parameter count includes an unused copy of the classifier

**Problem.** The pilot paper reports 373,350 trainable parameters (0.210%) for LoRA on mBERT.
With `TaskType.TOKEN_CLS`, PEFT wraps the classifier in a `ModulesToSaveWrapper`. The wrapper
keeps the original head and trains a copy (`modules_to_save.default`). The pilot then set
`requires_grad = True` on both. Only the copy is used in the forward pass, so the original
never receives a gradient, yet it was counted.

**Resolution.** `kazner.methods.count_parameters` skips parameters under `.original_module.`.
On mBERT with the 51 KazNERD labels this gives **334,131 of 177,596,979 (0.188%)**: 294,912 LoRA
weights plus 39,219 for the head. The pilot's counting is kept as `trainable_params_pilot` and
reproduces the paper exactly (373,350 of 177,636,198 = 0.2102%). A test on a tiny BERT
checks both formulas. Training is unaffected: the extra parameters get no gradient.

**Consequences.** The thesis reports the corrected share and notes the difference from the
paper. The conclusions do not change, since LoRA still trains about 0.2% of the parameters.

## 2026-10-13 — Loading a head for a different label set fails in transformers 5

**Problem.** The network test with `hf-internal-testing/tiny-random-bert` failed with
"You set `ignore_mismatched_sizes` to `False`". The checkpoint already contains a
token-classification head for another number of labels, and transformers 5 refuses to drop
it silently.

**Resolution.** `build_model` always passes `ignore_mismatched_sizes=True`, because kazner
always creates a fresh head for its own label set. The pilot's cross-lingual pipeline did the
same when loading the Turkish checkpoint. For mBERT, which has no head, nothing changes, and
the parameter counts above confirm it.

## 2026-10-13 — The same experiment got two run IDs because the data path was hashed

**Problem.** While writing the configs for experiments B and C, two configs described exactly
the same run: LoRA, lr 1e-4, r 8, seed 42, 10% KazNERD. One lives in `configs/`
(`data.dir: ../data/kaznerd`), the other in `configs/lora_sweep/` (`../../data/kaznerd`). The
run ID from #7 hashed `data.dir` as written, so the two configs produced different IDs, and the
same hour-long run would have been trained twice. The same would happen on any machine where
the data lives somewhere else.

**Resolution.** A path is a location, not content. `data.dir` is removed from the hashed
config, and `data_fingerprint` (SHA-256 over the split files, with CRLF normalised to LF) is
hashed instead. The same KazNERD release therefore gives the same ID from any directory and
with any line endings, while changed data gives a new ID. Across the 10 configs, two pairs now
share IDs (14 distinct runs instead of 15). `config.json` records the fingerprint.

## 2026-10-13 — Experiment design: seed scope, LoRA scaling and selection on validation

- **Seeds.** In experiment B only the training seed varies (42, 43, 44). The 10% subset stays
  fixed (`subsample_seed: 42`, as in the pilot), so the spread measures training variance,
  not sampling variance. Sampling variance is a separate question for later.
- **LoRA screen (C).** `alpha = 2r` keeps the LoRA scaling `alpha / r` constant when r goes
  from 8 to 16, so the screen varies capacity and learning rate, not the effective step size.
- **Selection.** The best LoRA setting is chosen by `validation_best` in `metrics.json`. Test
  F1 is reported only for the chosen setting, which is then re-run with three seeds. Choosing
  by test F1 would overfit the test set.

## 2026-10-13 — Aggregation warns on mixed code versions, not on mixed commits

Issue #10 asked for a warning when a group mixes runs from different git commits. Since #7,
however, a documentation-only commit deliberately keeps run IDs, so runs from different
commits can be identical computations. A warning on every such group would be noise. The
aggregator therefore warns when a group mixes different `src/kazner` tree hashes
(`environment.json → code.code_ref`), which is exactly when the code that produced the
numbers differs.

## 2026-10-13 — The pilot's frozen-encoder result depended on run order

**Problem.** Experiment B reproduced the pilot bit for bit for full fine-tuning (0.9020) and
LoRA (0.6858) at seed 42, but not for the frozen encoder (0.0781 vs 0.0919).

**Cause.** The pilot's file timestamps show that `run_peft_comparison.py` trained full
fine-tuning and the frozen encoder in **one process** on 12 June. Full finished at 01:18:45;
frozen finished at 01:25:20, 344 s of training later. LoRA was run in a separate process on 4–5
June. The pilot called `set_all_seeds(42)` once per process. Each model that was the first in
its process (full, LoRA) therefore started from the seeded state, while the frozen model's new
classification head was initialised from a random-number state already consumed by the full
fine-tuning run. kazner seeds immediately before every model, which is why full and LoRA match
and frozen does not.

**Consequences.** The published frozen value was a function of which experiment happened to run
before it. A frozen encoder trains only the randomly initialised head, so it is also the
setting most sensitive to that initialisation. The three-seed estimate, 0.0819 ± 0.0039, is
reported instead. Per-run seeding removes this hidden dependency on run order.

