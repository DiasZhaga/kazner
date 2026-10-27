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


## 2026-10-22 — Byte-level BPE tokenizers got pre-split words without their space marker

**Problem.** While planning the KazRoBERTa baseline, I checked how its tokenizer handles
kazner's word-level input (`is_split_into_words=True`). Byte-level BPE (RoBERTa family) marks
a word that follows a space with `Ġ`. Without `add_prefix_space=True`, each pre-split word is
encoded as if it started the text, so the marker is lost. In transformers 4.x the fast RoBERTa
tokenizer refused this combination. transformers 5.0.0 accepts it without an error or a
warning. On KazNERD validation, KazRoBERTa produced 1.569 subwords per word instead of 1.205.
Across all of KazNERD, 1,189,650 of 1,302,352 words got different token ids: every word
except the first of its sentence. The model would have been fine-tuned on ids that it saw in
pre-training only inside words or at the start of a text. Training would still have run and
reported a plausible, lower F1, and the loss would have been blamed on the model. The same
effect explains the gap between isolated and in-context fertility for KazRoBERTa in the
tokenizer comparison of 2026-10-13 (1.571 vs 1.211).

**Resolution.** The new `model.add_prefix_space` key is passed to the tokenizer only when it
is set. It is explicit rather than automatic, so that the value appears in `config.json` and
in the run ID. Before training, `prepare_corpus` checks that every training word gets the same
ids as pre-split input and as running text. Running-text tokens are mapped to words by
character offsets. If any word differs, training stops with an error that names the key. The
check has no tolerance: mBERT and XLM-R produce 0 mismatches on all 1.3 M KazNERD words, and
so does KazRoBERTa with the flag. Any mismatch therefore means a real problem.
`kazner fertility --add-prefix-space` reports the fertility that training actually sees.

**Consequences.** mBERT is unaffected: its tokenizer is created exactly as before when the key
is `null`, and setting the flag does not change its ids either (network test). Run IDs change,
as with every change to `src/kazner`. The offline tests use a byte-level BPE tokenizer trained
on the synthetic fixture and saved in KazRoBERTa's layout (`config.json`, `vocab.json`,
`merges.txt`). transformers builds such a tokenizer through a code path that `tokenizers`
deprecates, so those tests filter that one warning. Suites run with
`-W error::DeprecationWarning` stay green.

## 2026-10-22 — Identifying the model of a run: Hub revision and local checkpoints

**Problem.** A run ID covered the config, the code and the data, but the model only by its
name. A Hub repository can change under the same name; KazRoBERTa's was last modified in
October 2025. The vocabulary study (#35–#37) will also train from locally produced checkpoints.
For those, a path says nothing about the content, and a relative path passed straight to
`from_pretrained` would depend on the current working directory.

**Decisions.**
- `model.name` is a local directory only when written as a path: `./`, `../`, absolute, or
  with a drive letter. Hub IDs look like relative paths (`org/name`). Guessing by checking
  whether a directory exists would make the meaning of a config depend on the machine.
- A local directory is hashed by content, like the data. The fingerprint covers each file's
  relative path and content; dot-entries (caches, locks) are skipped. Line endings of `.json`
  and `.txt` files are normalised: `save_pretrained` on this Windows laptop wrote
  `config.json` and the tokenizer files with CRLF. Without normalisation, the same checkpoint
  saved on Windows and on Linux would get two run IDs. Binary files are hashed byte for byte.
- For Hub models, the requested `model.revision` is hashed. The commit that it resolved to is
  recorded in `environment.json` (`AutoConfig._commit_hash`), but it is not hashed. Hashing it
  would need the Hub or the cache just to compute IDs, and without a pin it would silently
  start new runs whenever the repository changed. Reported results should therefore pin a
  commit; that is a rule for the experiment configs (#32), not something the code enforces.
- `model.revision` together with a local directory is a configuration error rather than
  being ignored.

## 2026-10-23 — Baseline protocol: tune every model the same way

**Problem.** XLM-R and KazRoBERTa need their own learning rate. mBERT's 5e-5 was inherited
from the pilot and was never tuned. Tuning only the new models would compare tuned baselines
with an untuned reference and bias the comparison towards them.

**Decisions.**
- All three models get a four-point learning-rate screen at seed 42, with selection by
  validation F1 and the grid extended at an edge (the LoRA screen's best value was at the
  edge). Everything else stays at the pilot protocol, including no warm-up. XLM-R is known to
  be sensitive to warm-up. A collapsed run would be reported, and warm-up would then be a
  separate change applied to all three models.
- mBERT at 5e-5 is screened again with a pinned commit and the current code. It is a new run
  ID but the same computation, so it doubles as a regression check of #30 and #31 against the
  bit-exact pilot reproduction (F1 0.9020369285).
- Exact parameter counts were taken from the loaded models, not from the model cards:
  KazRoBERTa's card gives 83.5 M for the masked-LM model, while the token-classification model
  has 82.9 M.

**Practical problem.** Downloading the two checkpoints (1.45 GB) took about 4.5 hours on the
available connection (≈ 90 KB/s). Corpus downloads for #34 have to be sized and scheduled
with that in mind: streaming a fixed subsample, not downloading whole parquet shards.

## 2026-10-25 — Moving experiment B2 into the package: regression without scikit-learn

**Problem.** The first B2 analysis was a script in the git-ignored `results/` folder, so the
analysis behind a paper result was not versioned. It also used scikit-learn's
`LogisticRegression`, which is not a kazner dependency: it is only installed because seqeval
needs it.

**Decisions.**
- The regression is fitted by Newton's method in NumPy (about 15 lines). A test checks it
  against the closed form: with one binary predictor, the maximum-likelihood odds ratio equals
  the cross-product ratio of the 2 × 2 table. A ridge of 10⁻⁶ keeps the Hessian invertible
  when a run makes no error or a category separates the outcome; such a run gets `null`
  instead of a meaningless odds ratio.
- Re-running B2 with the package reproduced every table cell exactly but moved the odds ratios
  by up to 0.002. This was a genuine discrepancy, not noise. The script had stopped at
  scikit-learn's default tolerance (10⁻⁴). With a tolerance of 10⁻¹⁰, scikit-learn agrees with
  the package to six decimals (1.010645 vs 1.010646). The B2 conclusions are unchanged.
- The CI now covers the group's mean over seeds: every bootstrap replicate refits all seeds on
  the same resampled test sentences. The script bootstrapped seed 42 only.
- The reference tokenizer is fixed and separate from the runs' models. For the baselines and
  the vocabulary study, "fragmented" therefore always means fragmented under original mBERT,
  so every model is scored on the same entity subsets.
- The training subsample is defined in one place (`kazner.data.training_subsample`), shared by
  training and the analysis. Word frequencies and "seen" entities therefore come from exactly
  the sentences a run was trained on.

## 2026-10-25 — A reproducible corpus sample on a slow connection

**Problem.** Continued pre-training (#36) needs about 20 M words of unlabeled Kazakh text.
The FineWeb-2 `kaz_Cyrl` train split is 6.8 GB of parquet. On the available connection
(≈ 90 KB/s, measured when the baseline checkpoints were downloaded) that would take about 21
hours. The obvious bounded alternative, the first N streamed documents, is the head of one
file, not a sample of the corpus. The split that is small enough to download, FineWeb-2's
`test`, is explicitly marked "should not be trained on".

**Decisions.**
- Sample by parquet row groups. All row groups of both train files are put in an order fixed
  by a seed and read one at a time, text column only, until the cleaned text reaches the
  target. The download stays proportional to the sample, the sample spreads across the
  corpus, and the result is reproducible from (revision, seed, target, rules). The manifest
  lists every row group read.
- Clean at the line (paragraph) level. FineWeb-2 is deduplicated and language-identified at
  document level, but boilerplate lines repeat across documents and Kazakh pages contain
  Russian and English paragraphs. Russian lines would put Russian pieces into the extended
  vocabulary (#35). A line is therefore kept only if it contains a letter specific to Kazakh.
  Short Kazakh lines without one are lost; this is accepted and counted.
- Overlap with KazNERD is checked by containment, not by equality. KazNERD comes from Khabar
  news transcripts of 2020, and FineWeb-2 crawls run to 2024, so a web paragraph may contain a
  test sentence. Lines and sentences are compared as case-folded word sequences without
  punctuation, because KazNERD splits punctuation off (`барды .`). Sentences shorter than six
  words are not used: they are too generic to be evidence of overlap and would remove ordinary
  lines.
- The Hub path was not exercised before the first real download, to respect the decision not
  to download yet. `--dry-run` reads only the parquet footers (row groups and compressed text
  size per row group), so access and the expected download size can be checked first.

## 2026-10-25 — Read-ahead would have tripled the corpus download

**Problem.** The metadata-only dry run of the corpus script (#34) showed row groups of 1,000
documents with about 1.85 MiB of compressed text each. `HfFileSystem` opens files with
fsspec's default buffering: 5 MiB blocks with `readahead`. Every range read would have fetched
up to 5 MiB more than requested, so a 20 M-word sample would have cost about 380–580 MiB
instead of about 120–170 MiB. Without a measurement this would only have shown up as a slow
download.

**Resolution (#42).** The Hub files are opened with `cache_type="none"`. A thin
`CountingReader` sits between pyarrow and the file and counts the bytes actually read; the
manifest records them as `bytes_read`. Tracing pyarrow's reads showed that a row group costs
exactly one read of its text column chunk, while the footer is read once per file. The test
now asserts exactly that property, rather than an arbitrary fraction of the file. The dry run
against the Hub then measured the footers at 12.08 MiB for both files.

## 2026-10-26 — Extending mBERT's vocabulary without changing anything but fragmentation

**Problem.** The causal test (#37) needs an mBERT that fragments Kazakh less and is otherwise
the same model. Three pitfalls stood in the way.
- `tokenizer.add_tokens` adds "added tokens", which are matched as raw strings before WordPiece
  runs. They would cut words in the wrong places and cannot be `##` continuation pieces.
- A feasibility check showed that the learned pieces also eliminate mBERT's `[UNK]` words
  (0.79% of KazNERD words). Removing `[UNK]` is a second mechanism that would mix with
  fragmentation.
- The `tokenizers` WordPiece trainer is not deterministic. A test that learned candidates
  twice on the same synthetic corpus got two different orders: ties between equally frequent
  pairs are broken in hash order.

**Decisions.**
- New pieces are appended to the WordPiece model's vocabulary itself, keeping every old id. A
  test checks that an empty extension tokenizes all fixture splits identically, and a network
  test does the same for the real mBERT tokenizer.
- Only pieces that mBERT's vocabulary can spell without `[UNK]` are added, and no single
  characters. The extended tokenizer then marks exactly the same words as `[UNK]`. On KazNERD
  validation the `[UNK]` share is 0.788% before and after, for every size.
- New rows are initialised by FVT: the mean embedding, and the mean masked-LM bias, of the
  pieces that spell the new piece. The decoder stays tied to the embeddings, and a test checks
  both rows and bias.
- Candidates are ranked by use on the corpus, with ties broken by the piece. `new_pieces.txt`
  is authoritative, and `--pieces` rebuilds an extension byte for byte.
- N is chosen from KazNERD validation and corpus statistics (`docs/vocabulary.md`), never from
  test data. 10,000 is proposed for the preregistration of #37.

**Practical problem.** The size report took 5 h 12 min of wall-clock time instead of about 15
minutes. The Windows event log shows 55 minutes of Modern Standby, and the rest apparently ran
throttled while the display was off. Long CPU jobs get the same keep-awake wrapper as GPU
series.

## 2026-10-27 — Training two vocabularies identically

**Problem.** The causal test compares mBERT with and without the 10,000 added pieces after the
same continued pre-training. "The same" is not automatic. Standard masked-LM training packs
text into fixed-length token blocks and masks 15% of the *tokens*. With two vocabularies, the
same text then becomes different blocks, a different number of steps and different masked
content: the extended model would see more text per step and be asked to predict different
words.

**Decisions.**
- Chunks are defined in words, by one reference tokenizer (original mBERT) for both arms.
  Consecutive words are packed until 126 reference pieces. The extended model gets the same
  chunks in fewer tokens, which is exactly the difference under study.
- Masking is whole-word and decided per word from a generator seeded by (seed, chunk index).
  Both arms predict the same words with the same 80/10/10 decisions. A test checks this with
  two different tokenizers. Only the random replacement token differs, because the
  vocabularies differ.
- Masks are static, drawn once per chunk, which is enough for the planned single epoch. The
  same number of chunks and the same seed also give the same batch order.
- transformers 5 deprecates `warmup_ratio`; the warm-up fraction is passed as a float
  `warmup_steps`.
- Runs checkpoint every 2,000 steps and resume after an interruption, because the laptop's
  Modern Standby has already paused long runs. Pilots use `--max-chunks`.
- A model directory produced here is a local model for #31: its run ID uses the content
  fingerprint, and it can be passed directly to `kazner train`. A smoke test does exactly that.
