# Preregistration: does less fragmentation improve Kazakh NER? (#37)

**Status: DRAFT, 2026-10-27; revised 2026-10-28** (pilot results, the loss fix of #49, the
RTX 4090, ten fine-tuning seeds). This document becomes binding when it is merged into `main` and
the merge commit is tagged `prereg-37`. That must happen before the first run of #37 (the
continued pre-training of either arm included). Later changes are deviations: they are listed
in section 7 with a reason and reported in the paper.

## 1. Question and hypotheses

Experiment B2 (`docs/fragmentation.md`) found, observationally, that mBERT recognises unseen
test entities worse when their words are split into many pieces. Recall is about 0.79 for more
than 3 pieces per word, against about 0.88 for at most 2. Rarity and length were not
controlled within unseen entities, so the cause is open. This experiment changes only the
fragmentation of the input and keeps everything else equal.

- **H1 (primary).** Adding 10,000 Kazakh WordPiece pieces to mBERT's vocabulary (#35) and
  continuing masked-LM pre-training raises exact-span recall on *unseen, strongly fragmented*
  test entities. The comparison is with mBERT continued on the same text, in the same way,
  with its original vocabulary. H1 is tested separately for full fine-tuning and for LoRA.
  B2 predicts a larger effect for LoRA, which showed an independent effect of fragmentation,
  than for full fine-tuning, which did not.
- **H2 (specificity).** The gain on the primary subset is larger than the gain on unseen,
  weakly fragmented entities.
- **Negative control.** No meaningful gain is expected on entities seen in training.

## 2. Design

| | Arm A: original vocabulary | Arm B: extended vocabulary |
|---|---|---|
| Start model | mBERT `bert-base-multilingual-cased` at `3f076fd` | the same, plus 10,000 pieces (`models/mbert-kk-10k`, `new_pieces.txt`) with FVT initialisation |
| Continued pre-training | `configs/pretrain/mbert_kk_original.yaml` | `configs/pretrain/mbert_kk_ext10k.yaml` |
| NER fine-tuning | as below | as below |

- **Vocabulary.** N = 10,000, chosen on KazNERD validation and corpus statistics
  (`docs/vocabulary.md`). Validation pieces per word fall from 2.32 to 1.35, the `[UNK]`
  share is unchanged (0.788%), and 9,998 of the new pieces occur at least 100 times in the
  corpus. The extension was built on 2026-10-27 (`extension.json`, section 6). Its learning
  run found 52,104 candidates, against 52,108 in the size report, because the WordPiece
  trainer breaks ties in hash order. The validation statistics of the top 10,000 were
  identical to the report's, and `new_pieces.txt` is the authoritative list.
- **Continued pre-training (#36).**
  - Corpus: FineWeb-2 `kaz_Cyrl`, 20,000,040 words, SHA-256 `9ab7b0c5…74b0a`; lines that
    contain a KazNERD sentence were removed (#34). Like much Kazakh web text, the corpus
    contains gambling spam: 1.0% of its lines mention "казино", and the piece `казино` is the
    4th most frequent of the 10,000 new pieces. Both arms see exactly the same text, so this
    does not bias the comparison. It is reported as a property of the corpus.
  - Both arms get identical word chunks (cut by original mBERT, 126 pieces), identical
    whole-word masks (15%, 80/10/10), the same steps and the same batch order.
  - Settings: lr 5e-5, 1 epoch, batch 8 × gradient accumulation 4, 5% linear warm-up,
    weight decay 0.01, FP16, seed 42. The loss is averaged over the accumulation steps (the
    fix of #49, in the code since `aeb5b47`). One pre-training run per arm: variance between
    pre-training runs is not estimated (section 4).
  - The run IDs, and so the output directories, follow from the config, the code tree, the
    corpus SHA-256 and, for arm B, the fingerprint of `models/mbert-kk-10k`. They were computed
    at the freeze (section 6): arm A writes `models/4ea205f2243c`, arm B
    `models/cd81fd453b74`. The NER configs of the arms name these directories. A pre-training
    run on other inputs or other code gets another ID, and the NER configs then stop with a
    missing-directory error instead of using a different model.
- **NER fine-tuning (same for both arms; no tuning on the arms).**
  - 10% of KazNERD train (subsample seed 42), 3 epochs, max_len 128, batch 4 × 2, weight
    decay 0.01, FP16, best epoch by validation F1. **Seeds 42–51**, ten per arm and method.
    The settings, including batch 4 × 2, are those of the B2 runs, so the NER code computes
    exactly what it computed there (its logged training loss is twice the mean; see
    `docs/challenges.md`, #49).
  - Full fine-tuning at the learning rate selected for mBERT in #32 by validation F1:
    **lr = 5e-5**. The screen's best rate, 5e-5, was at the top of its grid; the extension to
    1e-4 diverged (validation F1 0.013).
  - LoRA: lr 1e-3, r 16, α 32, dropout 0.1, target modules query and value (the best LoRA
    setting of experiment C).
- **Reference R (descriptive, not part of the tests).** mBERT at `3f076fd` without continued
  pre-training, fine-tuned with the same settings and seeds on the same machine
  (`configs/causal/reference_{full,lora}.yaml`). The B2 runs (laptop, seeds 42–44) are only
  quoted in section 3.
- **Hardware.** Every run of #37 runs on one machine: a desktop NVIDIA RTX 4090 (24 GB) at
  Astana IT University, with the dependency versions pinned in `pyproject.toml` and PyTorch
  2.5.1 + CUDA 12.1 as on the laptop. Each run records its GPU, driver and library versions in
  `environment.json`. Another GPU uses other kernels, so its numbers are not bit-identical to
  the laptop's: runs from the laptop (B2, #32) never enter a comparison. The setup is
  described in `docs/causal-test-runbook.md`.

Configs: `configs/pretrain/mbert_kk_{original,ext10k}.yaml` and
`configs/causal/{arm_a,arm_b,reference}_{full,lora}.yaml`. That makes 2 pre-training runs, 40
NER runs in the tests (2 arms × 2 methods × 10 seeds) and 20 reference runs.

### Pilot runs (2026-10-27, not part of the test)

Both pre-training configs were run with `--max-chunks 2000` on the laptop (RTX 4050, 6 GB):
the first 2,000 corpus chunks, 88,417 words, 63 optimizer steps, on the fixed code (runs
`6e9a8d51ad1c` for A and `e63d7dabaa8e` for B, code tree `a9fdfbf`).

- **The arms see the same words.** Arm B needs 38% fewer pieces for them (157,147 against
  253,314) and predicts 22,922 masked pieces against 37,343, because the same words are
  masked in both arms.
- **The loss is right after the fix.** The pilot found that the logged loss, and the gradient,
  were four times too large (#49, `docs/challenges.md`). After the fix the first logged value
  (steps 1–5) is 4.89 for A and 8.14 for B, against 4.90 and 8.05 computed directly from the
  initial models on the same chunks. At the end of the pilot the logged loss was 4.56 (A) and
  6.49 (B). Losses per piece are not comparable between vocabularies; arm B starts higher
  because its 10,000 new embeddings start from FVT.
- **FP16 loss scaling** skipped 4 of the first 10 optimizer steps in both arms (the scheduler
  does not advance on a skipped step), so both arms made the same number of updates.
- **Speed and memory.** Arm A took 0.71 s per step. With arm B, GPU memory reached 5.9 of
  the laptop's 6.1 GB (desktop applications included), and the run slowed to 2.1 s per step,
  apparently because memory spilled into shared system memory.
  The full corpus (about 452,000 chunks, 14,100 steps) would take about 3 h for A and up to
  8 h for B on the laptop. This is why the test runs on the RTX 4090, where batch 8 × 4 fits
  without spilling; the batch is unchanged.

## 3. Fixed entity subsets

The subsets are defined from the test gold labels, the 10% training subsample (seed 42) and
the original mBERT tokenizer (`3f076fd`) only, with `kazner fragmentation`. They do not depend
on any model's predictions. A KazNERD test entity is *seen* if the same surface string with the
same type is an entity in the training subsample. "Pieces per word" is the mean number of
original-mBERT pieces over the entity's words.

| Subset | Definition | n |
|---|---|---|
| **Primary** | unseen, more than 3 pieces per word, no `[UNK]` word | **1,490** |
| Specificity comparison | unseen, at most 2 pieces per word, no `[UNK]` word | 2,605 |
| Negative control | seen, no `[UNK]` word | 7,177 |
| Reported separately | unseen with an `[UNK]` word | 168 |

The `[UNK]` exclusion keeps the comparison on fragmentation alone. The extended tokenizer marks
exactly the same words as `[UNK]` as mBERT (#35), but entities with `[UNK]` are a different
kind of hard case.

Reference values from mBERT without continued pre-training (B2 runs on the laptop, mean ± sd
over 3 seeds): recall on the primary subset 0.792 ± 0.006 (full) and 0.759 ± 0.001 (LoRA); on
the specificity subset 0.877 and 0.849; on seen entities 0.967 and 0.945. Reference R repeats
these runs with ten seeds on the test machine.

## 4. Outcomes and analysis

- **Primary outcome.** Δ = recall(B) − recall(A) on the primary subset, where each arm's
  recall is the mean over its ten fine-tuning seeds. It is computed per method.
- **Inference.** A paired bootstrap over test sentences: 10,000 replicates, generator seed 0.
  Each replicate resamples sentences once and recomputes both arms' seed-mean recall on the
  entities of the resampled sentences. The two-sided 95% percentile CI of Δ is reported. Two
  primary tests are run (full fine-tuning, LoRA); their p-values (bootstrap, two-sided) are
  Holm-corrected at α = 0.05. **H1 is supported for a method** if Δ > 0 and its corrected
  test rejects Δ = 0.
- **H2.** DiD = Δ(primary) − Δ(specificity subset), with the same bootstrap. It is supported
  if the 95% CI excludes 0 and DiD > 0.
- **Negative control.** Δ on seen entities, with a 95% CI. A gain on seen entities as large as
  the primary gain would point to a general improvement rather than one specific to
  fragmentation; this is discussed, not tested.
- **Manipulation check (descriptive).** Mean pieces per word of the primary subset under each
  tokenizer, before any NER run: **4.27 → 2.81 (−34%)**. The drop is smaller than for all
  fragmented validation entity words (4.52 → 2.45, −46%) because unseen names are rarer than
  the corpus pieces cover. The manipulation is therefore real but partial, which limits the
  effect that can be expected.
- **Secondary (descriptive).** Test F1 on all words (`test_contract`, mean ± sd per arm and
  method); the B2 odds ratio per extra piece in each arm (`kazner fragmentation`, reference
  tokenizer original mBERT); recall on the 168 entities with `[UNK]`.
- **Effect of continued pre-training alone (descriptive).** Δ = recall(A) − recall(R) per
  subset and method, with the same bootstrap and no hypothesis test. It shows how much of any
  change comes from more Kazakh text rather than from the vocabulary.
- **Pre-training seed (limitation).** Each arm has one pre-training run. The bootstrap covers
  test sentences, and the seed mean covers fine-tuning seeds, but nothing covers the variation
  between pre-training runs: Δ compares these two pre-trained models. Adding pre-training
  seeds later would be a deviation (section 7). It can only be decided before any validation
  or test result of the arms has been seen.
- **Sensitivity and power.** Null comparisons of seeds of one model (#32 runs, 10,000
  replicates) show that the standard error of Δ on the primary subset is
  σ·√(1/k_A + 1/k_B), with σ ≈ 0.006 per run for mBERT (0.005 for XLM-R and KazRoBERTa):
  0.0082–0.0084 for one run against one, 0.0073 for one against two. A real difference adds a
  part that more seeds do not remove, because the effect differs between entities: for mBERT
  against XLM-R (3 runs each) the standard error is 0.0077, where seed noise alone predicts
  0.0048. With ten seeds per arm, seed noise contributes 0.0026. The minimum detectable effect
  (80% power, two-sided 5%) is therefore about 0.7 recall points if the effect is the same for
  all entities, and up to 1.8 points if it varies between entities as much as the difference
  between mBERT and XLM-R. With three seeds it would be 1.3–2.2 points. A null result is
  reported as such, with its interval.

**Analysis code.** `kazner fragmentation` (#33) produces `entities.csv` with the subsets and
one hit column per run, for all 60 runs, with original mBERT at `3f076fd` as the reference
tokenizer. `kazner compare` then runs the analysis above. The primary call contains exactly
the two pairs, so Holm's correction is over these two tests (run IDs comma-separated, ten per
arm):

```bash
kazner compare --entities results/37/entities.csv --out results/37/comparison.json --pair full <A full runs> <B full runs> --pair lora <A LoRA runs> <B LoRA runs>
```

The descriptive comparison with the reference is a separate call
(`--out results/37/continued_pretraining.json`, pairs `full` and `lora` with R as the first and
A as the second arm); its p-values are not used. `docs/causal-test-runbook.md` lists the
commands that collect the run IDs.

It was written and tested before any arm exists; its commit is recorded in section 6. A dry
run on the existing mBERT runs of B2, 10,000 replicates each, checked that it behaves as
intended:

- **A real difference is found.** Comparing LoRA (as A) with full fine-tuning (as B), the
  primary Δ is +0.033 [+0.020, +0.045] (p = 0.0002, Holm p = 0.0002). Full fine-tuning is
  better on every subset (specificity Δ +0.028, seen Δ +0.021), so the DiD is +0.005
  [−0.011, +0.020] and H2 is correctly *not* supported: the gain is general, not specific to
  fragmentation.
- **No false positives between seeds of one model.** In the six seed-vs-seed comparisons
  (3 per method, one run per arm), the primary Δ ranges from −0.011 to +0.001, every p is
  above 0.17, and none is significant. The CI half-width with one run per arm is about 0.016.
  The bootstrap resamples test sentences only, not model seeds. These null comparisons show
  that seed-to-seed differences lie within its intervals.
- Run time: about 4 s for 10,000 replicates.

## 5. Stopping rules and exclusions

- Both pre-training runs and all 40 NER runs of arms A and B are completed before any arm's
  test results are looked at. Validation metrics may be watched to catch failures. The 20
  reference runs may finish later; they are not part of a test.
- A run that crashes is restarted with the same config, so it gets the same run ID and resumes.
  A run that finishes but performs badly is kept and reported; no run is dropped for its
  result.
- No hyperparameter is changed after an arm's validation or test results have been seen.
- Fine-tuning collapse (validation F1 < 0.5) is reported. If it happens in one arm only, the
  primary analysis is still run on all seeds, and the collapse is discussed.

## 6. Records filled in at freeze time

| Item | Value |
|---|---|
| Full fine-tuning learning rate (from #32) | 5e-5 (runs `268e26c874ea`, `ae02c9729dd2`) |
| `models/mbert-kk-10k/new_pieces.txt` SHA-256 | `6940c9a9888dd353627c61acad6893e32252b0f2f86216c745ba66a87ff7f9ae` |
| Manipulation check: pieces per word, primary subset, A → B | 4.27 → 2.81 |
| Comparison code commit | `59abfe2` (`kazner compare`), tests `7a4a0ca` |
| Pre-training loss fix (#49) | `aeb5b47` |
| Code tree `src/kazner` at the freeze | `82c1f9dbdd253eb7e3c67ea7ae3fcd6790c3914b` |
| Corpus SHA-256 | `9ab7b0c56c74685e0e105b46e0021bc97d369f4d458622ac7d0f06bc01e74b0a` |
| `models/mbert-kk-10k` fingerprint (`kazner.config.model_fingerprint`) | `661abc386fa31cbc3d417e4e0631737743db252656a9351a2f0cd452c5c7e9af` |
| Pre-training run IDs: arm A, arm B | `4ea205f2243c`, `cd81fd453b74` |
| NER seeds per arm and method | 42–51 |
| Test machine | NVIDIA RTX 4090 (24 GB), Astana IT University |
| Freeze commit (`prereg-37`) | ___ |

## 7. Deviations

None so far.
