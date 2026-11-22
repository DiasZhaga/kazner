# Results of the causal vocabulary test (#37)

**Question** (preregistration, `docs/preregistration-vocab.md`, frozen at the tag `prereg-37`).
Does adding 10,000 Kazakh WordPiece pieces to mBERT and continuing masked-LM pre-training raise
exact-span recall on unseen, strongly fragmented test entities, compared with mBERT continued
on the same text in the same way with its original vocabulary?

**Answer.** No.
- **H1 (primary):** not supported for either method.
  - **Full fine-tuning:** no effect. Δ = −0.15 recall points, 95% CI [−0.99, +0.67], Holm
    p = 0.70. The interval excludes gains larger than about 0.7 points, which is the
    preregistered minimum detectable effect.
  - **LoRA:** the extended vocabulary *lowers* recall. Δ = −1.42 points [−2.36, −0.49], Holm
    p = 0.006.
- **H2 (specificity):** not supported. For LoRA the loss is concentrated on the strongly
  fragmented entities: DiD = −1.32 points [−2.45, −0.23].
- **Continued pre-training on Kazakh text itself helps a lot** (descriptive, arm A against the
  reference R): +3.1 points with full fine-tuning and +4.4 with LoRA on the primary subset. The
  gain is largest on exactly the entities that fragmentation was suspected to hurt.

The analysis follows section 4 of the preregistration, with the code of the tag `prereg-37`.

## Runs

### Machine and environment

Deviation D1 of the preregistration: a rented GPU instead of the university's desktop. Its
details are recorded there.

| Item | Value |
|---|---|
| Provider | RunPod, on-demand pod (not spot), one instance for all 62 runs |
| GPU | NVIDIA GeForce RTX 4090, 24 GB; driver 595.112.07 |
| Image | RunPod PyTorch template, Ubuntu 24.04.5; 8 vCPU, 41 GB RAM, 50 GB container disk |
| Python and libraries | Python 3.11.18 (venv); torch 2.5.1+cu121, transformers 5.0.0, peft 0.19.1, datasets 4.5.0, accelerate 1.12.0, seqeval 1.2.2 |
| Code | tag `prereg-37`, commit `05786b0`, `src/kazner` tree `82c1f9d`, clean checkout |
| Price | USD 0.89 per hour (GPU) + 0.007 (disk); about 8.5 h in all, about USD 7.7 |

### Checks before the first run

All checks of the runbook, section 3, passed on the instance:
- the 293 offline tests at the tag pass;
- the data archive SHA-256 is `5c437045…d2c5`;
- the corpus SHA-256 is `9ab7b0c5…74b0a`;
- the `models/mbert-kk-10k` fingerprint is `661abc38…c7e9af`;
- `torch` reports `2.5.1+cu121 NVIDIA GeForce RTX 4090`.

### Timeline (2026-11-20, UTC)

| Time | Event |
|---|---|
| 07:15–07:23 | setup and checks |
| 07:23 / 07:28 | pre-training A / B start. Each wrote to the directory named in section 6: `models/4ea205f2243c` and `models/cd81fd453b74` |
| ~10:31 / ~11:04 | the GPU phase of A / B starts after about three hours of CPU preprocessing (see *Notes*) |
| 11:19 / 11:49 | pre-training A / B finish (exit 0) |
| 11:49–14:12 | the 40 NER runs of arms A and B (all exit 0) |
| 14:12–15:22 | the 20 reference runs (all exit 0) |

No run crashed, none was repeated, and none was dropped. The lowest best-epoch validation F1 of
any run is 0.864 (a reference LoRA run), so no fine-tuning collapse occurred (section 5).

### Notes on how the runs were executed

These do not change what was run (section 2), only how.
- **Concurrency.** The two pre-training runs ran at the same time on the one GPU. The NER
  configs of arms A and B then ran side by side, as the runbook allows; the reference ran
  after them. Both arms still got the same 462,598 chunks, the same masks and the same 14,457
  steps (`pretrain_metrics.json`).
- **Slow preprocessing.** Before the GPU phase, each pre-training run spent about three hours
  building its examples on one CPU core. The cause is a call of `len(tokenizer)` per chunk, which
  costs about 25 ms in transformers 5.0.0. The code is frozen by the tag, and any change would
  alter the run IDs, so it was left as it is; see `docs/challenges.md`.
- **Blinding.** Console output went to log files only. Progress was followed through a status
  file and validation F1, and the test results were opened after all 60 runs had finished.

### Run IDs

| Config | Seeds 42–51, in order |
|---|---|
| causal_arm_a_full | `12fcf0d309b0`, `206cdb2a9f4d`, `d44953a3b4f8`, `c9d07344f7e1`, `8ed77ad1dcd0`, `6467970e749b`, `4785fd5a6bdf`, `463d25897b95`, `cf74285c7575`, `9ba73286a971` |
| causal_arm_b_full | `a1af3750ed76`, `37869feb805e`, `a421d9d4e819`, `826c568b75b7`, `b0ff8a211add`, `dde31c0da661`, `068f37a4a7f5`, `1bb642fb946a`, `7a5037dfa925`, `db7a99155603` |
| causal_arm_a_lora | `d4dbfbf9d62e`, `943db70190b0`, `c7cb33090ea4`, `6e4a124b5270`, `54ec974efb54`, `3a3b484b5be4`, `be73d05852c6`, `4b3281dfba23`, `02e38e5fcc60`, `33a65dc5044b` |
| causal_arm_b_lora | `e6e74fddc71d`, `a174f55d9610`, `8865ae2f90b9`, `606cc29548fc`, `c7e6f6f4cd9d`, `9bf3bf6636d9`, `11b8e3274d85`, `299dfa1d3919`, `8577e3483c64`, `b269ae66a2f7` |
| causal_reference_full | `d9ac6f13a7c5`, `f4de41a587b9`, `be948f42dd84`, `64caff8060f3`, `4cbb7dadd7f0`, `2a2a655ef5c5`, `bf7dc10f29ef`, `29e10a9f1093`, `42ccf9ee5fdf`, `9982c98b492d` |
| causal_reference_lora | `e01a4e15946b`, `afbeab9b35c2`, `07ead8dc5f9f`, `a3e8ac98cef7`, `aabe911e9b8b`, `bd4b834ce896`, `32811375670f`, `da17c0113cbe`, `7d58bc6ba699`, `174d52178a8e` |

## Pre-training

| | Arm A: original vocabulary | Arm B: extended vocabulary |
|---|---|---|
| Output | `models/4ea205f2243c` | `models/cd81fd453b74` |
| Chunks, words | 462,598 chunks, 20,000,040 words | the same |
| Steps (batch 8 × 4) | 14,457 | 14,457 |
| Pieces in the corpus | 58,569,420 | 35,589,998 (−39%) |
| Masked pieces predicted | 8,647,526 | 5,201,912 |
| Loss per piece, first → last logging window | 4.56 → 3.48 | 7.29 → 4.09 |
| GPU time | 48 min | 45 min |

The losses are per piece, so they are not comparable across vocabularies.

**Manipulation check (section 4, recorded at the freeze).** Pieces per word on the primary
subset fall from 4.27 to 2.81 (−34%).

## Primary analysis (preregistered)

Exact-span recall, mean over ten fine-tuning seeds per arm; Δ = B − A. Each 95% CI is a paired
bootstrap over test sentences: 10,000 replicates, seed 0, 7,540 sentences with an entity of the
subsets. The two primary tests are Holm-corrected.

| Method | Subset | n | A | B | Δ (points) | 95% CI | p | Holm p |
|---|---|---|---|---|---|---|---|---|
| full | **primary** | 1,490 | 0.8207 | 0.8191 | **−0.15** | [−0.99, +0.67] | 0.70 | **0.70** |
| full | specificity | 2,605 | 0.8911 | 0.8926 | +0.15 | [−0.40, +0.70] | 0.57 | |
| full | negative control | 7,177 | 0.9715 | 0.9732 | +0.17 | [−0.02, +0.35] | 0.07 | |
| full | unseen with `[UNK]` | 168 | 0.8940 | 0.8946 | +0.06 | [−1.24, +1.35] | 0.95 | |
| full | DiD primary − specificity | | | | −0.31 | [−1.28, +0.66] | 0.51 | |
| LoRA | **primary** | 1,490 | 0.8034 | 0.7893 | **−1.42** | [−2.36, −0.49] | 0.003 | **0.006** |
| LoRA | specificity | 2,605 | 0.8709 | 0.8700 | −0.10 | [−0.68, +0.50] | 0.75 | |
| LoRA | negative control | 7,177 | 0.9592 | 0.9576 | −0.16 | [−0.39, +0.07] | 0.17 | |
| LoRA | unseen with `[UNK]` | 168 | 0.8690 | 0.8506 | −1.85 | [−4.04, +0.30] | 0.10 | |
| LoRA | DiD primary − specificity | | | | −1.32 | [−2.45, −0.23] | 0.02 | |

**Decisions, by the rules of section 4:**
- **H1, full fine-tuning: not supported.** Δ ≤ 0, and the test does not reject Δ = 0. The
  interval is narrower than the preregistered minimum detectable effect of 0.7–1.8 points, so
  this is an informative null result, not a lack of power.
- **H1, LoRA: not supported.** The test rejects Δ = 0 (Holm p = 0.006), but in the opposite
  direction: the extended vocabulary hurts.
- **H2: not supported for either method.** Both DiDs are below zero. For LoRA, the loss on the
  primary subset is larger than on weakly fragmented unseen entities, and the interval excludes 0.
- **Negative control.** No gain on seen entities: +0.17 (full) and −0.16 (LoRA) points,
  intervals including 0.

## Effect of continued pre-training alone (descriptive)

Arm A against the reference R: mBERT fine-tuned directly, on the same machine, with ten seeds.
Δ = A − R, same bootstrap. As preregistered, these p-values are not used for any decision.

| Method | Subset | R | A | Δ (points) | 95% CI |
|---|---|---|---|---|---|
| full | primary | 0.7899 | 0.8207 | **+3.08** | [+2.24, +3.96] |
| full | specificity | 0.8820 | 0.8911 | +0.91 | [+0.46, +1.38] |
| full | negative control | 0.9656 | 0.9715 | +0.59 | [+0.42, +0.77] |
| full | DiD primary − specificity | | | +2.17 | [+1.19, +3.15] |
| LoRA | primary | 0.7594 | 0.8034 | **+4.40** | [+3.35, +5.53] |
| LoRA | specificity | 0.8507 | 0.8709 | +2.02 | [+1.42, +2.62] |
| LoRA | negative control | 0.9449 | 0.9592 | +1.43 | [+1.14, +1.71] |
| LoRA | DiD primary − specificity | | | +2.38 | [+1.15, +3.65] |

The reference reproduces the laptop values of B2 on this machine. Primary recall is 0.790
(full) and 0.759 (LoRA), against 0.792 and 0.759 in the preregistration, section 3.

## Secondary outcomes (descriptive)

**Test F1, all words** (`test_contract`, mean ± sd over ten seeds):

| Method | R: mBERT | A: continued, original vocabulary | B: continued, extended vocabulary |
|---|---|---|---|
| full | 0.9007 ± 0.0027 | 0.9123 ± 0.0024 | 0.9140 ± 0.0019 |
| LoRA | 0.8674 ± 0.0021 | 0.8895 ± 0.0019 | 0.8886 ± 0.0032 |

**Odds ratio of an entity-word error per extra mBERT piece** (`kazner fragmentation`, as in B2):
frequency, length, B/I and type are controlled; geometric mean over seeds, 95% CI from a
sentence bootstrap.

| Method | R | A | B |
|---|---|---|---|
| full | 1.028 [0.987, 1.066] | 1.015 [0.977, 1.051] | 1.013 [0.974, 1.051] |
| LoRA | **1.060 [1.024, 1.095]** | 1.009 [0.975, 1.044] | 1.025 [0.993, 1.061] |

With ten seeds, the reference reproduces the LoRA effect found in B2 (1.053 with three seeds).
After continued pre-training it disappears, with or without the extended vocabulary.

## Interpretation

1. **Fewer pieces do not help.** The extension cut the primary subset's fragmentation by a third
   and gave no gain with full fine-tuning, and a loss with LoRA. Within this design, the
   fragmentation of mBERT's tokenizer is not the cause of its weaker recall on rare Kazakh
   names.
2. **More Kazakh text does help.** Continued pre-training with the original vocabulary raised
   recall most on unseen, strongly fragmented entities (DiD about +2.2 to +2.4 points) and
   removed LoRA's fragmentation effect. The observational link of B2 is therefore better
   explained by too little exposure to Kazakh than by the number of pieces.
3. **Why LoRA loses with the extended vocabulary** is not tested here. A plausible reading: the
   10,000 new embeddings saw only one epoch of 20 M words, and fine-tuning with LoRA cannot
   adapt the embedding matrix, while full fine-tuning can repair it.

## Limitations

- **One pre-training run per arm** (section 4). Variation between pre-training runs is not
  covered; Δ compares these two pre-trained models.
- **One extension.** One size (10,000 pieces), one initialisation (FVT) and one pre-training
  budget (one epoch, 20 M words). A larger corpus or longer pre-training could change the
  picture for arm B.
- **One task and one test set** (KazNERD NER).

## Reproducing the analysis

The run directories are git-ignored and come from the results archive.
- **Code:** analyse with the code of the tag, e.g. a worktree at `prereg-37` and
  `PYTHONPATH=<worktree>/src`.
- **Run IDs:** collect them per config with the snippet of the runbook, section 6.

```bash
kazner fragmentation --runs <the 60 run directories> --data data/kaznerd \
  --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --out results/37
kazner compare --entities results/37/entities.csv --out results/37/comparison.json \
  --pair full <A full> <B full> --pair lora <A LoRA> <B LoRA>
kazner compare --entities results/37/entities.csv --out results/37/continued_pretraining.json \
  --pair full <R full> <A full> --pair lora <R LoRA> <A LoRA>
```

On the laptop, `kazner fragmentation` over 60 runs takes about 15 minutes; each `kazner
compare` call takes a few seconds.
