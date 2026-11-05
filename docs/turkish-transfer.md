# Turkish → Kazakh transfer with an aligned classification head (#59)

**Question.** Does Turkish NER training help Kazakh NER at 1% of KazNERD, if the Kazakh head
starts from the Turkish head instead of from random weights? And is the pilot's negative
result real?

**Answer** (60 runs, 2026-11-05; details under [Results](#results)):
- **The aligned head helps.** At 1% of KazNERD it adds 1.55 F1 points over the pilot's
  transfer: 0.7484 against 0.7329, 95% CI +1.00 to +2.09, p = 0.0002. At 0.1% it adds 12.8.
- **The pilot's negative result was seed noise.** With a new head, transfer and direct training
  give the same F1 at 1% (difference −0.07, CI −0.68 to +0.62). By type the transfer still
  matters: +1.5 on the four types that have a Turkish counterpart, −0.8 on the other 21.

## Background: the pilot's transfer

The pilot (IEEE DG 2026 paper) trained mBERT in two stages:
1. on 10% of TWNERTC, Turkish with coarse labels: 52,669 sentences, 1 epoch;
2. on 1% of KazNERD: 903 sentences.

On the KazNERD test set the transfer gave F1 0.724, and training on KazNERD directly gave
0.740. Two things limit this result:
- **Only the encoder was transferred.** The Kazakh stage loaded the Turkish checkpoint with
  `ignore_mismatched_sizes=True`. The classification head was therefore re-initialised for
  KazNERD's 51 labels, and the Turkish head was discarded, although it already separates
  persons, places and organisations. Item 4 of the paper's Future Work proposes explicit
  label-space alignment.
- **One seed.** At 903 training sentences a gap of 1.6 points can be seed noise.

## Data

**Turkish checkpoint.** Stage 1 is not repeated. The pilot's checkpoint is copied to
`models/twnertc-coarse10-pilot/` (git-ignored); runs identify it by content, like every local
model directory.

| Item | Value |
|---|---|
| Source | `ner-project/results/cross_lingual_twnertc10_kaznerd1_ep3_len64_bs2/twnertc_pretrain` |
| Code | `src/run_cross_lingual_twnertc_kaznerd.py` at DiasZhaga/kaznerd-ner@d992fd9 |
| Settings | mBERT, 10% of TWNERTC coarse (seed 42), 1 epoch, lr 2e-5, batch 2, `max_len` 64; 26,335 steps, 1.4 h on the RTX 4050 |
| Labels | B-/I- PERSON, LOCATION, ORGANIZATION, MISC; O |
| `model.safetensors` SHA-256 | `6fd0da9e72248c986b4c50420d3c1b7e234521af4ab32bee1a05541d10b1a412` |
| Directory fingerprint (`kazner.config.model_fingerprint`) | `a6cbf8468bdcc0aac735eefe611ef5b677f1d20e6a475bfc093bd61e2affebe8` |

The copy holds `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and
the pilot's `run_config.json`. Its `tokenizer.json` is serialised differently from mBERT's at
commit `3f076fd`, but gives the same token ids on the whole KazNERD test set.

**TWNERTC.** TWNERTC [Şahin et al., 2017] annotates Turkish Wikipedia automatically from a
knowledge base, so its labels are silver: in the first held-out sentence, the name "Madero'nun"
is `O` and the title "başkanı" is MISC. The pilot converted three coarse-grained files with
`src/prepare_twnertc_coarse.py` and named them train, valid and test. **They are not splits.**
They are three noise-reduction variants of one corpus:

| Pilot file | TWNERTC file | Sentences | Also in "train" |
|---|---|---|---|
| `coarse_train.txt` | DomainDependent_NoiseReduction | 526,690 | — |
| `coarse_valid.txt` | DomainIndependent_NoiseReduction | 520,376 | 98.9% |
| `coarse_test.txt` | No_NoiseReduction | 534,768 | 98.3% |

The pilot skipped the Turkish evaluation, so none of its numbers is affected
(`docs/challenges.md`, 2026-11-03). Here the Turkish sets come from `coarse_train.txt` alone:

```bash
python scripts/make_twnertc_sets.py --source ../ner-project/data/coarse_train.txt
```

- `data/twnertc/train_subset.txt`: the pilot's training subsample, 52,669 sentences
  (`kazner.data.sample_indices`, fraction 0.1, seed 42; the pilot's own sampling rule).
- `data/twnertc/heldout.txt`: 5,000 other sentences (seed 42). None equals a training sentence,
  and each distinct sentence occurs once (`kazner.data.heldout_indices`).
- `data/twnertc/manifest.json`: settings and SHA-256 checksums. `coarse_train.txt` has SHA-256
  `d633f1586b46f391e94f04e069ef88609b5fd7b5139032183d41f8eb3d25ff00`.

## Design

**Arms.** Each arm is trained with seeds 42–51. Configs: `configs/turkish/`.

| Arm | Config | Start | Classification head |
|---|---|---|---|
| D, direct | `direct_1pct.yaml` | mBERT (Hub commit `3f076fd`) | new |
| T, transfer | `transfer_1pct.yaml` | Turkish checkpoint | new, as in the pilot |
| A, aligned transfer | `aligned_1pct.yaml` | Turkish checkpoint | 9 of 51 rows copied from the Turkish head, the rest new |

**Fixed settings.** These are the pilot's settings, identical in every arm:
- full fine-tuning;
- `max_len` 64, batch 2, grad accumulation 1, lr 2e-5, 3 epochs;
- the best epoch is chosen by validation F1;
- the pilot's 1% subsample of KazNERD train: `subsample_seed` 42, 903 sentences.

The fine-tuning seed is set before the model is built, so with one seed arms T and A get the
same random rows and the same training order. They differ only in the copied rows.

**Head mapping** (`configs/mappings/twnertc_to_kaznerd_head.yaml`, `kazner.head`). For each
pair, the weight row and the bias of the B- label and of the I- label are copied; the O row is
copied too.

| TWNERTC | KazNERD | Rows |
|---|---|---|
| PERSON | PERSON | B, I |
| LOCATION | LOCATION and GPE | B, I each |
| ORGANIZATION | ORGANISATION | B, I |
| MISC | none: its KazNERD counterpart is unclear | — |
| O | O | 1 |

The mapping must list every TWNERTC type, and every label it names must exist on its side.
The config key is `model.head_mapping`; it needs a local checkpoint, whose own head is read.
The file's content enters the run ID.

**Primary outcome.** Test entity-level micro F1 over all 25 types, with every word scored:
words cut off at `max_len` 64 count as `O` (the complete contract). Reported as mean ± sd over
the 10 seeds.

**Comparisons**, each arm B minus arm A, paired by seed (`kazner paired`):
1. **A − T: does the aligned head help?** The primary comparison.
2. T − D: a replication of the pilot.
3. A − D: is aligned transfer better than no transfer?

**Inference, fixed before the runs.** Δ is the difference of the seed means. Each of 10,000
bootstrap replicates resamples the seeds (the same draws in both arms) and the test sentences
(the same draws for every run), and recomputes Δ. This covers both seed variation and test
sampling. The report gives the percentile 95% CI and a two-sided bootstrap p-value. The
aligned head is said to help if Δ(A − T) > 0 and p < 0.05; this is the only test with a
decision. Comparisons 2 and 3 are reported with their CIs. An exact sign-flip test over the
10 per-seed differences, which treats the test set as fixed, is reported as a check.

The issue named `kazner compare` for this. That command bootstraps recall on the fixed entity
subsets of #37, so a new command compares F1 instead.

**Regression check.** Arms D and T at seed 42 must reproduce the pilot's 0.740 and 0.724 within
±0.01. An exact match is not expected: kazner scores all words, while the pilot dropped 274
truncated test words (`docs/reproduction.md` §1).

**Secondary analyses** (descriptive):
- **Per type group.** F1 on the four types with a Turkish counterpart (PERSON, GPE, LOCATION,
  ORGANISATION), and on the other 21 types: `kazner paired --types ...` and
  `--exclude-types ...`.
- **0.1% of KazNERD.** The same three arms on 91 sentences (`*_0p1pct.yaml`), where a head start
  should matter most.
- **Zero-shot transfer.** The Turkish checkpoint on the KazNERD test with no Kazakh training.
  Both sides are mapped to PER, LOC and ORG: KazNERD's with the #51 mapping, TWNERTC's with
  `twnertc_to_wikiann.yaml` (MISC → O).
- **Turkish F1.** The checkpoint on `heldout.txt`, against silver labels.
- **Subword overlap.** How many of KazNERD's mBERT pieces occur in the Turkish training text
  (`kazner overlap`).

**Expectation, stated before the runs.** The head is one linear layer, and 903 sentences train
it quickly, so the effect at 1% may be small. It should be larger at 0.1% and on the four
mapped types.

**Limitations.**
- **One Turkish checkpoint.** Variation from the Turkish stage is not covered; more stage-1
  seeds are possible on the RTX 4090.
- **Silver Turkish labels.** TWNERTC is annotated automatically.
- **Pilot settings.** The pilot's `max_len` of 64 truncates 274 KazNERD test words. It is kept so
  that the pilot's runs remain the reference.

## Commands

```bash
# Training series: 30 runs at 1%, then 30 at 0.1% (only with the user's go-ahead)
kazner train --config configs/turkish/direct_1pct.yaml
kazner train --config configs/turkish/transfer_1pct.yaml
kazner train --config configs/turkish/aligned_1pct.yaml

# Primary and secondary comparisons
kazner paired --runs runs --a turkish_transfer_1pct --b turkish_aligned_1pct \
  --out results/59/aligned_vs_transfer_1pct.json
kazner paired --runs runs --a turkish_direct_1pct --b turkish_transfer_1pct \
  --out results/59/transfer_vs_direct_1pct.json
kazner paired --runs runs --a turkish_direct_1pct --b turkish_aligned_1pct \
  --out results/59/aligned_vs_direct_1pct.json
kazner paired --runs runs --a turkish_transfer_1pct --b turkish_aligned_1pct \
  --types PERSON,GPE,LOCATION,ORGANISATION --out results/59/aligned_vs_transfer_1pct_mapped.json
kazner paired --runs runs --a turkish_transfer_1pct --b turkish_aligned_1pct \
  --exclude-types PERSON,GPE,LOCATION,ORGANISATION \
  --out results/59/aligned_vs_transfer_1pct_other.json
# ... the same for T - D and A - D, and for the *_0p1pct experiments

# Zero-shot transfer and Turkish F1
kazner predict --model models/twnertc-coarse10-pilot --data data/kaznerd/IOB2_test.txt \
  --max-len 64 --out results/59/zero_shot_kaznerd.csv
kazner evaluate --predictions results/59/zero_shot_kaznerd.csv \
  --map-gold configs/mappings/kaznerd_to_wikiann.yaml \
  --map-pred configs/mappings/twnertc_to_wikiann.yaml --out results/59/zero_shot_kaznerd.json
kazner predict --model models/twnertc-coarse10-pilot --data data/twnertc/heldout.txt \
  --max-len 64 --out results/59/turkish_heldout.csv
kazner evaluate --predictions results/59/turkish_heldout.csv

# Subword overlap
kazner overlap --source data/twnertc/train_subset.txt --target data/kaznerd/IOB2_test.txt \
  --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --out results/59/overlap_kaznerd_test.json
```

`kazner paired` selects an arm's runs by the `name` in their `config.json`, one per seed, and
stops if a seed has two runs.

## Results

### Subword overlap

Source: the Turkish training subset (52,669 sentences, 882,437 words, 23,423 distinct mBERT
pieces). Target: the KazNERD test set. The Turkish held-out set is shown as a control.

| Target | Words | Piece types also in Turkish training | Piece tokens covered | Words with every piece covered | Word types seen in Turkish |
|---|---|---|---|---|---|
| KazNERD test, all words | 129,824 | 15.9% | **7.7%** | 16.5% | 1.5% |
| KazNERD test, entity words | 26,555 | 22.4% | **7.4%** | 10.6% | 6.7% |
| Turkish held-out, all words (control) | 84,154 | 95.8% | 99.7% | 99.4% | 71.5% |

Only 7.7% of the pieces that mBERT produces for the KazNERD test text occur anywhere in the
Turkish training text, and only 16.5% of Kazakh words consist of such pieces alone. Turkish is
written in Latin script and Kazakh in Cyrillic, so almost every Kazakh piece embedding is
untouched by the Turkish stage, and the transfer has to go through the shared upper layers.

### Training series

**Runs.** 60 runs: three arms, two data sizes, seeds 42–51.
- **Machine and code.** The laptop's RTX 4050 (6 GB), code at `33574ad` (`src/kazner` tree
  `c1dcd8f`), 2026-11-04 and 05.
- **Time.** About 6.5 min per run at 1% and 2.9 min at 0.1%, 4.7 GPU hours in all.
- **Interruption.** The series was stopped once and resumed (`docs/challenges.md`, 2026-11-05).
  Complete runs were skipped, and the interrupted run started again from scratch.

**Regression check: passed.** Both arms reproduce the pilot within ±0.01. "All words" re-scores
the pilot's predictions with the complete contract; "as reported" is the pilot's own scoring,
which dropped 274 truncated words.

| Arm, seed 42 | kazner | Pilot, all words | Pilot, as reported |
|---|---|---|---|
| D, direct | 0.7333 | 0.7393 | 0.740 |
| T, transfer | 0.7232 | 0.7236 | 0.724 |

**Test F1**, mean ± sd over 10 seeds. "Mapped" means the four types with a Turkish counterpart
(PERSON, GPE, LOCATION, ORGANISATION); "other" means the remaining 21 types.

| KazNERD train | Types | D, direct | T, transfer | A, aligned transfer |
|---|---|---|---|---|
| 1% (903 sentences) | all 25 | 0.7336 ± 0.0078 | 0.7329 ± 0.0067 | **0.7484 ± 0.0055** |
| 1% | mapped 4 | 0.7382 ± 0.0101 | 0.7535 ± 0.0109 | **0.7794 ± 0.0052** |
| 1% | other 21 | 0.7314 ± 0.0094 | 0.7234 ± 0.0097 | 0.7345 ± 0.0079 |
| 0.1% (91 sentences) | all 25 | 0.0811 ± 0.0645 | 0.1796 ± 0.0713 | **0.3075 ± 0.0108** |
| 0.1% | mapped 4 | 0.0203 ± 0.0429 | 0.3842 ± 0.1585 | **0.6633 ± 0.0068** |
| 0.1% | other 21 | 0.0992 ± 0.0720 | 0.0649 ± 0.0369 | 0.0870 ± 0.0310 |

**Paired comparisons** (`kazner paired`, 10,000 replicates; Δ in F1).
- p = 0.0002 is the smallest bootstrap p that 10,000 replicates can give: no replicate crossed
  zero.
- p = 0.0020 is the smallest exact sign-flip p for 10 seeds.
- Only the first row carries a decision. The other rows are descriptive and not corrected for
  multiple comparisons.

| Train | Types | Comparison | Δ | 95% CI | p, bootstrap | p, sign-flip | Seeds with Δ > 0 |
|---|---|---|---|---|---|---|---|
| 1% | all | **A − T (primary)** | **+0.0155** | **[+0.0100, +0.0209]** | **0.0002** | 0.0039 | 9 of 10 |
| 1% | all | T − D | −0.0007 | [−0.0068, +0.0062] | 0.82 | 0.87 | 4 of 10 |
| 1% | all | A − D | +0.0148 | [+0.0093, +0.0206] | 0.0002 | 0.0020 | 10 of 10 |
| 1% | mapped | A − T | +0.0259 | [+0.0190, +0.0333] | 0.0002 | 0.0020 | 10 of 10 |
| 1% | mapped | T − D | +0.0152 | [+0.0065, +0.0241] | 0.0018 | 0.0059 | 9 of 10 |
| 1% | mapped | A − D | +0.0412 | [+0.0318, +0.0502] | 0.0002 | 0.0020 | 10 of 10 |
| 1% | other | A − T | +0.0111 | [+0.0030, +0.0197] | 0.0058 | 0.025 | 9 of 10 |
| 1% | other | T − D | −0.0080 | [−0.0158, +0.0005] | 0.065 | 0.076 | 2 of 10 |
| 1% | other | A − D | +0.0031 | [−0.0051, +0.0110] | 0.45 | 0.44 | 6 of 10 |
| 0.1% | all | A − T | +0.1279 | [+0.0873, +0.1759] | 0.0002 | 0.0020 | 10 of 10 |
| 0.1% | all | T − D | +0.0986 | [+0.0237, +0.1631] | 0.011 | 0.031 | 8 of 10 |
| 0.1% | all | A − D | +0.2265 | [+0.1843, +0.2638] | 0.0002 | 0.0020 | 10 of 10 |
| 0.1% | mapped | A − T | +0.2791 | [+0.1936, +0.3797] | 0.0002 | 0.0020 | 10 of 10 |
| 0.1% | mapped | T − D | +0.3639 | [+0.2428, +0.4650] | 0.0002 | 0.0039 | 9 of 10 |
| 0.1% | mapped | A − D | +0.6430 | [+0.6125, +0.6683] | 0.0002 | 0.0020 | 10 of 10 |
| 0.1% | other | A − T | +0.0222 | [−0.0001, +0.0443] | 0.051 | 0.10 | 7 of 10 |
| 0.1% | other | T − D | −0.0343 | [−0.0867, +0.0168] | 0.20 | 0.24 | 5 of 10 |
| 0.1% | other | A − D | −0.0122 | [−0.0646, +0.0347] | 0.67 | 0.66 | 7 of 10 |

**Findings:**
1. **Decision: the aligned head helps.** Δ(A − T) = +0.0155 > 0 with p = 0.0002, so the rule
   fixed before the runs is met. A beats T in 9 of 10 seeds and D in all 10.
2. **The pilot's gap was seed noise.** With a new head, T and D give the same F1 at 1%, and the
   CI of the difference is ±0.65 points. The pilot's 1.6-point gap came from one seed per arm;
   over 10 seeds, arm D alone ranges from 0.722 to 0.744.
3. **The overall null hides a split by type.** Without the aligned head, the Turkish stage adds
   1.5 points on the four mapped types and loses 0.8 on the other 21; the two cancel.
   - The aligned head keeps the gain on the mapped types and makes it larger: +4.1 over D.
   - On the other types, A is level with D (+0.3, CI includes zero), so the aligned head also
     removes T's small loss there. The copied `O` row is one possible reason; it was not tested
     separately.
4. **The effect grows as data shrinks, as expected before the runs.** At 0.1% of KazNERD (91
   sentences, 138 optimisation steps):
   - A reaches 0.663 on the mapped types, against 0.384 for T and 0.020 for D;
   - the 21 other types stay below 0.10 in every arm.
   With the pilot's 3 epochs, D and T are far from converged at this size, so part of A's gain
   at 0.1% is a faster start rather than a better end point.
5. **The aligned head makes runs more stable.** The sd over seeds is 0.0055 for A against 0.0067
   for T and 0.0078 for D at 1%. At 0.1% it is 0.011 for A against 0.071 for T and 0.065 for D.
   With a random head, some seeds at 0.1% barely learn the mapped types at all.

### Zero-shot transfer

The Turkish checkpoint on the KazNERD test set, with no Kazakh training. Both sides are mapped to
PER, LOC and ORG, so KazNERD's GPE and LOCATION both count as LOC. 274 words beyond `max_len` 64
count as `O`.

| Type | Precision | Recall | F1 | Gold entities |
|---|---|---|---|---|
| LOC | 0.605 | 0.678 | 0.640 | 1,954 |
| PER | 0.642 | 0.348 | 0.451 | 1,332 |
| ORG | 0.387 | 0.079 | 0.131 | 735 |
| **Micro** | 0.603 | 0.459 | **0.521** | 4,021 |

A model that saw no Kazakh text finds about half of KazNERD's persons, places and
organisations, although only 7.7% of the Kazakh piece tokens occur in its Turkish training
text. This number uses a different label set from the table above, so it is not directly
comparable with the mapped-type F1 of the trained arms.

### Turkish F1

The Turkish checkpoint on the 5,000 held-out TWNERTC sentences (84,154 words, 695 beyond
`max_len` 64), scored against TWNERTC's silver labels.

| Type | Precision | Recall | F1 | Gold entities |
|---|---|---|---|---|
| LOCATION | 0.631 | 0.738 | 0.680 | 2,656 |
| PERSON | 0.578 | 0.634 | 0.605 | 2,323 |
| MISC | 0.666 | 0.527 | 0.589 | 6,588 |
| ORGANIZATION | 0.433 | 0.441 | 0.437 | 499 |
| **Micro** | 0.626 | 0.591 | **0.608** | 12,066 |

The silver labels have their own errors (see Data), so this F1 measures agreement with the
automatic annotation rather than the checkpoint's true quality.

## Reference

H. B. Şahin, Ç. Tırkaz, E. Yıldız, M. T. Eren and O. Sönmez, "Automatically Annotated Turkish
Corpus for Named Entity Recognition and Text Categorization using Large-Scale Gazetteers,"
arXiv:1702.02363, 2017.
