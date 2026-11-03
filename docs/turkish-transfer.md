# Turkish → Kazakh transfer with an aligned classification head (#59)

**Question.** Does Turkish NER training help Kazakh NER at 1% of KazNERD, if the Kazakh head
starts from the Turkish head instead of from random weights? And is the pilot's negative
result real?

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
  --types PERSON,GPE,LOCATION,ORGANISATION

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

Pending: the runs start only with the user's go-ahead.

## Reference

H. B. Şahin, Ç. Tırkaz, E. Yıldız, M. T. Eren and O. Sönmez, "Automatically Annotated Turkish
Corpus for Named Entity Recognition and Text Categorization using Large-Scale Gazetteers,"
arXiv:1702.02363, 2017.
