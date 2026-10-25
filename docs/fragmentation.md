# Fragmentation and errors (experiment B2, #33)

**Question.** mBERT splits 60.6% of Kazakh words into several pieces. Are fragmented words
recognised worse, once their rarity in the training data is taken into account? The analysis
uses saved word-level test predictions (contract files); it trains nothing.

```bash
kazner fragmentation --data data/kaznerd --by method --out results/b2 \
  --runs runs/53a89e2dbc4c runs/6c8e0702563e runs/e8a361f0874d \
         runs/d7731ab072a6 runs/f3c937dd8c0a runs/fe15213d0f49
```

## Method

- **Reference tokenizer** (`--tokenizer`, default `bert-base-multilingual-cased`; `--revision`,
  `--add-prefix-space`). It defines the pieces per word: each word is tokenized on its own, as
  in `kazner fertility`; for mBERT this equals the in-context count. The tokenizer is fixed, so
  runs of different models (e.g. the baselines of #32 or the vocabulary study of #37) are
  scored on the same words and entities.
- **Training subsample.** It is read from the runs' `config.json` (`train_fraction`,
  `subsample_seed`, `max_train_samples`). All runs must agree on it, their
  `data_fingerprint` must match `--data`, and their test words and gold labels must be
  identical; otherwise the command stops.
- **Word table.** Each test word gets its pieces, whether a piece is `[UNK]`, its characters,
  and the frequency of the word form in the training subsample.
- **Entity table** (`entities.csv`). Each gold entity is found as in the evaluator (seqeval
  default mode). It gets its mean and maximum pieces per word and an `[UNK]` flag. It is
  *seen* if the same surface string with the same type is an entity in the training
  subsample. One `hit_<run_id>` column per run marks an exact predicted span with the gold
  type.
- **Tables per group** of runs (`--by`, config fields as in `kazner aggregate`). The error
  rate of entity words by pieces, raw and within training-frequency strata (unseen, 1–4, 5+),
  and exact-span recall by pieces per word for seen and unseen entities. Each is a mean ±
  sd over the group's runs; cells with fewer than 30 words or entities are left out.
- **Logistic regression** per run. The outcome is an error on an entity word. The predictors
  are pieces (clipped at 8), log(1 + training frequency), characters, B vs I, and the entity
  type. The model is fitted by Newton's method in NumPy; a ridge of 10⁻⁶ only keeps
  degenerate cases finite. The reported odds ratio per extra piece is the geometric mean
  over the group's runs. Its 95% CI comes from a bootstrap over test sentences (200
  replicates, seed 0), in which every run is refitted on the same resample. A run without
  errors has no defined odds ratio (`null`).

## Results: mBERT, 10% KazNERD, 3 epochs, seeds 42–44

`full` = full fine-tuning (test F1 0.899 ± 0.003); `lora` = the best LoRA setting, lr 1e-3,
r 16 (0.865 ± 0.003). There are 129,824 test words, 26,555 entity words and 13,508 gold
entities; 197 of these entities contain an `[UNK]` word.

Entity words, error rate by pieces:

| Pieces | n | Unseen in training | Error, full | Error, LoRA |
|---|---|---|---|---|
| 1 | 8,608 | 3.7% | 6.7% | 7.8% |
| 2 | 6,796 | 6.8% | 6.3% | 7.5% |
| 3 | 5,840 | 15.3% | 8.0% | 9.2% |
| 4 | 3,434 | 27.4% | 9.8% | 11.8% |
| 5+ | 1,877 | 36.8% | 10.8% | 13.4% |

Within frequency strata, full fine-tuning:

| Training frequency | 1 piece | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|
| unseen | 15.3% | 17.4% | 19.4% | 18.3% | 19.5% |
| 1–4 | 8.6% | 12.3% | 10.4% | 11.2% | 9.6% |
| 5+ | 6.2% | 4.4% | 4.2% | 4.1% | 2.4% |

Odds ratio of an error per extra piece, controlled for frequency, length, B/I and type:

| | Per seed (42, 43, 44) | Geometric mean | 95% CI (sentence bootstrap) |
|---|---|---|---|
| full | 1.011, 1.002, 1.005 | 1.006 | [0.967, 1.043] |
| LoRA | 1.059, 1.047, 1.052 | 1.053 | [1.016, 1.089] |

Exact-span recall by mean pieces per word:

| | Seen: 1.0 | (1,2] | (2,3] | >3 | Unseen: 1.0 | (1,2] | (2,3] | >3 |
|---|---|---|---|---|---|---|---|---|
| n | 1,913 | 1,848 | 1,767 | 1,678 | 360 | 2,365 | 2,079 | 1,498 |
| full | 0.966 | 0.966 | 0.965 | 0.969 | 0.862 | 0.879 | 0.836 | 0.793 |
| LoRA | 0.951 | 0.950 | 0.945 | 0.933 | 0.835 | 0.851 | 0.808 | 0.758 |

**Reading.**
1. Most of the raw link between fragmentation and errors is rarity: fragmented words are much
   more often unseen in training (3.7% → 36.8%). With frequency controlled, full fine-tuning
   shows no independent effect of the number of pieces.
2. LoRA does show one: about +5% error odds per extra piece, with a CI that excludes 1. A
   plausible but untested mechanism is that LoRA adapts only the attention query and value
   matrices, which gives it less capacity than full fine-tuning to compose fragmented words.
3. Fragmentation matters for unseen entities. Exact-span recall falls from about 0.88 to 0.79
   (full) and from 0.85 to 0.76 (LoRA) for entities with more than 3 pieces per word, while
   seen entities are unaffected. Within unseen entities, rarity and length are not controlled.

**Limits.** The study is observational: pieces, length and rarity are correlated, and the
regression controls only what was measured. It uses 10% of the data, one model and one
tokenizer. The causal test is #37.

## Agreement with the first analysis

The first B2 analysis was a script in the git-ignored `results/` folder. On the same six runs
the package reproduces every table cell (maximum absolute difference 6 × 10⁻¹⁷) and every
count. The odds ratios differ by up to 0.002 because the script used scikit-learn's default
convergence tolerance (10⁻⁴). On the same design matrix, scikit-learn with a tolerance of
10⁻¹⁰ gives 1.010645 for seed 42 of full fine-tuning; the package gives 1.010646, while the
script reported 1.0101. The script reported a CI for seed 42 only. The package's CI is for the
group's mean over seeds, so it is narrower (full: [0.967, 1.043] vs [0.972, 1.053]; LoRA:
[1.016, 1.089] vs [1.009, 1.111]). The conclusions do not change.
