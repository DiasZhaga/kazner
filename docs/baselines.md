# Baselines: XLM-R base and KazRoBERTa (#32)

Two more encoders are fine-tuned on 10% KazNERD next to mBERT, with the protocol of
experiment B (`docs/reproduction.md` §4). They are **baselines, not causal evidence about
tokenization**: pre-training data, model size, depth and vocabulary all change together. The
causal test of fragmentation is the vocabulary-extension study (#37).

## Models

| | mBERT | XLM-R base | KazRoBERTa |
|---|---|---|---|
| Hub ID | `bert-base-multilingual-cased` | `FacebookAI/xlm-roberta-base` | `kz-transformers/kaz-roberta-conversational` |
| Pinned commit (`model.revision`) | `3f076fd` | `e73636d` | `43077c2` |
| Layers × hidden size | 12 × 768 | 12 × 768 | 6 × 768 |
| Parameters with the 51-label head | 177,302,067 | 277,492,275 | 82,899,507 |
| of which input embeddings | 91,812,096 | 192,001,536 | 39,936,000 |
| Tokenizer, vocabulary | WordPiece, 119,547 | SentencePiece, 250,002 | byte-level BPE, 52,000 |
| KazNERD fertility, in context | 2.325 | 1.773 | 1.211 |
| Mean training sequence length, 10% subsample, incl. special tokens | 28.6 | 22.4 | 15.9 |
| Pre-training text | Wikipedia, 104 languages | CC-100, 100 languages (Kazakh: 476 M tokens) | MDBKD (Kazakh CC-100, OSCAR, books, news, Leipzig) + customer-support dialogues |
| Licence | Apache-2.0 | MIT | Apache-2.0 |
| kazner setting | — | — | `add_prefix_space: true` (#30) |

Parameter counts and commits were read from the loaded models (`kazner.methods.build_model`).
The fertility values come from the tokenizer comparison of 2026-10-13. The weights were
downloaded as `model.safetensors` at the pinned commits, and their size and SHA-256 were
checked against the Hub's LFS metadata.

**Possible overlap with the test set.** KazNERD's sentences come from Khabar Agency TV-news
transcripts collected in 2020. MDBKD contains 1.04 B tokens of Kazakh news whose sources are
not stated, so KazRoBERTa may have seen unlabelled KazNERD text during pre-training. mBERT's
Wikipedia and XLM-R's CC-100 (CommonCrawl, 2018) make such overlap less likely, but it cannot
be ruled out for KazNERD's archive items. This has to be stated next to any KazRoBERTa result.

## Protocol

Everything as in experiment B's full fine-tuning except the learning rate: 10% of the KazNERD
training split (`subsample_seed: 42`, 9,023 sentences), 3 epochs, `max_len` 128, batch 4 ×
gradient accumulation 2, weight decay 0.01, FP16 on CUDA, no warm-up, best epoch by
validation F1. The test metric is `test_contract` (all words).

1. **Memory probe** for XLM-R. Weights, gradients and the two Adam moments of 277.5 M FP32
   parameters take ≈ 4.4 GB of the laptop's 6 GB:
   `kazner train --config configs/baselines/xlmr_lr2e-5.yaml --max-train-samples 200`
   (its own run ID).
2. **Learning-rate screen** at seed 42 (`configs/baselines/`, 12 runs):

   | Model | Rates |
   |---|---|
   | mBERT | 1e-5, 2e-5, 3e-5, 5e-5 |
   | XLM-R base | 1e-5, 2e-5, 3e-5, 5e-5 |
   | KazRoBERTa | 2e-5, 3e-5, 5e-5, 1e-4 |

   The rate is selected by **validation** F1 (`metrics.json → validation_best.eval_f1`),
   never by test. If the best rate is at the edge of its grid, the grid is extended by one
   step before choosing. mBERT gets the same screen, so that a tuned XLM-R is not compared
   with an untuned mBERT. Its 5e-5 run repeats the pilot setting with the current code and a
   pinned commit, so it doubles as a regression check: test F1 must again be 0.9020369285
   (run `53a89e2dbc4c`).
3. **Seeds 43 and 44** at each selected rate. Configs `configs/baselines/<model>_best.yaml`
   with `seeds: [42, 43, 44]`: seed 42 has the same run ID as its screen run and is skipped.
4. `kazner aggregate --runs runs --by name` → mean ± sd; results below.

If a run collapses (validation F1 far below the others, e.g. a model that predicts only `O`),
it is reported as such. Adding warm-up would change the protocol for all three models, so it
would be a separate issue.

**Time.** Measured for mBERT: 14–19 min per run, depending on the laptop's power state.
Estimated: XLM-R slightly slower (same encoder, shorter sequences, more embedding
parameters to update), KazRoBERTa roughly half (6 layers, about half the tokens). Screen ≈
2.5–3.5 h, seeds ≈ 1.2–1.7 h.

## Results

### Learning-rate screen (seed 42)

All 12 runs completed on code tree `44db383` (commit `a7ec27d`). Selection uses validation F1
only; the test column is shown for completeness.

| Model | lr | Validation F1 | Test F1 | Run |
|---|---|---|---|---|
| mBERT | 1e-5 | 0.8612 | 0.8612 | `f111c4c893de` |
| mBERT | 2e-5 | 0.8901 | 0.8902 | `3de8e7c6ab14` |
| mBERT | 3e-5 | 0.8942 | 0.8959 | `199ca7f67789` |
| mBERT | **5e-5** | **0.9012** | 0.9020 | `268e26c874ea` |
| XLM-R base | 1e-5 | 0.8444 | 0.8399 | `296c02e7fe65` |
| XLM-R base | 2e-5 | 0.8987 | 0.8955 | `7da0fdf64f2d` |
| XLM-R base | **3e-5** | **0.9096** | 0.9085 | `595d27dab710` |
| XLM-R base | 5e-5 | 0.9091 | 0.9090 | `2eddccf7d232` |
| KazRoBERTa | 2e-5 | 0.8844 | 0.8898 | `0a5cd4604099` |
| KazRoBERTa | 3e-5 | 0.8996 | 0.9018 | `f09fc95dcd86` |
| KazRoBERTa | 5e-5 | 0.9105 | 0.9082 | `61c660ea37a2` |
| KazRoBERTa | **1e-4** | **0.9115** | 0.9088 | `0ec53f106e14` |

- **Regression check passed.** mBERT at 5e-5, with the current code and a pinned commit,
  reproduces the pilot run `53a89e2dbc4c` bit for bit: test F1 0.9020369284876905, and the
  same precision, recall, accuracy and loss.
- **Selection rule.** XLM-R's best rate, 3e-5, lies inside its grid and is selected. For mBERT
  (5e-5) and KazRoBERTa (1e-4) the best rate is the top of the grid, so each grid is extended
  by one step (mBERT 1e-4, KazRoBERTa 2e-4) before choosing:
  `configs/baselines/mbert_lr1e-4.yaml`, `kazroberta_lr2e-4.yaml`.
- **Every run peaked at its last epoch.** The best validation epoch is 3 of 3 in all 12 runs, so
  the comparison is one of equal training budget: all models were still improving when
  training stopped.
- **Run times are not comparable.** The laptop entered Modern Standby during the series
  (`docs/challenges.md`), which paused two runs (XLM-R 2e-5, mBERT 3e-5) for hours. The
  results are unaffected. Peak GPU memory was 5,899 MiB of 6,141 MiB, including about 700 MiB
  used by the desktop, so XLM-R never spilled into shared memory.

### Grid extension (seed 42)

| Model | lr | Validation F1 | Test F1 | Run |
|---|---|---|---|---|
| mBERT | 1e-4 | 0.0130 | 0.0134 | `ae02c9729dd2` |
| KazRoBERTa | 2e-4 | 0.8947 | 0.8925 | `2fcd1244acfe` |

mBERT diverged at 1e-4: validation F1 was 0.0007, 0.013 and 0.004 after the three epochs, so
the model stopped predicting entities. Full fine-tuning of BERT without warm-up at batch 8
does not tolerate this rate. The selected rate for mBERT is therefore **5e-5**: the step above
it is clearly worse (`configs/baselines/mbert_best.yaml`). The collapsed run is kept and
reported, as the protocol requires.

KazRoBERTa at 2e-4 trained normally but reached a lower validation F1 (0.8947) than at 1e-4
(0.9115), so **1e-4** is selected and now lies inside the extended grid
(`configs/baselines/kazroberta_best.yaml`).

**Selected rates:** mBERT 5e-5, XLM-R base 3e-5, KazRoBERTa 1e-4.

### Three seeds at the selected rates

The extension runs and seeds 43–44 ran on the screen's code tree (a worktree at `a7ec27d`), so
all runs of #32 share one code version (`44db383`). Seed 42 of each `*_best.yaml` is the
screen run. Test metric: `test_contract`, mean ± sample sd over seeds 42, 43, 44.

| Model | lr | Test F1 | Precision | Recall | Runs (seeds 42, 43, 44) |
|---|---|---|---|---|---|
| mBERT | 5e-5 | 0.8994 ± 0.0027 | 0.8902 ± 0.0034 | 0.9089 ± 0.0022 | `268e26c874ea`, `f56bf59907b4`, `9eb5a5465396` |
| XLM-R base | 3e-5 | **0.9093 ± 0.0007** | 0.8992 ± 0.0009 | 0.9195 ± 0.0016 | `595d27dab710`, `cf7766617b2a`, `24e58bec57d8` |
| KazRoBERTa | 1e-4 | **0.9089 ± 0.0013** | 0.8973 ± 0.0023 | 0.9208 ± 0.0008 | `0ec53f106e14`, `a2eed7ada3ff`, `c7518a30999b` |

- mBERT's seeds 43 and 44 give exactly the test F1 of experiment B (0.8996 and 0.8967): the
  current code with a pinned commit computes the same thing as two weeks earlier.
- XLM-R base and KazRoBERTa are about 1 F1 point above mBERT, which is clearly more than the
  seed spread. They are tied with each other, although KazRoBERTa has 30% of XLM-R's
  parameters and half of its layers.
- All nine runs peaked at the last of their 3 epochs. The comparison is at an equal training
  budget.
- **Not causal evidence.** The models differ in pre-training data, size, depth and
  vocabulary. KazRoBERTa's pre-training news may overlap with KazNERD (see *Possible
  overlap* above).

### Fragmentation analysis per model

`kazner fragmentation` on the nine runs, grouped by model. The reference tokenizer is
original mBERT at `3f076fd` for every model, so "pieces per word" always means mBERT pieces.
The table gives exact-span recall, mean over 3 seeds.

| | Seen: 1.0 | (1,2] | (2,3] | >3 | Unseen: 1.0 | (1,2] | (2,3] | >3 |
|---|---|---|---|---|---|---|---|---|
| n | 1,913 | 1,848 | 1,767 | 1,678 | 360 | 2,365 | 2,079 | 1,498 |
| mBERT | 0.966 | 0.966 | 0.965 | 0.969 | 0.862 | 0.879 | 0.836 | 0.793 |
| XLM-R base | 0.964 | 0.971 | 0.969 | 0.968 | 0.862 | 0.890 | 0.861 | 0.828 |
| KazRoBERTa | 0.971 | 0.980 | 0.981 | 0.988 | 0.854 | 0.887 | 0.846 | 0.811 |

Odds ratio of an entity-word error per extra mBERT piece, with frequency, length, B/I and type
controlled: mBERT 1.006 [0.967, 1.043], XLM-R 0.969 [0.925, 1.011], KazRoBERTa 0.979
[0.938, 1.020]. No full-fine-tuning model shows an independent effect.

**Reading.**
- **Entities that mBERT fragments strongly are hard for every model**, including models whose
  own tokenizers split them far less (XLM-R 1.77, KazRoBERTa 1.21 pieces per word). Unseen
  recall falls from (1,2] to >3 by 6.2 points for XLM-R, 7.6 for KazRoBERTa and 8.6 for
  mBERT. Most of the drop therefore belongs to the entities themselves: they are long, rare
  and morphologically complex.
- **The part that could be mBERT-specific is small.** Relative to mBERT, XLM-R gains 3.5
  points on unseen >3 entities but only 1.1 on unseen (1,2], a difference of about 2.4
  points; KazRoBERTa's difference is about 1.0 point. Both are upper bounds for a pure
  tokenization effect, because the models differ in everything else too.
- **Consequence for the causal test (#37).** An effect of roughly 1–2 recall points on the
  primary subset is a realistic expectation, and the manipulation there is partial (4.27 →
  2.81 pieces per word). With three fine-tuning seeds per arm, the preregistration estimates
  the minimum detectable effect at about 2.3 points, so the test may be underpowered. More
  seeds per arm lower the model noise, which dominates the seed-to-seed differences.
- KazRoBERTa is the best model on **seen** entities, including fragmented ones (0.988 at >3).
  This is consistent with strong memorisation of frequent Kazakh word forms. Its pre-training
  overlap with KazNERD (see above) is an alternative explanation.
