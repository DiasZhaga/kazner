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

### Still to run

The remaining runs use the screen's code tree (a worktree at `a7ec27d`), so the whole series
shares one code version, and seed 42 of `xlmr_best.yaml` reuses the screen run:

1. Grid extension at seed 42: mBERT 1e-4, KazRoBERTa 2e-4.
2. Seeds 43 and 44 at each selected rate (`<model>_best.yaml`; XLM-R: 3e-5).
