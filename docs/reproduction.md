# Reproduction against the pilot

kazner was refactored from the pilot repository (`ner-project`, commit `b6ee209`). This page
records the checks showing that the refactoring changed the structure but not the behaviour.
None of them retrains a model: they reuse artefacts saved by the pilot. Section 3 adds the
training-based reproduction (#11), and section 4 the first multi-seed results built on it.

Environment of the checks below: Windows 11, Python 3.11.9, kazner `0.1.0.dev0`,
seqeval 1.2.2, transformers 5.0.0. The pilot's files are read in place (`--pilot ../ner-project`);
nothing derived from KazNERD is committed.

## 1. Evaluator vs the pilot's stored metrics (#5)

### How the pilot computed F1

Every test F1 the pilot stored comes from `trainer.evaluate(test_tok)`. Its `compute_metrics`
called `evaluate.load("seqeval")` in the default mode over the first-subword positions: label
`-100` was skipped, so words cut off at `max_len` were not scored. That metric takes
`classification_report(...)["micro avg"]` for P/R/F1 and `accuracy_score` for accuracy.
The prediction files were written separately, from `trainer.predict` on the same model.
`kazner.evaluate` calls seqeval directly with the same functions and mode.

| Pilot file | Columns | Stored metrics |
|---|---|---|
| `results/cross_lingual_*/kaznerd_1pct_*/test_predictions.csv` (`run_cross_lingual_twnertc_kaznerd.py`, `max_len = 64`) | `sentence_id, token, true_label, pred_label` | `test_metrics.json` in the same directory |
| `results/test_predictions_mbert.csv` (`train_ner_mbert.py`, full train split, 1 epoch, batch 8, `max_len = 128`) | `sent_id, token_id, token, gold, pred` | none (the script only printed them) |

The PEFT run behind 0.9020 (`run_peft_comparison.py`) wrote no prediction file, so it cannot
be checked this way.

### Command

```bash
python scripts/reproduce_pilot_metrics.py --pilot ../ner-project
```

Each file is converted to the contract (`kazner.contract.convert_pilot_predictions`) in two
variants and scored with `kazner.evaluate`:

- *as-is*: only the words the pilot model saw, i.e. exactly what the pilot scored;
- *complete*: words lost to truncation appended from `IOB2_test.txt` with `pred = O`, i.e.
  the contract's rule, under which every gold word is scored.

### Result

| Run | Variant | Precision | Recall | F1 | Accuracy | Truncated words |
|---|---|---|---|---|---|---|
| matched baseline, direct full FT | stored | 0.7244637022 | 0.7563779294 | 0.7400769175 | 0.9452798147 | |
| | as-is | 0.7244637022 | 0.7563779294 | 0.7400769175 | 0.9452798147 | 0 |
| | complete | 0.7243216366 | 0.7548859935 | 0.7392880447 | 0.9448869238 | 274 |
| TWNERTC → KazNERD, full FT | stored | 0.7108287529 | 0.7385048947 | 0.7244025752 | 0.9429641065 | |
| | as-is | 0.7108287529 | 0.7385048947 | 0.7244025752 | 0.9429641065 | 0 |
| | complete | 0.7106146049 | 0.7369706840 | 0.7235527129 | 0.9425761030 | 274 |
| TWNERTC → KazNERD, LoRA | stored | 0.5868506494 | 0.1072382082 | 0.1813393529 | 0.8111771517 | |
| | as-is | 0.5868506494 | 0.1072382082 | 0.1813393529 | 0.8111771517 | 0 |
| | complete | 0.5868506494 | 0.1070476755 | 0.1810668670 | 0.8110672911 | 274 |
| TWNERTC → KazNERD, frozen encoder | stored | 0.5306334372 | 0.0378967665 | 0.0707413304 | 0.8009648784 | |
| | as-is | 0.5306334372 | 0.0378967665 | 0.0707413304 | 0.8009648784 | 0 |
| | complete | 0.5306334372 | 0.0378294344 | 0.0706240066 | 0.8008765714 | 274 |
| `train_ner_mbert.py`, full data | as-is | 0.9394515539 | 0.9510660349 | 0.9452231174 | 0.9880448933 | 0 |
| | complete | 0.9394515539 | 0.9510660349 | 0.9452231174 | 0.9880376510 | 5 |

**As-is: identical to the pilot.** All four stored runs are reproduced to every printed
digit (difference 0 in P, R, F1 and accuracy). The new evaluator computes exactly what the
pilot's Trainer-based metric computed.

**Complete: explained differences.** At `max_len = 64`, 274 test words are cut off and were
never scored by the pilot. Under the contract they count, with `pred = O`. Recall therefore
drops by up to 0.0015 and F1 by 0.00012–0.00085. Precision falls slightly for the two
full fine-tuning runs, because an entity cut at the boundary can no longer match. At
`max_len = 128` (mBERT run), one test sentence (index 3130) loses 5 words, `I-LAW O O O O`:
the gold `LAW` entity at words 59–65 loses its last word. The model predicted
`ORGANISATION` (59–61) there, so the entity is missed in both variants and F1 is unchanged.
Only accuracy moves (−7·10⁻⁶).

**Provenance of the paper's Table I.** The pilot never stored the test metrics of
`train_ner_mbert.py`. Re-scoring its prediction file gives P/R/F1/accuracy =
0.939 / 0.951 / 0.945 / 0.988, which is exactly Table I ("Fine-tuned mBERT") of the pilot paper.
The 100% data-efficiency run (`results/data_efficiency/mbert_100pct`) has recall 0.9515, which
rounds to 0.952. So Table I comes from the `train_ner_mbert.py` run (4 Feb 2026), not from
the data-efficiency series. Both runs give F1 = 0.945.

## 2. Tokenizer fertility (#6)

### Command

```bash
kazner fertility --data ../ner-project/data --tokenizer bert-base-multilingual-cased --out fertility.json
```

The full corpus (1.3 M words) runs in about 20 s on a laptop CPU, because the
tokenization of every distinct word is cached.

### Result

| Split | Words | Subword tokens | Fertility | Split words | UNK rate | Max |
|---|---|---|---|---|---|---|
| train | 1,043,305 | 2,426,503 | 2.325785 | 60.6421% | 0.8288% | 24 |
| validation | 129,223 | 300,050 | 2.321955 | 60.4730% | 0.7870% | 11 |
| test | 129,824 | 301,359 | 2.321289 | 60.5527% | 0.8365% | 11 |
| **overall** | **1,302,352** | **3,027,912** | **2.324957** | **60.6164%** | **0.8254%** | **24** |

The pilot's `results/tokenizer_fertility/mbert_fertility_results.csv` was compared for
11 metrics in each of the 4 rows: word and subword counts, fertility, single-token and
split-word rates, mean subwords of split words, median, standard deviation, maximum, unknown
words and UNK rate. All values are **identical** (absolute difference < 10⁻¹²).
The expected overall fertility of 2.325 is reproduced, as is Table III of the pilot paper.
The unit tests also compare `kazner.fertility` with a verbatim copy of the pilot's
`RunningStats` / `TokenizerAnalyzer` on the fixtures.

## 3. Training-based reproduction of F1 = 0.9020 (#11)

```bash
python scripts/download_kaznerd.py
kazner train --config configs/pilot_full_10pct.yaml
```

Environment:

- Windows 11, RTX 4050 Laptop GPU (6 GB);
- torch `2.5.1+cu121`, the same build as the pilot; transformers 5.0.0, peft 0.19.1, datasets 4.5.0;
- `.venv-gpu`, an environment separate from the CPU test environment;
- code: commit `94b6868`, `src/kazner` tree `be80673`, clean working tree;
- run ID `53a89e2dbc4c`, training time 772 s.

Compared metric (as agreed in #11): `metrics.json → test_trainer`, computed as the pilot
computed its numbers. Tolerance: 0.9020 ± 0.01.

| Test metric | kazner | Pilot (`full_10pct_metrics.json`) | Difference |
|---|---|---|---|
| Precision | 0.8928778648 | 0.8928778648 | 0 |
| Recall | 0.9113858454 | 0.9113858454 | 0 |
| **F1** | **0.9020369285** | **0.9020369285** | **0** |
| Accuracy | 0.9790169390 | 0.9790169390 | 0 |
| Loss | 0.1097337380 | 0.1097337380 | 0 |

**Result: reproduced bit for bit, far inside the ±0.01 tolerance.** The refactored pipeline
reproduces the pilot's computation exactly: same subset, seeding before the model is built,
same head initialisation, same Trainer arguments and same evaluation. The contract-based F1
(all words) is also 0.9020369285. The 5 truncated test words change only accuracy
(0.9790100444).

## 4. Multi-seed PEFT comparison (experiment B) and LoRA screen (experiment C)

All runs: 10% KazNERD (9,023 sentences, `subsample_seed` 42), mBERT, 3 epochs, batch 4 × gradient
accumulation 2, max_len 128, FP16, the same code tree `be80673`. Metrics: `test_contract`.
Raw rows: `kazner aggregate --runs runs --by method,fraction,lr,r`.

### B — the pilot's PEFT comparison with three seeds (42, 43, 44)

| Method (pilot settings) | Test F1, mean ± sd (n = 3) | Pilot paper (1 seed) |
|---|---|---|
| Full fine-tuning (lr 5e-5) | **0.8994 ± 0.0027** | 0.902 |
| LoRA r 8, α 16 (lr 1e-4) | **0.6861 ± 0.0035** | 0.686 |
| Frozen encoder (lr 5e-5) | **0.0819 ± 0.0039** | 0.092 |

- Full FT and LoRA with seed 42 reproduce the pilot exactly: 0.9020 and 0.6858.
- The published full-FT value is the best of the three seeds. The multi-seed estimate is
  0.899 ± 0.003.
- Frozen with seed 42 gives 0.0781, not the pilot's 0.0919. The pilot trained frozen in the
  same process right after full fine-tuning and seeded the process only once, so the frozen
  head was initialised from a random-number state consumed by the preceding run (see
  `docs/challenges.md`). The published frozen value therefore depended on run order. It lies
  about 2.6 sd above the multi-seed mean.

### C — LoRA learning rate × rank (seed 42, α = 2r, selection by validation F1)

| lr | r | Trainable params | Validation F1 (best epoch) | Test F1 |
|---|---|---|---|---|
| 1e-4 | 8 | 334,131 (0.188%) | 0.6903 (3) | 0.6858 |
| 1e-4 | 16 | 629,043 (0.354%) | 0.7227 (3) | 0.7173 |
| 3e-4 | 8 | 334,131 | 0.8048 (3) | 0.8023 |
| 3e-4 | 16 | 629,043 | 0.8322 (3) | 0.8304 |
| 1e-3 | 8 | 334,131 | 0.8578 (3) | 0.8617 |
| **1e-3** | **16** | **629,043** | **0.8686 (3)** | **0.8657** |

**Reading.** The pilot's LoRA was **under-trained**: its learning rate was too small. At the
pilot's rank 8, raising lr from 1e-4 to 1e-3 lifts test F1 from 0.686 to 0.862. The best
setting by validation (lr 1e-3, r 16) reaches 0.866, so the gap to full fine-tuning shrinks
from 0.213 to about 0.034. LoRA then trains 0.35% of the parameters, and training is 28%
faster (559 s vs 772 s).

**Caveats.**

- The C results come from one seed. The best setting still has to be re-run with seeds 43 and 44.
- The best setting lies at the edge of the grid, and validation F1 is still rising at
  epoch 3 (0.792 → 0.848 → 0.869). A larger lr or more epochs may close the gap further.

