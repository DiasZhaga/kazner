# Reproduction against the pilot

kazner was refactored from the pilot repository (`ner-project`, commit `b6ee209`). This page
records the checks showing that the refactoring changed the structure but not the behaviour.
None of them retrains a model: they reuse artefacts saved by the pilot. Training-based
reproduction (test F1 0.9020 for full fine-tuning on 10% KazNERD) is #11, milestone v0.2.0.

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

Pending — added with #6.
