# Vocabulary extension (#35)

The causal test of fragmentation (#37) compares mBERT with and without Kazakh pieces added to
its vocabulary, after the same continued pre-training (#36). This document records how the
pieces are chosen and how many to add. The method is described in the README
("Vocabulary extension") and in `kazner.vocab`.

## Size report

```bash
kazner extend-vocab --corpus data/kk_corpus/fineweb2_kaz_Cyrl.txt --data data/kaznerd \
  --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --sizes 5000,10000,20000,30000 \
  --out models/report
```

- Corpus: FineWeb-2 `kaz_Cyrl` sample from #34 (20,000,040 words, SHA-256 `9ab7b0c5…74b0a`).
- Learner: WordPiece, vocabulary size 60,000, minimum pair frequency 5, mBERT's normalizer
  and pre-tokenizer. Of the 60,000 learned pieces, 6,726 are already in mBERT, 690 are single
  characters, 29 cannot be spelled without `[UNK]`, and 447 are never used by the learner on
  the corpus. That leaves **52,108 candidates**.
- Pieces per word are measured on KazNERD **validation**: 129,223 words, 26,277 entity words,
  and 5,172 entity words that mBERT splits into 4 or more pieces ("fragmented").
- "Use in corpus" counts how often each new piece occurs when the corpus is tokenized with the
  extended tokenizer.

| N added | Pieces/word, all | Entity words | Fragmented entity words | Words with `[UNK]` | New pieces used < 100 times | Median use |
|---|---|---|---|---|---|---|
| 0 (mBERT) | 2.322 | 2.377 | 4.523 | 0.788% | — | — |
| 5,000 | 1.496 | 1.607 | 2.858 | 0.788% | 0 | 1,604 |
| 10,000 | 1.346 | 1.465 | 2.447 | 0.788% | 2 | 870 |
| 20,000 | 1.232 | 1.352 | 2.124 | 0.788% | 28 | 419 |
| 30,000 | 1.185 | 1.300 | 1.964 | 0.788% | 219 | 253 |

For reference, the tokenizer comparison of 2026-10-13 measured XLM-R at 1.773 and KazRoBERTa
at 1.211 pieces per word on full KazNERD.

**Reading.**
- The share of words with `[UNK]` is identical for every N. The extension changes
  fragmentation only, by construction.
- The gain per added piece shrinks quickly. For fragmented entity words, going from 5k to 10k
  pieces saves 0.41 pieces per word, 10k to 20k saves 0.32, and 20k to 30k saves 0.16.
- Rare new pieces get few masked-LM updates in a 20 M-word corpus. At 10k, 9,998 of the new
  pieces occur at least 100 times (median 870); at 30k, 219 occur fewer than 100 times and one
  never occurs.

**Proposed N = 10,000**, to be fixed in the preregistration of #37. It cuts mBERT's
fragmentation from 2.32 to 1.35 pieces per word, below XLM-R and close to KazRoBERTa, and from
4.52 to 2.45 for the fragmented entity words that the primary outcome is about (−46%). Nearly
every new piece is frequent enough in the corpus to be learned. The extension adds 7.7 M
parameters (10,000 × 768, +4.3%).

## Reproducibility

The WordPiece trainer of `tokenizers` breaks ties between equally frequent pairs in hash order.
Two learning runs on the same synthetic corpus gave different piece orders, which a test
caught. Candidates are therefore ranked by use on the corpus, with ties broken by the piece
itself. The learned vocabulary can still differ slightly between runs, so `new_pieces.txt` is
the authoritative list, and `--pieces new_pieces.txt` rebuilds an extension exactly. How much
two learning runs on the real corpus differ has not been measured yet.
