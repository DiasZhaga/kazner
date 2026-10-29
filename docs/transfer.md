# Transfer to WikiANN with an explicit label mapping (#51)

**Question.** How well do the models fine-tuned on KazNERD (#32) recognise entities in other
Kazakh text, annotated by other guidelines? This addresses two Future Works of the IEEE DG 2026
paper: explicit label-space alignment, and Kazakh data beyond KazNERD. No model is trained on
WikiANN.

## Data

**WikiANN** [Pan et al., 2017; Rahimi et al., 2019] provides NER data for 176 languages, built
automatically from Wikipedia links. Its Kazakh part (`kk`) has 1,000 sentences in each of
train, validation and test, annotated with PER, LOC and ORG:

```bash
python scripts/download_wikiann_kk.py        # -> data/wikiann_kk/
```

The script downloads the three parquet files of `unimelb-nlp/wikiann` at commit `f0a3be6`
(217,890 bytes) and verifies them against the Hub's SHA-256. It checks the `ner_tags` class
names stored in each file and writes `IOB2_train.txt`, `IOB2_valid.txt` and `IOB2_test.txt`
plus a `manifest.json` with every checksum. Only the test split is used here.

| Split | Sentences | Tokens | PER | LOC | ORG |
|---|---|---|---|---|---|
| test | 1,000 | 7,936 | 377 | 383 | 355 |

- **Silver annotation.** The labels come from Wikipedia links, not from annotators. Many
  sentences are fragments: titles, list items, or a name with a short gloss. Missing and
  inconsistent labels count as model errors.
- **Licence.** The dataset card gives none; the text is from Wikipedia (CC BY-SA). The data is
  downloaded by the script and never committed.

## Label mapping

`configs/mappings/kaznerd_to_wikiann.yaml` assigns every KazNERD type to a WikiANN type or to
`O`. `kazner transfer` stops if a predicted or gold type is not listed, or if a target is not
PER, LOC, ORG or O.

| KazNERD | WikiANN | Reason |
|---|---|---|
| PERSON | PER | |
| GPE, LOCATION | LOC | WikiANN has no GPE: countries, regions and cities are LOC |
| ORGANISATION | ORG | |
| FACILITY | O | Buildings, airports and roads have no clear WikiANN type. The variant `kaznerd_to_wikiann_facility_loc.yaml` maps them to LOC, as a sensitivity check |
| the other 20 types | O | NORP, EVENT, PROJECT, PRODUCT, ART, LAW, LANGUAGE, DISEASE, ADAGE, NON_HUMAN, MISCELLANEOUS, POSITION, CONTACT and the numeric and temporal types lie outside WikiANN's label space |

The B-/I- prefix is kept, so `B-GPE I-GPE` becomes `B-LOC I-LOC`, and two adjacent entities
that map to the same type, such as `B-GPE B-LOCATION`, stay two entities. The mapping files
live in `configs/mappings/` rather than next to the run configs (as the issue proposed),
because every YAML file elsewhere under `configs/` is tested as an experiment config.

## Design: label space and domain separately

For each model, `kazner transfer` reports three entity-level scores (seqeval, default mode;
mean ± sd over seeds 42–44):

1. **KazNERD, 25 types:** the #32 result.
2. **KazNERD, mapped:** gold and predictions both mapped to PER/LOC/ORG. The data are still
   KazNERD; only the label space is WikiANN's.
3. **WikiANN:** the predictions mapped, the gold labels as published.

The change from step 1 to step 2 is the effect of the label space alone. The change from step
2 to step 3 is the effect of other text and another annotation scheme. Step 3 also gets a 95%
percentile CI for the seed-mean F1, from a bootstrap over the 1,000 test sentences. Per-type
F1 is reported for steps 2 and 3. A secondary table gives the exact-span recall of the WikiANN
entities by mean mBERT pieces per word, in the bins of `kazner fragmentation`, to see whether
the fragmentation pattern of B2 also holds on other data.

## Runs

The #32 runs kept only their predictions, not their weights, so the three selected settings
are retrained with the WikiANN test as an extra test set (`extra_tests` in
`configs/transfer/{mbert,xlmr,kazroberta}_wikiann.yaml`). Everything else equals
`configs/baselines/*_best.yaml`. After training on KazNERD, each run predicts the KazNERD test
as before and then the WikiANN test, written as `predictions_wikiann_kk.csv`. In that file,
`gold` holds WikiANN's labels and `pred` the model's KazNERD labels. The extra test set does
not change the KazNERD results: it is predicted afterwards and consumes no random numbers.
Regression check: mBERT seed 42 must again reach test F1 0.9020369284876905.

```bash
kazner train --config configs/transfer/mbert_wikiann.yaml          # and xlmr_, kazroberta_
kazner transfer --mapping configs/mappings/kaznerd_to_wikiann.yaml --extra wikiann_kk \
  --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --out results/51 --runs runs/<id> ...
```

## Results

Pending: the nine runs started on 2026-10-29 (code tree `5378898`).

## Caveats

- WikiANN is silver data with its own guidelines. Some disagreements with it are annotation
  noise, not model errors, and they lower every model's score.
- mBERT was pre-trained on Wikipedia, Kazakh Wikipedia included, so it may have seen WikiANN's
  sentences, without labels. XLM-R (CommonCrawl) and KazRoBERTa (web, news, books) may have
  seen some of them too.
- 1,000 test sentences, with about 1,100 entities, give intervals of a few F1 points.
