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

Step 3 is also broken down by kind of sentence: redirect lines, article first lines and the
rest, the last also without repeats. Each kind gets its own scores, CI and recall table. This
breakdown was added after the first results; see [What the WikiANN test
contains](#what-the-wikiann-test-contains).

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

Nine runs on 2026-10-29 (code tree `5378898`, RTX 4050 laptop GPU, 3 h 12 min in total). Every
run reproduced its #32 counterpart on KazNERD bit for bit: the same test F1 and an identical
`predictions.csv`. The outputs are in `results/51/main` and `results/51/facility_loc`
(git-ignored). All numbers below use the main mapping and give the mean ± sd over seeds 42–44.

### Label space and domain

| Model | KazNERD, 25 types | KazNERD, mapped | WikiANN | WikiANN 95% CI | WikiANN P / R |
|---|---|---|---|---|---|
| mBERT | 0.899 ± 0.003 | 0.910 ± 0.003 | 0.490 ± 0.035 | [0.464, 0.516] | 0.468 / 0.515 |
| XLM-R | 0.909 ± 0.001 | 0.922 ± 0.003 | 0.562 ± 0.016 | [0.536, 0.589] | 0.538 / 0.590 |
| KazRoBERTa | 0.909 ± 0.001 | 0.916 ± 0.002 | 0.532 ± 0.025 | [0.504, 0.558] | 0.514 / 0.551 |

F1 per type, PER / LOC / ORG:

| Model | KazNERD, mapped | WikiANN |
|---|---|---|
| mBERT | 0.945 / 0.939 / 0.779 | 0.641 / 0.473 / 0.272 |
| XLM-R | 0.962 / 0.935 / 0.819 | 0.739 / 0.561 / 0.287 |
| KazRoBERTa | 0.955 / 0.950 / 0.765 | 0.704 / 0.567 / 0.238 |

1. **The label space costs nothing.** Mapping KazNERD to PER/LOC/ORG raises F1 by 0.7–1.3
   points: confusions between GPE and LOCATION disappear, and so do the hardest types. The
   entire loss, 36–42 points, comes from the other text and its annotation.
2. **ORG transfers worst** (0.24–0.29). WikiANN takes the type from the linked article, and many
   of its ORG spans are not organisations in KazNERD's sense. Examples from the test include a
   province (Куэнка), a prayer (Намаз), a theorem (Эйлер теоремасы) and a desert (Бетпақдала).

### What the WikiANN test contains

This breakdown is exploratory. It was added after the first results, when the errors turned out
to cluster in a few kinds of line. `kazner transfer` sorts the sentences by their first word
(`kazner.transfer.sentence_kind`):

| Kind | Sentences | Gold entities | What it is |
|---|---|---|---|
| `redirect` | 176 | 176 | `АЙДАУ X`, Kazakh Wikipedia's redirect line. X is the single entity |
| `bold_title` | 174 | 177 | An article's first line, which starts with the remains of the bold-title markup, as in `' '' Ивака '' ' — Ресейдегі өзен .` ("Ivaka is a river in Russia"). The title is `O`; the linked country word is LOC. There are 83 such river stubs |
| `other` | 650 | 762 | The rest. 86 of these sentences repeat an earlier one, and one of them occurs 33 times |
| `other_unique` | 564 | 633 | `other`, with each distinct sentence counted once |

F1 by kind, with a 95% CI for `other_unique`:

| Model | `redirect` | `bold_title` | `other` | `other_unique` | `other_unique` P / R |
|---|---|---|---|---|---|
| mBERT | 0.288 ± 0.146 | 0.368 ± 0.026 | 0.580 ± 0.011 | 0.635 ± 0.013 [0.600, 0.669] | 0.648 / 0.622 |
| XLM-R | 0.414 ± 0.040 | 0.720 ± 0.058 | 0.555 ± 0.009 | 0.624 ± 0.007 [0.590, 0.659] | 0.646 / 0.603 |
| KazRoBERTa | 0.484 ± 0.049 | 0.704 ± 0.047 | 0.495 ± 0.022 | 0.558 ± 0.021 [0.525, 0.595] | 0.571 / 0.545 |

3. **The Wikipedia lines decide the ranking on the whole test.** On running text without
   repeats, mBERT (0.635) and XLM-R (0.624) are level, with overlapping intervals, and
   KazRoBERTa is lower (0.558). mBERT loses its place on the other two kinds of line:
   - **Redirect lines.** mBERT tags the redirect keyword `АЙДАУ` itself as an entity in 108–162
     of the 176 lines, depending on the seed; XLM-R does so in 66–94 and KazRoBERTa in 2–34.
   - **River stubs.** mBERT never tags `Ресейдегі` ("in Russia") in the 83 river stubs, in any
     seed, while XLM-R and KazRoBERTa tag it every time. The full KazNERD training split
     contains the word 14 times, 10 of them as GPE. The number of pieces does not explain the
     difference: the word is a single piece for both mBERT and KazRoBERTa, while XLM-R splits
     it into `▁Ресей` + `дегі`.
4. **Precision stays low even on running text** (0.57–0.65). WikiANN's labels come from
   Wikipedia links, so a name the annotation missed counts as a false positive.
5. **Repeats matter.** Removing the repeats from `other` raises F1 by 5–7 points for every
   model. The sentence that occurs 33 times, `Кастилия — Ла-Манча автономды бірлестігіне
   қарасты Куэнка провинциясының құрамына кіреді .`, labels `Кастилия — Ла-Манча` as one LOC and
   Куэнка, a province, as ORG. All nine runs tag Куэнка as LOC, and only one (mBERT, seed 42)
   finds the LOC span. This one repeated sentence thus accounts for 33–66 missed entities per run.

### Fragmentation

Exact-span recall by the entity's mean mBERT pieces per word, on `other_unique`:

| Model | 1.0 (n = 38) | (1,2] (137) | (2,3] (199) | >3 (259) |
|---|---|---|---|---|
| mBERT | 0.746 ± 0.080 | 0.611 ± 0.008 | 0.553 ± 0.026 | 0.663 ± 0.012 |
| XLM-R | 0.746 ± 0.040 | 0.572 ± 0.015 | 0.551 ± 0.013 | 0.638 ± 0.012 |
| KazRoBERTa | 0.675 ± 0.040 | 0.557 ± 0.011 | 0.489 ± 0.041 | 0.562 ± 0.019 |

- **The whole test is misleading here.** The 1.0 bin has 154 entities, and 83 of them are
  `Ресейдегі` and 10 are `Франциядағы` from the stubs. So the bin measured point 3, not
  fragmentation: recall was 0.24 for mBERT and 0.79–0.83 for the others.
- **On running text the pattern is a U shape.** Recall falls from one piece to (2,3] and rises
  again above three pieces, for all three models.
- **This is not a test of B2.** B2's decline for unseen KazNERD entities (0.88 → 0.79) does not
  reappear here. But this table is not stratified by training frequency or entity type, so it
  neither confirms nor refutes B2. The causal test is #37.

### FACILITY → LOC

With FACILITY mapped to LOC (`kaznerd_to_wikiann_facility_loc.yaml`), WikiANN F1 falls for every
model:

| Model | Whole test | `other_unique` |
|---|---|---|
| mBERT | 0.482 (−0.8) | 0.626 (−0.8) |
| XLM-R | 0.550 (−1.2) | 0.618 (−0.5) |
| KazRoBERTa | 0.529 (−0.3) | 0.554 (−0.4) |

Precision drops while recall barely changes: WikiANN rarely labels a building, an airport or a
road as LOC. The main mapping stays, and no conclusion changes.

### Conclusions

- A model fine-tuned on KazNERD loses 36–42 F1 points on WikiANN `kk`, and the label mapping is
  not the cause.
- WikiANN `kk` is a weak benchmark for Kazakh NER:
  - a third of its test consists of redirect lines and article first lines, which follow their
    own labelling conventions;
  - its running text contains repeats;
  - its ORG labels are noisy.
- The whole-test score ranks models largely by how they treat those lines. Any comparison on
  WikiANN `kk` should report `other_unique` next to the whole test.

## Caveats

- WikiANN is silver data with its own guidelines. Some disagreements with it are annotation
  noise, not model errors, and they lower every model's score.
- mBERT was pre-trained on Wikipedia, Kazakh Wikipedia included, so it may have seen WikiANN's
  sentences, without labels. XLM-R (CommonCrawl) and KazRoBERTa (web, news, books) may have
  seen some of them too.
- 1,000 test sentences, with about 1,100 entities, give intervals of a few F1 points.
