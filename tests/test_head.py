import json
from pathlib import Path

import pytest

from kazner.config import ConfigError, LoraConfig, MethodConfig
from kazner.head import (
    HeadMapping,
    checkpoint_head,
    load_head_mapping,
    parse_head_mapping,
    prepare_head_init,
)
from kazner.methods import build_model

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ["B-LOC", "B-MISC", "B-PER", "I-LOC", "I-MISC", "I-PER", "O"]
TARGET = [
    "B-DATE",
    "B-GPE",
    "B-LOCATION",
    "B-PERSON",
    "I-DATE",
    "I-GPE",
    "I-LOCATION",
    "I-PERSON",
    "O",
]
RAW = {
    "source": "toy-tr",
    "target": "toy-kk",
    "outside": True,
    "mapping": {"PER": ["PERSON"], "LOC": ["LOCATION", "GPE"], "MISC": []},
}
HIDDEN = 16


def ids(labels):
    return {label: i for i, label in enumerate(labels)}


@pytest.fixture(scope="module")
def source_model(tmp_path_factory):
    """A tiny BERT with a trained-looking head over SOURCE labels, saved locally (offline)."""
    import torch
    from transformers import BertConfig, BertForTokenClassification

    path = tmp_path_factory.mktemp("source-ner")
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=60, hidden_size=HIDDEN, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=len(SOURCE), label2id=ids(SOURCE),
        id2label=dict(enumerate(SOURCE)),
    )  # fmt: skip
    model = BertForTokenClassification(config)
    with torch.no_grad():  # distinctive rows: row i is all i + 1, bias i is 10 * (i + 1)
        for i in range(len(SOURCE)):
            model.classifier.weight[i].fill_(i + 1.0)
            model.classifier.bias[i] = 10.0 * (i + 1)
    model.save_pretrained(path)
    return path


def test_row_pairs_cover_b_i_and_o():
    pairs = parse_head_mapping(RAW).row_pairs(ids(SOURCE), ids(TARGET))
    assert pairs == [
        ("O", "O"),
        ("B-PER", "B-PERSON"),
        ("I-PER", "I-PERSON"),
        ("B-LOC", "B-LOCATION"),
        ("I-LOC", "I-LOCATION"),
        ("B-LOC", "B-GPE"),
        ("I-LOC", "I-GPE"),
    ]
    without_o = parse_head_mapping({**RAW, "outside": False}).row_pairs(ids(SOURCE), ids(TARGET))
    assert ("O", "O") not in without_o and len(without_o) == 6


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"PER": ["PERSON"], "LOC": ["LOCATION"]}, r"types \['MISC'\] are not listed"),
        ({**RAW["mapping"], "ORG": []}, r"\['ORG'\] are not entity types of the checkpoint"),
        ({"PER": ["PERSON"], "LOC": ["PERSON"], "MISC": []}, r"\['PERSON'\] are listed more"),
        ({"PER": ["TIME"], "LOC": [], "MISC": []}, "target label set has no label 'B-TIME'"),
    ],
)
def test_row_pairs_are_checked_against_both_label_sets(mapping, message):
    with pytest.raises(ConfigError, match=message):
        parse_head_mapping({**RAW, "mapping": mapping}).row_pairs(ids(SOURCE), ids(TARGET))


def test_a_source_type_without_i_label_is_an_error():
    source = [s for s in SOURCE if s != "I-PER"]
    with pytest.raises(ConfigError, match="checkpoint has no label 'I-PER'"):
        parse_head_mapping(RAW).row_pairs(ids(source), ids(TARGET))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": 1}, "unknown key"),
        ({"outside": "yes"}, "outside: expected true or false"),
        ({"target": ""}, "target: expected a non-empty string"),
        ({"mapping": {}}, "expected a non-empty mapping"),
        ({"mapping": {"B-PER": ["PERSON"]}}, "invalid source type 'B-PER'"),
        ({"mapping": {"PER": "PERSON"}}, r"PER: expected a list of entity types"),
        ({"mapping": {"PER": ["O"]}}, r"PER: expected a list of entity types"),
    ],
)
def test_invalid_head_mappings_are_rejected(change, message):
    with pytest.raises(ConfigError, match=message):
        parse_head_mapping({**RAW, **change})
    with pytest.raises(ConfigError, match="missing key"):
        parse_head_mapping({k: v for k, v in RAW.items() if k != "outside"})
    with pytest.raises(ConfigError, match="expected a mapping"):
        parse_head_mapping(["PER"])


def test_load_head_mapping_names_the_file(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(json.dumps({**RAW, "outside": 1}), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"m\.yaml: head mapping\.outside"):
        load_head_mapping(path)


def test_shipped_head_mapping_matches_the_issue():
    m = load_head_mapping(ROOT / "configs" / "mappings" / "twnertc_to_kaznerd_head.yaml")
    assert m == HeadMapping(
        "TWNERTC",
        "KazNERD",
        True,
        {
            "PERSON": ("PERSON",),
            "LOCATION": ("LOCATION", "GPE"),
            "ORGANIZATION": ("ORGANISATION",),
            "MISC": (),
        },
    )


def test_checkpoint_head_reads_weights_and_labels(source_model):
    weight, bias, labels = checkpoint_head(source_model)
    assert labels == ids(SOURCE)
    assert tuple(weight.shape) == (len(SOURCE), HIDDEN)
    assert weight[2].tolist() == [3.0] * HIDDEN and bias[2].item() == 30.0


def test_checkpoint_without_a_head_is_rejected(tmp_path):
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=60, hidden_size=HIDDEN, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32,
    )  # fmt: skip
    BertModel(config).save_pretrained(tmp_path)
    with pytest.raises(ConfigError, match="has no token-classification head"):
        checkpoint_head(tmp_path)


@pytest.mark.parametrize("method", ["full", "lora"])
def test_build_model_copies_exactly_the_mapped_rows(source_model, tmp_path, method):
    """Same seed: with and without a head mapping the heads differ only in the copied rows."""
    import torch

    mapping = tmp_path / "head.yaml"
    mapping.write_text(json.dumps(RAW), encoding="utf-8")
    target = ids(TARGET)
    init = prepare_head_init(mapping, source_model, target)
    cfg = MethodConfig(method, LoraConfig())

    def head(head_init):
        torch.manual_seed(1)
        model = build_model(cfg, str(source_model), target, dict(enumerate(TARGET)),
                            head_init=head_init).model  # fmt: skip
        classifier = model.get_submodule("base_model.model.classifier.modules_to_save.default") \
            if method == "lora" else model.classifier  # fmt: skip
        return classifier.weight.detach().clone(), classifier.bias.detach().clone()

    (w_new, b_new), (w_copy, b_copy) = head(None), head(init)
    copied = {target[t] for _, t in init.pairs}
    for label, i in target.items():
        if i in copied:
            s = ids(SOURCE)[dict((t, s) for s, t in init.pairs)[label]]
            assert w_copy[i].tolist() == [s + 1.0] * HIDDEN
            assert b_copy[i].item() == 10.0 * (s + 1)
        else:  # untouched rows keep the seed's random initialisation
            assert torch.equal(w_copy[i], w_new[i]) and torch.equal(b_copy[i], b_new[i])
    assert set(range(len(TARGET))) - copied == {target["B-DATE"], target["I-DATE"]}


def test_hidden_size_mismatch_is_reported(source_model, tmp_path):
    from transformers import BertConfig, BertForTokenClassification

    mapping = tmp_path / "head.yaml"
    mapping.write_text(json.dumps(RAW), encoding="utf-8")
    init = prepare_head_init(mapping, source_model, ids(TARGET))
    config = BertConfig(
        vocab_size=60, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, num_labels=len(TARGET),
    )  # fmt: skip
    with pytest.raises(ValueError, match="hidden size 8 of the model differs from 16"):
        init.apply(BertForTokenClassification(config), ids(TARGET))
