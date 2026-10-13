import pytest

from kazner.config import LoraConfig, MethodConfig
from kazner.methods import build_model, register_method, registered_methods

LABELS = ["B-LOC", "B-PER", "I-LOC", "I-PER", "O"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
HIDDEN, LAYERS, R = 16, 2, 8
CLASSIFIER = HIDDEN * len(LABELS) + len(LABELS)
LORA = LAYERS * 2 * (R * HIDDEN + HIDDEN * R)  # query and value, A and B per layer


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    """A tiny BERT saved locally, so the builders run offline."""
    from transformers import BertConfig, BertForTokenClassification

    path = tmp_path_factory.mktemp("tiny-bert")
    config = BertConfig(
        vocab_size=60,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=len(LABELS),
    )
    BertForTokenClassification(config).save_pretrained(path)
    return str(path)


def build(name, model_name, **lora):
    return build_model(MethodConfig(name, LoraConfig(**lora)), model_name, LABEL2ID, ID2LABEL)


def trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def test_full_trains_everything(tiny_bert):
    built = build("full", tiny_bert)
    assert built.trainable_params == built.total_params
    assert built.trainable_percent == 100.0
    assert built.model.config.id2label == ID2LABEL


def test_frozen_trains_only_the_classifier(tiny_bert):
    built = build("frozen", tiny_bert)
    assert all(n.startswith("classifier.") for n in trainable_names(built.model))
    assert built.trainable_params == CLASSIFIER
    assert built.trainable_params_pilot == CLASSIFIER


def test_lora_count_excludes_the_inactive_classifier_copy(tiny_bert):
    built = build("lora", tiny_bert)
    assert built.trainable_params == LORA + CLASSIFIER
    # The pilot also counted PEFT's inactive original classifier (mBERT: 373,350 vs 334,131).
    assert built.trainable_params_pilot == LORA + 2 * CLASSIFIER
    assert built.total_params_pilot - built.total_params == CLASSIFIER
    frozen = build("frozen", tiny_bert)
    assert frozen.trainable_percent < built.trainable_percent < 100.0


def test_lora_settings_come_from_the_config(tiny_bert):
    built = build("lora", tiny_bert, r=4, alpha=8, dropout=0.0, target_modules=("query",))
    assert built.trainable_params == LAYERS * (4 * HIDDEN + HIDDEN * 4) + CLASSIFIER
    peft_config = built.model.peft_config["default"]
    assert (peft_config.r, peft_config.lora_alpha, peft_config.lora_dropout) == (4, 8, 0.0)


def test_unknown_method_lists_the_registered_ones(tiny_bert):
    with pytest.raises(
        ValueError, match=r"unknown adaptation method 'qlora'.*\['frozen', 'full', 'lora'\]"
    ):
        build("qlora", tiny_bert)


def test_registry_rejects_duplicates_and_accepts_new_methods(tiny_bert):
    assert registered_methods() == ["frozen", "full", "lora"]
    with pytest.raises(ValueError, match="already registered"):
        register_method("full")(lambda model, method: model)

    @register_method("head_and_last_layer")
    def _custom(model, method):
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("classifier.") or ".layer.1." in n
        return model

    try:
        built = build("head_and_last_layer", tiny_bert)
        assert CLASSIFIER < built.trainable_params < built.total_params
    finally:
        from kazner import methods

        methods._REGISTRY.pop("head_and_last_layer")


@pytest.mark.network
def test_methods_on_tiny_random_bert():
    for name in ("full", "frozen", "lora"):
        built = build(name, "hf-internal-testing/tiny-random-bert")
        assert 0 < built.trainable_params <= built.total_params
