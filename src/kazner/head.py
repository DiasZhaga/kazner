"""Classification head initialised from a checkpoint's own head (#59, ``model.head_mapping``).

A model fine-tuned for NER on another dataset has a head over that dataset's labels (TWNERTC:
PERSON, LOCATION, ORGANIZATION and MISC). Loaded for KazNERD, the head is replaced by a new one
for KazNERD's labels (``ignore_mismatched_sizes``), as in the pilot, and only the encoder is
transferred. A head mapping copies rows of the old head into the new one: for each source
entity type, the weight rows and biases of its ``B-`` and ``I-`` labels go to the ``B-`` and
``I-`` labels of every listed target type, and with ``outside: true`` the ``O`` row goes to
``O``. Every other row keeps the new random initialisation, so with the same seed a run with a
head mapping differs from one without only in the copied rows.

The mapping must list every entity type of the checkpoint (an empty list copies nothing), and
every label it names must exist on its side. The file is part of the run ID by content
(:func:`kazner.config.head_mapping_fingerprint`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kazner.config import ConfigError

OUTSIDE = "O"
PREFIXES = ("B", "I")
HEAD_MAPPING_KEYS = ("source", "target", "outside", "mapping")
#: Parameter names of the token-classification head in BERT and RoBERTa checkpoints.
HEAD_WEIGHT, HEAD_BIAS = "classifier.weight", "classifier.bias"


@dataclass(frozen=True)
class HeadMapping:
    """Source entity type -> target entity types whose rows start from the source's rows."""

    source: str
    target: str
    outside: bool
    types: Mapping[str, tuple[str, ...]]

    def row_pairs(
        self, source_labels: Mapping[str, int], target_labels: Mapping[str, int]
    ) -> list[tuple[str, str]]:
        """``(source label, target label)`` per copied row, validated against both label sets."""
        source_types = _entity_types(source_labels)
        missing = sorted(source_types - set(self.types))
        if missing:
            raise ConfigError(f"head mapping: the checkpoint's types {missing} are not listed")
        unknown = sorted(set(self.types) - source_types)
        if unknown:
            raise ConfigError(f"head mapping: {unknown} are not entity types of the checkpoint")
        targets = [t for listed in self.types.values() for t in listed]
        repeated = sorted({t for t in targets if targets.count(t) > 1})
        if repeated:
            raise ConfigError(f"head mapping: target types {repeated} are listed more than once")

        pairs = [(OUTSIDE, OUTSIDE)] if self.outside else []
        for source, listed in self.types.items():
            for target in listed:
                pairs += [(f"{p}-{source}", f"{p}-{target}") for p in PREFIXES]
        for src, tgt in pairs:
            if src not in source_labels:
                raise ConfigError(f"head mapping: the checkpoint has no label {src!r}")
            if tgt not in target_labels:
                raise ConfigError(f"head mapping: the target label set has no label {tgt!r}")
        return pairs


def _entity_types(labels: Mapping[str, int]) -> set[str]:
    return {label.partition("-")[2] for label in labels if label != OUTSIDE}


def parse_head_mapping(raw: Any) -> HeadMapping:
    """Build a :class:`HeadMapping` from a parsed YAML mapping, checking its form."""
    if not isinstance(raw, dict):
        raise ConfigError(f"head mapping: expected a mapping, got {raw!r}")
    unknown = [k for k in raw if k not in HEAD_MAPPING_KEYS]
    if unknown:
        raise ConfigError(
            f"head mapping: unknown key(s) {unknown}; expected {list(HEAD_MAPPING_KEYS)}"
        )
    missing = [k for k in HEAD_MAPPING_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"head mapping: missing key(s) {missing}")
    for key in ("source", "target"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ConfigError(f"head mapping.{key}: expected a non-empty string")
    if not isinstance(raw["outside"], bool):
        raise ConfigError("head mapping.outside: expected true or false")
    types = raw["mapping"]
    if not isinstance(types, dict) or not types:
        raise ConfigError("head mapping.mapping: expected a non-empty mapping of entity types")
    parsed = {}
    for source, targets in types.items():
        if not isinstance(source, str) or not source or "-" in source or source == OUTSIDE:
            raise ConfigError(f"head mapping.mapping: invalid source type {source!r}")
        if not isinstance(targets, list) or not all(
            isinstance(t, str) and t and "-" not in t and t != OUTSIDE for t in targets
        ):
            raise ConfigError(
                f"head mapping.mapping.{source}: expected a list of entity types (may be empty)"
            )
        parsed[source] = tuple(targets)
    return HeadMapping(raw["source"], raw["target"], raw["outside"], parsed)


def load_head_mapping(path: str | Path) -> HeadMapping:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return parse_head_mapping(raw)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


# --------------------------------------------------------------------------- weights


@dataclass(frozen=True)
class HeadInit:
    """Rows to copy into a new head, read once per experiment and applied in every run."""

    weight: Any  # torch.Tensor, source labels x hidden size
    bias: Any  # torch.Tensor, source labels
    source_labels: Mapping[str, int]
    pairs: tuple[tuple[str, str], ...]

    def apply(self, model: Any, target_labels: Mapping[str, int]) -> None:
        """Copy every pair's weight row and bias into ``model.classifier``."""
        import torch

        classifier = getattr(model, "classifier", None)
        if classifier is None or not hasattr(classifier, "weight"):
            raise AttributeError("the model has no linear token-classification head 'classifier'")
        if classifier.weight.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"head mapping: hidden size {classifier.weight.shape[1]} of the model differs "
                f"from {self.weight.shape[1]} of the checkpoint's head"
            )
        with torch.no_grad():
            for src, tgt in self.pairs:
                s, t = self.source_labels[src], target_labels[tgt]
                classifier.weight[t].copy_(self.weight[s])
                classifier.bias[t].copy_(self.bias[s])


def checkpoint_head(model_dir: str | Path) -> tuple[Any, Any, dict[str, int]]:
    """The head of a saved token-classification model: weight, bias and its label -> id.

    A checkpoint without a head (e.g. a bare encoder) is an error: ``from_pretrained`` would
    create a random head, and copying its rows would be meaningless.
    """
    from transformers import AutoModelForTokenClassification

    model, info = AutoModelForTokenClassification.from_pretrained(
        str(model_dir), output_loading_info=True
    )
    missing = sorted(k for k in info["missing_keys"] if k in (HEAD_WEIGHT, HEAD_BIAS))
    if missing:
        raise ConfigError(
            f"model.head_mapping: {model_dir} has no token-classification head ({missing})"
        )
    weight = model.classifier.weight.detach().clone()
    bias = model.classifier.bias.detach().clone()
    labels = {str(label): int(i) for label, i in model.config.label2id.items()}
    del model
    return weight, bias, labels


def prepare_head_init(
    mapping_path: str | Path, model_dir: str | Path, target_labels: Mapping[str, int]
) -> HeadInit:
    """Read the mapping and the checkpoint's head and check them against the target labels."""
    mapping = load_head_mapping(mapping_path)
    weight, bias, source_labels = checkpoint_head(model_dir)
    pairs = mapping.row_pairs(source_labels, target_labels)
    return HeadInit(weight, bias, source_labels, tuple(pairs))
