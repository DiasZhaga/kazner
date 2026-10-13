"""Adaptation-method registry (Factory Method): ``full``, ``frozen`` and ``lora``.

Each builder creates a token-classification model exactly as the pilot did
(``ner-project/src/run_peft_comparison.py``: ``prepare_model_for_method``) and reports its
trainable-parameter count. Adding a method means one ``@register_method`` function and one
config value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kazner.config import MethodConfig

#: Parameters of the inactive original module inside PEFT's ``ModulesToSaveWrapper``.
#: With ``TaskType.TOKEN_CLS`` PEFT trains a copy of the classifier (``modules_to_save``)
#: and keeps the original; only the copy is used in the forward pass.
_INACTIVE = ".original_module."


@dataclass(frozen=True)
class BuiltModel:
    model: Any
    trainable_params: int
    total_params: int
    #: Counted as the pilot counted (every parameter with ``requires_grad``), for comparison
    #: with the pilot paper. Differs from ``trainable_params`` only for LoRA.
    trainable_params_pilot: int
    total_params_pilot: int

    @property
    def trainable_percent(self) -> float:
        return 100.0 * self.trainable_params / self.total_params


Builder = Callable[[Any, MethodConfig], Any]
_REGISTRY: dict[str, Builder] = {}


def register_method(name: str) -> Callable[[Builder], Builder]:
    """Register ``builder(model, method_cfg) -> model`` under ``name``."""

    def decorator(builder: Builder) -> Builder:
        if name in _REGISTRY:
            raise ValueError(f"adaptation method {name!r} is already registered")
        _REGISTRY[name] = builder
        return builder

    return decorator


def registered_methods() -> list[str]:
    return sorted(_REGISTRY)


def count_parameters(model: Any) -> tuple[int, int, int, int]:
    """``(trainable, total, trainable_pilot, total_pilot)``; inactive PEFT copies excluded."""
    trainable = total = trainable_pilot = total_pilot = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_pilot += n
        trainable_pilot += n if param.requires_grad else 0
        if _INACTIVE in name:
            continue
        total += n
        trainable += n if param.requires_grad else 0
    return trainable, total, trainable_pilot, total_pilot


def build_model(
    method: MethodConfig,
    model_name: str,
    label2id: Mapping[str, int],
    id2label: Mapping[int, str],
) -> BuiltModel:
    """Create a token-classification model for ``model_name`` and adapt it with ``method``."""
    if method.name not in _REGISTRY:
        raise ValueError(
            f"unknown adaptation method {method.name!r}; registered: {registered_methods()}"
        )
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=dict(id2label),
        label2id=dict(label2id),
        # A fresh head is always created for our label set; a checkpoint that already has a
        # head for another label set (tiny test models, TWNERTC transfer) is re-initialised,
        # as the pilot's cross-lingual pipeline did. mBERT has no head, so nothing changes.
        ignore_mismatched_sizes=True,
    )
    model = _REGISTRY[method.name](model, method)
    return BuiltModel(model, *count_parameters(model))


def _freeze_encoder_keep_classifier_trainable(model: Any) -> None:
    """Pilot's ``freeze_encoder_keep_classifier_trainable``."""
    for param in model.parameters():
        param.requires_grad = False
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise AttributeError("the model has no token-classification head named 'classifier'")
    for param in classifier.parameters():
        param.requires_grad = True


@register_method("full")
def _full(model: Any, method: MethodConfig) -> Any:
    return model


@register_method("frozen")
def _frozen(model: Any, method: MethodConfig) -> Any:
    _freeze_encoder_keep_classifier_trainable(model)
    return model


@register_method("lora")
def _lora(model: Any, method: MethodConfig) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    _freeze_encoder_keep_classifier_trainable(model)
    lora = method.lora
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
    )
    model = get_peft_model(model, config)
    for param in model.classifier.parameters():  # as in the pilot: both copies flagged
        param.requires_grad = True
    return model
