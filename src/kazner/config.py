"""Experiment configuration: YAML -> frozen, validated dataclasses; deterministic run IDs.

One YAML file defines one experiment. Its seed list expands into one run per seed, and every
run gets a ``run_id`` derived from the resolved configuration and the code that executes it
(the git tree hash of ``src/kazner``). Defaults equal the pilot's ``TrainingArguments``.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import platform
import re
import subprocess
import sys
import types
import typing
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

import kazner

RUN_ID_LENGTH = 12
#: Familiar Hugging Face / pilot names, suggested when they appear as unknown keys.
ALIASES = {
    "lr": "learning_rate",
    "num_train_epochs": "epochs",
    "batch_size": "train_batch_size",
    "per_device_train_batch_size": "train_batch_size",
    "per_device_eval_batch_size": "eval_batch_size",
    "gradient_accumulation_steps": "grad_accumulation",
    "max_length": "max_len",
    "seed": "seeds",
}
#: Fields that label or place a run but do not change what it computes; not hashed.
#: The data directory is replaced by a fingerprint of the data content (see data_fingerprint).
UNHASHED_FIELDS = ("name", "output_root")
#: A ``model.name`` with one of these prefixes, or with a drive letter, is a local directory;
#: anything else is a Hugging Face Hub ID (so ``org/name`` is never read as a path).
LOCAL_MODEL_PREFIXES = ("./", "../", ".\\", "..\\", "/", "\\")
#: Files of a model directory whose line endings are normalised in its fingerprint:
#: ``save_pretrained`` writes them with CRLF on Windows.
TEXT_SUFFIXES = (".json", ".txt")


class ConfigError(ValueError):
    """A configuration file or value is invalid; the message names the offending key."""


def is_local_model(name: str) -> bool:
    """``True`` if ``model.name`` is a path (see :data:`LOCAL_MODEL_PREFIXES`)."""
    return name.startswith(LOCAL_MODEL_PREFIXES) or PureWindowsPath(name).is_absolute()


# --------------------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class DataConfig:
    dir: str
    train_fraction: float = 1.0
    subsample_seed: int = 42
    max_train_samples: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "bert-base-multilingual-cased"
    max_len: int = 128
    #: Passed to the tokenizer only when set; ``None`` keeps the tokenizer's default. Byte-level
    #: BPE tokenizers (RoBERTa family) need ``True`` for pre-split words (see train.py).
    add_prefix_space: bool | None = None
    #: Hub branch, tag or commit for the tokenizer and the model; ``None`` = default branch.
    #: Pin a commit for reported results. Not allowed for a local directory.
    revision: str | None = None


@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")


@dataclass(frozen=True)
class MethodConfig:
    name: str = "full"
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 5e-5
    epochs: float = 3.0
    train_batch_size: int = 4
    eval_batch_size: int = 8
    grad_accumulation: int = 2
    weight_decay: float = 0.01
    fp16: str = "auto"  # "auto" = use FP16 when CUDA is available, as in the pilot
    load_best_model_at_end: bool = True
    logging_steps: int = 50
    keep_checkpoints: bool = False


@dataclass(frozen=True)
class RunConfig:
    """One training run: an experiment with a single seed."""

    name: str
    data: DataConfig
    model: ModelConfig
    method: MethodConfig
    training: TrainingConfig
    seed: int
    output_root: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def hashable_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        for key in UNHASHED_FIELDS:
            d.pop(key)
        d["data"].pop("dir")  # a location, not content: the data fingerprint is hashed instead
        if is_local_model(self.model.name):
            d["model"].pop("name")  # likewise: the model fingerprint is hashed instead
        return d


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seeds: tuple[int, ...] = (42,)
    output_root: str = "runs"
    #: Directory of the YAML file; relative paths are resolved against it. Not hashed.
    base_dir: Path = field(default=Path("."), compare=False, repr=False)

    def expand(self) -> list[RunConfig]:
        """One :class:`RunConfig` per seed."""
        return [
            RunConfig(
                name=self.name,
                data=self.data,
                model=self.model,
                method=self.method,
                training=self.training,
                seed=seed,
                output_root=self.output_root,
            )
            for seed in self.seeds
        ]

    def resolve(self, path: str) -> Path:
        """Resolve a path from the config against the YAML file's directory."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def model_location(self) -> str:
        """What ``from_pretrained`` gets: a Hub ID as written, a local directory resolved."""
        name = self.model.name
        return str(self.resolve(name)) if is_local_model(name) else name


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            f.name: _to_plain(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.compare  # skips base_dir
        }
    if isinstance(obj, tuple | list):
        return [_to_plain(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- loading

_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _coerce(value: Any, annotation: Any, key: str) -> Any:
    """Check and convert one scalar or tuple value against a field annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(value, inner, key)
    if origin is tuple:
        if not isinstance(value, list | tuple) or not value:
            raise ConfigError(f"{key}: expected a non-empty list, got {value!r}")
        return tuple(_coerce(v, args[0], f"{key}[{i}]") for i, v in enumerate(value))
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected true or false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        # PyYAML (YAML 1.1) reads "5e-5" as a string; accept numeric strings explicitly.
        if isinstance(value, str) and _FLOAT_RE.match(value.strip()):
            return float(value)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key}: expected a non-empty string, got {value!r}")
        return value
    raise ConfigError(f"{key}: unsupported type {annotation!r}")  # pragma: no cover


def _build(cls: type, raw: Any, prefix: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix.rstrip('.') or 'config'}: expected a mapping, got {raw!r}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init and f.compare}
    unknown = [k for k in raw if k not in fields]
    if unknown:
        key = unknown[0]
        hint = [ALIASES[key]] if ALIASES.get(key) in fields else []
        hint = hint or difflib.get_close_matches(str(key), list(fields), n=1)
        suggestion = f" (did you mean '{hint[0]}'?)" if hint else ""
        raise ConfigError(f"{prefix}{key}: unknown key{suggestion}")
    kwargs = {}
    for name, f in fields.items():
        key = f"{prefix}{name}"
        if name not in raw:
            no_default = f.default is dataclasses.MISSING
            if no_default and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"{key}: required key is missing")
            continue
        annotation = hints[name]
        if dataclasses.is_dataclass(annotation):
            kwargs[name] = _build(annotation, raw[name], f"{key}.")
        else:
            kwargs[name] = _coerce(raw[name], annotation, key)
    return cls(**kwargs)


def build_dataclass(cls: type, raw: Any, prefix: str = "") -> Any:
    """Build a (nested) config dataclass from a parsed YAML mapping with the same key checks
    and type coercion as experiment configs; used by other config types (``kazner.pretrain``)."""
    return _build(cls, raw, prefix)


def _validate(cfg: ExperimentConfig) -> None:
    def check(ok: bool, key: str, message: str) -> None:
        if not ok:
            raise ConfigError(f"{key}: {message}")

    check(0 < cfg.data.train_fraction <= 1, "data.train_fraction", "must be in (0, 1]")
    mts = cfg.data.max_train_samples
    check(mts is None or mts > 0, "data.max_train_samples", "must be positive")
    check(cfg.model.max_len > 0, "model.max_len", "must be positive")
    check(
        cfg.model.revision is None or not is_local_model(cfg.model.name),
        "model.revision",
        "only applies to Hugging Face Hub models, not to a local directory",
    )
    check(cfg.method.lora.r > 0, "method.lora.r", "must be positive")
    check(cfg.method.lora.alpha > 0, "method.lora.alpha", "must be positive")
    check(0 <= cfg.method.lora.dropout < 1, "method.lora.dropout", "must be in [0, 1)")
    t = cfg.training
    check(t.learning_rate > 0, "training.learning_rate", "must be positive")
    check(t.epochs > 0, "training.epochs", "must be positive")
    for key in ("train_batch_size", "eval_batch_size", "grad_accumulation", "logging_steps"):
        check(getattr(t, key) > 0, f"training.{key}", "must be positive")
    check(t.weight_decay >= 0, "training.weight_decay", "must not be negative")
    check(t.fp16 in ("auto", "true", "false"), "training.fp16", "must be auto, true or false")
    check(len(set(cfg.seeds)) == len(cfg.seeds), "seeds", "must not contain duplicates")


def parse_config(raw: Any, base_dir: Path = Path(".")) -> ExperimentConfig:
    """Build and validate an :class:`ExperimentConfig` from a parsed YAML mapping."""
    training = raw.get("training") if isinstance(raw, dict) else None
    if isinstance(training, dict) and isinstance(training.get("fp16"), bool):
        raw = {**raw, "training": {**training, "fp16": str(training["fp16"]).lower()}}
    cfg = _build(ExperimentConfig, raw, "")
    cfg = dataclasses.replace(cfg, base_dir=Path(base_dir))
    _validate(cfg)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment file; relative paths in it are relative to the file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return parse_config(raw, base_dir=path.resolve().parent)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


# --------------------------------------------------------------------------- code identity


@dataclass(frozen=True)
class CodeState:
    """What code a run executes: git tree of ``src/kazner`` or, outside git, the version."""

    code_ref: str
    source: str  # "git-tree" or "package-version"
    commit: str | None = None
    dirty: bool | None = None
    package_dirty: bool | None = None


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_state(package_dir: Path | None = None) -> CodeState:
    """Identify the code: ``git rev-parse HEAD:<package path>`` plus commit and dirty flags."""
    package_dir = Path(package_dir or Path(kazner.__file__).parent).resolve()
    try:
        top = Path(_git(["rev-parse", "--show-toplevel"], package_dir)).resolve()
        rel = package_dir.relative_to(top).as_posix()
        tree = _git(["rev-parse", f"HEAD:{rel}"], top)
        commit = _git(["rev-parse", "HEAD"], top)
        dirty = bool(_git(["status", "--porcelain"], top))
        package_dirty = bool(_git(["status", "--porcelain", "--", rel], top))
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return CodeState(code_ref=f"kazner=={kazner.__version__}", source="package-version")
    if package_dirty:
        warnings.warn(
            f"uncommitted changes under {rel}: the run ID does not describe the code that runs",
            stacklevel=2,
        )
    return CodeState(tree, "git-tree", commit, dirty, package_dirty)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def data_fingerprint(data_dir: str | Path, files: tuple[str, ...] | None = None) -> str:
    """SHA-256 over the split files' content, with CRLF normalised to LF.

    Identifies the data by content, not by path: the same KazNERD release in another
    directory, or checked out with Windows line endings, gives the same fingerprint.
    """
    from kazner.data import KAZNERD_SPLITS

    data_dir = Path(data_dir)
    digest = hashlib.sha256()
    for name in sorted(files or KAZNERD_SPLITS.values()):
        content = (data_dir / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def model_fingerprint(model_dir: str | Path) -> str:
    """SHA-256 over every file of a local model directory (relative path + content).

    Identifies a locally produced checkpoint by content, not by path. Line endings of text
    files (:data:`TEXT_SUFFIXES`) are normalised, so a model saved on Windows and on Linux
    gets the same fingerprint. Files and directories whose name starts with a dot are skipped.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ConfigError(f"model.name: local model directory {model_dir} does not exist")
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in model_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(model_dir).parts)
    )
    for path in files:
        content = hashlib.sha256()
        if path.suffix in TEXT_SUFFIXES:
            content.update(path.read_bytes().replace(b"\r\n", b"\n"))
        else:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    content.update(chunk)
        name = path.relative_to(model_dir).as_posix()
        digest.update(name.encode("utf-8") + b"\0" + content.digest())
    return digest.hexdigest()


def run_id(run: RunConfig, code_ref: str, data_ref: str = "", model_ref: str = "") -> str:
    """First 12 hex chars of sha256(hashed config + code reference + data and model
    fingerprints); ``model_ref`` is empty for Hub models, which are hashed by name."""
    payload = canonical_json(
        {"config": run.hashable_dict(), "code": code_ref, "data": data_ref, "model": model_ref}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:RUN_ID_LENGTH]


# --------------------------------------------------------------------------- environment

PACKAGES = ("kazner", "torch", "transformers", "datasets", "peft", "accelerate", "seqeval")


def environment(state: CodeState) -> dict[str, Any]:
    """Snapshot of code, library versions and hardware for ``environment.json``."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    env: dict[str, Any] = {
        "code": dataclasses.asdict(state),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": False,
        "cuda_version": None,
        "gpu": None,
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if env["cuda_available"]:
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass
    return env


def write_run_files(
    run_dir: Path,
    run: RunConfig,
    rid: str,
    env: dict[str, Any],
    data_ref: str = "",
    model_ref: str = "",
) -> None:
    """Write ``config.json`` (resolved run config) and ``environment.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": rid,
        "data_fingerprint": data_ref,
        "model_fingerprint": model_ref or None,
        **run.to_dict(),
    }
    for name, payload in (("config.json", config), ("environment.json", env)):
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        (run_dir / name).write_text(text, encoding="utf-8", newline="\n")
