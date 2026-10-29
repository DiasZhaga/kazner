import json
import subprocess
from pathlib import Path

import pytest

from kazner.config import (
    CodeState,
    ConfigError,
    ExtraTestConfig,
    code_state,
    data_fingerprint,
    environment,
    extra_tests_fingerprint,
    is_local_model,
    load_config,
    model_fingerprint,
    parse_config,
    run_id,
    write_run_files,
)

ROOT = Path(__file__).resolve().parent.parent
MINIMAL = {"name": "exp", "data": {"dir": "data"}}


def cfg(**overrides):
    raw = json.loads(json.dumps(MINIMAL))
    for dotted, value in overrides.items():
        node = raw
        *parents, last = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[last] = value
    return parse_config(raw)


# --- loading and validation ------------------------------------------------------------


def test_defaults_equal_the_pilot_training_arguments():
    t = cfg().training
    assert (t.eval_batch_size, t.weight_decay, t.logging_steps) == (8, 0.01, 50)
    assert t.load_best_model_at_end is True
    assert t.fp16 == "auto"
    assert cfg().method.lora.target_modules == ("query", "value")


def test_pilot_config_reproduces_the_pilot_settings():
    c = load_config(ROOT / "configs" / "pilot_full_10pct.yaml")
    assert c.method.name == "full"
    assert (c.data.train_fraction, c.data.subsample_seed, c.seeds) == (0.10, 42, (42,))
    t = c.training
    assert (t.learning_rate, t.epochs, t.train_batch_size, t.grad_accumulation) == (5e-5, 3.0, 4, 2)
    assert c.model.max_len == 128
    assert c.resolve(c.data.dir) == (ROOT / "data" / "kaznerd").resolve()


EXPERIMENT_CONFIGS = sorted(
    p for p in (ROOT / "configs").rglob("*.yaml") if p.parent.name not in ("pretrain", "mappings")
)  # pre-training configs are tested in test_pretrain.py, label mappings in test_transfer.py


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_config_is_valid(path):
    c = load_config(path)
    assert c.resolve(c.data.dir) == (ROOT / "data" / "kaznerd").resolve()
    if path.parent.name == "baselines":  # reported results: model commit pinned
        assert len(c.model.revision) == 40
        assert c.model.add_prefix_space is ("roberta-conversational" in c.model.name or None)


def test_causal_test_arms_differ_only_in_the_start_model():
    """#37: per method, arms A, B and the reference share every fine-tuning setting, the
    settings of the B2 reference runs, and ten seeds; A and B start from their pre-training."""
    configs = {p.stem: load_config(p) for p in (ROOT / "configs" / "causal").glob("*.yaml")}
    arms, methods = ("arm_a", "arm_b", "reference"), ("full", "lora")
    assert sorted(configs) == [f"{arm}_{method}" for arm in arms for method in methods]
    b2 = {
        "full": load_config(ROOT / "configs" / "baselines" / "mbert_best.yaml"),
        "lora": load_config(ROOT / "configs" / "lora_best_10pct.yaml"),
    }
    for method, ref in b2.items():
        for arm in arms:
            c = configs[f"{arm}_{method}"]
            assert (c.method, c.training) == (ref.method, ref.training)
            same_data = ("train_fraction", "subsample_seed", "max_train_samples")
            assert all(getattr(c.data, k) == getattr(ref.data, k) for k in same_data)
            assert c.model.max_len == ref.model.max_len == 128
            assert c.seeds == tuple(range(42, 52))
            assert c.resolve(c.output_root) == (ROOT / "runs").resolve()
        assert configs[f"reference_{method}"].model == b2["full"].model  # pinned original mBERT
    start = {arm: configs[f"{arm}_full"].model_location() for arm in ("arm_a", "arm_b")}
    assert start["arm_a"] != start["arm_b"]
    for arm, location in start.items():
        assert configs[f"{arm}_lora"].model_location() == location
        assert Path(location).parent == (ROOT / "models").resolve()  # pretrain output_root


def test_transfer_configs_repeat_the_selected_baselines_with_wikiann():
    """#51: the WikiANN runs retrain the #32 selections unchanged and add one extra test set."""
    for key in ("mbert", "xlmr", "kazroberta"):
        t = load_config(ROOT / "configs" / "transfer" / f"{key}_wikiann.yaml")
        b = load_config(ROOT / "configs" / "baselines" / f"{key}_best.yaml")
        assert (t.data, t.model, t.method, t.training) == (b.data, b.model, b.method, b.training)
        assert (t.seeds, t.output_root) == (b.seeds, b.output_root)
        (extra,) = t.extra_tests
        assert extra.name == "wikiann_kk"
        assert t.resolve(extra.path) == (ROOT / "data" / "wikiann_kk" / "IOB2_test.txt").resolve()


def test_extra_tests_are_parsed_and_validated():
    c = parse_config({**MINIMAL, "extra_tests": [{"name": "wikiann_kk", "path": "w/test.txt"}]})
    assert c.extra_tests == (ExtraTestConfig("wikiann_kk", "w/test.txt"),)
    assert c.expand()[0].extra_tests == c.extra_tests
    with pytest.raises(ConfigError, match=r"extra_tests\[0\]\.file: unknown key"):
        parse_config({**MINIMAL, "extra_tests": [{"name": "a", "file": "x"}]})
    with pytest.raises(ConfigError, match=r"extra_tests\[0\]\.path: required key is missing"):
        parse_config({**MINIMAL, "extra_tests": [{"name": "a"}]})
    two = [{"name": "a", "path": "x"}, {"name": "a", "path": "y"}]
    with pytest.raises(ConfigError, match="names must be unique"):
        parse_config({**MINIMAL, "extra_tests": two})
    with pytest.raises(ConfigError, match=r"extra_tests\[0\]\.name: may only contain"):
        parse_config({**MINIMAL, "extra_tests": [{"name": "wiki ann", "path": "x"}]})


def test_extra_tests_enter_the_run_id_by_content_not_by_path(tmp_path):
    plain = cfg().expand()[0]
    assert "extra_tests" not in plain.hashable_dict()  # runs without them keep their payload
    assert run_id(plain, "code", "data") == run_id(plain, "code", "data", extra_ref="")
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "test.txt").write_bytes(b"x B-PER\n")

    def with_extra(folder):
        raw = {**MINIMAL, "extra_tests": [{"name": "w", "path": f"{folder}/test.txt"}]}
        return parse_config(raw, base_dir=tmp_path)

    a, b = with_extra("a"), with_extra("b")
    assert a.expand()[0].hashable_dict() == {
        **plain.hashable_dict(),
        "extra_tests": [{"name": "w"}],
    }
    assert extra_tests_fingerprint(a) == extra_tests_fingerprint(b)  # same content elsewhere
    ref = extra_tests_fingerprint(a)
    assert run_id(a.expand()[0], "code", "data", extra_ref=ref) != run_id(plain, "code", "data")
    (tmp_path / "b" / "test.txt").write_bytes(b"x B-PER\r\n")
    assert extra_tests_fingerprint(b) == ref  # line endings do not matter
    (tmp_path / "b" / "test.txt").write_bytes(b"x B-LOC\n")
    assert extra_tests_fingerprint(b) != ref
    assert extra_tests_fingerprint(cfg()) == ""
    with pytest.raises(ConfigError, match="does not exist"):
        extra_tests_fingerprint(with_extra("missing"))


def test_yaml_scientific_notation_without_dot_is_a_number(tmp_path):
    """PyYAML reads `5e-5` as a string; the loader must still accept it as a float."""
    path = tmp_path / "c.yaml"
    path.write_text("name: x\ndata: {dir: d}\ntraining: {learning_rate: 5e-5}\n", encoding="utf-8")
    assert load_config(path).training.learning_rate == 5e-5


def test_add_prefix_space_defaults_to_the_tokenizer_default():
    assert cfg().model.add_prefix_space is None
    assert cfg(**{"model.add_prefix_space": None}).model.add_prefix_space is None
    assert cfg(**{"model.add_prefix_space": True}).model.add_prefix_space is True


def test_fp16_accepts_booleans():
    assert cfg(**{"training.fp16": True}).training.fp16 == "true"
    assert cfg(**{"training.fp16": "false"}).training.fp16 == "false"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training.lr": 1e-4}, r"training\.lr: unknown key \(did you mean 'learning_rate'\?\)"),
        ({"seed": 1}, r"seed: unknown key \(did you mean 'seeds'\?\)"),
        ({"data.train_fraction": 1.5}, r"data\.train_fraction: must be in \(0, 1\]"),
        ({"data.train_fraction": "ten"}, r"data\.train_fraction: expected a number"),
        ({"model.max_len": 12.5}, r"model\.max_len: expected an integer"),
        ({"model.max_len": True}, r"model\.max_len: expected an integer"),
        ({"model.add_prefix_space": "yes"}, r"model\.add_prefix_space: expected true or false"),
        ({"model.revision": ""}, r"model\.revision: expected a non-empty string"),
        (
            {"model.name": "./models/m", "model.revision": "main"},
            r"model\.revision: only applies to Hugging Face Hub models",
        ),
        ({"training.epochs": 0}, r"training\.epochs: must be positive"),
        ({"training.fp16": "maybe"}, r"training\.fp16: must be auto, true or false"),
        ({"training.load_best_model_at_end": "yes"}, r"expected true or false"),
        ({"method.lora.dropout": 1.0}, r"method\.lora\.dropout: must be in \[0, 1\)"),
        ({"method.lora.target_modules": []}, r"target_modules: expected a non-empty list"),
        ({"method.name": ""}, r"method\.name: expected a non-empty string"),
        ({"seeds": [1, 1]}, r"seeds: must not contain duplicates"),
        ({"data.max_train_samples": 0}, r"data\.max_train_samples: must be positive"),
        ({"model": "mbert"}, r"model: expected a mapping"),
    ],
)
def test_invalid_values_name_the_offending_key(overrides, message):
    with pytest.raises(ConfigError, match=message):
        cfg(**overrides)


def test_missing_required_key():
    with pytest.raises(ConfigError, match="data: required key is missing"):
        parse_config({"name": "x"})


def test_file_errors_mention_the_file(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\ndata: {dir: d}\nmodle: {}\n", encoding="utf-8")
    with pytest.raises(
        ConfigError, match=r"bad\.yaml: modle: unknown key \(did you mean 'model'\?\)"
    ):
        load_config(path)


# --- seeds and run IDs -----------------------------------------------------------------


def test_seed_list_expands_into_one_run_per_seed():
    runs = cfg(seeds=[42, 43, 44]).expand()
    assert [r.seed for r in runs] == [42, 43, 44]
    assert len({run_id(r, "tree") for r in runs}) == 3


def test_same_config_gives_the_same_id():
    a, b = cfg().expand()[0], cfg().expand()[0]
    assert run_id(a, "tree") == run_id(b, "tree")
    assert len(run_id(a, "tree")) == 12


@pytest.mark.parametrize(
    "overrides",
    [
        {"data.train_fraction": 0.25},
        {"data.subsample_seed": 7},
        {"model.max_len": 64},
        {"model.add_prefix_space": True},
        {"model.revision": "f171d7baecaf37b5da5a3616d8833b9969753535"},
        {"method.name": "lora"},
        {"method.lora.r": 16},
        {"training.learning_rate": 1e-4},
        {"training.keep_checkpoints": True},
        {"seeds": [43]},
    ],
)
def test_any_computational_field_changes_the_id(overrides):
    base = run_id(cfg().expand()[0], "tree")
    assert run_id(cfg(**overrides).expand()[0], "tree") != base


def test_name_and_output_root_do_not_change_the_id():
    base = run_id(cfg().expand()[0], "tree")
    assert run_id(cfg(name="renamed", output_root="elsewhere").expand()[0], "tree") == base


def test_data_location_does_not_change_the_id_but_data_content_does(tmp_path):
    for folder, newline in (("a", b"\n"), ("b", b"\r\n")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "train.txt").write_bytes(b"x O" + newline + newline + b"y O" + newline)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "train.txt").write_bytes(b"x B-PER\n")
    a, b, c = (data_fingerprint(tmp_path / d, ("train.txt",)) for d in "abc")
    assert a == b  # same content, other directory and CRLF line endings
    assert a != c
    run_a = cfg(**{"data.dir": "a"}).expand()[0]
    run_b = cfg(**{"data.dir": "somewhere/else"}).expand()[0]
    assert run_id(run_a, "tree", a) == run_id(run_b, "tree", b)
    assert run_id(run_a, "tree", a) != run_id(run_a, "tree", c)


def test_code_reference_changes_the_id():
    run = cfg().expand()[0]
    assert run_id(run, "tree-a") != run_id(run, "tree-b")


# --- model identity --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "local"),
    [
        ("bert-base-multilingual-cased", False),
        ("kz-transformers/kaz-roberta-conversational", False),  # org/name is a Hub ID
        ("./models/mbert-kk", True),
        ("../models/mbert-kk", True),
        (".\\models\\mbert-kk", True),
        ("/srv/models/mbert-kk", True),
        ("C:\\models\\mbert-kk", True),
        ("C:/models/mbert-kk", True),
    ],
)
def test_local_model_names_are_explicit_paths(name, local):
    assert is_local_model(name) is local


def test_local_model_is_resolved_against_the_yaml_file(tmp_path):
    path = tmp_path / "configs" / "c.yaml"
    path.parent.mkdir()
    path.write_text("name: x\ndata: {dir: d}\nmodel: {name: ../models/m}\n", encoding="utf-8")
    assert load_config(path).model_location() == str((tmp_path / "models" / "m").resolve())
    assert cfg(**{"model.name": "org/model"}).model_location() == "org/model"


def write_model(directory: Path, config_text: bytes) -> Path:
    (directory / "sub").mkdir(parents=True)
    (directory / "config.json").write_bytes(config_text)
    (directory / "model.safetensors").write_bytes(b"\x00weights\r\n")
    (directory / "sub" / "vocab.txt").write_bytes(b"[PAD]\n[UNK]\n")
    return directory


def test_model_fingerprint_ignores_location_and_line_endings(tmp_path):
    lf = write_model(tmp_path / "a", b'{\n  "hidden_size": 8\n}\n')
    crlf = write_model(tmp_path / "elsewhere" / "b", b'{\r\n  "hidden_size": 8\r\n}\r\n')
    (crlf / ".cache").mkdir()
    (crlf / ".cache" / "lock").write_bytes(b"ignored")  # dot-entries are not part of the model
    assert model_fingerprint(lf) == model_fingerprint(crlf)

    a = cfg(**{"model.name": "./a"}).expand()[0]
    b = cfg(**{"model.name": "../elsewhere/b"}).expand()[0]
    assert run_id(a, "tree", "data", model_fingerprint(lf)) == run_id(
        b, "tree", "data", model_fingerprint(crlf)
    )


def test_model_fingerprint_changes_with_content(tmp_path):
    base = model_fingerprint(write_model(tmp_path / "a", b"{}\n"))
    other = write_model(tmp_path / "b", b"{}\n")
    (other / "model.safetensors").write_bytes(b"\x00weights\n")  # binary: CRLF not normalised
    assert model_fingerprint(other) != base
    renamed = write_model(tmp_path / "c", b"{}\n")
    (renamed / "sub" / "vocab.txt").rename(renamed / "sub" / "vocab2.txt")
    assert model_fingerprint(renamed) != base


def test_missing_model_directory_names_the_key(tmp_path):
    with pytest.raises(ConfigError, match=r"model\.name: local model directory .* does not exist"):
        model_fingerprint(tmp_path / "missing")


# --- code identity on a temporary git repository ---------------------------------------


def git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src" / "kazner").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "kazner" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "docs" / "notes.md").write_text("notes\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def commit_change(repo: Path, relpath: str, text: str) -> None:
    (repo / relpath).write_text(text, encoding="utf-8")
    git(repo, "commit", "-q", "-am", f"change {relpath}")


def test_docs_only_commit_keeps_the_run_id(repo):
    run = cfg().expand()[0]
    before = code_state(repo / "src" / "kazner")
    commit_change(repo, "docs/notes.md", "more notes\n")
    after = code_state(repo / "src" / "kazner")
    assert after.commit != before.commit
    assert run_id(run, after.code_ref) == run_id(run, before.code_ref)


def test_code_commit_changes_the_run_id(repo):
    run = cfg().expand()[0]
    before = code_state(repo / "src" / "kazner")
    commit_change(repo, "src/kazner/core.py", "x = 2\n")
    after = code_state(repo / "src" / "kazner")
    assert before.source == after.source == "git-tree"
    assert run_id(run, after.code_ref) != run_id(run, before.code_ref)


def test_uncommitted_package_changes_are_flagged(repo):
    (repo / "src" / "kazner" / "core.py").write_text("x = 3\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="uncommitted changes"):
        state = code_state(repo / "src" / "kazner")
    assert (state.dirty, state.package_dirty) == (True, True)


def test_outside_git_the_package_version_is_used(tmp_path):
    state = code_state(tmp_path)
    assert state.source == "package-version"
    assert state.code_ref.startswith("kazner==")


# --- run files -------------------------------------------------------------------------


def test_run_files_record_config_and_environment(tmp_path):
    run = cfg().expand()[0]
    state = CodeState("abc", "git-tree", "def", False, False)
    env = environment(state)
    write_run_files(tmp_path / "r", run, "0123456789ab", env, "f" * 64)

    config = json.loads((tmp_path / "r" / "config.json").read_text(encoding="utf-8"))
    saved_env = json.loads((tmp_path / "r" / "environment.json").read_text(encoding="utf-8"))
    assert config["run_id"] == "0123456789ab"
    assert config["data_fingerprint"] == "f" * 64
    assert config["model_fingerprint"] is None  # a Hub model is identified by name
    assert config["seed"] == 42 and config["data"]["dir"] == "data"
    assert saved_env["code"]["code_ref"] == "abc"
    assert saved_env["packages"]["transformers"] == "5.0.0"
    assert set(saved_env) >= {"python", "platform", "cuda_available", "gpu"}
