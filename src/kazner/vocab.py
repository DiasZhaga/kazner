"""Vocabulary extension for WordPiece models (#35): Kazakh pieces appended to mBERT's vocabulary.

1. **Candidates.** A WordPiece vocabulary is trained on the unlabeled corpus (#34) with the
   base tokenizer's own normalizer and pre-tokenizer. A learned piece is skipped if the base
   vocabulary already has it, if it is a single character (word-initial or ``##``), if the
   base vocabulary cannot spell it without ``[UNK]``, or if the learned tokenizer never uses
   it on the corpus. The extended tokenizer therefore marks exactly the same words as
   ``[UNK]`` as the base tokenizer: only fragmentation changes. Candidates are ranked by how
   often the learned tokenizer uses them on the corpus, ties by the piece itself. The trainer
   breaks ties between equally frequent pairs in hash order, so the learned vocabulary can
   differ slightly between runs; ``new_pieces.txt`` of an extension is therefore the
   authoritative list, and ``pieces=`` rebuilds an extension from it exactly.
2. **Extension.** The first N candidates are appended to the WordPiece vocabulary itself, with
   ids ``len(base) ... len(base) + N - 1``; every old id is unchanged. This is not
   ``tokenizer.add_tokens``: added tokens are matched as raw strings before WordPiece runs and
   would split words in the wrong places.
3. **Initialisation (FVT, Gee et al. 2022).** Each new row of the input embeddings (tied with
   the masked-LM decoder) is the mean of the rows of the base pieces that spell the new piece;
   the masked-LM output bias likewise.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PREFIX = "##"
DEFAULT_SIZES = (5000, 10000, 20000, 30000)
#: Base pieces per word from which a word counts as strongly fragmented in the report.
FRAGMENTED = 4


def spell(piece: str, vocab: Mapping[str, int], max_chars: int = 100) -> list[str] | None:
    """Greedy longest-match-first segmentation of ``piece`` with ``vocab``, as WordPiece does.

    A ``##`` piece is spelled with continuation pieces only. ``None`` if the vocabulary cannot
    spell it (WordPiece would produce ``[UNK]``).
    """
    continuation = piece.startswith(PREFIX)
    text = piece[len(PREFIX) :] if continuation else piece
    if not text or len(text) > max_chars:
        return None
    pieces, start = [], 0
    while start < len(text):
        end = len(text)
        while end > start:
            sub = text[start:end] if start == 0 and not continuation else PREFIX + text[start:end]
            if sub in vocab:
                break
            end -= 1
        else:
            return None
        pieces.append(sub)
        start = end
    return pieces


def base_vocab(tokenizer: Any) -> dict[str, int]:
    """The WordPiece vocabulary of a fast tokenizer; its ids must be 0 ... V-1."""
    vocab = tokenizer.backend_tokenizer.get_vocab(with_added_tokens=False)
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("the base vocabulary ids are not contiguous from 0")
    return vocab


def learn_pieces(
    tokenizer: Any, lines: Iterable[str], vocab_size: int = 60000, min_frequency: int = 5
) -> Any:
    """A WordPiece ``tokenizers.Tokenizer`` trained on ``lines`` with the base tokenizer's
    normalizer and pre-tokenizer."""
    from tokenizers import Tokenizer, models, trainers

    backend = tokenizer.backend_tokenizer
    unk = backend.model.unk_token
    learner = Tokenizer(models.WordPiece(unk_token=unk, continuing_subword_prefix=PREFIX))
    learner.normalizer = backend.normalizer
    learner.pre_tokenizer = backend.pre_tokenizer
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[unk],
        continuing_subword_prefix=PREFIX,
        show_progress=False,
    )
    learner.train_from_iterator(lines, trainer)
    return learner


def usage_counts(learner: Any, lines: Iterable[str]) -> Counter[str]:
    """How often ``learner`` uses each of its pieces when it tokenizes ``lines``."""
    counts = np.zeros(learner.get_vocab_size(), dtype=np.int64)
    batch: list[str] = []

    def flush() -> None:
        for encoding in learner.encode_batch(batch):
            counts[:] += np.bincount(encoding.ids, minlength=len(counts))[: len(counts)]
        batch.clear()

    for line in lines:
        batch.append(line)
        if len(batch) == 5000:
            flush()
    if batch:
        flush()
    return Counter({piece: int(counts[i]) for piece, i in learner.get_vocab().items()})


def select_candidates(
    used: Mapping[str, int], vocab: Mapping[str, int]
) -> tuple[list[str], dict[str, int]]:
    """New, multi-character, spellable, used pieces, most used first (ties by the piece), and
    why the others were skipped."""
    counts: Counter[str] = Counter(learned=len(used))
    candidates = []
    for piece, n in used.items():
        body = piece[len(PREFIX) :] if piece.startswith(PREFIX) else piece
        if piece in vocab:
            counts["in_base"] += 1
        elif len(body) < 2:
            counts["single_character"] += 1
        elif spell(piece, vocab) is None:
            counts["needs_unk"] += 1
        elif n == 0:
            counts["unused"] += 1
        else:
            candidates.append(piece)
    candidates.sort(key=lambda p: (-used[p], p))
    counts["candidates"] = len(candidates)
    return candidates, dict(counts)


def learn_candidates(
    tokenizer: Any, lines: Sequence[str] | Path, vocab_size: int = 60000, min_frequency: int = 5
) -> tuple[list[str], dict[str, int]]:
    """Learn pieces on a corpus (a list of lines or a text file) and select the candidates."""
    learner = learn_pieces(tokenizer, _lines(lines), vocab_size, min_frequency)
    return select_candidates(usage_counts(learner, _lines(lines)), base_vocab(tokenizer))


def _lines(source: Sequence[str] | Path) -> Iterable[str]:
    return read_lines(source) if isinstance(source, Path) else iter(source)


def extend_tokenizer(tokenizer: Any, pieces: Sequence[str]) -> Any:
    """A copy of ``tokenizer`` whose WordPiece vocabulary has ``pieces`` appended."""
    from tokenizers import models

    vocab = base_vocab(tokenizer)
    clash = [p for p in pieces if p in vocab]
    if clash or len(set(pieces)) != len(pieces):
        raise ValueError(f"pieces must be new and unique; already present: {clash[:5]}")
    extended = copy.deepcopy(tokenizer)
    backend = extended.backend_tokenizer
    old = backend.model
    backend.model = models.WordPiece(
        vocab={**vocab, **{p: len(vocab) + i for i, p in enumerate(pieces)}},
        unk_token=old.unk_token,
        continuing_subword_prefix=old.continuing_subword_prefix,
        max_input_chars_per_word=old.max_input_chars_per_word,
    )
    return extended


def fvt_sources(pieces: Sequence[str], vocab: Mapping[str, int]) -> list[list[int]]:
    """For every new piece, the ids of the base pieces that spell it."""
    sources = []
    for piece in pieces:
        spelled = spell(piece, vocab)
        if spelled is None:
            raise ValueError(f"{piece!r} cannot be spelled with the base vocabulary")
        sources.append([vocab[p] for p in spelled])
    return sources


def extend_model(model: Any, sources: Sequence[Sequence[int]]) -> Any:
    """Append one row per new piece to the embeddings and initialise it by FVT."""
    import torch

    old = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(old + len(sources), mean_resizing=False)
    embeddings = model.get_input_embeddings().weight
    output = model.get_output_embeddings()
    tied = output is not None and output.weight.data_ptr() == embeddings.data_ptr()
    with torch.no_grad():
        for i, ids in enumerate(sources):
            index = torch.tensor(ids)
            embeddings[old + i] = embeddings[index].mean(dim=0)
            if output is not None and not tied:
                output.weight[old + i] = output.weight[index].mean(dim=0)
            if output is not None and output.bias is not None:
                output.bias[old + i] = output.bias[index].mean()
    return model


# --------------------------------------------------------------------------- report


def word_pieces(
    tokenizer: Any, sentences: Sequence[Sequence[str]]
) -> tuple[np.ndarray, np.ndarray]:
    """Pieces per word and whether the word contains ``[UNK]``, for pre-split sentences."""
    unk = tokenizer.unk_token_id
    pieces, unknown = [], []
    for start in range(0, len(sentences), 2000):
        batch = [list(s) for s in sentences[start : start + 2000]]
        enc = tokenizer(batch, is_split_into_words=True, add_special_tokens=False)
        for i, words in enumerate(batch):
            counts = np.zeros(len(words), dtype=int)
            has_unk = np.zeros(len(words), dtype=bool)
            for word_id, token in zip(
                enc.word_ids(batch_index=i), enc["input_ids"][i], strict=True
            ):
                counts[word_id] += 1
                has_unk[word_id] |= token == unk
            pieces.append(counts)
            unknown.append(has_unk)
    return np.concatenate(pieces), np.concatenate(unknown)


def fertility_report(
    base: Any, extended: Any, sentences: Sequence[Sequence[str]], labels: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Pieces per word before and after, for all words, entity words and entity words with at
    least :data:`FRAGMENTED` base pieces, and the share of words containing ``[UNK]``."""
    before, unk_before = word_pieces(base, sentences)
    after, unk_after = word_pieces(extended, sentences)
    entity = np.array([tag != "O" for tags in labels for tag in tags])
    fragmented = entity & (before >= FRAGMENTED)
    report: dict[str, Any] = {"words": int(len(before))}
    for name, sel in (("all", slice(None)), ("entity", entity), ("fragmented_entity", fragmented)):
        n = int(before[sel].size)
        report[name] = {
            "n": n,
            "before": float(before[sel].mean()) if n else None,
            "after": float(after[sel].mean()) if n else None,
        }
    report["unk_words"] = {"before": float(unk_before.mean()), "after": float(unk_after.mean())}
    return report


def piece_usage(
    extended: Any, lines: Iterable[str], first_new_id: int, n_new: int
) -> dict[str, Any]:
    """How often each new piece occurs when ``lines`` are tokenized with the extended tokenizer."""
    counts = np.zeros(n_new, dtype=np.int64)
    batch: list[str] = []

    def flush() -> None:
        for ids in extended(batch, add_special_tokens=False)["input_ids"]:
            new = np.asarray(ids) - first_new_id
            counts[:] += np.bincount(new[new >= 0], minlength=n_new)[:n_new]
        batch.clear()

    for line in lines:
        batch.append(line)
        if len(batch) == 5000:
            flush()
    if batch:
        flush()
    return {
        "min": int(counts.min()) if n_new else 0,
        "p10": float(np.percentile(counts, 10)) if n_new else 0.0,
        "median": float(np.median(counts)) if n_new else 0.0,
        "below_100": int((counts < 100).sum()),
        "unused": int((counts == 0).sum()),
    }


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pieces(path: str | Path) -> list[str]:
    """A ``new_pieces.txt``: one piece per line, in order."""
    return [p for p in Path(path).read_text(encoding="utf-8").splitlines() if p]


def read_lines(path: str | Path) -> Iterable[str]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def extend_vocabulary(
    base_name: str,
    corpus: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    size: int | None = None,
    sizes: Sequence[int] = DEFAULT_SIZES,
    revision: str | None = None,
    vocab_size: int = 60000,
    min_frequency: int = 5,
    pieces: Sequence[str] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Learn candidates (or take ``pieces``, e.g. a saved ``new_pieces.txt``), report every
    size on KazNERD *validation* and on the corpus, and, if ``size`` is given, write the
    extended tokenizer and model to ``out_dir``.

    Without ``size`` only ``report.json`` is written: N is chosen from it, never from test data.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from kazner.config import data_fingerprint
    from kazner.data import read_splits

    kwargs = {} if revision is None else {"revision": revision}
    base = AutoTokenizer.from_pretrained(base_name, **kwargs)
    vocab = base_vocab(base)
    corpus = Path(corpus)
    if pieces is None:
        candidates, counts = learn_candidates(base, corpus, vocab_size, min_frequency)
        learner: dict[str, Any] = {"vocab_size": vocab_size, "min_frequency": min_frequency}
    else:
        candidates, counts = list(pieces), {"given": len(pieces)}
        fvt_sources(candidates, vocab)  # spellable; newness is checked by extend_tokenizer
        learner = {"pieces_given": True}
    log(f"{len(candidates):,} candidates ({counts})")
    wanted = sorted({*sizes, *([size] if size is not None else [])})
    if wanted and wanted[-1] > len(candidates):
        raise ValueError(f"only {len(candidates)} candidates for sizes {wanted}")

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    validation = read_splits(data_dir)["validation"]
    reports = {}
    for n in wanted:
        extended = extend_tokenizer(base, candidates[:n])
        report = fertility_report(base, extended, *validation)
        report["corpus_usage"] = piece_usage(extended, read_lines(corpus), len(vocab), n)
        reports[str(n)] = report
        frag = report["fragmented_entity"]
        log(
            f"N={n:>6}: pieces/word {number(report['all']['before'])} -> "
            f"{number(report['all']['after'])}, entity {number(report['entity']['after'])}, "
            f"fragmented entity {number(frag['before'])} -> {number(frag['after'])}, [UNK] "
            f"words {report['unk_words']['after']:.4%}, new pieces used < 100 times "
            f"{report['corpus_usage']['below_100']:,}"
        )

    manifest: dict[str, Any] = {
        "base": {"name": base_name, "revision": revision, "vocab_size": len(vocab)},
        "corpus": {"path": str(corpus), "sha256": sha256_of(corpus)},
        "kaznerd": {"dir": str(data_dir), "fingerprint": data_fingerprint(data_dir)},
        "learner": {**learner, **counts},
        "reports_on_validation": reports,
    }
    out_dir = Path(out_dir)
    if size is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        (out_dir / "report.json").write_text(text, encoding="utf-8", newline="\n")
        return manifest

    pieces = candidates[:size]
    model = AutoModelForMaskedLM.from_pretrained(base_name, **kwargs)
    extend_model(model, fvt_sources(pieces, vocab))
    extended = extend_tokenizer(base, pieces)
    manifest["extension"] = {
        "size": size,
        "first_new_id": len(vocab),
        "vocab_size": len(vocab) + size,
        "initialisation": "FVT: mean of the base pieces that spell the new piece",
    }
    write_extension(out_dir, extended, model, pieces, manifest)
    log(f"extended tokenizer and model ({len(vocab):,} + {size:,} pieces) -> {out_dir}")
    return manifest


def write_extension(
    out_dir: str | Path, tokenizer: Any, model: Any, pieces: Sequence[str], manifest: dict[str, Any]
) -> Path:
    """Save the extended tokenizer and model, ``new_pieces.txt`` and ``extension.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    (out_dir / "new_pieces.txt").write_text(
        "".join(p + "\n" for p in pieces), encoding="utf-8", newline="\n"
    )
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "extension.json").write_text(text, encoding="utf-8", newline="\n")
    return out_dir
