"""KenLM n-gram language model: corpus preparation, vocabulary cap, lmplz, binary, mixing.

Pipeline (documented in README.md):
  1. prepare_corpus: raw sentences -> subtitle clean-up -> Latin sentences dropped, digits
     verbalized -> normalize() (the SAME function used for references) -> one sentence per
     line, capped length. Sources: OpenSubtitles, Wikipedia, and everyday text dumps
     (iter_text_corpus: web, forums, messages; any number of them, named "text...").
  2. build_vocab: top-N words by frequency (N = lm.vocab_size). lmplz maps the rest to <unk>.
  3. lmplz -o ORDER --prune ... --limit_vocab_file -> ARPA; build_binary trie -> .bin.
     The ARPA is what export_lm.py turns into the phone format; the binary is what the
     Python beam decoder (pyctcdecode) loads for evaluation and tuning.
  4. mix_corpora: a speaker's own text mixed into the general text by repetition
     (count-level interpolation) before rebuilding.
"""

from __future__ import annotations

import gzip
import logging
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from voicetotext.config import Config
from voicetotext.phonetics.normalize import Normalizer
from voicetotext.phonetics.numbers import has_digits, verbalize_digits

log = logging.getLogger("voicetotext.lm")

_LATIN = re.compile(r"[A-Za-z]")
_SUB_BRACKETS = re.compile(r"[\[(][^\])]*[\])]")  # [ΓΕΛΙΑ], (μουσική)
_SUB_TAGS = re.compile(r"<[^>]+>")
_SUB_DASH = re.compile(r"^\s*[-–—]+\s*")
_SUB_SPEAKER = re.compile(r"^\s*[Α-ΩΆΈΉΊΌΎΏ][Α-ΩΆΈΉΊΌΎΏ .]{1,25}:\s*")  # ΓΙΑΝΝΗΣ: ...
_SUB_NOTE = re.compile(r"[♪♫#*]")
_SENT_SPLIT = re.compile(r"(?<=[.;!?·])\s+|\n+")


def iter_opensubtitles(path: Path) -> Iterator[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:  # type: ignore[operator]
            for line in f:
                line = line.strip()
                if line:
                    yield line
    except EOFError:  # a truncated download still yields everything before the cut
        log.warning("%s: truncated gzip stream, using what was readable", path)


def iter_wikipedia_parquet(path: Path, max_articles: int | None = None) -> Iterator[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column("text").to_pylist():
            n += 1
            if max_articles and n > max_articles:
                return
            for para in text.split("\n"):
                para = para.strip()
                if len(para) < 40 or para.endswith(":"):
                    continue  # headings, lists, infobox residue
                for sent in _SENT_SPLIT.split(para):
                    sent = sent.strip()
                    if sent:
                        yield sent


def iter_text_corpus(path: Path) -> Iterator[str]:
    """Everyday text (web, forums, comments, messages): the register people actually type.

    Accepts ``.txt``/``.txt.gz`` (one paragraph or document per line), ``.jsonl``/``.jsonl.gz``
    with a ``"text"`` field (HPLT, CulturaX, OSCAR style dumps) and ``.parquet`` with a
    ``text`` column. Paragraphs are split into sentences; cleaning, digit verbalization and
    the length cap happen in ``clean_sentences`` like for every other source.
    """
    import json

    name = str(path)

    def paragraphs() -> Iterator[str]:
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    yield from (text or "").split("\n")
            return
        opener = gzip.open if name.endswith(".gz") else open
        is_json = ".jsonl" in name or ".json." in name or name.endswith(".json")
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:  # type: ignore[operator]
                for line in f:
                    if is_json:
                        try:
                            text = json.loads(line).get("text", "")
                        except (ValueError, AttributeError):
                            continue
                        yield from text.split("\n")
                    else:
                        yield line
        except EOFError:
            log.warning("%s: truncated gzip stream, using what was readable", path)

    for para in paragraphs():
        para = para.strip()
        if len(para) < 20:
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if sent:
                yield sent


def clean_subtitle_line(line: str) -> str:
    """Subtitle-specific pass before normalize(): stage notes, tags, speaker dashes/names, music marks."""
    line = _SUB_TAGS.sub(" ", line)
    line = _SUB_BRACKETS.sub(" ", line)
    if _SUB_NOTE.search(line):
        return ""
    line = _SUB_DASH.sub("", line)
    line = _SUB_SPEAKER.sub("", line)
    if line.strip().endswith(":"):
        return ""
    return line.strip()


class CleanStats:
    def __init__(self) -> None:
        self.raw = self.kept = self.dropped_latin = self.dropped_digits = self.dropped_empty = self.dropped_long = self.verbalized = 0

    def as_dict(self) -> dict[str, float]:
        d = {k: v for k, v in self.__dict__.items()}
        if self.raw:
            d["kept_frac"] = self.kept / self.raw
            d["dropped_latin_frac"] = self.dropped_latin / self.raw
            d["digit_sentences_frac"] = (self.dropped_digits + self.verbalized) / self.raw
        return d


def clean_sentences(raw: Iterable[str], norm: Normalizer, *, digits: str, drop_latin: bool, max_words: int,
                    subtitles: bool = False, stats: CleanStats | None = None) -> Iterator[str]:
    """digits: 'verbalize' (spoken Greek forms) | 'drop' (whole sentence removed). Latin -> sentence dropped."""
    stats = stats or CleanStats()
    for s in raw:
        stats.raw += 1
        if subtitles:
            s = clean_subtitle_line(s)
        if drop_latin and _LATIN.search(s):
            stats.dropped_latin += 1
            continue
        if has_digits(s):
            if digits == "verbalize":
                s = verbalize_digits(s)
                stats.verbalized += 1
            else:
                stats.dropped_digits += 1
                continue
        t = norm(s)
        if not t:
            stats.dropped_empty += 1
            continue
        if t.count(" ") + 1 > max_words:
            stats.dropped_long += 1
            continue
        stats.kept += 1
        yield t


def prepare_corpus(cfg: Config, norm: Normalizer, sources: dict[str, Path], out_path: Path, wiki_max_articles: int | None = None) -> dict[str, dict]:
    """Write the normalized training text; returns per-source sentence counts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for name, path in sources.items():
            if name == "wikipedia":
                raw = iter_wikipedia_parquet(path, wiki_max_articles)
            elif name.startswith("text"):
                raw = iter_text_corpus(path)
            else:
                raw = iter_opensubtitles(path)
            n = 0
            # a "text..." source takes its own cap if set, else the shared "text" one
            cap = cfg.lm.max_sentences.get(name, cfg.lm.max_sentences.get("text") if name.startswith("text") else None)
            st = CleanStats()
            for sent in clean_sentences(raw, norm, digits=cfg.lm.digits, drop_latin=cfg.lm.drop_latin_sentences,
                                        max_words=cfg.lm.max_words_per_sentence, subtitles=(name == "opensubtitles"), stats=st):
                out.write(sent + "\n")
                n += 1
                if cap and n >= cap:
                    break
            counts[name] = st.as_dict()
            log.info("%s: %s", name, counts[name])
    return counts


def build_vocab(text_path: Path, vocab_path: Path, size: int) -> tuple[int, float]:
    """Top-``size`` words by frequency. Returns (types_total, coverage of tokens by the cap)."""
    c: Counter[str] = Counter()
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            c.update(line.split())
    total = sum(c.values())
    top = c.most_common(size)
    vocab_path.write_text("\n".join(w for w, _ in top) + "\n", encoding="utf-8")
    coverage = sum(n for _, n in top) / total if total else 0.0
    log.info("vocab: %d types, kept %d, token coverage %.4f", len(c), len(top), coverage)
    return len(c), coverage


def _run(cmd: list[str], **kw) -> None:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build_arpa(cfg: Config, text_path: Path, vocab_path: Path, arpa_path: Path, tmp_dir: Path | None = None) -> Path:
    lmplz = cfg.paths.kenlm / "lmplz"
    if not lmplz.exists():
        raise FileNotFoundError(f"{lmplz} not found; build KenLM (see README.md)")
    tmp_dir = tmp_dir or arpa_path.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [lmplz, "-o", cfg.lm.order, "--prune", *cfg.lm.prune.split(), "--limit_vocab_file", vocab_path,
           "--discount_fallback", "-S", "40%", "-T", tmp_dir, "--text", text_path, "--arpa", arpa_path]
    _run(cmd)
    return arpa_path


def build_binary(cfg: Config, arpa_path: Path, bin_path: Path) -> Path:
    bb = cfg.paths.kenlm / "build_binary"
    q = str(cfg.lm.quantize_bits)
    _run([bb, "-q", q, "-b", q, "-a", "22", "trie", arpa_path, bin_path])
    return bin_path


def load_lm(path: str | Path):
    import kenlm

    return kenlm.Model(str(path))


def perplexity(model, sentences: Iterable[str]) -> float:
    """Word-level perplexity including </s>; sentences already normalized."""
    logp, n = 0.0, 0
    for s in sentences:
        if not s:
            continue
        logp += model.score(s, bos=True, eos=True)  # log10
        n += len(s.split()) + 1
    return 10 ** (-logp / n) if n else float("inf")


def oov_rate(vocab: set[str], sentences: Iterable[str]) -> float:
    total = oov = 0
    for s in sentences:
        for w in s.split():
            total += 1
            oov += w not in vocab
    return oov / total if total else 0.0


def mix_corpora(general_txt: Path, personal_txt: Path, out_txt: Path, personal_weight: int) -> Path:
    """Count-level interpolation: the personal sentences are repeated ``personal_weight`` times
    after the general text, and the LM is rebuilt from the result. A speaker's own phrases then
    carry more weight without a separate model at decoding time."""
    with open(out_txt, "w", encoding="utf-8") as out:
        with open(general_txt, encoding="utf-8") as f:
            for line in f:
                out.write(line)
        personal = [l for l in open(personal_txt, encoding="utf-8") if l.strip()]
        for _ in range(personal_weight):
            out.writelines(personal)
    return out_txt


def build_general_lm(cfg: Config, norm: Normalizer, sources: dict[str, Path], name: str = "general_el", wiki_max_articles: int | None = None) -> dict:
    """End-to-end: corpus -> vocab -> ARPA -> binary under <lm_dir>. Returns a report dict."""
    lm_dir = cfg.paths.lm
    lm_dir.mkdir(parents=True, exist_ok=True)
    text, vocab, arpa, binp = lm_dir / f"{name}.txt", lm_dir / f"{name}.vocab", lm_dir / f"{name}.arpa", lm_dir / f"{name}.bin"
    counts = prepare_corpus(cfg, norm, sources, text, wiki_max_articles)
    types, coverage = build_vocab(text, vocab, cfg.lm.vocab_size)
    build_arpa(cfg, text, vocab, arpa)
    build_binary(cfg, arpa, binp)
    report = {
        "name": name, "sentences": counts, "types": types, "vocab_size": cfg.lm.vocab_size, "token_coverage": coverage,
        "order": cfg.lm.order, "prune": cfg.lm.prune, "arpa_mb": arpa.stat().st_size / 1e6, "bin_mb": binp.stat().st_size / 1e6,
        "text_mb": text.stat().st_size / 1e6, "arpa_path": str(arpa), "bin_path": str(binp),
    }
    log.info("LM report: %s", report)
    return report


def word_counts_of(text_path: Path) -> Counter:
    c: Counter[str] = Counter()
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            c.update(line.split())
    return c


def oov_report(counts: Counter, sizes: list[int], ref_sets: dict[str, list[str]]) -> list[dict]:
    """OOV rate of the top-N vocabulary on each reference set (normalized), for several N."""
    ranked = [w for w, _ in counts.most_common(max(sizes))]
    total_tokens = sum(counts.values())
    rows = []
    for n in sizes:
        vocab = set(ranked[:n])
        row = {"vocab_size": n, "types_total": len(counts), "token_coverage": sum(counts[w] for w in vocab) / total_tokens}
        for name, refs in ref_sets.items():
            row[f"oov_{name}"] = oov_rate(vocab, refs)
        rows.append(row)
    return rows
