"""WER / CER with substitution, deletion and insertion counts, plus breakdown helpers.

CER matters more than WER here: at high error rates WER saturates and stops discriminating
between changes. Report both, always. Corpus-level rates are total errors / total reference
length (not the mean of per-utterance rates).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class ErrorCounts:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0
    hyp_len: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            return 0.0 if self.hyp_len == 0 else float(self.hyp_len)
        return self.errors / self.ref_len

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.sub + other.sub,
            self.dele + other.dele,
            self.ins + other.ins,
            self.ref_len + other.ref_len,
            self.hyp_len + other.hyp_len,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.sub, "del": self.dele, "ins": self.ins,
            "ref_len": self.ref_len, "hyp_len": self.hyp_len,
            "errors": self.errors, "rate": self.rate,
        }


def edit_ops(ref: Sequence[T], hyp: Sequence[T]) -> ErrorCounts:
    """Levenshtein alignment counts (sub/del/ins). O(len(ref) * len(hyp)), pure Python."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return ErrorCounts(ins=m, ref_len=0, hyp_len=m)
    if m == 0:
        return ErrorCounts(dele=n, ref_len=n, hyp_len=0)
    # dp cells hold (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m  # type: ignore[list-item]
        ri = ref[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                c = prev[j - 1]
                cur[j] = c
                continue
            s = prev[j - 1]
            d = prev[j]
            a = cur[j - 1]
            best = s
            kind = 0
            if d[0] < best[0]:
                best, kind = d, 1
            if a[0] < best[0]:
                best, kind = a, 2
            if kind == 0:
                cur[j] = (best[0] + 1, best[1] + 1, best[2], best[3])
            elif kind == 1:
                cur[j] = (best[0] + 1, best[1], best[2] + 1, best[3])
            else:
                cur[j] = (best[0] + 1, best[1], best[2], best[3] + 1)
        prev = cur  # type: ignore[assignment]
    cost, sub, dele, ins = prev[m]
    return ErrorCounts(sub=sub, dele=dele, ins=ins, ref_len=n, hyp_len=m)


def word_counts(ref: str, hyp: str) -> ErrorCounts:
    return edit_ops(ref.split(), hyp.split())


def char_counts(ref: str, hyp: str) -> ErrorCounts:
    """Character-level counts; spaces count as characters (as in jiwer.cer)."""
    return edit_ops(list(ref), list(hyp))


def wer(ref: str, hyp: str) -> float:
    return word_counts(ref, hyp).rate


def cer(ref: str, hyp: str) -> float:
    return char_counts(ref, hyp).rate


def sum_counts(counts: Iterable[ErrorCounts]) -> ErrorCounts:
    total = ErrorCounts()
    for c in counts:
        total = total + c
    return total


LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16+", 16, 10**9),
)


def length_bucket(n_words: int) -> str:
    if n_words <= 0:
        return "0"
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n_words <= hi:
            return name
    return "16+"


def breakdown(
    rows: Iterable[Any],
    key: Callable[[Any], str],
    words: Callable[[Any], ErrorCounts],
    chars: Callable[[Any], ErrorCounts],
) -> dict[str, dict[str, Any]]:
    """Group rows by ``key`` and report n / WER / CER per group (corpus-level within group)."""
    groups: dict[str, list[Any]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for g, rs in groups.items():
        w = sum_counts(words(r) for r in rs)
        c = sum_counts(chars(r) for r in rs)
        out[g] = {"n": len(rs), "wer": w.rate, "cer": c.rate, "ref_words": w.ref_len, "ref_chars": c.ref_len}
    return dict(sorted(out.items(), key=lambda kv: _bucket_order(kv[0])))


def _bucket_order(name: str) -> tuple[int, str]:
    names = [b[0] for b in LENGTH_BUCKETS]
    return (names.index(name), name) if name in names else (len(names), name)


# ----------------------------------------------------------------------------- oracle spelling
# Greek has many spellings per sound (ι η υ ει οι υι -> /i/, ο ω -> /o/, αι ε -> /e/, plus stress).
# Collapsing them before CER gives the "oracle-spelling CER": the error that remains even if
# every homophonous spelling were fixed. The gap between plain CER and this number is the
# ceiling an LM / speller can recover; the remainder is acoustic.
_ORACLE_MAP: tuple[tuple[str, str], ...] = (
    ("ει", "i"), ("οι", "i"), ("υι", "i"), ("αι", "e"), ("ου", "u"),
    ("ι", "i"), ("η", "i"), ("υ", "i"), ("ο", "o"), ("ω", "o"), ("ε", "e"), ("ς", "σ"),
)


def oracle_spelling(text: str) -> str:
    """Collapse homophonous Greek spellings and drop stress marks (input: normalized text).

    Case-folded first: the digraph table below is lowercase, and without folding none of
    the substitutions would match uppercase text, so the oracle CER would collapse onto the
    plain CER.
    """
    import unicodedata

    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for a, b in _ORACLE_MAP:
        s = s.replace(a, b)
    return s


def oracle_char_counts(ref: str, hyp: str) -> ErrorCounts:
    return char_counts(oracle_spelling(ref), oracle_spelling(hyp))
