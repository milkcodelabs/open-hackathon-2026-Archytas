"""Text normalization into exactly the character set of the CTC vocabulary.

This is the single source of truth. It is applied identically to LM training text,
references and hypotheses. A mismatch here silently destroys beam-search quality, so the
spec is *derived from the label set*, never assumed.

Pipeline: NFC -> lowercase -> polytonic-to-monotonic -> optional accent stripping ->
per-character fitting into the alphabet (progressively dropping marks, else space) ->
final-sigma rule -> whitespace collapse.

Property: normalize(normalize(x)) == normalize(x).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from voicetotext.decoding.vocab import COMBINING_ACUTE, COMBINING_DIAERESIS, greek_letter_set

# Marks seen in Greek text. Everything that carries stress maps to the monotonic acute.
_STRESS_MARKS = {"\u0300", "\u0301", "\u0342", "\u0344"}  # varia, oxia/tonos, perispomeni, dialytika+oxia
_DROP_MARKS = {"\u0313", "\u0314", "\u0345", "\u0343"}  # psili, dasia, ypogegrammeni, koronis
_WS = re.compile(r"\s+")
_FINAL_SIGMA = re.compile(r"σ(?= |$)")


@dataclass(frozen=True)
class NormalizeSpec:
    alphabet: frozenset[str]  # single characters the output may contain, besides space
    strip_accents: bool
    keep_dialytika: bool
    final_sigma: bool  # vocabulary has ς
    # Case follows the vocabulary, it is not a preference: a CTC vocabulary can be
    # uppercase only. Lowercasing text for an uppercase vocabulary would map every letter
    # to space and silently produce empty hypotheses.
    lowercase: bool = True
    uppercase: bool = False

    @property
    def sorted_alphabet(self) -> list[str]:
        return sorted(self.alphabet)


def spec_from_labels(labels: list[str], strip_accents: bool | None = None) -> NormalizeSpec:
    # Greek letters only: apostrophes and any other non-letter labels map to space for LM
    # training and for scoring.
    chars = greek_letter_set(labels)
    lower = {c for c in chars if c.islower()}
    upper = {c for c in chars if c.isupper()}
    is_upper_vocab = len(upper) > len(lower)
    has_accents = any(COMBINING_ACUTE in unicodedata.normalize("NFD", c) for c in chars)
    has_dialytika = any(COMBINING_DIAERESIS in unicodedata.normalize("NFD", c) for c in chars)
    strip = (not has_accents) if strip_accents is None else bool(strip_accents)
    if strip:
        # never emit accented forms even if the vocab has them
        chars = {c for c in chars if COMBINING_ACUTE not in unicodedata.normalize("NFD", c)}
    return NormalizeSpec(
        alphabet=frozenset(chars),
        strip_accents=strip,
        keep_dialytika=has_dialytika,
        final_sigma="ς" in chars,
        lowercase=not is_upper_vocab,
        uppercase=is_upper_vocab,
    )


def _monotonic(text: str) -> str:
    """Polytonic -> monotonic: any stress mark becomes the acute; breathings etc. vanish."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if ch in _STRESS_MARKS:
            out.append(COMBINING_ACUTE)
        elif ch in _DROP_MARKS:
            continue
        else:
            out.append(ch)
    return "".join(out)


def _fit_char(ch: str, spec: NormalizeSpec) -> str:
    """Map one NFC character into the alphabet, dropping marks progressively; else space."""
    if ch in spec.alphabet:
        return ch
    if ch.isspace():
        return " "
    decomposed = unicodedata.normalize("NFD", ch)
    base = decomposed[0]
    marks = [m for m in decomposed[1:] if unicodedata.category(m) == "Mn"]
    if not marks:
        return " "
    # try: without acute, without diaeresis, bare base
    candidates = []
    if COMBINING_ACUTE in marks:
        candidates.append(unicodedata.normalize("NFC", base + "".join(m for m in marks if m != COMBINING_ACUTE)))
    if COMBINING_DIAERESIS in marks:
        candidates.append(unicodedata.normalize("NFC", base + "".join(m for m in marks if m != COMBINING_DIAERESIS)))
    candidates.append(base)
    for c in candidates:
        if c in spec.alphabet:
            return c
    return " "


@lru_cache(maxsize=64)
def _fitter(spec: NormalizeSpec):
    return lru_cache(maxsize=4096)(lambda ch: _fit_char(ch, spec))


def normalize(text: str, spec: NormalizeSpec) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    if spec.uppercase:
        t = t.upper()
    elif spec.lowercase:
        t = t.lower()
    t = _monotonic(t)
    if spec.strip_accents:
        t = t.replace(COMBINING_ACUTE, "")
    if not spec.keep_dialytika:
        t = t.replace(COMBINING_DIAERESIS, "")
    t = unicodedata.normalize("NFC", t)
    fit = _fitter(spec)
    t = "".join(fit(ch) for ch in t)
    t = _WS.sub(" ", t).strip()
    if spec.final_sigma:
        t = _FINAL_SIGMA.sub("ς", t)
    elif not spec.uppercase:
        t = t.replace("ς", "σ")
    return t


class Normalizer:
    """Callable bound to a spec; what the pipeline and harness hand around."""

    def __init__(self, spec: NormalizeSpec) -> None:
        self.spec = spec

    @classmethod
    def from_labels(cls, labels: list[str], strip_accents: bool | None = None) -> "Normalizer":
        return cls(spec_from_labels(labels, strip_accents))

    def __call__(self, text: str) -> str:
        return normalize(text, self.spec)
