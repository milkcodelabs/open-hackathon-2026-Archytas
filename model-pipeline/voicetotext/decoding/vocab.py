"""CTC label handling.

* Inspection report: what the label set actually contains (accents, case, punctuation,
  blank / delimiter tokens). Nothing downstream assumes; it reads this.
* Conversion to pyctcdecode's alphabet convention.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The CTC blank is written "<pad>" in our label files ("[PAD]" is accepted as well).
BLANK_TOKENS: tuple[str, ...] = ("<pad>", "[PAD]")
UNK_TOKENS: tuple[str, ...] = ("<unk>", "[UNK]")
BOS_EOS_TOKENS: tuple[str, ...] = ("<s>", "</s>", "[BOS]", "[EOS]")
WORD_DELIMITER = "|"
SPECIAL_TOKENS: tuple[str, ...] = UNK_TOKENS + BOS_EOS_TOKENS
UNUSED_PREFIX = "<unused"
BLANK_TOKEN = BLANK_TOKENS[0]  # canonical name used in reports

COMBINING_ACUTE = "\u0301"
COMBINING_DIAERESIS = "\u0308"


def save_labels(labels: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=0), encoding="utf-8")
    return path


def load_labels(path: str | Path) -> list[str]:
    return list(json.loads(Path(path).read_text(encoding="utf-8")))


def blank_index(labels: list[str]) -> int:
    for tok in BLANK_TOKENS:
        if tok in labels:
            return labels.index(tok)
    raise ValueError(f"no blank token {BLANK_TOKENS} in labels")


def is_blank(label: str) -> bool:
    return label in BLANK_TOKENS


def is_special(label: str) -> bool:
    """Blank, unk, bos/eos or unused-id filler: never part of the text."""
    return label in BLANK_TOKENS or label in SPECIAL_TOKENS or label.startswith(UNUSED_PREFIX)


def is_greek_letter(ch: str) -> bool:
    return is_letter(ch) and "GREEK" in unicodedata.name(ch, "")


def greek_letter_set(labels: list[str]) -> set[str]:
    """Greek letter labels only; anything else in a label set maps to space in normalized text."""
    return {l for l in labels if is_greek_letter(l)}


def delimiter_index(labels: list[str]) -> int | None:
    return labels.index(WORD_DELIMITER) if WORD_DELIMITER in labels else None


def is_letter(ch: str) -> bool:
    return len(ch) == 1 and unicodedata.category(ch).startswith("L")


def letter_set(labels: list[str]) -> set[str]:
    """Single-character letter labels: the characters the model can actually emit."""
    return {l for l in labels if is_letter(l)}


def has_mark(ch: str, mark: str) -> bool:
    return mark in unicodedata.normalize("NFD", ch)


@dataclass
class VocabReport:
    size: int
    blank: str | None
    delimiter: str | None
    specials: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    letters: list[str] = field(default_factory=list)
    accented: list[str] = field(default_factory=list)
    dialytika: list[str] = field(default_factory=list)
    uppercase: list[str] = field(default_factory=list)
    digits: list[str] = field(default_factory=list)
    punctuation: list[str] = field(default_factory=list)
    latin: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    has_final_sigma: bool = False
    has_space: bool = False

    def format(self) -> str:
        def j(xs: list[str]) -> str:
            return " ".join(xs) if xs else "(none)"

        lines = [
            f"vocab size           : {self.size}",
            f"blank token          : {self.blank!r}",
            f"word delimiter       : {self.delimiter!r}",
            f"special tokens       : {j(self.specials)}",
            f"unused ids           : {len(self.unused)}",
            f"letters ({len(self.letters):>3})        : {j(self.letters)}",
            f"accented vowels      : {j(self.accented)}",
            f"dialytika            : {j(self.dialytika)}",
            f"uppercase            : {j(self.uppercase)}",
            f"final sigma (ς)      : {self.has_final_sigma}",
            f"digits               : {j(self.digits)}",
            f"punctuation/symbols  : {j(self.punctuation)}",
            f"latin letters        : {j(self.latin)}",
            f"literal space label  : {self.has_space}",
            f"other                : {j(self.other)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def inspect_labels(labels: list[str]) -> VocabReport:
    r = VocabReport(size=len(labels), blank=None, delimiter=None)
    for lab in labels:
        if lab in BLANK_TOKENS:
            r.blank = lab
        elif lab == WORD_DELIMITER:
            r.delimiter = lab
        elif lab in SPECIAL_TOKENS:
            r.specials.append(lab)
        elif lab.startswith(UNUSED_PREFIX):
            r.unused.append(lab)
        elif lab == " ":
            r.has_space = True
        elif len(lab) == 1:
            cat = unicodedata.category(lab)
            if cat.startswith("L"):
                if "LATIN" in unicodedata.name(lab, ""):
                    r.latin.append(lab)
                    continue
                r.letters.append(lab)
                if lab.isupper():
                    r.uppercase.append(lab)
                if has_mark(lab, COMBINING_ACUTE):
                    r.accented.append(lab)
                if has_mark(lab, COMBINING_DIAERESIS):
                    r.dialytika.append(lab)
                if lab == "ς":
                    r.has_final_sigma = True
            elif cat.startswith("N"):
                r.digits.append(lab)
            elif cat.startswith(("P", "S")):
                r.punctuation.append(lab)
            else:
                r.other.append(repr(lab))
        else:
            r.other.append(lab)
    r.letters.sort()
    return r


# ----------------------------------------------------------------------------- pyctcdecode
def to_pyctcdecode_labels(labels: list[str]) -> list[str]:
    """Convert model labels to pyctcdecode's alphabet convention, keeping index alignment.

    blank -> "", word delimiter -> " ", unk -> "⁇", bos/eos/unused -> unique unusable tokens.
    pyctcdecode treats "" as blank and " " as the word boundary; every other label is a
    character. Labels must stay index-aligned with the emission columns.
    """
    out: list[str] = []
    for i, lab in enumerate(labels):
        if lab in BLANK_TOKENS:
            out.append("")
        elif lab == WORD_DELIMITER or lab == " ":
            out.append(" ")
        elif lab in UNK_TOKENS:
            out.append("⁇")
        elif lab in BOS_EOS_TOKENS or lab.startswith(UNUSED_PREFIX):
            out.append(f"<unused{i}>")
        else:
            out.append(lab)
    if out.count("") != 1:
        raise ValueError("expected exactly one blank label")
    return out
