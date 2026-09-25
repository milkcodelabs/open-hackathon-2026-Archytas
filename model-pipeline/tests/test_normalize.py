import random
import unicodedata
from pathlib import Path

import pytest

from voicetotext.phonetics.normalize import Normalizer, normalize, spec_from_labels

GOLDEN = Path(__file__).parent / "golden" / "normalize.tsv"


def _golden_cases() -> list[tuple[str, str]]:
    cases = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        inp, _, exp = line.rpartition("\t")
        cases.append((inp, exp))
    return cases


@pytest.fixture(scope="module")
def spec(labels):
    return spec_from_labels(labels)


def test_spec_follows_vocab(spec):
    assert spec.strip_accents is False  # this vocab has ά έ ή ί ό ύ ώ
    assert spec.keep_dialytika is True
    assert spec.final_sigma is True
    assert "'" not in spec.alphabet  # the apostrophe label is not a letter
    assert "ΰ" in spec.alphabet and "ϋ" in spec.alphabet and "ΐ" in spec.alphabet
    assert len(spec.alphabet) == 36


@pytest.mark.parametrize("inp,exp", _golden_cases())
def test_golden(spec, inp, exp):
    assert normalize(inp, spec) == exp


def test_idempotent_on_golden(spec):
    for inp, _ in _golden_cases():
        once = normalize(inp, spec)
        assert normalize(once, spec) == once


def test_idempotent_and_closed_on_random_text(spec):
    rng = random.Random(1234)
    pool = [chr(c) for c in range(0x0370, 0x03FF)] + [chr(c) for c in range(0x1F00, 0x1FFF)]
    pool += list("abcXYZ0123 .,;!'«»·\t\n") + [chr(0x301), chr(0x308), chr(0x344)]
    allowed = set(spec.alphabet) | {" "}
    for _ in range(300):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 30)))
        out = normalize(s, spec)
        assert normalize(out, spec) == out
        assert set(out) <= allowed
        assert out == out.strip() and "  " not in out
        assert unicodedata.normalize("NFC", out) == out


def test_strip_accents_override(labels):
    n = Normalizer.from_labels(labels, strip_accents=True)
    assert n("Καλημέρα ναΐ") == "καλημερα ναϊ"
    assert n(n("Καλημέρα")) == n("Καλημέρα")


def _upper_labels() -> list[str]:
    """An uppercase-only vocabulary without final sigma."""
    letters = list("ΆΈΉΊΌΎΏΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΪΫ")
    return letters + ["|", "'", "-", "M", "<pad>", "<unk>"]


def test_case_follows_the_vocabulary():
    """An uppercase CTC vocabulary must not be fed lowercased text: everything would
    fall outside the alphabet and normalize would return empty strings."""
    spec = spec_from_labels(_upper_labels())
    assert spec.uppercase is True and spec.lowercase is False
    assert spec.final_sigma is False
    out = normalize("Καλημέρα, τι κάνεις; Ο Οδυσσέας.", spec)
    # the vocabulary carries accented capitals, so accents survive the uppercasing
    assert out == "ΚΑΛΗΜΈΡΑ ΤΙ ΚΆΝΕΙΣ Ο ΟΔΥΣΣΈΑΣ"
    assert out.strip() != "" and set(out) <= set(spec.alphabet) | {" "}
    assert normalize(out, spec) == out


def test_lowercase_vocabulary_still_lowercases(labels):
    spec = spec_from_labels(labels)
    assert spec.uppercase is False and spec.lowercase is True
    assert normalize("ΚΑΛΗΜΕΡΑ", spec) == "καλημερα"


def test_final_sigma_only_when_vocabulary_has_it():
    upper = spec_from_labels(_upper_labels())
    assert normalize("ο Οδυσσέας", upper) == "Ο ΟΔΥΣΣΈΑΣ"  # Σ, never ς
    assert "ς" not in normalize("ας πας σε σας", upper)
