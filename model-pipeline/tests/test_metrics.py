import pytest

from voicetotext.eval.metrics import (
    ErrorCounts, breakdown, cer, char_counts, edit_ops, length_bucket, sum_counts, wer, word_counts,
)


def test_wer_basic():
    assert wer("α β γ", "α β γ") == 0.0
    assert wer("α β γ", "α χ γ") == pytest.approx(1 / 3)
    assert wer("α β γ", "α γ") == pytest.approx(1 / 3)
    assert wer("α β γ", "α β γ δ") == pytest.approx(1 / 3)
    assert wer("α β γ", "") == 1.0


def test_counts_kinds():
    c = word_counts("το σπίτι μου", "το σπιτι μου εδώ")
    assert (c.sub, c.dele, c.ins, c.ref_len, c.hyp_len) == (1, 0, 1, 3, 4)
    c = word_counts("ένα δύο τρία", "δύο")
    assert (c.sub, c.dele, c.ins) == (0, 2, 0)


def test_cer_includes_spaces():
    assert cer("αβ γ", "αβ γ") == 0.0
    assert cer("αβγ", "αβδ") == pytest.approx(1 / 3)
    assert char_counts("αβ γ", "αβγ").dele == 1


def test_empty_reference():
    c = edit_ops([], ["x", "y"])
    assert c.ins == 2 and c.ref_len == 0
    assert c.rate == 2.0  # errors reported against empty reference
    assert edit_ops([], []).rate == 0.0


def test_sum_counts_is_corpus_level():
    a = ErrorCounts(sub=1, ref_len=2)
    b = ErrorCounts(sub=0, ref_len=8)
    t = sum_counts([a, b])
    assert t.rate == pytest.approx(0.1)  # not the mean of 0.5 and 0.0


def test_length_bucket():
    assert [length_bucket(n) for n in (0, 1, 2, 3, 4, 7, 8, 15, 16, 100)] == [
        "0", "1", "2-3", "2-3", "4-7", "4-7", "8-15", "8-15", "16+", "16+",
    ]


def test_breakdown_groups():
    rows = [("a", ErrorCounts(1, 0, 0, 4, 4), ErrorCounts(2, 0, 0, 20, 20)),
            ("b", ErrorCounts(0, 0, 0, 4, 4), ErrorCounts(0, 0, 0, 20, 20)),
            ("a", ErrorCounts(1, 0, 0, 4, 4), ErrorCounts(0, 0, 0, 20, 20))]
    out = breakdown(rows, lambda r: r[0], lambda r: r[1], lambda r: r[2])
    assert out["a"]["n"] == 2 and out["a"]["wer"] == pytest.approx(0.25)
    assert out["a"]["cer"] == pytest.approx(0.05)
    assert out["b"]["wer"] == 0.0


def test_oracle_spelling_is_case_insensitive():
    """Uppercase-only CTC vocabularies exist; the homophone table is lowercase."""
    from voicetotext.eval.metrics import cer, oracle_spelling

    assert oracle_spelling("ΚΑΛΗΜΈΡΑ") == oracle_spelling("καλημέρα")
    # η, ι and ει all collapse to the same sound in both cases
    assert oracle_spelling("ΤΥΠΟΙ") == oracle_spelling("τίπι")
    # a homophone-only error must cost 0 oracle CER whatever the case
    ref, hyp = "ΟΙ ΤΎΠΟΙ", "Η ΤΊΠΗ"
    assert cer(ref, hyp) > 0
    assert cer(oracle_spelling(ref), oracle_spelling(hyp)) == 0.0
