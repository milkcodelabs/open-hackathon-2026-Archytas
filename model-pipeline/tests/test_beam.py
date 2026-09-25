import pytest

pytest.importorskip("pyctcdecode")  # the `decode` extra

from voicetotext.acoustic.emissions import greedy_decode, load_npz  # noqa: E402
from voicetotext.decoding.beam import BeamDecoder  # noqa: E402
from tests.conftest import FIXTURES, make_emissions  # noqa: E402


def test_beam_without_lm_matches_greedy_on_synthetic(labels):
    em = make_emissions(labels, "καλημέρα σας")
    dec = BeamDecoder(labels, n_best=20, beam_width=16)
    hyps = dec.decode(em)
    assert hyps[0].text == "καλημέρα σας"
    assert hyps[0].lm_score == pytest.approx(0.0, abs=1e-6) or hyps[0].lm_score <= 0.0
    assert [w.text for w in hyps[0].words] == ["καλημέρα", "σας"]
    assert hyps[0].words[0].end_frame <= hyps[0].words[1].start_frame
    assert 0.0 <= hyps[0].words[0].confidence <= 1.0


def test_beam_nbest_on_real_fixture(expected, fixture_ids, labels):
    em = load_npz(FIXTURES / "emissions" / f"{fixture_ids[0]}.npz")
    dec = BeamDecoder(labels, n_best=20, beam_width=32)
    hyps = dec.decode(em)
    assert 1 < len(hyps) <= 20
    assert hyps[0].text == greedy_decode(em)  # no LM: best path equals greedy view here
    scores = [h.acoustic_score + h.lm_score for h in hyps]
    assert scores == sorted(scores, reverse=True)
