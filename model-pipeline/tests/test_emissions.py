import numpy as np

from voicetotext.acoustic.emissions import (
    greedy_decode, greedy_hypothesis, greedy_path, greedy_words, load_npz, save_npz,
)
from tests.conftest import FIXTURES, make_emissions


def test_synthetic_greedy_roundtrip(labels):
    em = make_emissions(labels, "καλημέρα σας")
    assert greedy_decode(em) == "καλημέρα σας"
    words = greedy_words(em)
    assert [w.text for w in words] == ["καλημέρα", "σας"]
    assert words[0].start_frame == 0 and words[0].end_frame < words[1].start_frame
    assert 0.0 < words[0].confidence <= 1.0
    hyp = greedy_hypothesis(em)
    assert hyp.text == "καλημέρα σας" and hyp.lm_score == 0.0 and hyp.acoustic_score < 0


def test_repeated_letters_need_blank(labels):
    em = make_emissions(labels, "άλλο", frames_per_char=2)
    assert greedy_decode(em) == "άλλο"  # λλ separated by blank frames survives collapse
    path = greedy_path(em)
    assert [t.label for t in path] == list("άλλο")


def test_npz_roundtrip(tmp_path, labels):
    em = make_emissions(labels, "ναι", audio_id="abc")
    p = save_npz(em, tmp_path / "e.npz")
    back = load_npz(p)
    np.testing.assert_array_equal(back.logprobs, em.logprobs)
    assert back.labels == labels and back.audio_id == "abc" and back.model_id == "synthetic"
    assert back.frame_duration_ms == 20.0


def test_real_fixture_emissions(expected, fixture_ids, labels):
    for fid in fixture_ids:
        em = load_npz(FIXTURES / "emissions" / f"{fid}.npz")
        assert em.labels == labels
        assert em.num_frames == expected[fid]["frames"]
        # log-softmax rows sum to ~1 in probability space
        np.testing.assert_allclose(np.exp(em.logprobs).sum(axis=1), 1.0, atol=1e-3)
        assert greedy_decode(em) == expected[fid]["greedy"]
