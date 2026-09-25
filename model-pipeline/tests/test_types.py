import numpy as np
import pytest

from voicetotext.types import FINAL_SOURCES, Emissions, Hypothesis, Recognition, WordSpan


def test_final_sources():
    assert set(FINAL_SOURCES) == {"greedy", "beam"}


def test_recognition_json_roundtrip():
    rec = Recognition(
        audio_id="u1",
        greedy="καλημερα",
        n_best=[Hypothesis("καλημέρα", -3.5, -1.0, [WordSpan("καλημέρα", 0, 40, 0.8)])],
        final_text="καλημέρα",
        final_source="beam",
    )
    back = Recognition.from_json(rec.to_json())
    assert back == rec
    assert back.best is not None and back.best.total_score == pytest.approx(-4.5)


def test_invalid_final_source_rejected():
    with pytest.raises(ValueError):
        Recognition("u", "", [], "", "magic")


def test_emissions_shape_validation():
    with pytest.raises(ValueError):
        Emissions(np.zeros((10, 3), np.float32), ["a", "b"], 20.0, "u", "m")
    em = Emissions(np.zeros((10, 2), np.float64), ["a", "b"], 20.0, "u", "m")
    assert em.logprobs.dtype == np.float32
    assert em.num_frames == 10 and em.vocab_size == 2
    assert em.duration_s == pytest.approx(0.2)
