from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def labels() -> list[str]:
    from voicetotext.decoding.vocab import load_labels

    return load_labels(FIXTURES / "labels.json")


@pytest.fixture(scope="session")
def expected() -> dict:
    p = FIXTURES / "expected.json"
    if not p.exists():
        pytest.skip("fixture emissions not generated (scripts/make_fixtures.py)")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixture_ids(expected: dict) -> list[str]:
    return sorted(expected)


def make_emissions(labels: list[str], text: str, frames_per_char: int = 3, audio_id: str = "synthetic"):
    """Synthetic emissions that greedy-decode to ``text`` (blank frames between chars)."""
    from voicetotext.decoding.vocab import WORD_DELIMITER, blank_index
    from voicetotext.types import Emissions

    V = len(labels)
    blank = blank_index(labels)
    rows: list[np.ndarray] = []

    def frame(idx: int, p: float = 0.9) -> np.ndarray:
        probs = np.full(V, (1.0 - p) / (V - 1), dtype=np.float64)
        probs[idx] = p
        return np.log(probs)

    for ch in text:
        idx = labels.index(WORD_DELIMITER if ch == " " else ch)
        rows += [frame(idx)] * frames_per_char
        rows += [frame(blank, 0.95)]
    if not rows:
        rows = [frame(blank, 0.95)]
    return Emissions(
        logprobs=np.stack(rows).astype(np.float32),
        labels=labels,
        frame_duration_ms=20.0,
        audio_id=audio_id,
        model_id="synthetic",
    )
