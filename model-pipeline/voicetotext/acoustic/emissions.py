"""Emissions utilities: npz persistence and *views* over the (T, V) matrix.

Nothing here mutates or discards the matrix. ``greedy_decode`` is a convenience view; the
stored representation is always the full log-probability matrix.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicetotext.decoding.vocab import WORD_DELIMITER, blank_index, is_special
from voicetotext.types import Emissions, Hypothesis, WordSpan

# ----------------------------------------------------------------------------- persistence


def save_npz(em: Emissions, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logprobs=em.logprobs.astype(np.float32),
        labels=np.array(em.labels, dtype=np.str_),
        frame_duration_ms=np.float64(em.frame_duration_ms),
        audio_id=np.str_(em.audio_id),
        model_id=np.str_(em.model_id),
    )
    return path


def load_npz(path: str | Path) -> Emissions:
    with np.load(Path(path), allow_pickle=False) as z:
        return Emissions(
            logprobs=np.ascontiguousarray(z["logprobs"], dtype=np.float32),
            labels=[str(x) for x in z["labels"]],
            frame_duration_ms=float(z["frame_duration_ms"]),
            audio_id=str(z["audio_id"]),
            model_id=str(z["model_id"]),
        )


# ----------------------------------------------------------------------------- greedy view


@dataclass
class TokenSpan:
    """One collapsed CTC token on the argmax path."""

    label_id: int
    label: str
    start_frame: int
    end_frame: int  # exclusive
    prob: float  # mean posterior of the token over its frames


def greedy_path(em: Emissions) -> list[TokenSpan]:
    """Argmax per frame, collapse repeats, drop blanks. Keeps the word delimiter as a token."""
    blank = blank_index(em.labels)
    ids = em.logprobs.argmax(axis=1)
    maxp = np.exp(em.logprobs.max(axis=1))
    spans: list[TokenSpan] = []
    t = 0
    T = len(ids)
    while t < T:
        lab = int(ids[t])
        start = t
        while t < T and int(ids[t]) == lab:
            t += 1
        if lab != blank:
            spans.append(
                TokenSpan(
                    label_id=lab,
                    label=em.labels[lab],
                    start_frame=start,
                    end_frame=t,
                    prob=float(maxp[start:t].mean()),
                )
            )
    return spans


def _is_text_token(label: str) -> bool:
    return not is_special(label)


def greedy_words(em: Emissions) -> list[WordSpan]:
    """Words on the greedy path with frame spans and a simple confidence: the mean over the
    word's characters of each character's mean posterior."""
    words: list[WordSpan] = []
    chars: list[TokenSpan] = []

    def flush() -> None:
        if chars:
            words.append(
                WordSpan(
                    text=unicodedata.normalize("NFC", "".join(c.label for c in chars)),
                    start_frame=chars[0].start_frame,
                    end_frame=chars[-1].end_frame,
                    confidence=float(np.mean([c.prob for c in chars])),
                )
            )
            chars.clear()

    for tok in greedy_path(em):
        if tok.label == WORD_DELIMITER or tok.label == " ":
            flush()
        elif _is_text_token(tok.label):
            chars.append(tok)
    flush()
    return words


def greedy_decode(em: Emissions) -> str:
    """Unconstrained acoustic string. A view; never store it in place of the emissions."""
    return " ".join(w.text for w in greedy_words(em))


def greedy_hypothesis(em: Emissions) -> Hypothesis:
    words = greedy_words(em)
    return Hypothesis(
        text=" ".join(w.text for w in words),
        acoustic_score=float(em.logprobs.max(axis=1).sum()),
        lm_score=0.0,
        words=words,
    )


def posteriors(em: Emissions) -> np.ndarray:
    """(T, V) probabilities. Allocates; use sparingly."""
    return np.exp(em.logprobs)
