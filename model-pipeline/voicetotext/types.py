"""Shared types. Everything downstream imports from here; keep it dependency-light.

Design notes
------------
* ``Emissions`` is the unit that flows through the system: a ``(T, V)`` log-probability
  matrix plus the label set it is indexed by. Argmax strings are *views* computed on demand
  (see ``voicetotext.acoustic.emissions.greedy_decode``), never the stored representation.
* ``Recognition`` is what one utterance produces: the greedy view, the N-best list and the
  text the decoder settled on. ``audio_id`` is the utterance's key in evaluation results.
* Hot-path types are plain dataclasses (numpy arrays inside). Config and manifest records
  are pydantic models elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

import numpy as np

# Where the final text came from.
FinalSource = Literal["greedy", "beam"]
FINAL_SOURCES: tuple[str, ...] = get_args(FinalSource)


@dataclass
class Emissions:
    """CTC emission matrix for one utterance.

    logprobs: (T, V) float32, log-softmax over the label axis for every frame.
    labels:   index-aligned to V. ``labels[i]`` is the token whose log-prob is column ``i``.
    frame_duration_ms: stride of one frame (20 ms for the acoustic model).
    """

    logprobs: np.ndarray
    labels: list[str]
    frame_duration_ms: float
    audio_id: str
    model_id: str

    def __post_init__(self) -> None:
        if self.logprobs.ndim != 2:
            raise ValueError(f"logprobs must be (T, V), got shape {self.logprobs.shape}")
        if self.logprobs.shape[1] != len(self.labels):
            raise ValueError(
                f"label count {len(self.labels)} does not match V={self.logprobs.shape[1]}"
            )
        if self.logprobs.dtype != np.float32:
            self.logprobs = self.logprobs.astype(np.float32)

    @property
    def num_frames(self) -> int:
        return int(self.logprobs.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.logprobs.shape[1])

    @property
    def duration_s(self) -> float:
        return self.num_frames * self.frame_duration_ms / 1000.0

    def frame_to_seconds(self, frame: int) -> float:
        return frame * self.frame_duration_ms / 1000.0


@dataclass
class WordSpan:
    """A word with its frame alignment in the emission matrix and a 0..1 confidence."""

    text: str
    start_frame: int
    end_frame: int  # exclusive
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WordSpan":
        return cls(
            text=d["text"],
            start_frame=int(d["start_frame"]),
            end_frame=int(d["end_frame"]),
            confidence=float(d["confidence"]),
        )


@dataclass
class Hypothesis:
    """One entry of an N-best list.

    acoustic_score: sum of per-frame log-probs along the decoded path (higher is better).
    lm_score:       the language-model contribution alone (0.0 when no LM was used).
                    pyctcdecode reports a *combined* score under the name ``lm_score``;
                    the beam wrapper separates the two before building this object.
    """

    text: str
    acoustic_score: float
    lm_score: float
    words: list[WordSpan] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return self.acoustic_score + self.lm_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "acoustic_score": self.acoustic_score,
            "lm_score": self.lm_score,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hypothesis":
        return cls(
            text=d["text"],
            acoustic_score=float(d["acoustic_score"]),
            lm_score=float(d.get("lm_score", 0.0)),
            words=[WordSpan.from_dict(w) for w in d.get("words", [])],
        )


@dataclass
class Recognition:
    """Result of recognizing one utterance.

    greedy:        unconstrained acoustic view (argmax of the emissions). Always kept, never
                   overwritten by any later stage.
    n_best:        hypotheses from the decoder, best first. With greedy decoding this is a
                   single-element list.
    final_text:    the decoder's best text.
    final_source:  which decoder produced ``final_text``.
    """

    audio_id: str
    greedy: str
    n_best: list[Hypothesis]
    final_text: str
    final_source: FinalSource

    def __post_init__(self) -> None:
        if self.final_source not in FINAL_SOURCES:
            raise ValueError(f"final_source must be one of {FINAL_SOURCES}, got {self.final_source!r}")

    @property
    def best(self) -> Hypothesis | None:
        return self.n_best[0] if self.n_best else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "greedy": self.greedy,
            "n_best": [h.to_dict() for h in self.n_best],
            "final_text": self.final_text,
            "final_source": self.final_source,
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recognition":
        return cls(
            audio_id=d["audio_id"],
            greedy=d["greedy"],
            n_best=[Hypothesis.from_dict(h) for h in d.get("n_best", [])],
            final_text=d["final_text"],
            final_source=d["final_source"],
        )

    @classmethod
    def from_json(cls, s: str) -> "Recognition":
        return cls.from_dict(json.loads(s))
