"""pyctcdecode wrapper: Emissions -> full N-best list of Hypothesis.

This is the reference decoder the app's BeamSearch.kt reproduces; alpha/beta and the
pruning constants were tuned here.

Scores: pyctcdecode's ``logit_score`` is the acoustic log-prob sum along the path and its
``lm_score`` is the *combined* score (acoustic + alpha*LM + beta*words). We store
acoustic_score = logit_score and lm_score = combined - logit_score (the LM's contribution).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from voicetotext.decoding.vocab import to_pyctcdecode_labels
from voicetotext.types import Emissions, Hypothesis, WordSpan

log = logging.getLogger("voicetotext.beam")


class BeamDecoder:
    def __init__(
        self,
        labels: list[str],
        *,
        lm_path: str | Path | None = None,
        alpha: float = 0.7,
        beta: float = 3.0,
        unk_score_offset: float = -10.0,
        beam_width: int = 128,
        n_best: int = 20,
        unigrams: list[str] | None = None,
    ) -> None:
        """``unigrams``: the LM vocabulary. Without it pyctcdecode cannot tell in-vocabulary
        words from OOV ones (every partial word is scored through <unk>) and warns that
        results will be much worse. If None, ``<lm_path minus suffix>.vocab`` is used when present."""
        from pyctcdecode import build_ctcdecoder

        self.labels = labels
        self.alpha, self.beta = alpha, beta
        self.beam_width, self.n_best = beam_width, n_best
        self.lm_path = str(Path(lm_path).expanduser()) if lm_path else None
        if unigrams is None and self.lm_path:
            vp = Path(self.lm_path).with_suffix(".vocab")
            if vp.exists():
                unigrams = vp.read_text(encoding="utf-8").split()
        self.unigrams = unigrams
        kw: dict[str, Any] = {"alpha": alpha, "beta": beta, "unk_score_offset": unk_score_offset}
        self._decoder = build_ctcdecoder(to_pyctcdecode_labels(labels), kenlm_model_path=self.lm_path, unigrams=unigrams, **kw)
        log.info("beam decoder: lm=%s unigrams=%s alpha=%.2f beta=%.2f width=%d nbest=%d", self.lm_path, len(unigrams) if unigrams else 0, alpha, beta, beam_width, n_best)

    def reset_params(self, alpha: float | None = None, beta: float | None = None, unk_score_offset: float | None = None) -> None:
        """Change LM weights without rebuilding (used by alpha/beta tuning)."""
        self._decoder.reset_params(alpha=alpha, beta=beta, unk_score_offset=unk_score_offset)
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta

    def decode(self, em: Emissions) -> list[Hypothesis]:
        beams = self._decoder.decode_beams(em.logprobs, beam_width=self.beam_width)
        out: list[Hypothesis] = []
        for b in beams[: self.n_best]:
            text, _last_state, frames, logit_score, combined = b[0], b[1], b[2], b[3], b[4]
            words = [WordSpan(text=w, start_frame=int(s), end_frame=int(e), confidence=_span_conf(em, s, e)) for w, (s, e) in frames]
            out.append(Hypothesis(text=text, acoustic_score=float(logit_score), lm_score=float(combined - logit_score), words=words))
        return out


def _span_conf(em: Emissions, start: int, end: int) -> float:
    """Mean max-posterior over the word's frames."""
    end = max(end, start + 1)
    return float(np.exp(em.logprobs[start:end].max(axis=1)).mean())
