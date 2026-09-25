"""Pipeline: audio -> Emissions -> decoder (greedy, or beam search with the n-gram LM) -> Recognition.

The emitter is injectable so tests and the harness can run on fixture emissions without a
model.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from voicetotext.acoustic.emissions import greedy_decode, greedy_hypothesis, load_npz, save_npz
from voicetotext.audio.io import load_audio
from voicetotext.config import Config
from voicetotext.phonetics.normalize import Normalizer
from voicetotext.types import Emissions, FinalSource, Hypothesis, Recognition

log = logging.getLogger("voicetotext.pipeline")


class Emitter(Protocol):
    labels: list[str]
    model_id: str

    def emit(self, waveforms: list[np.ndarray], audio_ids: list[str] | None = None) -> list[Emissions]: ...


@dataclass
class AudioItem:
    audio: str | Path | np.ndarray
    audio_id: str | None = None


def make_audio_id(audio: str | Path | np.ndarray) -> str:
    if isinstance(audio, (str, Path)):
        p = Path(audio)
        h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
        return f"{p.stem}_{h}"
    return uuid.uuid4().hex[:12]


class Pipeline:
    def __init__(
        self,
        cfg: Config | None = None,
        *,
        emitter: Emitter | None = None,
        use_emissions_cache: bool = False,
    ) -> None:
        self.cfg = cfg or Config.load()
        self._emitter = emitter
        self._normalizer: Normalizer | None = None
        self._beam: Any = None
        self.use_emissions_cache = use_emissions_cache

    # ------------------------------------------------------------------ lazy parts
    @property
    def emitter(self) -> Emitter:
        if self._emitter is None:
            from voicetotext.acoustic.omni import OmniOnnxEmitter

            self._emitter = OmniOnnxEmitter(self.cfg.model, self.cfg.audio.sample_rate)
        return self._emitter

    @property
    def labels(self) -> list[str]:
        em = self.emitter
        if not em.labels and hasattr(em, "load"):
            em.load()  # type: ignore[attr-defined]
        return em.labels

    @property
    def normalizer(self) -> Normalizer:
        if self._normalizer is None:
            self._normalizer = Normalizer.from_labels(self.labels, self.cfg.normalize.strip_accents)
        return self._normalizer

    @property
    def beam(self) -> Any:
        if self._beam is None:
            from voicetotext.decoding.beam import BeamDecoder

            d = self.cfg.decoder
            self._beam = BeamDecoder(
                self.labels, lm_path=d.lm_path, alpha=d.alpha, beta=d.beta, unk_score_offset=d.unk_score_offset,
                beam_width=d.beam_width, n_best=d.n_best,
            )
        return self._beam

    # ------------------------------------------------------------------ public API
    def recognize(self, audio: str | Path | np.ndarray, *, audio_id: str | None = None) -> Recognition:
        return self.recognize_batch([AudioItem(audio, audio_id)])[0]

    def recognize_batch(self, items: list[AudioItem]) -> list[Recognition]:
        sr = self.cfg.audio.sample_rate
        wavs: list[np.ndarray] = []
        ids: list[str] = []
        for it in items:
            if isinstance(it.audio, np.ndarray):
                wav = np.asarray(it.audio, dtype=np.float32)
            else:
                wav, _ = load_audio(it.audio, sr)
            wavs.append(wav)
            ids.append(it.audio_id or make_audio_id(it.audio))
        return [self.recognize_emissions(em) for em in self.emit_cached(wavs, ids)]

    def emit_cached(self, wavs: list[np.ndarray], ids: list[str]) -> list[Emissions]:
        """Reuse emissions from <emissions_cache>/<model>/<id>.npz when enabled (decoder tuning)."""
        if not self.use_emissions_cache:
            return self.emitter.emit(wavs, ids)
        # Key the cache by the *emitter* id, not the configured model id: the sherpa export
        # and the phone graph produce slightly different emissions, and sharing a cache
        # directory between them would silently report one file's numbers for the other.
        key = re.sub(r"[^A-Za-z0-9._+-]", "__", self.emitter.model_id)
        cache = self.cfg.paths.emissions_cache / key
        result: list[Emissions | None] = [None] * len(ids)
        todo = [i for i, uid in enumerate(ids) if not (cache / f"{uid}.npz").exists()]
        for i in range(len(ids)):
            if i not in todo:
                result[i] = load_npz(cache / f"{ids[i]}.npz")
        if todo:
            fresh = self.emitter.emit([wavs[i] for i in todo], [ids[i] for i in todo])
            for i, em in zip(todo, fresh):
                save_npz(em, cache / f"{ids[i]}.npz")
                result[i] = em
        return [r for r in result if r is not None]

    def recognize_emissions(self, em: Emissions) -> Recognition:
        """Decode already-computed emissions. No audio involved."""
        greedy, n_best, source = self.decode(em)
        best = n_best[0] if n_best else Hypothesis(text="", acoustic_score=0.0, lm_score=0.0)
        return Recognition(audio_id=em.audio_id, greedy=greedy, n_best=n_best, final_text=best.text, final_source=source)

    def decode(self, em: Emissions) -> tuple[str, list[Hypothesis], FinalSource]:
        greedy = greedy_decode(em)
        kind = self.cfg.decoder.kind
        if kind == "greedy":
            return greedy, [greedy_hypothesis(em)], "greedy"
        if kind == "beam":
            n_best = self.beam.decode(em)
            if not n_best:
                return greedy, [greedy_hypothesis(em)], "greedy"
            return greedy, n_best, "beam"
        raise ValueError(f"unknown decoder kind {kind!r}")
