"""Integration: full pipeline and harness on fixture clips with a fixture emitter."""

from pathlib import Path

import numpy as np

from voicetotext.acoustic.emissions import load_npz
from voicetotext.config import Config
from voicetotext.eval.harness import ManifestEntry, run_eval, write_manifest
from voicetotext.pipeline import AudioItem, Pipeline
from voicetotext.types import Emissions
from tests.conftest import FIXTURES


class FixtureEmitter:
    """Stands in for the acoustic model: returns the stored emissions for known audio ids."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.model_id = "fixture"
        self.calls = 0

    def emit(self, waveforms: list[np.ndarray], audio_ids: list[str] | None = None) -> list[Emissions]:
        self.calls += 1
        assert audio_ids is not None
        return [load_npz(FIXTURES / "emissions" / f"{a}.npz") for a in audio_ids]


def _cfg(tmp_path: Path) -> Config:
    return Config.load(overrides={"paths": {"data_dir": str(tmp_path)}})


def test_pipeline_end_to_end(tmp_path, labels, expected, fixture_ids):
    emitter = FixtureEmitter(labels)
    pipe = Pipeline(_cfg(tmp_path), emitter=emitter)
    items = [AudioItem(FIXTURES / "audio" / f"{fid}.wav", fid) for fid in fixture_ids]
    recs = pipe.recognize_batch(items)
    assert emitter.calls == 1 and len(recs) == len(fixture_ids)
    for fid, rec in zip(fixture_ids, recs):
        assert rec.audio_id == fid
        assert rec.greedy == expected[fid]["greedy"]
        assert rec.final_text == rec.greedy and rec.final_source == "greedy"
        assert len(rec.n_best) == 1 and rec.n_best[0].words
        assert all(0.0 <= w.confidence <= 1.0 for w in rec.n_best[0].words)


def test_emissions_cache(tmp_path, labels, fixture_ids):
    emitter = FixtureEmitter(labels)
    pipe = Pipeline(_cfg(tmp_path), emitter=emitter, use_emissions_cache=True)
    wavs = [np.zeros(16000, np.float32) for _ in fixture_ids]
    first = pipe.emit_cached(wavs, fixture_ids)
    again = pipe.emit_cached(wavs, fixture_ids)
    assert emitter.calls == 1  # the second call is served from <emissions_cache>/fixture/
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a.logprobs, b.logprobs)


def test_harness_on_fixtures(tmp_path, labels, expected, fixture_ids):
    cfg = _cfg(tmp_path)
    entries = [
        ManifestEntry(audio_path=str(FIXTURES / "audio" / f"{fid}.wav"), reference=expected[fid]["reference"],
                      utterance_id=fid, speaker_id="spk1", severity="typical", split="test")
        for fid in fixture_ids
    ]
    manifest = write_manifest(entries, tmp_path / "m.jsonl")
    pipe = Pipeline(cfg, emitter=FixtureEmitter(labels))
    report = run_eval(cfg, manifest, "fixture_run", pipeline=pipe, out_dir=tmp_path / "out")
    assert report.n == len(fixture_ids)
    assert 0.0 <= report.final_cer <= 1.0 and report.final_wer == report.greedy_wer
    assert (tmp_path / "out" / "per_utterance.jsonl").exists()
    assert (tmp_path / "out" / "aggregate.json").exists()
    assert "typical" in report.by_severity and "spk1" in report.by_speaker
    assert report.total_audio_s > 0
