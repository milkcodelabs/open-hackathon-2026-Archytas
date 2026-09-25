"""Evaluation harness: manifest (JSONL) -> pipeline -> per-utterance + aggregate results.

Both the *final* text and the *greedy* acoustic view are scored on every run, so the
decoder can never hide a regression of the acoustic layer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from voicetotext.audio.io import audio_info
from voicetotext.config import Config
from voicetotext.eval.metrics import (
    ErrorCounts, breakdown, char_counts, length_bucket, oracle_char_counts, sum_counts, word_counts,
)
from voicetotext.pipeline import AudioItem, Pipeline

log = logging.getLogger("voicetotext.eval")


class ManifestEntry(BaseModel):
    audio_path: str
    reference: str
    utterance_id: str | None = None
    speaker_id: str = "unknown"
    severity: str | None = None  # e.g. typical
    split: str | None = None  # dev | test | ...
    duration_s: float | None = None
    source: str | None = None


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    entries: list[ManifestEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = ManifestEntry.model_validate_json(line)
            p = Path(e.audio_path)
            if not p.is_absolute():
                e.audio_path = str((path.parent / p).resolve())
            if e.utterance_id is None:
                e.utterance_id = Path(e.audio_path).stem
            entries.append(e)
    return entries


def write_manifest(entries: list[ManifestEntry], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json(exclude_none=True) + "\n")
    return path


@dataclass
class UttResult:
    utterance_id: str
    speaker_id: str
    severity: str | None
    split: str | None
    audio_s: float
    reference: str
    reference_norm: str
    greedy: str
    final: str
    final_norm: str
    final_source: str
    n_ref_words: int
    bucket: str
    wer: float
    cer: float
    greedy_wer: float
    greedy_cer: float
    oracle_cer: float = 0.0
    greedy_oracle_cer: float = 0.0
    words: ErrorCounts = field(default_factory=ErrorCounts)
    chars: ErrorCounts = field(default_factory=ErrorCounts)
    greedy_words: ErrorCounts = field(default_factory=ErrorCounts)
    greedy_chars: ErrorCounts = field(default_factory=ErrorCounts)
    oracle_chars: ErrorCounts = field(default_factory=ErrorCounts)
    greedy_oracle_chars: ErrorCounts = field(default_factory=ErrorCounts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("words", "chars", "greedy_words", "greedy_chars", "oracle_chars", "greedy_oracle_chars"):
            d[k] = getattr(self, k).to_dict()
        return d


@dataclass
class EvalReport:
    run_name: str
    manifest: str
    model_id: str
    decoder_kind: str
    config_fingerprint: str
    n: int
    total_audio_s: float
    wall_s: float
    final_wer: float
    final_cer: float
    greedy_wer: float
    greedy_cer: float
    final_oracle_cer: float
    greedy_oracle_cer: float
    by_speaker: dict[str, Any]
    by_severity: dict[str, Any]
    by_length: dict[str, Any]
    results: list[UttResult]

    @property
    def rtf(self) -> float:
        return self.wall_s / self.total_audio_s if self.total_audio_s else 0.0

    def aggregate_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name, "manifest": self.manifest, "model_id": self.model_id,
            "decoder_kind": self.decoder_kind, "config_fingerprint": self.config_fingerprint,
            "n": self.n, "total_audio_s": self.total_audio_s, "wall_s": self.wall_s, "rtf": self.rtf,
            "final": {"wer": self.final_wer, "cer": self.final_cer, "oracle_cer": self.final_oracle_cer},
            "greedy": {"wer": self.greedy_wer, "cer": self.greedy_cer, "oracle_cer": self.greedy_oracle_cer},
            "by_speaker": self.by_speaker, "by_severity": self.by_severity, "by_length": self.by_length,
        }


def run_eval(
    cfg: Config,
    manifest_path: str | Path,
    run_name: str,
    *,
    limit: int | None = None,
    pipeline: Pipeline | None = None,
    chunk: int = 8,
    out_dir: Path | None = None,
    use_emissions_cache: bool = False,
) -> EvalReport:
    entries = load_manifest(manifest_path)
    if limit:
        entries = entries[:limit]
    pipe = pipeline or Pipeline(cfg, use_emissions_cache=use_emissions_cache)
    norm = pipe.normalizer
    results: list[UttResult] = []
    t0 = time.perf_counter()
    for start in range(0, len(entries), chunk):
        batch = entries[start : start + chunk]
        items = [AudioItem(e.audio_path, e.utterance_id) for e in batch]
        recs = pipe.recognize_batch(items)
        for e, rec in zip(batch, recs):
            audio_s = e.duration_s if e.duration_s is not None else audio_info(e.audio_path)[0]
            ref_n = norm(e.reference)
            fin_n = norm(rec.final_text)
            gr_n = norm(rec.greedy)
            w, c = word_counts(ref_n, fin_n), char_counts(ref_n, fin_n)
            gw, gc = word_counts(ref_n, gr_n), char_counts(ref_n, gr_n)
            oc, goc = oracle_char_counts(ref_n, fin_n), oracle_char_counts(ref_n, gr_n)
            n_words = len(ref_n.split())
            results.append(
                UttResult(
                    utterance_id=rec.audio_id, speaker_id=e.speaker_id, severity=e.severity, split=e.split,
                    audio_s=audio_s, reference=e.reference, reference_norm=ref_n, greedy=rec.greedy,
                    final=rec.final_text, final_norm=fin_n, final_source=rec.final_source,
                    n_ref_words=n_words, bucket=length_bucket(n_words),
                    wer=w.rate, cer=c.rate, greedy_wer=gw.rate, greedy_cer=gc.rate,
                    oracle_cer=oc.rate, greedy_oracle_cer=goc.rate,
                    words=w, chars=c, greedy_words=gw, greedy_chars=gc, oracle_chars=oc, greedy_oracle_chars=goc,
                )
            )
        log.info("%d/%d done", min(start + chunk, len(entries)), len(entries))
    wall = time.perf_counter() - t0
    tw, tc = sum_counts(r.words for r in results), sum_counts(r.chars for r in results)
    gw, gc = sum_counts(r.greedy_words for r in results), sum_counts(r.greedy_chars for r in results)
    oc, goc = sum_counts(r.oracle_chars for r in results), sum_counts(r.greedy_oracle_chars for r in results)
    report = EvalReport(
        run_name=run_name, manifest=str(manifest_path), model_id=pipe.emitter.model_id,
        decoder_kind=cfg.decoder.kind, config_fingerprint=cfg.fingerprint(), n=len(results),
        total_audio_s=sum(r.audio_s for r in results), wall_s=wall,
        final_wer=tw.rate, final_cer=tc.rate, greedy_wer=gw.rate, greedy_cer=gc.rate,
        final_oracle_cer=oc.rate, greedy_oracle_cer=goc.rate,
        by_speaker=breakdown(results, lambda r: r.speaker_id, lambda r: r.words, lambda r: r.chars),
        by_severity=breakdown(results, lambda r: r.severity or "n/a", lambda r: r.words, lambda r: r.chars),
        by_length=breakdown(results, lambda r: r.bucket, lambda r: r.words, lambda r: r.chars),
        results=results,
    )
    write_report(report, out_dir or (cfg.paths.results / run_name))
    return report


def write_report(report: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_utterance.jsonl", "w", encoding="utf-8") as f:
        for r in report.results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / "aggregate.json").write_text(
        json.dumps(report.aggregate_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(format_summary(report), encoding="utf-8")
    return out_dir


def _table(title: str, groups: dict[str, Any]) -> list[str]:
    lines = [f"**{title}**", "", "| group | n | WER | CER |", "|---|---:|---:|---:|"]
    for g, v in groups.items():
        lines.append(f"| {g} | {v['n']} | {v['wer']:.3f} | {v['cer']:.3f} |")
    return lines + [""]


def format_summary(r: EvalReport) -> str:
    lines = [
        f"# {r.run_name}", "",
        f"- manifest: `{r.manifest}`",
        f"- model: `{r.model_id}`  decoder: `{r.decoder_kind}`  config: `{r.config_fingerprint}`",
        f"- utterances: {r.n}  audio: {r.total_audio_s:.1f}s  wall: {r.wall_s:.1f}s  RTF: {r.rtf:.2f}", "",
        "| view | WER | CER | oracle-spelling CER |", "|---|---:|---:|---:|",
        f"| final ({r.decoder_kind}) | {r.final_wer:.3f} | {r.final_cer:.3f} | {r.final_oracle_cer:.3f} |",
        f"| greedy | {r.greedy_wer:.3f} | {r.greedy_cer:.3f} | {r.greedy_oracle_cer:.3f} |", "",
    ]
    lines += _table("by speaker", r.by_speaker)
    lines += _table("by severity", r.by_severity)
    lines += _table("by reference length (words)", r.by_length)
    return "\n".join(lines)


def results_row(r: EvalReport, note: str = "") -> str:
    """One Markdown table row: date, run, manifest, n, decoder, final and greedy rates, RTF, note."""
    date = time.strftime("%Y-%m-%d")
    return (
        f"| {date} | {r.run_name} | {Path(r.manifest).stem} | {r.n} | "
        f"{r.decoder_kind} | {r.final_wer:.3f} | {r.final_cer:.3f} | {r.final_oracle_cer:.3f} | "
        f"{r.greedy_wer:.3f} | {r.greedy_cer:.3f} | {r.greedy_oracle_cer:.3f} | {r.rtf:.2f} | {note} |"
    )
