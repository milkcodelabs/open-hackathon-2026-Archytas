"""alpha/beta grid search for beam+LM decoding on a DEV manifest, using cached emissions."""

from __future__ import annotations

import logging
from pathlib import Path

from voicetotext.audio.io import load_audio
from voicetotext.config import Config
from voicetotext.eval.harness import load_manifest
from voicetotext.eval.metrics import char_counts, sum_counts, word_counts
from voicetotext.pipeline import Pipeline

log = logging.getLogger("voicetotext.tune")


def tune_alpha_beta(cfg: Config, manifest: Path, alphas: list[float], betas: list[float], *, limit: int | None = None, metric: str = "cer") -> dict:
    if "test" in Path(manifest).stem.lower():
        raise ValueError("refusing to tune on a manifest named like a test split")
    entries = load_manifest(manifest)[: limit or None]
    pipe = Pipeline(cfg, use_emissions_cache=True)
    norm = pipe.normalizer
    # emissions once (cached on disk for later runs)
    ems = []
    for i in range(0, len(entries), 8):
        batch = entries[i : i + 8]
        wavs = [load_audio(e.audio_path, cfg.audio.sample_rate)[0] for e in batch]
        ems.extend(pipe.emit_cached(wavs, [e.utterance_id or "" for e in batch]))
        log.info("emissions %d/%d", min(i + 8, len(entries)), len(entries))
    refs = [norm(e.reference) for e in entries]
    beam = pipe.beam
    rows = []
    for a in alphas:
        for b in betas:
            beam.reset_params(alpha=a, beta=b)
            hyps = [norm(h[0].text) if (h := beam.decode(em)) else "" for em in ems]
            w = sum_counts(word_counts(r, h) for r, h in zip(refs, hyps))
            c = sum_counts(char_counts(r, h) for r, h in zip(refs, hyps))
            rows.append({"alpha": a, "beta": b, "wer": w.rate, "cer": c.rate})
            log.info("alpha=%.2f beta=%.2f wer=%.4f cer=%.4f", a, b, w.rate, c.rate)
    best = min(rows, key=lambda r: r[metric])
    lines = ["| alpha | beta | WER | CER |", "|---:|---:|---:|---:|"] + [f"| {r['alpha']} | {r['beta']} | {r['wer']:.4f} | {r['cer']:.4f} |" for r in rows]
    return {"manifest": str(manifest), "n": len(entries), "lm_path": cfg.decoder.lm_path, "rows": rows, "best": best, "table": "\n".join(lines)}
