"""Build the test fixtures from real audio (needs the acoustic model; unit tests then don't).

    python scripts/make_fixtures.py --model ~/voicetotext_data/models/omni.onnx
    python scripts/make_fixtures.py --model ... --manifest ~/voicetotext_data/manifests/fleurs_el_gr_test.jsonl

Without --manifest the clips already in tests/fixtures/audio/ are re-run and their
references kept. With --manifest the N shortest clips are copied in first. For each clip the
emission matrix (npz) and the greedy string go to tests/fixtures/, with the label list in
tests/fixtures/labels.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from voicetotext.acoustic.emissions import greedy_decode, save_npz
from voicetotext.audio.io import load_audio
from voicetotext.config import Config
from voicetotext.decoding.vocab import save_labels
from voicetotext.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="omni.onnx (phone graph) or the sherpa export")
    ap.add_argument("--manifest", default=None, help="pick new clips from this manifest")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=6.0)
    args = ap.parse_args()

    cfg = Config.load(overrides={"model": {"onnx_path": args.model}})
    pipe = Pipeline(cfg)
    (FIX / "audio").mkdir(parents=True, exist_ok=True)
    (FIX / "emissions").mkdir(parents=True, exist_ok=True)
    old_path = FIX / "expected.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}

    clips: list[tuple[str, dict]] = []
    if args.manifest:
        from voicetotext.eval.harness import load_manifest

        entries = load_manifest(args.manifest)
        entries = [e for e in entries if (e.duration_s or 99) <= args.max_seconds] or entries
        entries.sort(key=lambda e: e.duration_s or 99)
        for i, e in enumerate(entries[: args.n]):
            fid = f"clip{i + 1}"
            shutil.copyfile(e.audio_path, FIX / "audio" / f"{fid}.wav")
            clips.append((fid, {"source_utterance_id": e.utterance_id, "source": e.source,
                                "reference": e.reference, "duration_s": e.duration_s}))
    else:
        for wav in sorted((FIX / "audio").glob("*.wav")):
            meta = {k: v for k, v in old.get(wav.stem, {}).items()
                    if k in ("source_utterance_id", "source", "reference", "duration_s")}
            clips.append((wav.stem, meta))

    save_labels(pipe.labels, FIX / "labels.json")
    expected: dict[str, dict] = {}
    for fid, meta in clips:
        wav, _ = load_audio(FIX / "audio" / f"{fid}.wav", cfg.audio.sample_rate)
        em = pipe.emitter.emit([wav], [fid])[0]
        npz = save_npz(em, FIX / "emissions" / f"{fid}.npz")
        expected[fid] = {
            **meta,
            "greedy": greedy_decode(em),
            "frames": em.num_frames,
            "frame_duration_ms": em.frame_duration_ms,
            "model_id": em.model_id,
        }
        print(f"{fid}: frames={em.num_frames} npz={npz.stat().st_size // 1024}KB")
        print(f"   ref   : {meta.get('reference', '')}")
        print(f"   greedy: {expected[fid]['greedy']}")
    old_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", old_path)


if __name__ == "__main__":
    main()
