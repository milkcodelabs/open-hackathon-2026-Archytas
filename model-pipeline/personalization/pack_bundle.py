"""Build personal_bundle.zip, the input of the Colab fine-tune (finetune.ipynb).

    uv run python personalization/pack_bundle.py <recordings dir> <out dir> [--lm ...] [--cv-eval ...] [--cv-full ...]

Contents (all audio 16 kHz mono wav, manifests JSONL with paths relative to the bundle):
  recordings/          the speaker's recordings (metadata.csv + wav/ from recorder.html), split by session
  eval/fleurs_dev_clean, eval/fleurs_test_clean, eval/cv300_clean   regression sets
  reg/cv_reg           Common Voice clips for rehearsal during training: the full CV test split
                       (clean), minus the 300 eval clips and minus every speaker that occurs in them
  lm/                  KenLM binary + unigram vocab (the model of configs/omni_beam.yaml)
  voicetotext/         this package (normalize, metrics, beam decoder), imported by the notebook
  finetune.ipynb

Manifests come from VOICETOTEXT_DATA_DIR/manifests (default ~/voicetotext_data), made with
`voicetotext data fleurs`, `voicetotext data common-voice` and `voicetotext data clean`
(README.md next to this file).
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
DATA = Path(os.environ.get("VOICETOTEXT_DATA_DIR", "~/voicetotext_data")).expanduser()

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("recordings", type=Path, help="folder with metadata.csv and wav/ (unzipped recorder.html export)")
ap.add_argument("out", type=Path, help="bundle folder; the zip is written next to it as <out>.zip")
ap.add_argument("--lm", type=Path, default=DATA / "lm" / "general_el_300k_p02233.bin",
                help="KenLM binary; <lm>.vocab must sit next to it")
ap.add_argument("--fleurs-dev", default="fleurs_el_gr_dev_clean")
ap.add_argument("--fleurs-test", default="fleurs_el_gr_test_clean")
ap.add_argument("--cv-eval", default="cv_el_test_300_clean", help="manifest of the 300 CV eval clips")
ap.add_argument("--cv-full", default="cv_el_test_full_clean", help="manifest of the whole CV test split")
args = ap.parse_args()
OUT = args.out


def read(name):
    return [json.loads(l) for l in open(DATA / "manifests" / f"{name}.jsonl", encoding="utf-8")]


def copy_set(rows, sub):
    d = OUT / sub
    (d / "wav").mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        src = Path(r["audio_path"])
        info = sf.info(src)
        assert info.samplerate == 16000 and info.channels == 1, (src, info)
        dst = d / "wav" / src.name
        shutil.copyfile(src, dst)
        out.append({"audio": f"{sub}/wav/{src.name}", "text": r["reference"],
                    "utterance_id": r["utterance_id"], "speaker_id": r["speaker_id"],
                    "duration_s": r.get("duration_s") or info.duration})
    with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{sub}: {len(out)} clips, {sum(r['duration_s'] for r in out) / 60:.1f} min")


assert (args.recordings / "metadata.csv").exists(), f"{args.recordings}: no metadata.csv"
assert args.lm.with_suffix(".vocab").exists(), f"{args.lm.with_suffix('.vocab')} not found"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
shutil.copytree(args.recordings, OUT / "recordings")
copy_set(read(args.fleurs_dev), "eval/fleurs_dev_clean")
copy_set(read(args.fleurs_test), "eval/fleurs_test_clean")
cv300 = read(args.cv_eval)
copy_set(cv300, "eval/cv300_clean")
ids = {r["utterance_id"] for r in cv300}
spk = {r["speaker_id"] for r in cv300}
reg = [r for r in read(args.cv_full) if r["utterance_id"] not in ids and r["speaker_id"] not in spk]
copy_set(reg, "reg/cv_reg")
(OUT / "lm").mkdir()
shutil.copyfile(args.lm, OUT / "lm" / "lm.bin")
shutil.copyfile(args.lm.with_suffix(".vocab"), OUT / "lm" / "lm.vocab")
shutil.copyfile(HERE / "finetune.ipynb", OUT / "finetune.ipynb")
shutil.copytree(PIPELINE / "voicetotext", OUT / "voicetotext", ignore=shutil.ignore_patterns("__pycache__"))

zp = OUT.with_suffix(".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))
print(zp, f"{zp.stat().st_size / 1e6:.0f} MB")
