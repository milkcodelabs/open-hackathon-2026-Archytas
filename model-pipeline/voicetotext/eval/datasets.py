"""Public evaluation data: FLEURS el_gr (CC-BY 4.0) and Common Voice Greek.

Both are read speech by typical speakers. FLEURS files are fetched with huggingface_hub
directly (tsv + audio tar per split), so the ``datasets`` library is not required.
"""

from __future__ import annotations

import logging
import random
import tarfile
from pathlib import Path

from voicetotext.audio.io import audio_info
from voicetotext.eval.harness import ManifestEntry, write_manifest

log = logging.getLogger("voicetotext.data")

FLEURS_REPO = "google/fleurs"
FLEURS_LANG = "el_gr"


def _parse_fleurs_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[0] == "id":
                continue
            rows.append({"id": parts[0], "file": parts[1], "raw": parts[2], "norm": parts[3]})
    return rows


def build_fleurs_manifest(
    data_dir: Path,
    manifests_dir: Path,
    split: str = "test",
    limit: int | None = 100,
    seed: int = 0,
    keep_archive: bool = False,
) -> Path:
    """Download one FLEURS split, keep ``limit`` clips (seeded sample), write a manifest."""
    from huggingface_hub import hf_hub_download

    dl_dir = data_dir / "fleurs" / "_downloads"
    out_dir = data_dir / "fleurs" / FLEURS_LANG / split
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = Path(hf_hub_download(FLEURS_REPO, f"data/{FLEURS_LANG}/{split}.tsv", repo_type="dataset", local_dir=dl_dir))
    rows = _parse_fleurs_tsv(tsv)
    rows.sort(key=lambda r: (r["id"], r["file"]))
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
        rows.sort(key=lambda r: (r["id"], r["file"]))
    wanted = {r["file"] for r in rows}
    missing = {f for f in wanted if not (out_dir / f).exists()}
    if missing:
        tar_path = Path(
            hf_hub_download(FLEURS_REPO, f"data/{FLEURS_LANG}/audio/{split}.tar.gz", repo_type="dataset", local_dir=dl_dir)
        )
        log.info("extracting %d clips from %s", len(missing), tar_path.name)
        with tarfile.open(tar_path, "r|gz") as tf:
            for m in tf:
                name = Path(m.name).name
                if name in missing and m.isfile():
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    (out_dir / name).write_bytes(src.read())
                    missing.discard(name)
                    if not missing:
                        break
        if not keep_archive:
            tar_path.unlink(missing_ok=True)
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips not found in the archive: {sorted(missing)[:5]}")
    entries: list[ManifestEntry] = []
    for r in rows:
        wav = out_dir / r["file"]
        dur = audio_info(wav)[0]
        entries.append(
            ManifestEntry(
                audio_path=str(wav.resolve()),
                reference=r["raw"],
                utterance_id=f"fleurs_{split}_{Path(r['file']).stem}",
                speaker_id="fleurs",
                severity="typical",
                split=split,
                duration_s=dur,
                source=f"{FLEURS_REPO}/{FLEURS_LANG}",
            )
        )
    path = write_manifest(entries, manifests_dir / f"fleurs_{FLEURS_LANG}_{split}.jsonl")
    log.info("wrote %s (%d entries, %.1f s audio)", path, len(entries), sum(e.duration_s or 0 for e in entries))
    return path


# --------------------------------------------------------------------------- Common Voice
# Common Voice is distributed by the Mozilla Data Collective (account + terms), so the corpus
# must be downloaded manually: cv-corpus-<ver>/el/{test.tsv, clips/*.mp3}. This builds a
# manifest from that local extract, decoding the chosen clips to 16 kHz wav.


def _read_tsv_dicts(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(dict(zip(header, parts)))
    return rows


def build_common_voice_manifest(
    cv_lang_dir: Path,
    data_dir: Path,
    manifests_dir: Path,
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    sample_rate: int = 16000,
) -> Path:
    from voicetotext.audio.io import load_audio, save_wav

    cv_lang_dir = Path(cv_lang_dir).expanduser()
    tsv = cv_lang_dir / f"{split}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"{tsv} not found; expected a Common Voice language folder (el/) with {split}.tsv and clips/")
    rows = _read_tsv_dicts(tsv)
    rows = [r for r in rows if r.get("path") and r.get("sentence")]
    rows.sort(key=lambda r: r["path"])
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
        rows.sort(key=lambda r: r["path"])
    out_dir = data_dir / "common_voice" / cv_lang_dir.name / split
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    for i, r in enumerate(rows):
        src = cv_lang_dir / "clips" / r["path"]
        wav_path = out_dir / (Path(r["path"]).stem + ".wav")
        if not wav_path.exists():
            wav, sr = load_audio(src, sample_rate)
            save_wav(wav_path, wav, sr)
        dur = audio_info(wav_path)[0]
        entries.append(
            ManifestEntry(
                audio_path=str(wav_path.resolve()),
                reference=r["sentence"],
                utterance_id=f"cv_{split}_{Path(r['path']).stem}",
                speaker_id=(r.get("client_id") or "unknown")[:12],
                severity="typical",
                split=split,
                duration_s=dur,
                source=f"common_voice/{cv_lang_dir.name}",
            )
        )
        if (i + 1) % 100 == 0:
            log.info("decoded %d/%d clips", i + 1, len(rows))
    path = write_manifest(entries, manifests_dir / f"cv_{cv_lang_dir.name}_{split}.jsonl")
    log.info("wrote %s (%d entries, %.1f s audio)", path, len(entries), sum(e.duration_s or 0 for e in entries))
    return path
