"""Audio loading: any format soundfile/torchaudio can read -> float32 mono at the target rate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Return ``(waveform, sample_rate)``: float32 mono in [-1, 1], resampled to ``target_sr``."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    except Exception as e:  # noqa: BLE001 - fall back to torchaudio for exotic containers
        wav, sr = _load_with_torchaudio(path, e)
    wav = to_mono_float32(wav)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def _load_with_torchaudio(path: Path, original_error: Exception) -> tuple[np.ndarray, int]:
    try:
        import torchaudio

        t, sr = torchaudio.load(str(path))
    except Exception as e2:  # noqa: BLE001
        raise RuntimeError(
            f"could not read {path}: soundfile: {original_error}; torchaudio: {e2}"
        ) from e2
    return t.numpy().T, int(sr)


def to_mono_float32(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.dtype.kind in "iu":
        wav = wav.astype(np.float32) / float(np.iinfo(wav.dtype).max)
    if wav.ndim == 2:
        # accept (n, ch) or (ch, n); the long axis is time
        wav = wav.mean(axis=1) if wav.shape[0] >= wav.shape[1] else wav.mean(axis=0)
    return wav.astype(np.float32, copy=False)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    return AF.resample(t, orig_freq=sr, new_freq=target_sr).numpy()


def save_wav(path: str | Path, wav: np.ndarray, sr: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")
    return path


def duration_seconds(wav: np.ndarray, sr: int) -> float:
    return float(len(wav)) / float(sr)


def audio_info(path: str | Path) -> tuple[float, int, int]:
    """(duration_s, sample_rate, channels) without decoding the whole file."""
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate), int(info.channels)
