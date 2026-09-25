import numpy as np
import pytest
import soundfile as sf

from voicetotext.audio.io import audio_info, load_audio, save_wav, to_mono_float32


def test_load_resamples_and_downmixes(tmp_path):
    pytest.importorskip("torchaudio")  # resampling needs the `data` extra
    sr = 44100
    t = np.arange(int(sr * 0.5)) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1).astype(np.float32) * 0.5
    p = tmp_path / "x.wav"
    sf.write(str(p), stereo, sr)
    wav, out_sr = load_audio(p, 16000)
    assert out_sr == 16000 and wav.ndim == 1 and wav.dtype == np.float32
    assert abs(len(wav) - 8000) <= 2
    assert np.abs(wav).max() <= 1.0
    assert audio_info(p)[1] == 44100


def test_downmix_without_resampling(tmp_path):
    sr = 16000
    stereo = np.stack([np.full(sr, 0.2), np.full(sr, 0.4)], axis=1).astype(np.float32)
    p = tmp_path / "s.wav"
    sf.write(str(p), stereo, sr)
    wav, out_sr = load_audio(p, 16000)
    assert out_sr == 16000 and wav.shape == (sr,)
    assert wav[100] == pytest.approx(0.3, abs=1e-3)


def test_save_wav_roundtrip(tmp_path):
    wav = (np.random.RandomState(0).randn(16000) * 0.1).astype(np.float32)
    p = save_wav(tmp_path / "sub" / "y.wav", wav, 16000)
    back, sr = load_audio(p, 16000)
    assert sr == 16000 and len(back) == 16000
    assert np.abs(back - wav).max() < 1e-3  # PCM_16 quantization


def test_to_mono_int16():
    x = np.array([[0, 32767], [0, -32768]], dtype=np.int16)  # (n=2, ch=2)
    m = to_mono_float32(x)
    assert m.dtype == np.float32 and m.shape == (2,)
    assert m[0] == pytest.approx(0.5, abs=1e-4)
