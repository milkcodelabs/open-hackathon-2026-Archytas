"""Single source of configuration: pydantic models + one YAML file.

Resolution order for the YAML path: explicit argument > ``VOICETOTEXT_CONFIG`` env var >
``configs/default.yaml`` next to the package. ``VOICETOTEXT_DATA_DIR`` overrides
``paths.data_dir``. No paths or model ids are hardcoded anywhere else in the package.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


class ModelConfig(BaseModel):
    """Meta Omnilingual ASR CTC 300M, run with onnxruntime (acoustic/omni.py)."""

    model_id: str = "omniASR_CTC_300M"
    # Either the sherpa-onnx export (model.onnx with tokens.txt next to it) or the phone
    # graph written by `voicetotext export omni-phone` (omni.onnx with omni.labels.json).
    onnx_path: str | None = None
    onnx_threads: int | None = None  # intra-op threads for onnxruntime (None -> default)
    # Sherpa export only: what to do with probability mass on non-Greek symbols.
    omni_foreign: Literal["drop", "letters"] = "letters"


class AudioConfig(BaseModel):
    sample_rate: int = 16000


class DecoderConfig(BaseModel):
    kind: Literal["greedy", "beam"] = "greedy"
    n_best: int = 20  # always keep the full N-best, even when only the 1-best is used
    beam_width: int = 128
    alpha: float = 0.7  # LM weight; tuned on FLEURS dev (never on a test split)
    beta: float = 3.0  # word insertion bonus; tuned with alpha
    lm_path: str | None = None  # KenLM binary/ARPA; None -> beam search without LM
    unk_score_offset: float = -10.0  # pyctcdecode penalty for out-of-vocabulary words


class NormalizeConfig(BaseModel):
    # None -> derived from the CTC vocabulary at runtime (accents kept iff the vocab has them).
    strip_accents: bool | None = None


class LMConfig(BaseModel):
    """n-gram build parameters (decoding/lm.py). The phone model's values are in configs/lm_phone.yaml."""

    order: int = 5
    vocab_size: int = 300000
    prune: str = "0 0 1 1 2"  # lmplz --prune per order
    # cap on normalized sentences taken per corpus (None = all); keeps lmplz within CPU/disk budget
    max_sentences: dict[str, int | None] = Field(default_factory=lambda: {"opensubtitles": 15_000_000, "wikipedia": None})
    # digits: "verbalize" -> spoken Greek cardinals (phonetics/numbers.py); "drop" -> whole sentence removed.
    # Latin letters cannot be emitted by the CTC alphabet, so such sentences are dropped entirely
    # (never truncated). Evaluation uses the same rule: references with digits/Latin are excluded
    # from the primary ("clean") manifests instead of being scored truncated.
    digits: Literal["verbalize", "drop"] = "verbalize"
    drop_latin_sentences: bool = True
    max_words_per_sentence: int = 60
    quantize_bits: int = 8  # build_binary -q / -b


class PathsConfig(BaseModel):
    data_dir: str = "~/voicetotext_data"
    results_dir: str | None = None  # None -> <data_dir>/results
    manifests_dir: str | None = None  # None -> <data_dir>/manifests
    emissions_cache_dir: str | None = None  # None -> <data_dir>/emissions_cache
    lm_dir: str | None = None  # None -> <data_dir>/lm
    kenlm_bin: str = "~/tools/kenlm/build/bin"

    @property
    def data(self) -> Path:
        return _expand(self.data_dir)

    @property
    def results(self) -> Path:
        return _expand(self.results_dir) if self.results_dir else self.data / "results"

    @property
    def manifests(self) -> Path:
        return _expand(self.manifests_dir) if self.manifests_dir else self.data / "manifests"

    @property
    def emissions_cache(self) -> Path:
        return _expand(self.emissions_cache_dir) if self.emissions_cache_dir else self.data / "emissions_cache"

    @property
    def lm(self) -> Path:
        return _expand(self.lm_dir) if self.lm_dir else self.data / "lm"

    @property
    def kenlm(self) -> Path:
        return _expand(self.kenlm_bin)


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    lm: LMConfig = Field(default_factory=LMConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    source_path: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _apply_env(self) -> "Config":
        env_data = os.environ.get("VOICETOTEXT_DATA_DIR")
        if env_data:
            self.paths.data_dir = env_data
        return self

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        cfg_path = Path(path) if path else Path(os.environ.get("VOICETOTEXT_CONFIG", DEFAULT_CONFIG_PATH))
        data: dict[str, Any] = {}
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        if overrides:
            data = _deep_merge(data, overrides)
        cfg = cls.model_validate(data)
        cfg.source_path = str(cfg_path)
        return cfg

    @classmethod
    def default(cls) -> "Config":
        return cls.load(DEFAULT_CONFIG_PATH)

    # ------------------------------------------------------------------ identity
    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)

    def fingerprint(self) -> str:
        """Short stable hash of the effective configuration; stored with every evaluation run."""
        return hashlib.sha1(self.to_json().encode("utf-8")).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_override(s: str) -> dict[str, Any]:
    """Parse ``a.b.c=value`` (CLI ``--set``) into a nested dict; the value goes through YAML."""
    key, sep, raw = s.partition("=")
    if not sep:
        raise ValueError(f"override must look like section.key=value, got {s!r}")
    value = yaml.safe_load(raw)
    node: dict[str, Any] = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node
