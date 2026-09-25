"""Meta Omnilingual ASR CTC 300M (Apache 2.0) through the sherpa-onnx ONNX export.

The model has one character vocabulary for 1600+ languages (9812 symbols) and no language
input. For Greek voice typing the emission matrix is restricted to the columns that can
occur in Greek text and renormalized over them:

    blank (token 0), word space, apostrophe, and the monotonic lowercase Greek letters.

Measured on the fixture clips, blank + Greek columns already hold >= 98.6% of the mass in
every frame, so the restriction mostly removes the chance of emitting another script. The
emissions stay a proper (T, V) log-softmax, only with V = 39 instead of 9812, so storage,
beam search and scoring work unchanged.

Without a language input the model sometimes writes a short Greek clip in another script
("antropaemo akusta su lego" for "άνθρωπε μου ακούς τι σου λέγω"). ``omni_foreign``:

* ``drop``: condition on the Greek columns only. Where a Latin letter won, blank now wins
  and the letter is lost.
* ``letters``: mass on non-Greek symbols means "a letter was spoken here"; it is moved to
  the Greek letters in proportion to their own probability in that frame. Blank, space and
  apostrophe keep their probability. This is what the app ships.

Input: raw 16 kHz waveform, zero mean / unit variance (as in sherpa-onnx's test.py).
Output: 20 ms frames.

``export_phone_graph`` bakes the restriction into the ONNX graph for the app (omni.onnx +
omni.labels.json); ``OmniOnnxEmitter`` runs either file, so the phone graph can be
evaluated exactly as shipped.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from voicetotext.config import ModelConfig
from voicetotext.types import Emissions

log = logging.getLogger("voicetotext.omni")

BLANK = "<pad>"        # our label for the blank column
BLANK_ID = 0           # the model's CTC blank is token 0 ("<s>" in tokens.txt), not "<pad>"
DELIM = "|"


def read_tokens(path: str | Path) -> dict[int, str]:
    """sherpa-onnx tokens.txt: '<token> <id>' per line; a line with only an id is the space."""
    out: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) == 1:
            out[int(f[0])] = " "
        elif len(f) == 2:
            out[int(f[1])] = f[0]
    return out


def _monotonic_greek(ch: str) -> bool:
    if len(ch) != 1 or not ("Ͱ" <= ch <= "Ͽ"):     # excludes Greek Extended (polytonic)
        return False
    return "GREEK SMALL LETTER" in unicodedata.name(ch, "")


def greek_subset(id2tok: dict[int, str]) -> tuple[list[int], list[str]]:
    """Column ids to keep and their labels, blank first and the space renamed to '|'."""
    ids, labels = [], []
    for i, t in sorted(id2tok.items()):
        if i == BLANK_ID:
            ids.insert(0, i)
            labels.insert(0, BLANK)
        elif t == " ":
            ids.append(i)
            labels.append(DELIM)
        elif t == "'" or _monotonic_greek(t):
            ids.append(i)
            labels.append(t)
    if not labels or labels[0] != BLANK or DELIM not in labels:
        raise ValueError("tokens.txt lacks the blank or the space token")
    return ids, labels


def phone_labels_path(onnx_path: str | Path) -> Path:
    """Where export_phone_graph writes the label list: omni.onnx -> omni.labels.json."""
    return Path(onnx_path).with_suffix(".labels.json")


class OmniOnnxEmitter:
    """Waveforms -> Emissions over the Greek columns.

    ``cfg.onnx_path`` is either the phone graph (``<name>.labels.json`` next to it; the
    graph already outputs the restricted log-probabilities) or the sherpa-onnx export
    (``tokens.txt`` next to it; the Greek restriction runs here in numpy).
    """

    def __init__(self, cfg: ModelConfig, sample_rate: int = 16000) -> None:
        if not cfg.onnx_path:
            raise ValueError("model.onnx_path must point at the Omnilingual ONNX file")
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.path = Path(cfg.onnx_path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.frame_duration_ms = 20.0
        self.foreign = cfg.omni_foreign
        tokens = self.path.parent / "tokens.txt"
        phone_labels = phone_labels_path(self.path)
        # the phone graph may sit in the sherpa folder, next to tokens.txt: its labels win
        if phone_labels.exists():
            self.phone_graph = True
            self.labels = json.loads(phone_labels.read_text(encoding="utf-8"))
            self.model_id = f"{cfg.model_id}+phone:{self.path.name}"
        elif tokens.exists():
            self.phone_graph = False
            self.id2tok = read_tokens(tokens)
            self.cols, self.labels = greek_subset(self.id2tok)
            self._cols = np.array(self.cols)
            self._letter = np.array([len(l) == 1 and l.isalpha() for l in self.labels])
            suffix = "" if self.foreign == "drop" else f"+{self.foreign}"
            self.model_id = f"{cfg.model_id}+onnx:{self.path.name}+greek{len(self.labels)}{suffix}"
        else:
            raise FileNotFoundError(f"neither {tokens} nor {phone_labels} found next to the model")
        self._sess: Any = None
        self.last_greek_mass: float = 1.0   # min over frames of the mass kept, for diagnostics

    @property
    def session(self) -> Any:
        if self._sess is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if self.cfg.onnx_threads:
                opts.intra_op_num_threads = int(self.cfg.onnx_threads)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            log.info("loading %s (%d labels)", self.path, len(self.labels))
            self._sess = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        return self._sess

    def load(self) -> None:
        _ = self.session

    def emit(self, waveforms: list[np.ndarray], audio_ids: list[str] | None = None) -> list[Emissions]:
        sess = self.session
        name = sess.get_inputs()[0].name
        if audio_ids is None:
            audio_ids = [f"utt{i}" for i in range(len(waveforms))]
        out: list[Emissions] = []
        for wav, aid in zip(waveforms, audio_ids):
            x = np.asarray(wav, dtype=np.float32)
            x = ((x - x.mean()) / np.sqrt(x.var() + 1e-5))[None, :].astype(np.float32)
            logits = sess.run(None, {name: x})[0][0].astype(np.float64)
            if self.phone_graph:
                lp = logits
            else:
                full = logits - _logsumexp(logits)
                sub = full[:, self._cols]
                kept = _logsumexp(sub)
                self.last_greek_mass = float(np.exp(kept.min()))
                lp = transfer_foreign(sub, self._letter) if self.foreign == "letters" else sub - kept
            out.append(Emissions(
                logprobs=np.ascontiguousarray(lp, dtype=np.float32),
                labels=self.labels,
                frame_duration_ms=self.frame_duration_ms,
                audio_id=aid,
                model_id=self.model_id,
            ))
        return out


def transfer_foreign(sub: np.ndarray, letter: np.ndarray) -> np.ndarray:
    """``sub``: full-vocabulary log-probs of the kept columns. Returns a log-softmax over them
    where the mass the model put on non-Greek symbols is added to the Greek letters in
    proportion to their own probability: p'(l) = p(l) * (1 + p(foreign) / p(letters))."""
    kept = _logsumexp(sub)
    letters = _logsumexp(sub[:, letter])
    foreign = np.log1p(-np.minimum(np.exp(kept), 1.0 - 1e-12))
    lp = sub.copy()
    lp[:, letter] += np.logaddexp(0.0, foreign - letters)
    return lp - _logsumexp(lp)        # numerical tidy-up only


def _logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=-1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=-1, keepdims=True))


def export_phone_graph(src: str | Path, dst: str | Path, foreign: str = "letters", tokens: str | Path | None = None) -> Path:
    """Append the Greek-column restriction (and the foreign-mass transfer) to the ONNX graph.

    ``tokens`` is the sherpa ``tokens.txt`` (default: next to ``src``). The app's CtcModel
    then feeds ``input_values`` (normalized waveform) and reads (N, T, 39) log-probabilities
    whose columns are ``<dst>.labels.json``. Log-softmax in the app is a no-op on log-probs.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    src, dst = Path(src), Path(dst)
    id2tok = read_tokens(Path(tokens) if tokens else src.parent / "tokens.txt")
    cols, labels = greek_subset(id2tok)
    letter = [i for i, l in enumerate(labels) if len(l) == 1 and l.isalpha()]
    m = onnx.load(str(src))
    g = m.graph
    old_in, old_out = g.input[0].name, g.output[0].name
    for n in g.node:                               # rename the input to what the app feeds
        n.input[:] = ["input_values" if i == old_in else i for i in n.input]
    g.input[0].name = "input_values"

    def const(name, arr):
        g.initializer.append(numpy_helper.from_array(arr, name))
        return name

    c_cols = const("greek_cols", np.array(cols, dtype=np.int64))
    nodes = [
        helper.make_node("LogSoftmax", [old_out], ["g_full"], axis=-1),
        helper.make_node("Gather", ["g_full", c_cols], ["g_sub"], axis=2),
    ]
    out = "g_sub_norm"
    if foreign == "letters":
        c_letter = const("greek_letter_idx", np.array(letter, dtype=np.int64))
        c_mask = const("greek_letter_mask", np.array([1.0 if i in letter else 0.0 for i in range(len(labels))], dtype=np.float32))
        c_one = const("g_one", np.array(1.0, dtype=np.float32))
        c_eps = const("g_eps", np.array(1e-12, dtype=np.float32))
        nodes += [
            helper.make_node("ReduceLogSumExp", ["g_sub"], ["g_kept"], axes=[2], keepdims=1),
            helper.make_node("Gather", ["g_sub", c_letter], ["g_let"], axis=2),
            helper.make_node("ReduceLogSumExp", ["g_let"], ["g_letters"], axes=[2], keepdims=1),
            helper.make_node("Exp", ["g_kept"], ["g_kept_p"]),
            helper.make_node("Sub", [c_one, "g_kept_p"], ["g_for_p"]),
            helper.make_node("Max", ["g_for_p", c_eps], ["g_for_p2"]),
            helper.make_node("Log", ["g_for_p2"], ["g_foreign"]),
            helper.make_node("Sub", ["g_foreign", "g_letters"], ["g_ratio"]),
            helper.make_node("Softplus", ["g_ratio"], ["g_boost"]),
            helper.make_node("Mul", ["g_boost", c_mask], ["g_boost_m"]),
            helper.make_node("Add", ["g_sub", "g_boost_m"], ["g_sub2"]),
            helper.make_node("LogSoftmax", ["g_sub2"], [out], axis=-1),
        ]
    else:
        nodes.append(helper.make_node("LogSoftmax", ["g_sub"], [out], axis=-1))
    g.node.extend(nodes)
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, ["N", "T", len(labels)]))
    onnx.checker.check_model(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    phone_labels_path(dst).write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    return dst
