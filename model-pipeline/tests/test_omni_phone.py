"""The phone graph computes exactly what the Python emitter computes on the sherpa export."""

import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")  # the `export` extra
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from voicetotext.acoustic.omni import (  # noqa: E402
    OmniOnnxEmitter, _logsumexp, export_phone_graph, greek_subset, read_tokens, transfer_foreign,
)
from voicetotext.config import ModelConfig  # noqa: E402

TOKENS = "<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n4\nα 5\nZ 6\nά 7\nἀ 8\n' 9\nς 10\nΑ 11\nж 12\n"


def _identity_model(path, vocab: int) -> None:
    """Stand-in for the acoustic model: its (1, T, V) input comes out as the logits."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "T", vocab])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "T", vocab])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["logits"])], "g", [x], [y])
    # an IR version every onnxruntime release loads (newer onnx packages default higher)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


@pytest.mark.parametrize("foreign", ["letters", "drop"])
def test_phone_graph_matches_numpy(tmp_path, foreign):
    (tmp_path / "tokens.txt").write_text(TOKENS, encoding="utf-8")
    src = tmp_path / "model.onnx"
    _identity_model(src, 13)
    dst = export_phone_graph(src, tmp_path / "phone" / "omni.onnx", foreign)
    labels = json.loads((tmp_path / "phone" / "omni.labels.json").read_text(encoding="utf-8"))
    cols, expected_labels = greek_subset(read_tokens(tmp_path / "tokens.txt"))
    assert labels == expected_labels

    import onnxruntime as ort

    logits = np.random.RandomState(0).randn(1, 20, 13).astype(np.float32) * 3
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].name == "input_values"
    got = sess.run(None, {"input_values": logits})[0][0]

    full = logits[0].astype(np.float64) - _logsumexp(logits[0].astype(np.float64))
    sub = full[:, cols]
    letter = np.array([len(l) == 1 and l.isalpha() for l in labels])
    want = transfer_foreign(sub, letter) if foreign == "letters" else sub - _logsumexp(sub)
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_emitter_reads_the_phone_graph(tmp_path):
    (tmp_path / "tokens.txt").write_text(TOKENS, encoding="utf-8")
    src = tmp_path / "model.onnx"
    _identity_model(src, 13)
    # written next to tokens.txt on purpose: the phone labels must win
    dst = export_phone_graph(src, tmp_path / "omni.onnx")
    em = OmniOnnxEmitter(ModelConfig(onnx_path=str(dst)))
    assert em.phone_graph and em.labels == ["<pad>", "|", "α", "ά", "'", "ς"]
    sherpa = OmniOnnxEmitter(ModelConfig(onnx_path=str(src)))
    assert not sherpa.phone_graph and sherpa.labels == em.labels
