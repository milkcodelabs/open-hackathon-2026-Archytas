# Test fixtures

* `labels.json` - the 39 labels of the phone graph `omni.onnx` (blank, word delimiter,
  apostrophe and the 36 monotonic lowercase Greek letters), in column order.
* `audio/clip*.wav`, `emissions/clip*.npz`, `expected.json` - three short clips from
  FLEURS `el_gr` (Google, CC-BY 4.0, https://huggingface.co/datasets/google/fleurs) with
  the emission matrices produced by `omni.onnx` and their greedy text. `clip1.wav` is the
  app's `test.wav`. Regenerate with `scripts/make_fixtures.py`. Unit tests never download
  models or data.
