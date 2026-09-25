"""Score sentences with the exported GPT-2 and re-rank an N-best list, as NeuralRescorer.kt does.

    voicetotext gpt2-probe el_gpt2.int8.onnx lighteternal/gpt2-finetuned-greek \\
        "αναφέρθηκε στις φήμες ως πολιτικό κουτσομπολιό και ανοησίες"

prints the token ids and the sentence log-probability. The app logs the same probe when it
loads the model (logcat tag NeuralRescorer). The release check: identical ids, logp -69.53
here against -69.35 on a Galaxy A55 (int8 kernels differ between x86 and ARM).

Re-ranking:

    for the top K hypotheses (K = 20):
        total = acoustic + ngram + GAMMA * log P_gpt2(sentence | context)
    a hypothesis whose acoustic score is more than ACOUSTIC_GUARD nats below the best one
    cannot win.

The app additionally subtracts 1000 from an excluded hypothesis's language-model score, so
it cannot show up as a likely alternative on screen; that does not change which sentence wins.
Weights chosen on FLEURS dev + half of Common Voice: WER 0.129 -> 0.114 on FLEURS test,
0.132 -> 0.123 on the other Common Voice half.
"""

from __future__ import annotations

from dataclasses import dataclass

GAMMA = 0.5
NGRAM_WEIGHT = 1.0
ACOUSTIC_GUARD = 8.0
TOP_K = 20
START_ID = 0                 # <|endoftext|>
MAX_CONTEXT_TOKENS = 48
PROBE = "αναφέρθηκε στις φήμες ως πολιτικό κουτσομπολιό και ανοησίες"


class Gpt2Scorer:
    """Sentence log-probabilities from el_gpt2.int8.onnx."""

    def __init__(self, onnx_path: str, tokenizer_id: str, threads: int = 4) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def score(self, sentences: list[str], context: str = "") -> list[float]:
        import numpy as np

        ctx = context.strip()
        prefix = [START_ID] + (self.ids(ctx)[-MAX_CONTEXT_TOKENS:] if ctx else [])
        lead = " " if ctx else ""
        seqs = [prefix + self.ids(lead + s) for s in sentences]
        n = max(len(x) for x in seqs)
        ids = np.full((len(seqs), n), START_ID, dtype=np.int64)
        mask = np.zeros((len(seqs), n), dtype=np.int64)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = x
            mask[j, :len(x)] = 1
        lp = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        return [float(lp[j, len(prefix) - 1:len(x) - 1].sum()) for j, x in enumerate(seqs)]


@dataclass
class Hyp:
    text: str
    acoustic: float
    ngram: float


def rerank(hyps: list[Hyp], neural: list[float]) -> list[Hyp]:
    """Best first. ``neural[i]`` is the GPT-2 score of ``hyps[i]`` for the first TOP_K."""
    head = hyps[:TOP_K]
    best_ac = head[0].acoustic
    eligible, excluded = [], []
    for h, n in zip(head, neural):
        total = h.acoustic + NGRAM_WEIGHT * h.ngram + GAMMA * n
        (excluded if best_ac - h.acoustic > ACOUSTIC_GUARD else eligible).append((total, h))
    eligible.sort(key=lambda x: -x[0])
    return [h for _, h in eligible] + [h for _, h in excluded] + hyps[TOP_K:]
