"""Export the layer-2c neural LM (a small Greek GPT-2) to the ONNX graph the app runs.

    voicetotext export gpt2 lighteternal/gpt2-finetuned-greek out_dir

Writes el_gpt2.onnx (fp32, not shipped), el_gpt2.int8.onnx, el_gpt2.vocab.json,
el_gpt2.merges.txt and el_gpt2.meta.json. Model:
https://huggingface.co/lighteternal/gpt2-finetuned-greek (Apache-2.0, 124M, Greek byte-level
BPE, 50,257 tokens). Needs the ``neural`` extra (torch, transformers) and the ``export`` extra (onnx).

Graph contract (NeuralRescorer.kt):
    inputs   input_ids       int64 [batch, tokens]   right-padded; any id may pad
             attention_mask  int64 [batch, tokens]   1 for real tokens, 0 for padding
    output   token_logp      float [batch, tokens-1] natural-log P(token t | tokens < t) for
                                                     t = 1..tokens-1; 0 at padded positions
Sequence = [start token 0 (<|endoftext|>)] + context tokens + sentence tokens; the app sums
the outputs that belong to the sentence.

Two export details that matter:
  * transformers builds its causal mask with vmap, which the ONNX exporter cannot trace, so
    the forward pass is written out with the model's own modules and a plain triangular mask,
    and checked against transformers before exporting (the release differed by at most 5e-5).
  * int8 dynamic quantization includes Gather, so the embedding table is quantized too
    (otherwise the file is 282 MB instead of 164 MB).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def export(model_id: str, out_dir: str | Path) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    model.config.use_cache = False

    class MiniGPT2(torch.nn.Module):
        """GPT-2 forward pass with the model's own modules and a plain triangular mask."""

        def __init__(self, hf):
            super().__init__()
            t = hf.transformer
            self.wte, self.wpe, self.h, self.ln_f = t.wte, t.wpe, t.h, t.ln_f
            self.n_head = hf.config.n_head
            self.n_embd = hf.config.n_embd

        def forward(self, input_ids):
            b, n = input_ids.shape
            pos = torch.arange(n, device=input_ids.device).unsqueeze(0)
            x = self.wte(input_ids) + self.wpe(pos)
            causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=input_ids.device))
            hd = self.n_embd // self.n_head
            for blk in self.h:
                q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(self.n_embd, dim=2)
                q = q.view(b, n, self.n_head, hd).transpose(1, 2)
                k = k.view(b, n, self.n_head, hd).transpose(1, 2)
                v = v.view(b, n, self.n_head, hd).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
                att = att.masked_fill(~causal, torch.finfo(att.dtype).min)
                y = (torch.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(b, n, self.n_embd)
                x = x + blk.attn.c_proj(y)
                x = x + blk.mlp(blk.ln_2(x))
            return self.ln_f(x) @ self.wte.weight.T

    mini = MiniGPT2(model).eval()
    with torch.no_grad():
        probe = torch.tensor([[tok.eos_token_id] + tok("καλημέρα σε όλους, τι κάνετε;", add_special_tokens=False)["input_ids"]])
        diff = (mini(probe) - model(input_ids=probe).logits).abs().max().item()
    if diff > 1e-3:
        raise RuntimeError(f"re-implemented GPT-2 differs from transformers by {diff}")

    class TokenLogp(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            logits = self.m(input_ids)
            logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            nxt = input_ids[:, 1:].unsqueeze(-1)
            return torch.gather(logp, 2, nxt).squeeze(-1) * attention_mask[:, 1:].float()

    wrap = TokenLogp(mini).eval()
    ids = torch.tensor([tok("καλημέρα σε όλους", add_special_tokens=False)["input_ids"]] * 2)
    ids = torch.cat([torch.full((2, 1), tok.eos_token_id), ids], dim=1)
    mask = torch.ones_like(ids)
    fp32 = out / "el_gpt2.onnx"
    torch.onnx.export(
        wrap, (ids, mask), str(fp32), opset_version=17,
        input_names=["input_ids", "attention_mask"], output_names=["token_logp"],
        dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
                      "token_logp": {0: "batch", 1: "steps"}},
        dynamo=False,
    )
    int8 = out / "el_gpt2.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), op_types_to_quantize=["MatMul", "Gemm", "Gather"], weight_type=QuantType.QInt8)

    tmp = out / "_tok"
    tok.save_pretrained(tmp)
    shutil.copyfile(tmp / "vocab.json", out / "el_gpt2.vocab.json")
    shutil.copyfile(tmp / "merges.txt", out / "el_gpt2.merges.txt")
    shutil.rmtree(tmp)
    (out / "el_gpt2.meta.json").write_text(json.dumps({
        "model_id": model_id, "start_token_id": tok.eos_token_id, "vocab_size": len(tok),
        "max_positions": model.config.n_positions, "reimplementation_max_abs_diff": diff,
    }, indent=2), encoding="utf-8")
    return int8
