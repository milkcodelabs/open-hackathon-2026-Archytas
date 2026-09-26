# Personal acoustic model: fine-tune layer 1 (Omnilingual CTC 300M v2) on one speaker's recordings.
# Cells are separated by "# %%"; make_notebook.py turns this file into finetune.ipynb (Colab, GPU).
# The recipe and its measurements are in README.md next to this file.
#
# Data: personal_bundle.zip (pack_bundle.py) in MyDrive/voicetotext/.
#   recordings/  the speaker's recordings (recorder.html); sessions 1-4 train, 5 dev (checkpoint choice), 6 test
#   eval/        FLEURS dev/test clean + Common Voice test 300 clean: does typical speech still work?
#   reg/         Common Voice clips (speaker-disjoint from the 300) replayed during training
#   lm/          the KenLM binary + vocab of configs/omni_beam.yaml, for beam-search scoring
#   voicetotext/ the model-pipeline package, for the same normalize / metrics / beam decoder
# Output (MyDrive/voicetotext/personal/): results, the best checkpoint, and phone/ with the three
# files the app loads as its personal model: omni.personal.onnx (input_values -> (N, T, 39)
# log-probs, int8 per-channel), omni.personal.labels.json, omni.personal.json.
# %%
import subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "NO GPU: Runtime > Change runtime type > T4 GPU")
from google.colab import drive
drive.mount("/content/drive")
BUNDLE = "/content/drive/MyDrive/voicetotext/personal_bundle.zip"
OUT = "/content/drive/MyDrive/voicetotext/personal"
W = "/content/voicetotext"
subprocess.run(["mkdir", "-p", W, OUT], check=True)
subprocess.run(["unzip", "-q", "-o", BUNDLE, "-d", W], check=True)
print(subprocess.run(["ls", W], capture_output=True, text=True).stdout)

# %%
# pyctcdecode + kenlm: the beam decoder of the pipeline. Everything else
# (torch, transformers, soundfile, onnx) is taken from the Colab image and printed.
import os, re
def sh(cmd, env=None, log=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(cmd[:70], "->", r.returncode)
    if log:
        open(log, "w").write(r.stdout + r.stderr)
    if r.returncode:
        txt = r.stdout + r.stderr
        lines = [l for l in txt.splitlines() if "Building CXX" not in l]
        print("\n".join(lines[-60:]))
    return r.returncode
import importlib.util
NEED_INSTALL = importlib.util.find_spec("kenlm") is None     # after an install: restart session, run all
if NEED_INSTALL:
  sh("pip install -q pyctcdecode onnx onnxruntime")
  sh("apt-get -qq install -y build-essential cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev > /dev/null")
  # kenlm's shipped kenlm.cpp predates Python 3.13: regenerate it with a current Cython first
  sh("rm -rf /content/kenlm && git clone -q --depth 1 https://github.com/kpu/kenlm /content/kenlm && pip install -q cython && cd /content/kenlm && cython --cplus -3 python/kenlm.pyx -o python/kenlm.cpp")
  sh("pip install --no-build-isolation /content/kenlm",
     env={**os.environ, "CMAKE_POLICY_VERSION_MINIMUM": "3.5", "MAX_JOBS": "12"}, log="/content/kenlm_build.log")
  raise SystemExit("installed: Runtime > Restart session, then Run all")
import sys, torch, transformers, kenlm, pyctcdecode
sys.path.insert(0, W)
print(sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)

# %%
CFG = dict(
    base="aadel4/omniASR-CTC-300M-v2",   # fairseq2 -> Wav2Vec2ForCTC, parity-verified (1e-4)
    train_sessions=["1", "2", "3", "4"], dev_session="5", test_session="6",
    freeze_below=16,        # transformer layers < this stay frozen (0..23); CNN always frozen
    lr=3e-5, lr_head=1e-4, weight_decay=0.01, warmup=0.1,
    epochs=30, patience=30, batch=16,
    rehearsal=2.0,          # general clips per speaker clip, resampled every epoch
    rehearsal_sets=["fleurs_train", "cv_reg"],
    sel_beam_width=32,      # beam width for the per-epoch checkpoint choice
    fleurs_dev_tol=0.03,    # a checkpoint may lose at most this much FLEURS dev beam WER
    speed=(0.9, 1.0, 1.1), noise_snr_db=(20, 40), gain_db=6.0,
    mask_time_prob=0.05, mask_time_length=5,
    alpha=0.7, beta=3.0, unk=-10.0, beam_width=128,   # configs/omni_beam.yaml
    seed=0,
    speaker="",             # shown on the app screen with the personal model
    run="r5",               # output file prefix in OUT (r5 = the run measured in README.md)
    kd_weight=1.0, kd_temp=1.0,   # rehearsal clips: KL to the frozen base model
)
import random, numpy as np
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# %%
import csv, json, unicodedata
import soundfile as sf

def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and x.ndim == 1, (path, sr, x.shape)
    return x

hx = list(csv.DictReader(open(f"{W}/recordings/metadata.csv", encoding="utf-8-sig")))
def hx_rows(sessions):
    return [{"audio": f"{W}/recordings/{r['file']}", "text": r["text"], "id": r["file"], "said": r["pronounced_as"]}
            for r in hx if r["session"] in sessions and r["set"] != "demo"]
def manifest(sub):
    return [{"audio": f"{W}/{r['audio']}", "text": r["text"], "id": r["utterance_id"]}
            for r in map(json.loads, open(f"{W}/{sub}/manifest.jsonl", encoding="utf-8"))]

SETS = {
    "hx_train": hx_rows(CFG["train_sessions"]),
    "hx_dev": hx_rows([CFG["dev_session"]]),
    "hx_test": hx_rows([CFG["test_session"]]),
    "fleurs_dev": manifest("eval/fleurs_dev_clean"),
    "fleurs_test": manifest("eval/fleurs_test_clean"),
    "cv300": manifest("eval/cv300_clean"),
    "cv_reg": manifest("reg/cv_reg"),
}

# General Greek speech for rehearsal: FLEURS el_gr train (tsv + audio tar from the Hub, as in
# voicetotext/eval/datasets.py). Sentences with Latin letters or digits are left out (the CTC
# alphabet cannot spell them). FLEURS splits are sentence-disjoint, so dev/test stay clean.
import os, re, tarfile
from huggingface_hub import hf_hub_download
def fleurs_split(split):
    tsv = hf_hub_download("google/fleurs", f"data/el_gr/{split}.tsv", repo_type="dataset")
    tar = hf_hub_download("google/fleurs", f"data/el_gr/audio/{split}.tar.gz", repo_type="dataset")
    d = f"/content/fleurs_{split}"
    if not os.path.isdir(d):
        with tarfile.open(tar) as tf:
            tf.extractall(d)
    wavs = {f: os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs}
    rows = []
    for line in open(tsv, encoding="utf-8"):
        p = line.rstrip(chr(10)).split(chr(9))
        if len(p) >= 4 and p[1] in wavs and not re.search(r"[A-Za-z0-9]", p[2]):
            rows.append({"audio": wavs[p[1]], "text": p[2], "id": "fleurs_train_" + p[0]})
    return rows
SETS["fleurs_train"] = fleurs_split("train")
for k, v in SETS.items():
    print(f"{k:12s} {len(v):5d}")
AUDIO = {}
for rows in SETS.values():
    for r in rows:
        AUDIO[r["audio"]] = load_wav(r["audio"])
print("clips loaded", len(AUDIO))

# %%
# Model, vocabulary, Greek columns. Same selection as voicetotext/acoustic/omni.py:
# blank, word space, apostrophe, monotonic lowercase Greek letters.
from transformers import Wav2Vec2ForCTC, AutoProcessor
proc = AutoProcessor.from_pretrained(CFG["base"])
model = Wav2Vec2ForCTC.from_pretrained(CFG["base"])
tok = proc.tokenizer
id2tok = {i: t for t, i in tok.get_vocab().items()}
print("vocab", len(id2tok), "head", model.config.vocab_size, "pad/blank", model.config.pad_token_id,
      repr(id2tok.get(model.config.pad_token_id)), "id0", repr(id2tok.get(0)))
print("feat_extract_norm", model.config.feat_extract_norm, "do_normalize", getattr(proc.feature_extractor, "do_normalize", None))
print("first tokens", [id2tok[i] for i in range(12)])

def monotonic_greek(ch):
    return len(ch) == 1 and "Ͱ" <= ch <= "Ͽ" and "GREEK SMALL LETTER" in unicodedata.name(ch, "")

BLANK_ID = model.config.pad_token_id
SPACE_TOKENS = {" ", "▁", "|"}
cols, labels = [BLANK_ID], ["<pad>"]
for i in sorted(id2tok):
    t = id2tok[i]
    if i == BLANK_ID:
        continue
    if t in SPACE_TOKENS and "|" not in labels:
        cols.append(i); labels.append("|")
    elif t == "'" or monotonic_greek(t):
        cols.append(i); labels.append(t)
print(len(labels), "".join(l for l in labels if len(l) == 1))
assert "|" in labels and len(labels) >= 38, "space token or Greek letters not found: inspect the vocab above"

# %%
# Emissions -> greedy and beam+LM text, scored with the project's normalize and metrics.
import time
from voicetotext.types import Emissions
from voicetotext.acoustic.emissions import greedy_decode
from voicetotext.acoustic.omni import transfer_foreign
from voicetotext.decoding.beam import BeamDecoder
from voicetotext.phonetics.normalize import Normalizer
from voicetotext.eval.metrics import word_counts, char_counts, oracle_char_counts, sum_counts

NORM = Normalizer.from_labels(labels)
LETTER = np.array([len(l) == 1 and l.isalpha() for l in labels])
COLS = torch.tensor(cols)
BEAM = BeamDecoder(labels, lm_path=f"{W}/lm/lm.bin", alpha=CFG["alpha"], beta=CFG["beta"],
                   unk_score_offset=CFG["unk"], beam_width=CFG["beam_width"], n_best=1)

def norm_wave(x):
    return (x - x.mean()) / np.sqrt(x.var() + 1e-5)    # as omni.py / CtcModel.kt

@torch.no_grad()
def emissions(m, wav, restricted):
    m.eval()
    x = torch.from_numpy(norm_wave(wav))[None].to(DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        logits = m(x).logits[0].float()
    if restricted:                                     # fine-tuned head: already 39 columns
        lp = torch.log_softmax(logits, -1).cpu().numpy().astype(np.float64)
    else:                                              # base: full vocab, foreign mass -> letters
        full = torch.log_softmax(logits, -1)[:, COLS.to(DEV)].cpu().numpy().astype(np.float64)
        lp = transfer_foreign(full, LETTER)
    return Emissions(logprobs=lp.astype(np.float32), labels=labels, frame_duration_ms=20.0, audio_id="x", model_id="personal")

def evaluate(m, rows, restricted, beam=True, show=0, decoder=None):
    g, b = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        em = emissions(m, AUDIO[r["audio"]], restricted)
        ref = NORM(r["text"])
        hg = NORM(greedy_decode(em))
        g.append((ref, hg))
        if beam:
            hb = NORM((decoder or BEAM).decode(em)[0].text)
            b.append((ref, hb))
        if i < show:
            print(f"  ref  {ref}\n  said {r.get('said', '')}\n  grd  {hg}" + (f"\n  beam {hb}" if beam else ""))
    def rates(pairs):
        if not pairs:
            return None
        w = sum_counts(word_counts(a, h) for a, h in pairs)
        c = sum_counts(char_counts(a, h) for a, h in pairs)
        o = sum_counts(oracle_char_counts(a, h) for a, h in pairs)
        return {"wer": round(w.rate, 4), "cer": round(c.rate, 4), "oracle_cer": round(o.rate, 4)}
    return {"n": len(rows), "greedy": rates(g), "beam": rates(b), "sec": round(time.time() - t0, 1)}

def report(tag, res):
    for k, v in res.items():
        gb = v["greedy"]; bb = v["beam"] or {}
        print(f"{tag:6s} {k:12s} n={v['n']:4d}  greedy WER {gb['wer']:.3f} CER {gb['cer']:.3f}"
              + (f"   beam WER {bb['wer']:.3f} CER {bb['cer']:.3f} oracle {bb['oracle_cer']:.3f}" if bb else ""))

# %%
# BEFORE: the base model on everything. Sanity: FLEURS test clean with v1 (sherpa) was
# greedy 0.371 / beam 0.139 (model-pipeline/README.md); v2 should be in that neighbourhood.
model.to(DEV)
EVAL_SETS = ["hx_dev", "hx_test", "fleurs_test", "cv300"]
if "before" not in globals():       # the base model does not change between runs
    before = {k: evaluate(model, SETS[k], restricted=False, show=3 if k == "hx_dev" else 0) for k in EVAL_SETS}
if "fleurs_dev" not in before:      # reference point of the forgetting constraint
    before["fleurs_dev"] = evaluate(model, SETS["fleurs_dev"], restricted=False)
report("before", before)
json.dump({"cfg": CFG, "before": before}, open(f"{OUT}/results_before.json", "w"), indent=1)

# %%
# Restrict the head to the Greek columns (the rows of the original head, so step 0 equals
# the base model without the foreign-mass transfer), blank at index 0.
old = model.lm_head
head = torch.nn.Linear(old.in_features, len(cols)).to(DEV)
with torch.no_grad():
    head.weight.copy_(old.weight[COLS.to(DEV)])
    head.bias.copy_(old.bias[COLS.to(DEV)])
model.lm_head = head
model.config.vocab_size = len(cols)
model.config.pad_token_id = 0
model.config.ctc_loss_reduction = "mean"
model.config.ctc_zero_infinity = True
model.config.apply_spec_augment = True
model.config.mask_time_prob = CFG["mask_time_prob"]
model.config.mask_time_length = CFG["mask_time_length"]
model.config.layerdrop = 0.0

model.freeze_feature_encoder()
for p in model.wav2vec2.parameters():
    p.requires_grad = False
layers = model.wav2vec2.encoder.layers
for i in range(CFG["freeze_below"], len(layers)):
    for p in layers[i].parameters():
        p.requires_grad = True
for p in model.wav2vec2.encoder.layer_norm.parameters():
    p.requires_grad = True
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"layers {len(layers)}, trainable {n_tr / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

L2I = {l: i for i, l in enumerate(labels)}
def encode(text):
    ids = [L2I["|" if ch == " " else ch] for ch in NORM(text) if ch == " " or ch in L2I]
    assert ids, text
    return ids
dropped = {ch for r in SETS["hx_train"] for ch in NORM(r["text"]) if ch != " " and ch not in L2I}
print("chars without a column (dropped from targets):", dropped or "none")

def augment(x):
    s = random.choice(CFG["speed"])
    if s != 1.0:
        x = np.interp(np.arange(0, len(x) - 1, s), np.arange(len(x)), x).astype(np.float32)
    x = x * 10 ** (random.uniform(-CFG["gain_db"], CFG["gain_db"]) / 20)
    snr = random.uniform(*CFG["noise_snr_db"])
    x = x + np.random.randn(len(x)).astype(np.float32) * np.sqrt(x.var() / 10 ** (snr / 10) + 1e-12)
    return x

REG_AUDIO = {r["audio"] for k in CFG["rehearsal_sets"] for r in SETS[k]}
def batches(rows):
    rows = rows[:]
    random.shuffle(rows)
    bs = CFG["batch"]
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        wavs = [norm_wave(augment(AUDIO[r["audio"]])) for r in chunk]
        tl = [encode(r["text"]) for r in chunk]
        T, U = max(map(len, wavs)), max(map(len, tl))
        x = np.zeros((len(chunk), T), np.float32); am = np.zeros((len(chunk), T), np.int64)
        y = np.full((len(chunk), U), -100, np.int64)
        for j, (w, t) in enumerate(zip(wavs, tl)):
            x[j, :len(w)] = w; am[j, :len(w)] = 1; y[j, :len(t)] = t
        isreg = torch.tensor([r["audio"] in REG_AUDIO for r in chunk])
        yield torch.from_numpy(x), torch.from_numpy(am), torch.from_numpy(y), isreg

SEL = BeamDecoder(labels, lm_path=f"{W}/lm/lm.bin", alpha=CFG["alpha"], beta=CFG["beta"],
                  unk_score_offset=CFG["unk"], beam_width=CFG["sel_beam_width"], n_best=1)
def dev_scores(m):
    hx_ = evaluate(m, SETS["hx_dev"], restricted=True, decoder=SEL)
    fl_ = evaluate(m, SETS["fleurs_dev"], restricted=True, decoder=SEL)
    return hx_, fl_
step0 = dev_scores(model)
FLEURS_BASE = step0[1]["beam"]["wer"]     # restricted head = base model without the transfer
print(f"step0 hx_dev beam WER {step0[0]['beam']['wer']:.3f}  fleurs_dev beam WER {FLEURS_BASE:.3f}")

# %%
# Training. Rehearsal: each epoch adds rehearsal x |hx_train| general clips (FLEURS train +
# CV reg), resampled. Checkpoint choice: lowest hx_dev beam WER (width sel_beam_width) among
# epochs whose FLEURS dev beam WER is within fleurs_dev_tol of the base model. The test
# session and FLEURS test / CV 300 are not touched here.
# kd_weight > 0: on rehearsal clips the loss also has KL(base || model) per frame, so the
# model keeps its behaviour on typical speech while it adapts to the speaker.
import copy
from transformers import get_linear_schedule_with_warmup

use_mask = model.config.feat_extract_norm == "layer"
head_p = list(model.lm_head.parameters())
body_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("lm_head")]
opt = torch.optim.AdamW([{"params": body_p, "lr": CFG["lr"]}, {"params": head_p, "lr": CFG["lr_head"]}],
                        weight_decay=CFG["weight_decay"])
REG = [r for k in CFG["rehearsal_sets"] for r in SETS[k]]
n_reg = min(int(len(SETS["hx_train"]) * CFG["rehearsal"]), len(REG))
steps_ep = -(-(len(SETS["hx_train"]) + n_reg) // CFG["batch"])
sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps_ep * CFG["epochs"]), steps_ep * CFG["epochs"])
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
teacher = None
if CFG["kd_weight"] > 0:            # the model here is still the (restricted) base model
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

def kd_loss(x, am, logits):
    with torch.no_grad():
        t = teacher(x, attention_mask=am if use_mask else None).logits.float()
    T = CFG["kd_temp"]
    lt, ls = torch.log_softmax(t / T, -1), torch.log_softmax(logits.float() / T, -1)
    flen = model._get_feat_extract_output_lengths(am.sum(-1))
    mask = (torch.arange(ls.shape[1], device=ls.device)[None] < flen[:, None]).float()
    kl = (lt.exp() * (lt - ls)).sum(-1)
    return (kl * mask).sum() / mask.sum()
print(f"rehearsal pool {len(REG)}, {n_reg} per epoch, {steps_ep} steps per epoch")

def trainable_state():
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

best = {"key": (step0[0]["beam"]["wer"], step0[0]["beam"]["cer"]), "epoch": 0, "state": trainable_state()}
best_any = dict(best)               # ignoring the FLEURS limit, for the report only
history, bad = [], 0
for ep in range(1, CFG["epochs"] + 1):
    model.train()
    rows = SETS["hx_train"] + random.sample(REG, n_reg)
    tot, tkd, nb, t0 = 0.0, 0.0, 0, time.time()
    for x, am, y, isreg in batches(rows):
        x, am, y, isreg = x.to(DEV), am.to(DEV), y.to(DEV), isreg.to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            out = model(x, attention_mask=am if use_mask else None, labels=y)
            loss = out.loss
            if teacher is not None and bool(isreg.any()):
                kd = kd_loss(x[isreg], am[isreg], out.logits[isreg])
                loss = loss + CFG["kd_weight"] * kd
                tkd += kd.item()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(body_p + head_p, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot += out.loss.item(); nb += 1
    hx_, fl_ = dev_scores(model)
    dev, gen = hx_["beam"], fl_["beam"]
    ok = gen["wer"] <= FLEURS_BASE + CFG["fleurs_dev_tol"]
    key = (dev["wer"], dev["cer"])
    mark = "" if ok else "  (fleurs over limit)"
    if key < best_any["key"]:
        best_any = {"key": key, "epoch": ep, "state": trainable_state()}
    if ok and key < best["key"]:
        best, bad, mark = {"key": key, "epoch": ep, "state": trainable_state()}, 0, "  *best"
    else:
        bad += 1
    history.append({"epoch": ep, "loss": tot / nb, "hx_dev": hx_, "fleurs_dev": fl_})
    print(f"ep {ep:2d} loss {tot / nb:.3f} kd {tkd / nb:.3f}  hx_dev beam WER {dev['wer']:.3f} CER {dev['cer']:.3f}  "
          f"fleurs_dev beam WER {gen['wer']:.3f}  {time.time() - t0:.0f}s{mark}")
    if bad >= CFG["patience"]:
        print("early stop"); break
print("best epoch", best["epoch"], best["key"], "| best ignoring the FLEURS limit", best_any["epoch"], best_any["key"])

# %%
# AFTER: best checkpoint on everything (greedy + beam/LM), saved to Drive with the numbers.
# If the best epoch ignoring the FLEURS limit differs, it is scored too (report only).
def load_state(st):
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, v in st.items():
            params[n].copy_(v.to(DEV))
after_any = None
if best_any["epoch"] != best["epoch"]:
    load_state(best_any["state"])
    after_any = {k: evaluate(model, SETS[k], restricted=True) for k in EVAL_SETS}
load_state(best["state"])
after = {k: evaluate(model, SETS[k], restricted=True, show=3 if k == "hx_test" else 0) for k in EVAL_SETS}
report("before", before)
report("after", after)
if after_any:
    report(f"any{best_any['epoch']:02d}", after_any)
json.dump({"cfg": CFG, "labels": labels, "best_epoch": best["epoch"], "best_any_epoch": best_any["epoch"],
           "history": history, "before": before, "after": after, "after_any": after_any},
          open(f"{OUT}/{CFG['run']}_results.json", "w"), indent=1, ensure_ascii=False)
model.save_pretrained(f"{OUT}/{CFG['run']}_hf")
json.dump(labels, open(f"{OUT}/{CFG['run']}_hf/labels.json", "w"), ensure_ascii=False)
print("saved to", OUT)

# %%
# Save the best checkpoint ignoring the FLEURS limit too (the speaker's own model).
if after_any:
    load_state(best_any["state"])
    d = f"{OUT}/{CFG['run']}_any{best_any['epoch']:02d}_hf"
    model.save_pretrained(d)
    json.dump(labels, open(f"{d}/labels.json", "w"), ensure_ascii=False)
    print("saved", d)
    load_state(best["state"])

# %%
# Phone export: input_values (N, samples, normalized) -> (N, T, 39) log-probs, labels.json.
# Runs in a fresh process from the saved checkpoint: the kernel already holds an ml_dtypes
# that is too old for the image's onnx. int8 = MatMul only, like the sherpa export on the phone.
# Then parity against torch and greedy WER of fp32 / int8 here, with onnxruntime only.
import onnxruntime as ort
src = f"{OUT}/{CFG['run']}_hf"
fp32, int8 = f"/content/{CFG['run']}.onnx", f"/content/{CFG['run']}.int8.onnx"
open("/content/export_onnx.py", "w").write('''
import sys, torch
from transformers import Wav2Vec2ForCTC
src, fp32, int8 = sys.argv[1:4]
m = Wav2Vec2ForCTC.from_pretrained(src).eval()
class E(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, input_values):
        return torch.log_softmax(s.m(input_values).logits, -1)
torch.onnx.export(E(m), (torch.randn(1, 32000),), fp32, input_names=["input_values"], output_names=["logprobs"],
                  dynamic_axes={"input_values": {0: "N", 1: "S"}, "logprobs": {0: "N", 1: "T"}},
                  opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(fp32, int8, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
print("exported")
''')
sh("pip install -q -U ml_dtypes onnx")
sh(f"python /content/export_onnx.py {src} {fp32} {int8}", log="/content/export.log")
print(open("/content/export.log").read()[-600:])

class Export(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, input_values):
        return torch.log_softmax(self.m(input_values).logits, -1)
cpu_model = copy.deepcopy(model).float().cpu().eval()

def onnx_eval(path, rows):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    pairs, dmax = [], 0.0
    for r in rows:
        w = norm_wave(AUDIO[r["audio"]])[None].astype(np.float32)
        lp = s.run(None, {"input_values": w})[0][0]
        with torch.no_grad():
            ref_lp = Export(cpu_model)(torch.from_numpy(w))[0].numpy()
        dmax = max(dmax, float(np.abs(np.exp(lp) - np.exp(ref_lp)).max()))
        em = Emissions(logprobs=lp.astype(np.float32), labels=labels, frame_duration_ms=20.0, audio_id="x", model_id="onnx")
        pairs.append((NORM(r["text"]), NORM(greedy_decode(em)), NORM(BEAM.decode(em)[0].text)))
    w = sum_counts(word_counts(a, h) for a, h, _ in pairs)
    b = sum_counts(word_counts(a, h) for a, _, h in pairs)
    return {"greedy_wer": round(w.rate, 4), "beam_wer": round(b.rate, 4), "max_abs_dp": round(dmax, 5)}

for p in (fp32, int8):
    print(p.split("/")[-1], subprocess.run(["du", "-h", p], capture_output=True, text=True).stdout.split()[0],
          onnx_eval(p, SETS["hx_test"]))
for p in (fp32, int8):
    subprocess.run(["cp", p, OUT], check=True)
json.dump(labels, open(f"{OUT}/{CFG['run']}.labels.json", "w"), ensure_ascii=False)
print("copied to", OUT)

# %%
# int8 variants for the phone: per-channel weights, optionally keeping the CTC head in fp32.
open("/content/quant_variants.py", "w").write('''
import sys, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
fp32, out_pc, out_pc_head = sys.argv[1:4]
quantize_dynamic(fp32, out_pc, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8, per_channel=True)
m = onnx.load(fp32, load_external_data=False)
last = [n.name for n in m.graph.node if n.op_type == "MatMul"][-1:]
quantize_dynamic(fp32, out_pc_head, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
                 per_channel=True, nodes_to_exclude=last)
print("ok", last)
''')
pc, pch = f"/content/{CFG['run']}.int8pc.onnx", f"/content/{CFG['run']}.int8pch.onnx"
sh(f"python /content/quant_variants.py {fp32} {pc} {pch}", log="/content/quant.log")
print(open("/content/quant.log").read()[-300:])
for p in (pc, pch):
    print(p.split("/")[-1], subprocess.run(["du", "-h", p], capture_output=True, text=True).stdout.split()[0],
          onnx_eval(p, SETS["hx_test"]))
subprocess.run(["cp", pc, f"{OUT}/{CFG['run']}.int8pc.onnx"], check=True)   # the phone candidate
print("copied", pc, "to", OUT)

# %%
# The three files the app loads as its personal model (Recognizer.kt: omni.personal.*). Copy
# them to the phone's model folder, or publish them on the release and put their sizes and
# SHA-256 in ModelDownloader.PERSONAL. omni.personal.json is one line on the app's screen.
import hashlib, shutil
PHONE = f"{OUT}/phone"
os.makedirs(PHONE, exist_ok=True)
shutil.copyfile(pc, f"{PHONE}/omni.personal.onnx")
json.dump(labels, open(f"{PHONE}/omni.personal.labels.json", "w"), ensure_ascii=False)
hx_wer = after["hx_test"]["beam"]["wer"]
json.dump({"speaker": CFG["speaker"], "base": CFG["base"].split("/")[-1], "trained": f"{CFG['run']}, epoch {best['epoch']}",
           "note": f"WER session {CFG['test_session']}: {hx_wer:.3f} (base {before['hx_test']['beam']['wer']:.3f})"},
          open(f"{PHONE}/omni.personal.json", "w"), ensure_ascii=False)
for f in sorted(os.listdir(PHONE)):
    b = open(f"{PHONE}/{f}", "rb").read()
    print(f"{f:28s} {len(b):>12,d}  {hashlib.sha256(b).hexdigest()}")
