# Personal acoustic model from a handful of sentences (a real speaker with dysarthria: 15 read
# sentences, 82 s of speech). Cells are separated by "# %%"; `python make_notebook.py few_sentences.py`
# writes few_sentences.ipynb. README.md next to this file has the recipe and the measurements.
#
# Data: personal_bundle.zip (pack_bundle.py) in MyDrive/voicetotext/. One row of
# recordings/metadata.csv is set=demo: never trained on, never used for a choice.
# Steps: base model -> 7-fold cross-validation of the LoRA recipe (every training sentence scored by
# a model that did not hear it) -> per-sentence table -> final model on all training sentences ->
# int8 ONNX and the three phone files omni.personal.<name>.* (the app offers every such model).

# %%
import os, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before CUDA starts
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "NO GPU: Runtime > Change runtime type > A100")
from google.colab import drive
drive.mount("/content/drive")
BUNDLE = "/content/drive/MyDrive/voicetotext/personal_bundle.zip"
OUT = "/content/drive/MyDrive/voicetotext/few_sentences"
W = "/content/voicetotext"
subprocess.run(["mkdir", "-p", W, OUT], check=True)
if not os.path.isdir(f"{W}/voicetotext"):
    subprocess.run(["unzip", "-q", "-o", BUNDLE, "-d", W], check=True)
print(sorted(os.listdir(W)))

# %%
# pyctcdecode + kenlm (beam decoder); everything else comes from the Colab image.
import re, importlib.util
def sh(cmd, env=None, log=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(cmd[:70], "->", r.returncode)
    if log:
        open(log, "w").write(r.stdout + r.stderr)
    if r.returncode:
        print("\n".join([l for l in (r.stdout + r.stderr).splitlines() if "Building CXX" not in l][-60:]))
    return r.returncode
if importlib.util.find_spec("kenlm") is None:     # after an install: restart session, run all
  sh("pip install -q pyctcdecode onnx onnxruntime")
  sh("apt-get -qq install -y build-essential cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev > /dev/null")
  sh("rm -rf /content/kenlm && git clone -q --depth 1 https://github.com/kpu/kenlm /content/kenlm && pip install -q cython && cd /content/kenlm && cython --cplus -3 python/kenlm.pyx -o python/kenlm.cpp")
  sh("pip install --no-build-isolation /content/kenlm",
     env={**os.environ, "CMAKE_POLICY_VERSION_MINIMUM": "3.5", "MAX_JOBS": "12"}, log="/content/kenlm_build.log")
  raise SystemExit("installed: Runtime > Restart session, then Run all")
import sys, torch, transformers, kenlm, pyctcdecode
sys.path.insert(0, W)
print(sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)

# %%
CFG = dict(
    base="aadel4/omniASR-CTC-300M-v2",
    name="s2",              # phone files: omni.personal.<name>.onnx / .labels.json / .json
    speaker="",             # shown on the app screen (the "speaker" of the .json)
    folds=7, seed=0,
    epochs=10, eval_epochs=[4, 6, 8, 10], fleurs_epochs=[10],
    export_epoch=4,         # the cross-validation optimum (README.md)
    batch=16,
    repeat=10,              # every speaker clip appears this many times per epoch, augmented differently
    rehearsal=1.0,          # general clips per (repeated) speaker clip
    rehearsal_sets=["fleurs_train", "cv_reg"],
    reg_max_s=10.0,         # rehearsal clips up to this long (LoRA keeps activations of all 24 layers)
    kd_weight=1.0, kd_temp=1.0,   # rehearsal clips: KL to the frozen base model
    speed=(0.8, 0.9, 1.0, 1.1, 1.2), noise_snr_db=(15, 40), gain_db=6.0,
    mask_time_prob=0.1, mask_time_length=5,
    warmup=0.1, weight_decay=0.01,
    lora=dict(rank=16, alpha=32, layers=list(range(24)), lr=5e-4, lr_head=1e-4),
    alpha=0.7, beta=3.0, unk=-10.0,   # configs/omni_beam.yaml, as on the phone
)
import csv, json, math, random, time, copy, gc, tarfile, unicodedata, shutil, hashlib
import numpy as np
import soundfile as sf
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# %%
def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and x.ndim == 1, (path, sr, x.shape)
    return x

REC = [{"audio": f"{W}/recordings/{r['file']}", "text": r["text"], "id": r["file"].split("/")[-1][:-4], "set": r["set"]}
       for r in csv.DictReader(open(f"{W}/recordings/metadata.csv", encoding="utf-8-sig"))]
DEMO = [r for r in REC if r["set"] == "demo"]
TRAIN = [r for r in REC if r["set"] != "demo"]
assert len(DEMO) == 1, "mark exactly one sentence set=demo in metadata.csv"
order = list(range(len(TRAIN)))
random.Random(CFG["seed"]).shuffle(order)
FOLDS = [[TRAIN[i] for i in order[k::CFG["folds"]]] for k in range(CFG["folds"])]
print("demo:", DEMO[0]["id"], DEMO[0]["text"])
for k, f in enumerate(FOLDS):
    print("fold", k, [r["id"] for r in f])

def manifest(sub):
    return [{"audio": f"{W}/{r['audio']}", "text": r["text"], "id": r["utterance_id"]}
            for r in map(json.loads, open(f"{W}/{sub}/manifest.jsonl", encoding="utf-8"))]
SETS = {"fleurs_dev": manifest("eval/fleurs_dev_clean"), "cv_reg": manifest("reg/cv_reg")}

# General Greek speech for rehearsal: FLEURS el_gr train, without sentences that have Latin
# letters or digits (the CTC alphabet cannot spell them).
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
AUDIO = {r["audio"]: load_wav(r["audio"]) for rows in [REC] + list(SETS.values()) for r in rows}
REG = [r for k in CFG["rehearsal_sets"] for r in SETS[k] if len(AUDIO[r["audio"]]) <= CFG["reg_max_s"] * 16000]
REG_AUDIO = {r["audio"] for r in REG}
print({k: len(v) for k, v in SETS.items()}, "rehearsal pool", len(REG))

# %%
# Base model, Greek columns (blank, word space, apostrophe, monotonic lowercase letters), and the
# restricted base: its head keeps only those rows, so at step 0 it is the base model.
from transformers import Wav2Vec2ForCTC, AutoProcessor
proc = AutoProcessor.from_pretrained(CFG["base"])
BASE = Wav2Vec2ForCTC.from_pretrained(CFG["base"]).eval()
id2tok = {i: t for t, i in proc.tokenizer.get_vocab().items()}
def monotonic_greek(ch):
    return len(ch) == 1 and "Ͱ" <= ch <= "Ͽ" and "GREEK SMALL LETTER" in unicodedata.name(ch, "")
BLANK_ID = BASE.config.pad_token_id
cols, labels = [BLANK_ID], ["<pad>"]
for i in sorted(id2tok):
    t = id2tok[i]
    if i == BLANK_ID:
        continue
    if t in {" ", "▁", "|"} and "|" not in labels:
        cols.append(i); labels.append("|")
    elif t == "'" or monotonic_greek(t):
        cols.append(i); labels.append(t)
assert "|" in labels and len(labels) >= 38
COLS = torch.tensor(cols)

BASE_R = copy.deepcopy(BASE)
old = BASE_R.lm_head
head = torch.nn.Linear(old.in_features, len(cols))
with torch.no_grad():
    head.weight.copy_(old.weight[COLS]); head.bias.copy_(old.bias[COLS])
BASE_R.lm_head = head
BASE_R.config.vocab_size = len(cols)
BASE_R.config.pad_token_id = 0
BASE_R.config.ctc_loss_reduction = "mean"
BASE_R.config.ctc_zero_infinity = True
BASE_R.config.apply_spec_augment = True
BASE_R.config.mask_time_prob = CFG["mask_time_prob"]
BASE_R.config.mask_time_length = CFG["mask_time_length"]
BASE_R.config.layerdrop = 0.0
TEACHER = copy.deepcopy(BASE_R).to(DEV).eval()
for p in TEACHER.parameters():
    p.requires_grad = False
USE_MASK = BASE_R.config.feat_extract_norm == "layer"

# %%
# Scoring with the pipeline's normalize / metrics / beam decoder, and layer 2c as the phone runs it:
# the release's int8 GPT-2 re-ranks the beam's N-best (voicetotext/decoding/gpt2_rescore.py).
from voicetotext.types import Emissions
from voicetotext.acoustic.emissions import greedy_decode
from voicetotext.acoustic.omni import transfer_foreign
from voicetotext.decoding.beam import BeamDecoder
from voicetotext.decoding.gpt2_rescore import Gpt2Scorer, Hyp, rerank, TOP_K
from voicetotext.phonetics.normalize import Normalizer
from voicetotext.eval.metrics import word_counts, char_counts, sum_counts

NORM = Normalizer.from_labels(labels)
LETTER = np.array([len(l) == 1 and l.isalpha() for l in labels])
L2I = {l: i for i, l in enumerate(labels)}
NBEST = BeamDecoder(labels, lm_path=f"{W}/lm/lm.bin", alpha=CFG["alpha"], beta=CFG["beta"],
                    unk_score_offset=CFG["unk"], beam_width=128, n_best=TOP_K)
GPT2_ONNX = "/content/el_gpt2.int8.onnx"
if not os.path.exists(GPT2_ONNX):
    subprocess.run(["wget", "-q", "-O", GPT2_ONNX,
                    "https://github.com/milkcodelabs/open-hackathon-2026-Archytas/releases/download/models-v1/el_gpt2.int8.onnx"], check=True)
GPT2 = Gpt2Scorer(GPT2_ONNX, "lighteternal/gpt2-finetuned-greek")

def encode(text):
    ids = [L2I["|" if ch == " " else ch] for ch in NORM(text) if ch == " " or ch in L2I]
    assert ids, text
    return ids

def norm_wave(x):
    return (x - x.mean()) / np.sqrt(x.var() + 1e-5)    # as CtcModel.kt

@torch.no_grad()
def emissions(m, wav, restricted=True):
    m.eval()
    x = torch.from_numpy(norm_wave(wav))[None].to(DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        logits = m(x).logits[0].float()
    if restricted:
        return torch.log_softmax(logits, -1).cpu().numpy().astype(np.float32)
    full = torch.log_softmax(logits, -1)[:, COLS.to(DEV)].cpu().numpy().astype(np.float64)
    return transfer_foreign(full, LETTER).astype(np.float32)

def em_of(lp):
    return Emissions(logprobs=lp, labels=labels, frame_duration_ms=20.0, audio_id="x", model_id="few_sentences")

def decode(lp):
    """(greedy, beam + 3-gram, beam + GPT-2), normalized; layer 2b is left out."""
    em = em_of(lp)
    hyps = NBEST.decode(em)
    beam = NORM(hyps[0].text) if hyps else ""
    if len(hyps) < 2:
        return NORM(greedy_decode(em)), beam, beam
    hs = [Hyp(h.text, h.acoustic_score, h.lm_score) for h in hyps]
    return NORM(greedy_decode(em)), beam, NORM(rerank(hs, GPT2.score([h.text for h in hs]))[0].text)

def rates(pairs):
    w = sum_counts(word_counts(a, h) for a, h in pairs)
    c = sum_counts(char_counts(a, h) for a, h in pairs)
    return round(w.rate, 3), round(c.rate, 3)

def greedy_rates(m, rows):
    return rates([(NORM(r["text"]), NORM(greedy_decode(em_of(emissions(m, AUDIO[r["audio"]]))))) for r in rows])

# %%
# BASE: v2 with the foreign-mass transfer (as the phone runs the general model) on every recording.
BASE_LP = {r["id"]: emissions(BASE.to(DEV), AUDIO[r["audio"]], restricted=False) for r in REC}
BASE.cpu()
BASE_OUT = {r["id"]: decode(BASE_LP[r["id"]]) for r in REC}
for r in REC:
    g, b, c = BASE_OUT[r["id"]]
    print(f"{r['id']} {'DEMO ' if r['set'] == 'demo' else ''}ref  {NORM(r['text'])}\n     greedy {g}\n     +2c    {c}")
for name, rows in (("training sentences", TRAIN), ("demo", DEMO)):
    print(f"base v2, {name}: greedy", rates([(NORM(r["text"]), BASE_OUT[r["id"]][0]) for r in rows]),
          " beam", rates([(NORM(r["text"]), BASE_OUT[r["id"]][1]) for r in rows]),
          " +2c", rates([(NORM(r["text"]), BASE_OUT[r["id"]][2]) for r in rows]))
print("restricted base, FLEURS dev greedy WER/CER", greedy_rates(TEACHER, SETS["fleurs_dev"]))

# %%
# LoRA: low-rank updates on attention q/k/v/out and both FFN matrices of every transformer layer;
# the CTC head (39 x 1024) trains too. fold() merges the updates into plain weights for export.
class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale
    def merged(self):
        with torch.no_grad():
            self.base.weight += (self.B @ self.A) * self.scale
        return self.base

def make_model():
    spec = CFG["lora"]
    m = copy.deepcopy(BASE_R)
    for p in m.parameters():
        p.requires_grad = False
    body = []
    for i in spec["layers"]:
        L = m.wav2vec2.encoder.layers[i]
        for parent, name in [(L.attention, "q_proj"), (L.attention, "k_proj"), (L.attention, "v_proj"),
                             (L.attention, "out_proj"), (L.feed_forward, "intermediate_dense"),
                             (L.feed_forward, "output_dense")]:
            w = LoRALinear(getattr(parent, name), spec["rank"], spec["alpha"])
            setattr(parent, name, w)
            body += [w.A, w.B]
    head_p = list(m.lm_head.parameters())
    for p in body + head_p:
        p.requires_grad = True
    return m.to(DEV), body, head_p

def fold(m):
    for L in m.wav2vec2.encoder.layers:
        for parent in (L.attention, L.feed_forward):
            for name, mod in list(parent.named_children()):
                if isinstance(mod, LoRALinear):
                    setattr(parent, name, mod.merged())
    return m

# %%
# Training: speaker clips x repeat (each copy augmented), plus as many rehearsal clips with KD.
from transformers import get_linear_schedule_with_warmup

def augment(x):
    s = random.choice(CFG["speed"])
    if s != 1.0:
        x = np.interp(np.arange(0, len(x) - 1, s), np.arange(len(x)), x).astype(np.float32)
    x = x * 10 ** (random.uniform(-CFG["gain_db"], CFG["gain_db"]) / 20)
    snr = random.uniform(*CFG["noise_snr_db"])
    return x + np.random.randn(len(x)).astype(np.float32) * np.sqrt(x.var() / 10 ** (snr / 10) + 1e-12)

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

def kd_loss(m, x, am, logits):
    with torch.no_grad():
        t = TEACHER(x, attention_mask=am if USE_MASK else None).logits.float()
    T = CFG["kd_temp"]
    lt, ls = torch.log_softmax(t / T, -1), torch.log_softmax(logits.float() / T, -1)
    flen = m._get_feat_extract_output_lengths(am.sum(-1))
    mask = (torch.arange(ls.shape[1], device=ls.device)[None] < flen[:, None]).float()
    return ((lt.exp() * (lt - ls)).sum(-1) * mask).sum() / mask.sum()

def run(train_rows, watch_rows, keep_states=False, tag=""):
    """Trains one model; returns {epoch: {id: logprobs}} for watch_rows, FLEURS dev greedy per
    fleurs epoch, and (keep_states) the trainable weights per eval epoch."""
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    m, body, head_p = make_model()
    opt = torch.optim.AdamW([{"params": body, "lr": CFG["lora"]["lr"]}, {"params": head_p, "lr": CFG["lora"]["lr_head"]}],
                            weight_decay=CFG["weight_decay"])
    n_spk = len(train_rows) * CFG["repeat"]
    n_reg = min(int(n_spk * CFG["rehearsal"]), len(REG))
    E = CFG["epochs"]
    steps_ep = -(-(n_spk + n_reg) // CFG["batch"])
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps_ep * E), steps_ep * E)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    watch, fleurs, states, t0 = {}, {}, {}, time.time()
    for ep in range(1, E + 1):
        m.train()
        tot = nb = 0
        for x, am, y, isreg in batches(train_rows * CFG["repeat"] + random.sample(REG, n_reg)):
            x, am, y, isreg = x.to(DEV), am.to(DEV), y.to(DEV), isreg.to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                out = m(x, attention_mask=am if USE_MASK else None, labels=y)
                loss = out.loss
                if CFG["kd_weight"] > 0 and bool(isreg.any()):
                    loss = loss + CFG["kd_weight"] * kd_loss(m, x[isreg], am[isreg], out.logits[isreg])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(body + head_p, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += out.loss.item(); nb += 1
        if ep in CFG["eval_epochs"]:
            watch[ep] = {r["id"]: emissions(m, AUDIO[r["audio"]]) for r in watch_rows}
            if keep_states:
                states[ep] = {n: p.detach().cpu().clone() for n, p in m.named_parameters() if p.requires_grad}
        if ep in CFG["fleurs_epochs"]:
            fleurs[ep] = greedy_rates(m, SETS["fleurs_dev"])
        if ep in CFG["eval_epochs"] or ep == 1:
            print(f"  {tag} ep {ep:2d} ctc {tot / nb:.3f} {'fleurs dev greedy ' + str(fleurs[ep]) if ep in fleurs else ''} {time.time() - t0:.0f}s")
    del m; gc.collect(); torch.cuda.empty_cache()
    return watch, fleurs, states

# %%
# Cross-validation: every training sentence scored by the model of the fold that held it out.
# Per checkpoint: greedy, beam + 3-gram, beam + GPT-2; then every sentence at export_epoch.
import pickle
CV = []
for k, held in enumerate(FOLDS):
    watch, fleurs, _ = run([r for r in TRAIN if r not in held], held, tag=f"fold{k}")
    CV.append({"held": [r["id"] for r in held], "watch": watch, "fleurs": fleurs})
    pickle.dump({"cfg": CFG, "labels": labels, "cv": CV}, open(f"{OUT}/cv.pkl", "wb"))
TEXT = {r["id"]: NORM(r["text"]) for r in REC}
DEC = {ep: {rid: decode(lp) for f in CV for rid, lp in f["watch"][ep].items()} for ep in CFG["eval_epochs"]}
print(f"== {len(TRAIN)} held-out sentences, WER/CER (base v2: see above)")
for ep in CFG["eval_epochs"]:
    print(f"  ep {ep:2d}  greedy {rates([(TEXT[i], d[0]) for i, d in DEC[ep].items()])}"
          f"  beam {rates([(TEXT[i], d[1]) for i, d in DEC[ep].items()])}"
          f"  +2c {rates([(TEXT[i], d[2]) for i, d in DEC[ep].items()])}")
print("  FLEURS dev greedy (mean over folds):", {ep: tuple(np.round(np.mean([f["fleurs"][ep] for f in CV], 0), 3)) for ep in CFG["fleurs_epochs"]})
ep = CFG["export_epoch"]
for rid in sorted(DEC[ep], key=lambda i: rates([(TEXT[i], DEC[ep][i][2])])):
    print(f"  {rid}  +2c WER/CER {rates([(TEXT[rid], DEC[ep][rid][2])])}  base {rates([(TEXT[rid], BASE_OUT[rid][2])])}\n"
          f"      ref {TEXT[rid]}\n      +2c {DEC[ep][rid][2]}")

# %%
# Final model on all training sentences; the demo sentence is only decoded.
watch, fleurs, STATES = run(TRAIN, REC, keep_states=True, tag="final")
d = DEMO[0]
print("demo:", NORM(d["text"]), "\n  base v2 +2c:", BASE_OUT[d["id"]][2])
for ep in CFG["eval_epochs"]:
    g, b, c = decode(watch[ep][d["id"]])
    print(f"  ep {ep:2d} greedy {g}\n        beam   {b}\n        +2c    {c}")

# %%
# Export export_epoch in the phone's contract: LoRA folded into plain weights, save_pretrained, ONNX
# (input_values -> (N, T, 39) log-probs, opset 17) in a fresh process, int8 per-channel MatMul.
# The int8 graph is run here on the demo sentence, then the three phone files are written.
import onnxruntime as ort
EP = CFG["export_epoch"]
m, _, _ = make_model()
with torch.no_grad():
    params = dict(m.named_parameters())
    for n, v in STATES[EP].items():
        params[n].copy_(v.to(DEV))
m = fold(m).float().cpu().eval()
HF = f"{OUT}/ep{EP}_hf"
m.save_pretrained(HF)
json.dump(labels, open(f"{HF}/labels.json", "w"), ensure_ascii=False)
del m
open("/content/export_personal.py", "w").write('''
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
quantize_dynamic(fp32, int8, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8, per_channel=True)
print("exported")
''')
sh("pip install -q -U ml_dtypes onnx")      # the export needs a newer onnx than the kernel holds
FP32, INT8 = "/content/personal.onnx", "/content/personal.int8pc.onnx"
sh(f"python /content/export_personal.py {HF} {FP32} {INT8}", log="/content/export_personal.log")
print(open("/content/export_personal.log").read()[-400:])
sess = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])
lp = sess.run(None, {"input_values": norm_wave(AUDIO[DEMO[0]["audio"]])[None].astype(np.float32)})[0][0]
print("int8 ONNX demo (greedy, beam, +2c):", decode(lp))
PH = f"{OUT}/phone"
os.makedirs(PH, exist_ok=True)
stem = f"omni.personal.{CFG['name']}"
shutil.copyfile(INT8, f"{PH}/{stem}.onnx")
json.dump(labels, open(f"{PH}/{stem}.labels.json", "w"), ensure_ascii=False)
json.dump({"speaker": CFG["speaker"] or CFG["name"], "base": CFG["base"].split("/")[-1],
           "trained": f"LoRA, {len(TRAIN)} προτάσεις, epoch {EP}"}, open(f"{PH}/{stem}.json", "w"), ensure_ascii=False)
for f in sorted(os.listdir(PH)):
    b = open(f"{PH}/{f}", "rb").read()
    print(f"{f:34s} {len(b):>12,d}  {hashlib.sha256(b).hexdigest()}")
