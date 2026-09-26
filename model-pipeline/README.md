# Model pipeline

The Python code behind the files the Android app downloads, and the evaluation that set
the app's decoding parameters. Nothing here runs on the phone.

| file the app downloads | made by |
|---|---|
| `omni.onnx`, `omni.labels.json` | `voicetotext export omni-phone` (`voicetotext/acoustic/omni.py`) |
| `el_3gram.gvtlm` | `voicetotext lm prepare/build` + `voicetotext lm export-phone` (`voicetotext/decoding/lm.py`, `export_lm.py`) |
| `el_homophones.bin` | `voicetotext homophones` (`voicetotext/phonetics/homophones.py`) |
| `el_gpt2.int8.onnx`, `el_gpt2.vocab.json`, `el_gpt2.merges.txt` | `voicetotext export gpt2` (`voicetotext/decoding/gpt2_export.py`) |
| `test.wav` | `tests/fixtures/audio/clip1.wav` (FLEURS, CC-BY 4.0) |
| `omni.personal.onnx`, `omni.personal.labels.json`, `omni.personal.json` (optional) | speaker fine-tune on Colab, `personalization/` (its own README) |

## Layout

| path | what |
|---|---|
| `voicetotext/acoustic/` | Omnilingual CTC 300M through onnxruntime, Greek-column restriction, phone-graph export; emission utilities |
| `voicetotext/decoding/` | corpus cleaning and KenLM build (`lm.py`), the phone LM format (`export_lm.py`), pyctcdecode beam search (`beam.py`), GPT-2 export and re-ranking, label handling |
| `voicetotext/phonetics/` | text normalization, Greek number verbalization, the homophone index |
| `voicetotext/eval/` | FLEURS / Common Voice manifests, WER/CER harness, alpha/beta tuning |
| `voicetotext/pipeline.py`, `cli.py`, `config.py` | audio -> emissions -> decoder, the `voicetotext` command, configuration |
| `configs/` | `default.yaml`, `omni_greedy.yaml`, `omni_beam.yaml`, `lm_phone.yaml` |
| `personalization/` | personal acoustic model: recording page, Colab fine-tune, phone export |
| `scripts/make_fixtures.py` | regenerates the test fixtures from the acoustic model |
| `tests/` | unit and integration tests; no model or network needed |

## Setup

```
py -m uv sync --group dev                            # core + pytest (`uv` itself is `py -m uv` here)
uv sync --group dev --extra export --extra neural --extra data   # phone graphs, GPT-2, FLEURS
uv sync --group dev --extra decode --extra lm        # beam search + KenLM (Python 3.11/3.12: pyctcdecode needs numpy<2)
uv run pytest
```

Data, manifests, emissions cache and results live under `~/voicetotext_data`
(`VOICETOTEXT_DATA_DIR` overrides it). The n-gram build calls KenLM's `lmplz` and
`build_binary`, built from source (https://github.com/kpu/kenlm, needs cmake, boost and a
C++ compiler); `paths.kenlm_bin` points at their folder.

## The acoustic model: `omni.onnx`

Meta Omnilingual ASR CTC 300M (v1, Apache 2.0, 325M parameters, 20 ms frames), from the
sherpa-onnx int8 export
(https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12,
`model.int8.onnx` + `tokens.txt`). The model has one
9,812-symbol vocabulary for 1,600+ languages and no language input, so two adaptations are
baked into the graph the phone runs:

* **Greek columns only**: blank, space, apostrophe and the 36 monotonic lowercase Greek
  letters (V = 39). The CTC blank is token 0 (`<s>` in `tokens.txt`), not `<pad>`.
* **Foreign-script mass goes to Greek letters** (`foreign: letters`). On short clips the
  model sometimes writes Greek in Latin script; keeping only the Greek columns would then
  let blank win those frames and the letters vanish. The non-Greek mass is moved onto the
  Greek letters in proportion to their own probability; blank and space are left alone.

```
voicetotext export omni-phone <sherpa export folder>/<int8 model>.onnx omni.onnx \
    --verify tests/fixtures/audio/clip1.wav
```

This writes `omni.onnx` (input `input_values`, output (N, T, 39) log-probabilities) and
`omni.labels.json`. `--verify` compares it with the Python emitter on the sherpa file;
on 8 FLEURS / Common Voice clips the difference was at most 7e-6 in log-probability with
identical greedy text. The app normalizes the waveform to zero mean and unit variance
before feeding it, like `OmniOnnxEmitter`.

## The language model: `el_3gram.gvtlm`

1. **Corpus** (`voicetotext lm prepare`): OPUS OpenSubtitles v2018 mono `el` (first 15M
   cleaned sentences) and Greek Wikipedia 20231101 shard 1/3 (`wikimedia/wikipedia`, all
   1.83M kept sentences), about 16.8M sentences.
   * Subtitle pass: `<tags>`, `[stage notes]`, `(notes)` removed; speaker dashes and
     `ΟΝΟΜΑ:` prefixes stripped; lines with ♪/♫/#/* or ending in ":" dropped.
   * Sentences with Latin letters are dropped whole: the model cannot emit them, and
     truncating would teach broken sentences.
   * Digits are verbalized as neuter Greek cardinals ("1980" -> "χίλια εννιακόσια
     ογδόντα", "3,5%" -> "τρία κόμμα πέντε τοις εκατό").
   * Everything goes through the same `normalize()` as references and hypotheses:
     Greek letters of the model's alphabet only, lowercase, monotonic accents kept,
     punctuation to space, final sigma.

   | corpus | raw sentences | kept | dropped: Latin | with digits (verbalized, kept) |
   |---|---:|---:|---:|---:|
   | OpenSubtitles el (first 31.4M lines read) | 31,434,779 | 95.4% | 4.1% | 2.1% |
   | Wikipedia el shard 1/3 | 2,386,571 | 76.6% | 22.7% | 30.3% |

2. **Vocabulary** (`voicetotext lm oov`): the top 300k words. A word outside the LM
   vocabulary can never be produced by the beam search, so OOV is a hard ceiling.

   | vocab | token coverage | OOV FLEURS dev (clean) | OOV CV 6.1 test (clean) |
   |---:|---:|---:|---:|
   | 70k | 0.966 | 4.76% | 9.16% |
   | 150k | 0.984 | 2.45% | 5.23% |
   | 300k | 0.993 | 1.73% | 3.72% |
   | 500k | 0.997 | 1.22% | 2.82% |

3. **n-gram** (`voicetotext lm build -c configs/lm_phone.yaml`): `lmplz -o 3 --prune 0 2 4
   --limit_vocab_file`, which writes the ARPA and a KenLM binary (the binary is for the
   Python beam search).
4. **Phone format** (`voicetotext lm export-phone <name>.arpa el_3gram.gvtlm`): KenLM's
   binary is a C++ trie, so the ARPA is rewritten as sorted arrays that `NgramLm.kt`
   memory-maps and binary-searches (layout in `export_lm.py`; the file starts with `NGRAM1`). The homophone index starts with `HOMIDX1`. The published file: order
   3, 300,003 words (300k plus `<s>`, `</s>`, `<unk>`), 2,997,665 bigrams, 3,173,808
   trigrams, 95 MB. The app's lookup and backoff (`NgramLm.kt`) were checked against an
   unquantized KenLM binary on 3,000 real lookups: identical scores. The same KenLM binary
and vocabulary build the homophone index below.

## The spelling index: `el_homophones.bin`

```
voicetotext homophones el_homophones.bin --kenlm <name>.bin --vocab <name>.vocab
```

For each pronunciation, the vocabulary words spelled that way, most frequent first.
`--arpa <name>.arpa` reads the unigram probabilities without the kenlm module; near-equal
spellings can then come out in a slightly different order than the published file, which
used the 8-bit quantized KenLM binary. The published file has 233,383 sound keys, 45,478
of them with more than one spelling.

The sound key is the same function as `SoundKey.kt` (the two must stay identical, because
the phone looks words up in this file): lowercase, drop stress, αυ/ευ/ηυ -> av/ev/iv,
ει/οι/υι -> i, αι -> e, ου -> u, ι/η/υ -> i, ο/ω -> o, ε -> e, ς -> σ, doubled consonants
single (γγ stays), and a diaeresis makes ι/υ a stand-alone /i/. `δήμου` and `δίμου` both
become `δiμu` (μ is not rewritten).

Layer 2b (`SpellingRescorer.kt`) offers up to 6 spellings of the same sound for every word
of the 50 best sentences, and a small dynamic programme picks the sequence the 3-gram likes
in context. Replacing a word already in the vocabulary costs 1.5 nats; an unknown word is
replaced for free.

## The GPT-2 rescorer: `el_gpt2.*`

https://huggingface.co/lighteternal/gpt2-finetuned-greek (Apache 2.0, 124M, Greek byte-level BPE).

```
voicetotext export gpt2 lighteternal/gpt2-finetuned-greek out_dir
voicetotext gpt2-probe out_dir/el_gpt2.int8.onnx lighteternal/gpt2-finetuned-greek \
    "αναφέρθηκε στις φήμες ως πολιτικό κουτσομπολιό και ανοησίες"
```

The probe prints token ids and the sentence log-probability. The app logs the same sentence
when it loads the model (`NeuralRescorer`): the release check was identical ids, logp
-69.53 here against -69.35 on a Galaxy A55. The graph takes `input_ids` and `attention_mask`
and returns the natural-log probability of each next token; int8 quantization includes the
embedding Gather, which is what brings the file from 282 MB down to 164 MB.

Re-ranking (`gpt2_rescore.py`, the same rule as `NeuralRescorer.kt`): for the top 20
sentences, `total = acoustic + ngram + 0.5 * log P_gpt2(sentence | context)`. A sentence
whose acoustic score is more than 8 nats below the best cannot win. Weights chosen on
FLEURS dev plus half of Common Voice: WER 0.129 -> 0.114 on FLEURS test, 0.132 -> 0.123 on
the Common Voice half it never saw. nikokons/gpt2-greek was better on FLEURS and worse on
Common Voice; Qwen3-0.6B-Base was too slow for a phone.

## Decoding parameters

The app's `BeamSearch.kt` is the same CTC prefix beam search as pyctcdecode, with the same
constants: label threshold e^-5 (`token_min_logp`), hypotheses more than 10 (natural log)
below the best dropped, beam width 128, unknown-word offset -10, LM score once per
completed word. alpha 0.7 / beta 3.0 were tuned on FLEURS dev clean
(`voicetotext tune`, `configs/omni_beam.yaml`).

## Evaluation

```
voicetotext data fleurs --split dev --split test --limit 100
voicetotext data common-voice ~/cv-corpus/el --limit 300
voicetotext data clean ~/voicetotext_data/manifests/fleurs_el_gr_test.jsonl
voicetotext eval ~/voicetotext_data/manifests/fleurs_el_gr_test_clean.jsonl -n omni_beam_test -c configs/omni_beam.yaml --cache-emissions
voicetotext tune ~/voicetotext_data/manifests/fleurs_el_gr_dev_clean.jsonl -c configs/omni_beam.yaml --alphas 0.5,0.7,0.9 --betas 2,3,4
```

* The primary sets are the `_clean` manifests: utterances whose reference contains digits
  or Latin letters are left out rather than scored truncated (FLEURS dev 71/100, test
  75/100, CV 6.1 test 300/300).
* References and hypotheses go through the same `normalize()` before WER/CER.
  Corpus-level rates: total errors / total reference tokens.
* The greedy view is scored on every run next to the decoder's output, so the decoder
  cannot hide an acoustic regression. The oracle-spelling CER collapses homophone
  spellings (ι/η/υ/ει/οι, ο/ω, αι/ε, stress): the gap to the plain CER is what spelling
  correction can still recover.
* alpha/beta are tuned on dev only; `voicetotext tune` refuses a manifest named like a test
  split.
* FLEURS and Common Voice are read speech by typical speakers.

WER / CER, Omnilingual CTC 300M, 300k-word n-gram LM:

| | FLEURS dev clean (71) | FLEURS test clean (75) | CV 6.1 test 300 clean |
|---|---:|---:|---:|
| greedy | 0.360 / 0.091 | 0.371 / 0.095 | 0.294 / 0.075 |
| greedy, foreign mass -> letters | 0.360 / 0.091 | 0.371 / 0.095 | 0.293 / 0.072 |
| + beam / LM, a0.7 b3 (what the app runs) | 0.128 / 0.045 | 0.139 / 0.045 | 0.132 / 0.035 |
| int8 (phone graph), greedy | | 0.371 / 0.095 | 0.296 / 0.077 |

* The LM does most of the work (0.371 -> 0.139 on FLEURS test): the model's errors are
  mostly spelling of the right sounds, which an n-gram fixes.
* int8 costs nothing measurable.
* On CV the errors are mostly phonetic spelling (ίσε / είσαι, κράτι / Σωκράτη) plus the
  script confusion on very short clips.

On the phone (Samsung Galaxy A55, fixture clip 1, 4.86 s): acoustic ~780 ms, beam + LM
146-168 ms, total 922-953 ms (RTF 0.19-0.20), with the reference text word for word.

## Licences of the inputs

Omnilingual ASR and lighteternal/gpt2-finetuned-greek: Apache 2.0. FLEURS (`test.wav`):
CC BY 4.0. Greek Wikipedia text: CC BY-SA. OpenSubtitles text comes from OPUS; check its
terms before any use beyond research or a prototype. KenLM is LGPL and is a build tool,
not shipped in the app.
