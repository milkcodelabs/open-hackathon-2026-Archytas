# Personal acoustic model

Fine-tunes layer 1 (Omnilingual CTC 300M v2) on one speaker's recordings and exports it in
the format the app loads as its **personal model** (`omni.personal.*`, the
"Προσωπικό μοντέλο" switch). The general model writes what it hears; a speaker whose ρ comes
out as λ gets "λίξε" for "ρίξε". After fine-tuning, the model has learnt that this speaker's λ
may be a ρ and puts probability on both, and the language model picks the right word.

| file | what |
|---|---|
| `recorder.html` | offline recording page: prompts, sessions, export as a ZIP (`metadata.csv` + `wav/`) |
| `pack_bundle.py` | recordings + evaluation sets + LM + the `voicetotext` package -> `personal_bundle.zip` |
| `finetune.py` | the training notebook as a script, cells split by `# %%` |
| `make_notebook.py` | a notebook script -> its `.ipynb` (stdlib only): `python make_notebook.py [few_sentences.py]` |
| `finetune.ipynb` | what runs on Colab; regenerate it after editing `finetune.py` |
| `few_sentences.py`, `few_sentences.ipynb` | the recipe for a handful of recordings (a real speaker, 14 sentences), below |

Recordings and trained weights never go into git. They stay on the recording device, the
machine that packs the bundle, and the account's own Google Drive.

## Results (one speaker, simulated profile)

380 recordings (16.9 min) of one speaker reading prompts with a **simulated** dysarthric
profile: ρ -> λ; cluster reduction (σ + stop, stop + ρ/λ); an unstressed initial vowel
dropped. For example, "χορέψαμε μέχρι το πρωί" is said as "χολέψαμε μέχι το πωί". The label
is the intended text. Sessions 1-4 are for training (257 clips), session 5 is dev (63,
checkpoint choice) and session 6 is test (60). Every sentence occurs once, so dev and test
are sentences the model never heard. **This is not real dysarthric speech.** It shows that
the method works, not what target speakers will get.

All WER below use beam search with the 300k-word LM (alpha 0.7, beta 3, width 128) and the
same normalization.

| run | what changed | dev | **test** | FLEURS test | CV 300 |
|---|---|---:|---:|---:|---:|
| base v2 | - | 0.500 | **0.480** | 0.131 | 0.141 |
| r1 | top 12 layers, lr 5e-5, 58 CV clips replayed, choice by greedy WER | 0.208 | 0.219 | 0.285 | 0.255 |
| r2/r3 | + FLEURS train replay 1:1 / 2:1, top 8 layers, choice by beam WER | stopped: FLEURS dev over the limit every epoch | | | |
| r4 ep 1 | + KD (KL to the frozen base on replayed clips), FLEURS dev <= base + 0.01 | 0.396 | 0.369 | 0.138 | 0.156 |
| **r5** | same, FLEURS dev <= base + 0.03, 30 epochs, best epoch 7 | **0.133** | **0.140** | 0.160 | 0.171 |

- **Phone export of r5:**
  - fp32 ONNX: 0.140 on test (max |Δp| 5e-4 against torch).
  - int8 MatMul per-tensor: 0.151.
  - **int8 per-channel: 0.147 (340 MB). This is the file the app uses.**
  - int8 per-channel with the head kept in fp32: 0.151.
- **The greedy view barely moves (0.638 -> 0.599), but oracle-spelling CER falls from 0.132 to 0.024.**
  - The gain comes from the model putting probability on both readings of a sound.
  - The LM then chooses between them, so the fine-tune and the LM work together.
- **Forgetting is the cost.**
  - Without replay (r1), errors on typical speech double.
  - FLEURS replay alone was not enough. The per-frame KL to the base model on replayed clips is what keeps it in check.
  - What remains is about +3 points on FLEURS and CV. That is why this is a per-speaker model and the general one stays the default.
- **The useful training is short.** With about 250 clips, dev is best at epoch 7 and drifts upwards afterwards.

## Few sentences: a real speaker with dysarthria (speaker 2)

A person with dysarthria read 15 sentences once each: 82 s of speech after trimming the silence
(recorded at 48 kHz stereo with identical channels, converted to 16 kHz mono). One sentence is
the demo, never trained on; the other 14 train. The speech is severely affected: the general
model gets almost no word right. With so little data the r5 recipe does not carry over, so
everything below was chosen by **7-fold cross-validation**: each of the 14 sentences is scored
by a model trained on the other 12, i.e. on a sentence it never heard.

| layer 1 | decoding | WER | CER |
|---|---|---:|---:|
| base v2 | beam + 3-gram | 0.92 | 0.74 |
| LoRA r16, all 24 layers, KD rehearsal (round 1, epoch 6 of 40) | beam + 3-gram | 0.72 | 0.36 |
| same | beam + GPT-2 (2c) | 0.67 | 0.33 |
| **same, 10-epoch schedule, checkpoint 4** (two fold draws) | beam + GPT-2 (2c) | **0.66 / 0.69** | **0.34 / 0.36** |
| top 8 layers (the r5 recipe, lr 3e-5) | beam + 3-gram | 0.90 | 0.63 |

- **LoRA works and the r5 recipe does not.**
  - LoRA: rank 16 on attention q/k/v/out and both FFN matrices of every layer, plus the CTC head.
  - Training data per epoch: every speaker clip 10 times, each copy augmented differently (speed 0.8-1.2, noise 15-40 dB SNR, gain ±6 dB, SpecAugment 0.1). Next to them, as many general clips of up to 10 s, with the KL to the base model.
  - Top-8 fine-tuning at r5's learning rate barely moves. LHUC was not run to the end.
- **Stop early.** With a 40-epoch schedule the held-out error is lowest at epoch 6-8 and grows after it (memorising 12 sentences). A 10-epoch schedule is flat, and checkpoint 4 is exported.
- **Decoding side.**
  - GPT-2 re-ranking (the app's 2c weights) helps.
  - A heavier n-gram weight hurts: it glues words together.
  - A confusion transform estimated on the training sentences (e.g. λ -> ρ) gained only 0.01-0.03 and is left out.
- **Cost on typical speech:** FLEURS dev greedy WER 0.35 -> 0.41. A per-speaker model, like r5.
- **Per sentence the spread is large.** WER goes from 0.16 to 0.93.
  - The best is «Προγραμματίζω τις διακοπές μου για τον επόμενο μήνα». It came out word for word in 3 of 4 held-out runs; the base model gives «μα τι τι διακοπτόμενα».
  - It became the demo after this table, so it shows the best case, and the cross-validated numbers above are the typical one.
  - The final model, trained on the other 14, writes it exactly, and so does its int8 ONNX.

The measurements above used the float GPT-2 (`lighteternal/gpt2-finetuned-greek`) with the app's weights. `few_sentences.py` uses the release's int8 GPT-2, the one on the phone.

`few_sentences.py` is the cleaned recipe: base model, cross-validation with a per-sentence table, final model, int8 export and the phone files. The comparisons above came from the exploratory notebook it was distilled from, and this form has not been run end to end. It reads the same `personal_bundle.zip`: mark the demo sentence `set=demo` in `metadata.csv`, and set `name` and `speaker` in the config cell.

The release has this model under a pseudonym, and the recordings are not published:

| file | bytes | SHA-256 |
|---|---:|---|
| `omni.personal.s2.onnx` | 356,199,242 | `a34cf06470e1712fbddaa581b03a6c0787961c05b9d499a2ec38c7f775c62381` |
| `omni.personal.s2.labels.json` | 235 | `0098d2e62f383f6cbb9a7450669ce22831b7c32204ab98b96059e52f4dc5d07c` |
| `omni.personal.s2.json` | 205 | `841b3ffe9bf1fb7c638051112e45d70523b7d7a73fbe09b0b535d172625deb56` |

The app does not download it by itself. Copy the three files over USB or with **Εισαγωγή αρχείων**. The app lists every `omni.personal.<name>.*` it finds, next to `omni.personal.*`, under «Προσωπικό μοντέλο».

## Recipe (r5)

- **Base model:** `aadel4/omniASR-CTC-300M-v2`, a Wav2Vec2ForCTC conversion of Omnilingual CTC 300M v2 (parity-checked against the original).
- **Head:** cut to the 39 Greek columns (blank = token 0, word space, apostrophe, monotonic lowercase letters). Its rows are copied from the original head, so at step 0 it is the base model.
- **Frozen parts:** the CNN and transformer layers 0-15. Layers 16-23 and the final layer norm train.
  - lr 3e-5 (head 1e-4), weight decay 0.01, 10% warmup, batch 16, 30 epochs, fp16 autocast.
- **Augmentation:** speed 0.9/1.0/1.1, gain ±6 dB, noise at 20-40 dB SNR, SpecAugment time masks (p 0.05, length 5).
- **Rehearsal:** every epoch adds 2× as many general clips as speaker clips, resampled.
  - Sources: FLEURS el_gr train without sentences that contain Latin letters or digits, plus Common Voice test clips that share no speaker with the CV 300 eval set.
  - On these clips the loss adds KL(base || model) per frame (weight 1, temperature 1).
- **Checkpoint choice:** after every epoch, dev beam WER (width 32).
  - Only epochs whose FLEURS dev beam WER stays within 0.03 of the base qualify.
  - The test session, FLEURS test and CV 300 are only scored before and after training.
- **Export:**
  - `input_values` (N, samples; zero mean, unit variance) -> `logprobs` (N, T, 39), opset 17.
  - Dynamic int8 quantization of MatMul weights, per channel.

## Steps

### 1. Record

1. Open `recorder.html` in a browser on a phone or laptop. It works offline and keeps the takes in the browser's IndexedDB.
2. Pick the session (1-8; 5 is dev and 6 is test) and record the 380 training prompts across sessions 1-6.
   - Record on different days or at different times, so the sessions differ the way real use does.
3. The 20 demo phrases are `set=demo` and are never used for training.
4. **Εξαγωγή > Κατέβασε το ZIP** writes the export: `wav/` (16 kHz mono 16-bit), `metadata.csv` and `README.txt`.

`metadata.csv` columns: `file, text, pronounced_as, set, category, session, duration_s, sample_rate, recorded_at`.
- `text` is the training label: what the speaker means.
- `pronounced_as` and the "Πώς προφέρω" tab belong to the simulated profile. A real speaker simply speaks naturally, and the column is ignored in training.

### 2. Evaluation data and LM (once, with the pipeline set up as in `../README.md`)

```
voicetotext data fleurs --split dev --split test --limit 100
voicetotext data clean ~/voicetotext_data/manifests/fleurs_el_gr_dev.jsonl
voicetotext data clean ~/voicetotext_data/manifests/fleurs_el_gr_test.jsonl
```

For Common Voice, both manifests are written as `cv_el_test.jsonl`, so rename each one before making the next.

- The 300-clip eval sample goes to `cv_el_test_300.jsonl`.
- The whole test split goes to `cv_el_test_full.jsonl`. It is the source of the rehearsal clips.

```
voicetotext data common-voice <cv-corpus>/el --limit 300
mv ~/voicetotext_data/manifests/cv_el_test.jsonl ~/voicetotext_data/manifests/cv_el_test_300.jsonl
voicetotext data common-voice <cv-corpus>/el
mv ~/voicetotext_data/manifests/cv_el_test.jsonl ~/voicetotext_data/manifests/cv_el_test_full.jsonl
voicetotext data clean ~/voicetotext_data/manifests/cv_el_test_300.jsonl
voicetotext data clean ~/voicetotext_data/manifests/cv_el_test_full.jsonl
```

The LM is the KenLM binary of `configs/omni_beam.yaml`, with `<name>.vocab` next to it (`voicetotext lm build`, `../README.md`).
- r5 used `general_el_300k_p02233.bin`. Any 300k-word build is fine for scoring.

### 3. Pack

```
uv run python personalization/make_notebook.py
uv run python personalization/pack_bundle.py <unzipped recorder export> ~/personal_bundle
```

This writes `~/personal_bundle.zip`. Other manifest or LM names can be passed with `--lm`, `--cv-eval`, `--cv-full`, `--fleurs-dev` and `--fleurs-test`.

### 4. Train on Colab

1. Upload `personal_bundle.zip` to `MyDrive/voicetotext/`.
2. Open `finetune.ipynb` in Colab and choose a GPU runtime (T4 works; r5 ran on an A100). Set `speaker` in the config cell.
3. **Run all.** The first run builds kenlm and stops on purpose. Restart the session, then run all again.

Outputs go to `MyDrive/voicetotext/personal/`:
- `r5_results*.json`: before and after, for every set, plus the per-epoch history.
- `r5_hf/`: the best checkpoint.
- `r5.onnx`, `r5.int8.onnx` and `r5.int8pc.onnx`.
- `phone/`: the three files the app needs. The last cell prints each file's size and SHA-256.

### 5. Put it on the phone

Either copy the files over USB. The switch appears on the next start.

```
adb push omni.personal.onnx omni.personal.labels.json omni.personal.json /sdcard/Android/data/com.openhackathon.voicetotext/files/
```

Or publish them on the release. Upload the three files to `models-v1`, then update their sizes and hashes in `ModelDownloader.PERSONAL`. The app then offers **Λήψη προσωπικού μοντέλου** in the Models card, and downloads only on request.

The files currently on the release are the r5 model of the simulated profile above:

| file | bytes | SHA-256 |
|---|---:|---|
| `omni.personal.onnx` | 356,199,242 | `83acf5aa777f83b5f78b535980f58a4185cfde2c1690799173ce3d32c12e3757` |
| `omni.personal.labels.json` | 235 | `0098d2e62f383f6cbb9a7450669ce22831b7c32204ab98b96059e52f4dc5d07c` |
| `omni.personal.json` | 176 | `8ace0f8c832942e28ccefd455b80c529bce75382824f5cedd846ac48ff37968f` |

How the app uses them:
- The personal model replaces `omni.onnx` only when both its graph and its labels are present and the switch is on.
- Its labels are read from its own file, because the letter order differs from the base model's.
- The files form one download group, so a graph is never paired with another model's labels.
- On a Samsung Galaxy A55, the whole recognition of the 4.9-second test clip took 1.4 s with the personal model.
