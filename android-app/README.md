# Voice-to-text (Android)

Greek voice typing that runs entirely on the phone. A floating bubble works on top of any
app: tap it, speak, tap again, and the text is typed into the text field that has focus.
Audio never leaves the device; the network is used only once, to download the models.

## How it works

- **Layer 1, acoustic model (CTC):** Meta Omnilingual ASR CTC 300M, int8, restricted to the
  39 Greek symbols inside the model file (`omni.onnx`). It outputs a probability for every
  Greek letter every 20 ms.
- **Layer 2a, beam search + Greek 3-gram language model** (`decoding/BeamSearch.kt`,
  `el_3gram.ngram`, 300k words): turns the letter probabilities into the most likely
  sentences. It continues the text already in the field and knows the speaker's own words
  from `my_words.txt`.
- **Layer 2b, spelling** (`decoding/SpellingRescorer.kt`, `el_homophones.bin`): the acoustic
  model cannot tell ι/η/υ/ει/οι, ο/ω, ε/αι apart; for every word the real spellings of the
  same sound are offered and the language model picks, in context.
- **Candidate sentences** (`decoding/Candidates.kt`): the ten best sentences, with their
  relative probability and the words that differ from the best one highlighted. Tap one to
  choose it and copy it.

About 1 second for a 5-second sentence on a Samsung Galaxy A55.

## Code layout

| package | what |
|---|---|
| `MainActivity`, `ui/` | the setup screen (Jetpack Compose) |
| `service/` | the floating bubble (`OverlayService`) and the typing service (`TypingAccessibilityService`) |
| `audio/` | microphone recording |
| `asr/` | the acoustic model (`CtcModel`), `Emissions`, and `Recognizer`, which owns it |
| `decoding/` | layer 2: beam search, n-gram LM, homophone index, spelling pass, the candidate sentences |
| `models/` | `ModelDownloader`: fetches the models from GitHub Releases |

## Build

Open this `android-app` folder in Android Studio and run it, or from a terminal:
`gradlew.bat :app:assembleDebug` (Windows) / `./gradlew :app:assembleDebug`.
Only `arm64-v8a` is packaged, so it runs on real arm64 phones, not on an x86 emulator.

## Models

The models are too large for the APK. On first start the app downloads them from the
release `models-v1` of this repository (automatically on Wi-Fi, with the **Λήψη μοντέλων**
button otherwise), shows progress, speed and time left, resumes an interrupted download and
checks every file's SHA-256 before using it. The release must be reachable without signing
in, i.e. the repository must be public; otherwise use **Εισαγωγή αρχείων** in the app or
push them over USB:

| file | size | what |
|---|---:|---|
| `omni.onnx` | 365 MB | layer 1, Omnilingual CTC 300M (default engine) |
| `omni.labels.json` | 1 KB | its 39 Greek labels |
| `el_3gram.ngram` | 95 MB | layer 2a, Greek language model |
| `el_homophones.bin` | 10 MB | layer 2b, sound -> spellings index |
| `test.wav` | 0.3 MB | optional, for the in-app test button |
| `my_words.txt` | tiny | optional, the speaker's names and words, one per line |
| `omni.personal.onnx` + `.labels.json` + `.json` | 356 MB | optional, the speaker's own layer 1 (downloaded only on request; `model-pipeline/personalization/`) |

```
adb push omni.onnx /sdcard/Android/data/com.openhackathon.voicetotext/files/
```

## On the phone, once

- Allow the microphone and "display over other apps" from the app.
- Settings > Accessibility > Installed apps > Voice-to-text > on. This lets the bubble type
  into other apps. Samsung may ask for "Allow restricted settings" first (App info, top-right
  menu).

| gesture on the bubble | effect |
|---|---|
| tap | start listening (red) |
| tap again | stop, recognise, type the text into the focused field |
| long press | run on `test.wav` instead of the microphone |
| drag | move it; it snaps to the screen edge |
