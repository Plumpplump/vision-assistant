# Plan — VisionAssist

Decisions behind the demo. Product goals and the original shortlist live in
`plan.txt`; that file is background, this one is the current intent.

Product shape: a Flet app that watches a camera, warns about obstacles **without
talking**, and answers questions about what is in front of the user — on a
laptop for the demo, on Android later.

## Status (2026-11-15)

| Phase | State |
| --- | --- |
| P0 skeleton — domain/adapters/app/ui boundaries, tests | **done** (186 unit tests) |
| P1 say the scene — OpenCV camera, YOLO, tracker, Paradee TTS | **done** |
| P2 reflex loop — corridor filter, metric distance bands, soft chimes, haptics | **done** |
| P3 ask — VAD + Parakeet ASR | **desktop push-to-talk + "Hey Vision" wake word wired, auto-stops on silence**; acoustic KWS not wired |
| P4 find — open-vocabulary search | **done** (MobileCLIP2-B, threshold 0.22) |
| P5 describe — Qwen VLM loop | **wired** (128 px, brief prompt, 3–5 s on CPU); template fallback; free-form visual questions route to it |
| P6 mobile — flet build, on-device TTS/ASR | not started |

### Remaining work (in priority order)

1. **Acoustic wake word** — "Hey Vision" is matched on the Parakeet transcript.
   Ducking + tolerant matching make it usable, but a custom openWakeWord model
   removes "transcribe everything" and the ~1 s wake latency. Plan and acceptance
   tests: `docs/wake-word.md`.
2. **P6 mobile** — audio is cross-platform now (`flet-audio-recorder`, PCM16
   streaming, RECORD_AUDIO declared); packaging is prepared
   (`tools/package_models.py`: `apk` = 169 MB, `apk-vision` = 907 MB,
   `apk-full` = 1.5 GB). Still needed: a `flet-camera` frame source (OpenCV
   cannot open the Android camera), the APK build on a machine with the Android
   SDK, and echo cancellation for voice barge-in.
3. **Safety/UX audit** — voice "stop" during TTS (needs AEC or a ducker), obstacle
   classes COCO lacks (stairs, glass doors), TalkBack semantics pass, spoken
   startup state.
4. **Evaluation + ops** — a small labelled set for detection/intents, latency on
   a phone, `git init`, CI, `cmake` so Flet integration tests can run, licence
   attribution screen (AGPL YOLO, CC-BY Parakeet, Apple CLIP licence).

Camera decision: **OpenCV only**. `flet_camera` was dropped: Flutter's camera
plugin supports Android/iOS/web but not Linux desktop, and the demo runs on a
laptop. `VISIONASSIST_SOURCE` selects camera / video file / synthetic `demo`.

## Model selection (decided 2026-11-15)

Rule: ONNX (or GGUF) only. No torch, no transformers, no cloud in the runtime
path. Every model sits behind a protocol + registry entry so a swap is one
adapter file, never a UI change.

### Copied from sibling projects (no download)

| Role | Component | Location | Size | Licence | Source |
| --- | --- | --- | --- | --- | --- |
| Detection | YOLO11n ONNX + `config.json` | `src/assets/models/yolo11n/` | 11 MB | AGPL-3.0 (Ultralytics upstream) | `schooluniform` |
| Open-vocabulary search | MobileCLIP2-**B** (image + text towers) | `src/assets/models/mobileclip2-b/` | 577 MB | `apple-amlr` | `schooluniform`; `--vision` |
| Scene description | Qwen3.5-0.8B VLM q4 ONNX (decoder + embed + vision encoder) | `src/assets/models/qwen3.5-0.8b/` | ≈704 MB | Apache-2.0 | `schooluniform`; default describer when present |
| Speech out | Paradee-8M v1.3 int8 + `config.json` + LICENSE | `src/assets/models/paradee/` | 8.5 MB | Apache-2.0 | `SAT-Words` |
| G2P lexicons | misaki `{us,gb}_{gold,silver}.json` | `src/assets/models/g2p/` | 13 MB | Apache-2.0 | `SAT-Words` |

Learned details:

- YOLO11n's `config.json` maps all 80 COCO labels — no re-export needed to see
  `cup`, `chair`, `bottle`, `dining table`. The allowlist lives in
  `adapters/detection/yolo_onnx.py`.
- MobileCLIP2 needs `config.json` (224×224, mean 0/1, 512-d), `tokenizer.json`
  and the tokenizer files next to the towers. Prompt ensembles are averaged and
  L2-normalised; threshold 0.22.
- Paradee is torch-free and Android-ready in `SAT-Words`
  (`local_tts.py`, `g2p.py`, `vocab.py`, `fetch_tts_assets.py`). The port here
  is `adapters/speech/{paradee,g2p,vocab}.py`. The fetch script extracts the
  lexicons from the **misaki wheel data directory**, so misaki's `Python<3.13`
  pin never applies.
- Paradee accepts at most 512 phoneme ids → `ParadeeTts.synthesize` splits on
  sentence boundaries and recurses by words if a sentence is still too long.
- Only the four English voices are usable with the bundled lexicons; the
  Hindi/French/Spanish voices additionally need eSpeak NG.

### Downloads

| Role | Component | Repo | Size | Licence | State |
| --- | --- | --- | --- | --- | --- |
| Speech gating | Silero VAD int8 | `onnx-community/silero-vad` | 0.64 MB | MIT | downloaded |
| Speech in (**default**) | Parakeet TDT 110M int8 (offline) | `akm9191/sherpa-onnx-parakeet-tdt-110m-int8` (copy of k2-fsa's `sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8`) | 136 MB | CC-BY-4.0 | downloaded |
| Speech in, live partials (**optional**) | Streaming zipformer EN int8, chunk-16 left-64 | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` | 73 MB | Apache-2.0 | `--partials` |

Benchmarked on the same audio (`docs/asr-benchmark.md`, 73 LibriSpeech dummy
clips, 481 s, references provided):

| Model | Size | Load | RTF | First partial | WER | CER |
| --- | --- | --- | --- | --- | --- | --- |
| **Parakeet TDT 110M int8** | 136 MB | 1.77 s | **0.055** | n/a (offline) | **4.6 %** | **1.6 %** |
| zipformer 2023-06-26 int8 | 73 MB | 4.48 s | 0.129 | 0.13 s | 9.2 % | 4.5 % |

Parakeet is both more accurate and faster, so it is the default. The streaming
zipformer is kept behind `--partials` because its only remaining advantage is
word-by-word feedback. Single-pass Parakeet still supports barge-in: Silero VAD
hears the user and stops playback, the transcript arrives 0.36 s after the
sentence ends. Both are English-only; Parakeet adds punctuation and capitals.

The 2023-06-26 zipformer is a drop-in for the older 2023-02-21 one (same
encoder/decoder/joiner + `tokens.txt` interface) but 43 % smaller int8 and with
two left-context variants: `left-64` (chosen here, lower latency) and
`left-128` (more context, upstream only).

`sherpa-onnx` loads the VAD ONNX itself; it is not a separate dependency. It has
cp314 manylinux **and** Android wheels, which is why it is the speech runtime.

### Deferred or rejected

| Model | Why not now |
| --- | --- |
| Qwen3-ASR-0.6B (987 MB int8 sherpa bundle) | superseded for English: Parakeet 110M is 136 MB, faster, and CC-BY-4.0. Qwen only wins on multilingual; revisit then |
| Quantised MobileCLIP2-S0 (`felixhrdyn`, 107 MB) | **rejected after testing**: towers misaligned (dog photo scored phone 0.045 vs dog 0.044). B is proven and 577 MB fits the 1.7 GB phone budget |
| Qwen3.5-0.8B VLM (704 MB) | **in use** for "describe it"; still fetched by `--vision` so the small default bundle stays 169 MB |
| Kokoro-82M / Piper | redundant: Paradee is 8.5 MB and already integrated |
| `qualcomm/YOLOv11-Detection` | AGPL + QNN export path; the plain YOLO11n works on CPU |
| OWLv2-base int8 (167 MB) | true open-vocab boxes; only if CLIP gallery + COCO labels fail P4 |
| Depth-Anything-V2-S int8 (27 MB) | only if the ground-plane heuristic fails P2 |
| LFM3-Omni-2B Q4 + mmproj (1.4 GB) | P5 head-to-head vs Qwen only; llama.cpp has no 3.14 wheels and `cmake` is absent |
| MobileCLIP2-S2 ONNX (≈100 MB) | P6 if APK size / phone CLIP latency hurts |
| YuNet face detector | no face feature in this app |

Size budget, smallest useful demo first:

| Bundle | Contents | Size |
| --- | --- | --- |
| **Default** | YOLO11n 11 + Paradee 8.5 + g2p 13 + VAD 0.6 + Parakeet 136 | **≈169 MB** |
| `apk-vision` | + Qwen3.5-0.8B (704 MB) | 907 MB |
| `apk-full` | + MobileCLIP2-B (577 MB) | 1.5 GB |
| `full` | + streaming zipformer if fetched (`--partials`, 73 MB) | 1.6 GB |

The default is what ships in a small APK and what the demo needs. No model is
committed to git.

### Asset layout and rules

```
src/assets/models/           # gitignored except README.md
  yolo11n/ paradee/ g2p/ silero-vad/ asr-zipformer-en/
  mobileclip2-b/ qwen3.5-0.8b/     # optional, --vision
  README.md                  # provenance, size, licence per model
tools/fetch_models.py        # copies from siblings, downloads the rest
```

- Weights live under the app root (user decision), so `flet build` bundles
  them: `flet build` is ~1.5 GB once CLIP/Qwen are present. The web build
  excludes `assets/models` because Pyodide has no onnxruntime.
- Paths resolve through `adapters/paths.py` (`FLET_ASSETS_DIR`,
  `VISIONASSIST_MODELS_DIR`, `VISIONASSIST_YOLO_MODEL`).
- Each entry in `fetch_models.py` prints whether it copied, skipped or
  downloaded, and `--verify` exercises YOLO and one TTS sentence.

### Language

English-only for v1, end to end (both ASR models are English, and Paradee only
has English lexicons bundled). Hindi/French/Spanish would require eSpeak NG on
device plus a multilingual ASR (Qwen3-ASR or Parakeet 0.6B v3, 671 MB int8).

## Runtime shape

Three loops with separate latency budgets.

| Loop | Cadence | Models | Output | Budget |
| --- | --- | --- | --- | --- |
| Reflex | ~6 Hz, always on | YOLO + tracker + band policy | chime/haptic only | <200 ms |
| Cognition | on request (Talk) | live VAD endpoint → Parakeet final (→ rules → template/→ Qwen) → Paradee | one or two sentences | ~0.8 s endpoint + 0.1-0.4 s decode |
| Search | ~1.4 Hz while asked | labels first, MobileCLIP2-B for open vocab | one announcement | <15 s to find |

Measured on this laptop (640×360 webcam): YOLO ≈0.18–0.24 s/frame, model load
0.4 s, Paradee ≈62× realtime.

### Alerts are not speech

Speech is serial, slow, masks the ambient sound a blind user navigates by, and
is conspicuous in public. So:

| Information | Signal |
| --- | --- |
| distance | **silent beyond 1.0 m**; one quiet chime at 0.6–1.0 m; quick chime + haptic under 0.6 m |
| direction | stereo `balance` (-1 left … 1 right) |
| floor vs head hazard | pitch (G4 / C5 / E5); ticks use 240/320/420 Hz |
| new object | one soft double blip |
| found / not found | rising / falling two-note cue |
| imminent | haptic pulse, not a louder siren |

Styles, selectable in Settings: `soft` (default, two-note chime), `ticks`,
`voice` (silent until close, then one word "Careful."), `haptic`, `off`.
Continuous tones and buzzers are explicitly rejected: they read as alarm and
mask the environment. `tools/audition_alerts.py` renders every cue to WAV for
listening tests.

### Startup latency

TTS → greeting → ASR, with the camera loop frozen meanwhile; the VLM loads on
first use. Measured (headless, demo source): TTS 1.1 s, greeting 1.7 s, ASR done
at 3.6 s. Loading all three models up front used to postpone the greeting by
~30 s under CPU contention.

### Activation and wake word

Implemented: **"Hey Vision"** via VAD-segmented transcripts (works today, no new
model), wake-only earcon instead of speech, 8 s command window, 6 s no-wake
follow-up after each answer, shake-to-talk and the Talk button as non-voice
paths, and a startup greeting by name. The alert chime ducks to 12 % while the
user speaks (otherwise the wake rate at volume 0.45 was 0/3; now 5/5 at 0.45
and 0.8), and wake matching tolerates mangled greetings.

Target: openWakeWord (Apache-2.0, ONNX) as the always-on trigger with a 2 s
pre-roll ring buffer, so only the command goes to Parakeet. "Hey Vision" needs
a custom trained model; acceptance criteria and training outline are in
`docs/wake-word.md`. Do not enable an always-on acoustic trigger in production
until it passes those tests.

### Distance policy

Monocular distance, not apparent size: the box bottom is treated as the floor
contact point and `Z = h / tan(pitch + atan((y_bottom - cy) / f))`, with `f`
derived from the horizontal FOV. A contact point at or above the horizon has no
finite floor distance and is never alerted — that is what keeps a building
across the street silent while a chair at 1 m beeps.

Calibratable in Settings: camera height (default 1.0 m), downward tilt (20°),
HFOV (65°) and a `distance_scale` correction. Alert range: sound only at
**≤ 1.0 m**, escalate at **≤ 0.6 m**, with 0.1 m hysteresis; beyond 1.0 m the
assistant stays silent. The object must also sit in the middle third of the
view (`corridor = 0.35`, 0.05 exit margin): a cup at the frame edge is
described on request but never beeps.
Speech quotes metres in half-metre steps below 3 m, whole metres above.

Clipping: a box filling ≥75 % of the frame, or a large box whose bottom is in
the lowest ~1.5 % of the image, has its contact point outside the frame; the
estimate saturates at the frame-edge distance (~1 m with the default camera),
so the policy forces `CLOSE` and says "very close" instead of quoting a number.
`distance_scale` cannot fix that range.

Surface plane: defaults follow the laptop demo (`camera_height_m = 1.1`,
`surface_height_m = 0.75`). `surface_height_m` (0 = floor) is applied to
`SURFACE_LABELS` only, so table objects are measured against the desk while
people/chairs keep floor geometry. Calibration (`distance_scale`) is a separate
one-number correction applied after that. The ground-plane
model needs the box bottom to be on the floor; a laptop webcam watching a desk
over-reports by a roughly constant factor, so Settings offers a "place an object
at a known distance → Calibrate" correction instead of asking users to guess
height and tilt. Wearable use should leave the scale at 1.0.

Why not area ratio alone (the original suggestion): a door at 4 m and a mug at
0.5 m can occupy the same share of the frame, so size cannot answer "is it
close?"; it stays only as a fallback for objects above the horizon and as a
noise gate (`min_size_ratio`, default 3 % of frame height).

Hysteresis lives in `domain/distance.py`; the corridor/size/distance policy
lives in `domain/hazards.py`; the choice of one sound per event belongs to
`app/assistant.py` (`_handle_events`).
