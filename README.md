# VisionAssist

A Flet demo of a wearable-style assistant for blind users: a camera watches the
path, warns with **quiet non-verbal sounds**, and answers spoken/typed questions
about what is in front of you. Runs on a laptop now, Android later.

Product goals: `plan.txt`. Decisions and model selection: `plan.md`.
Complete inventory of AI models, Python packages, Flet APIs and project modules:
`model.txt`.

## What works today

- Live OpenCV camera preview at ~12 fps with labelled boxes.
- YOLO11n detection over all 80 COCO classes (household allowlist), tracked with
  stable ids and a two-hit confirmation.
- Path hazards: walking-corridor filter, **metric distance** from ground-plane
  geometry (camera height + tilt, calibratable), and a **soft two-note chime**
  whose rhythm encodes distance, whose stereo balance encodes direction, and
  whose pitch encodes floor vs head height. With calibrated distances, sound
  only starts at **1.0 m or closer**, and escalates under 0.6 m; anything
  farther is silent. The object must also be **in or near the middle of the
  view** (|pan| ≤ 0.35, the middle third): something at the edge of the frame
  never triggers an alarm, though it is still described if you ask.
- **Wake word "Hey Vision"** or push-to-talk: Silero VAD segments speech,
  Parakeet 110M transcribes (4.6 % WER, RTF 0.055, offline), and a wake-only
  phrase opens an 8 s listening window with an earcon. Details and the path to
  an acoustic wake-word model: `docs/wake-word.md`.
- Spoken answers through Paradee-8M v1.3 (8.5 MB int8 ONNX, offline, no torch):
  "what is in front of me", "describe it", "find the cup", "stop", "mute", "help".
- **"Find my keys / a mug"** works for objects outside YOLO's 80 classes:
  **MobileCLIP2-B** embeds the query and scores frames while you pan (~0.6 s per
  frame on this CPU), then localises left/centre/right from three strips.
- **"Describe it"** sends the object crop to **Qwen3.5-0.8B** (int4 ONNX) with a
  prompt that asks for one short sentence. A soft cue plays while it thinks;
  ~3–5 s on this laptop CPU. Without the bundle (or on failure) the template
  describer answers instead.
- Qwen3.5-0.8B VLM slot in the model registry (adapter present, generation loop
  not wired yet — the template describer answers meanwhile).
- Settings dialog: detection confidence, minimum object size, corridor width,
  alert style (soft / ticks / voice / haptics / off), volumes, voice, speed,
  description backend.

Not wired yet: an **acoustic** wake-word model (today the wake phrase is
matched on the Parakeet transcript, which transcribes every utterance; see
`docs/wake-word.md`), live word-by-word partials (optional zipformer),
open-vocab search (CLIP bundle not copied), Android microphone (use
`flet-audio-recorder` there; sounddevice has no APK support), mobile build.

## Run

```bash
# uses ~/Coding/pyprojects/venv (Python 3.14, flet 1.1.2)
flet run                 # desktop, hot reload
flet run --web           # UI only: Pyodide has no onnxruntime
```

The app needs models under `src/assets/models`. Copy/download them first:

```bash
python tools/fetch_models.py             # ≈169 MB: yolo11n, Paradee+g2p, VAD, Parakeet
python tools/fetch_models.py --partials  # +73 MB: streaming zipformer (live partials)
python tools/fetch_models.py --vision    # +~1.3 GB: MobileCLIP2-B (find) + Qwen (describe)
python tools/fetch_models.py --verify    # runs YOLO, TTS and one ASR sentence
```

The Qwen bundle is what makes "describe it" a real visual description rather
than a label; without it the app still runs and falls back to the template.

**Startup order.** TTS loads first, the greeting plays (~2 s on this laptop),
then Parakeet loads behind it (ready ~4 s). The 700 MB VLM is **not** loaded at
startup: the first description or free-form question pays a one-off ~5 s load
(with a cue) and is instant-ish afterwards. The camera loop is paused while the
voice loads so the greeting is not delayed by YOLO.

Sizes and speeds measured here: YOLO 0.18–0.24 s/frame; Paradee ≈62× realtime;
Parakeet ASR 4.6 % WER, RTF 0.055 on LibriSpeech dummy (`docs/asr-benchmark.md`);
streaming zipformer 9.2 % WER, RTF 0.129, first partial 0.13 s.

No camera? `VISIONASSIST_SOURCE=demo flet run` uses a synthetic feed;
`VISIONASSIST_SOURCE=/path/clip.mp4 flet run` replays a video.

Camera says "busy or unavailable"? Another process is holding `/dev/video0`,
often a leftover app from an earlier run. `fuser -v /dev/video0` shows the PID;
kill that PID. The source retries every 2 s, so it recovers on its own once the
device is free.

## Using it

- **Say "Hey Vision"** followed by the request ("Hey Vision, is there a cup in
  front of me?") and it answers in one go. If the alert chime is playing it
  **ducks to 12 %** as soon as you start speaking, so it cannot mask the wake
  phrase; matching also tolerates what ASR does to it under noise ("visions",
  "day vision", "a vision", or just "vision, …"). Or tap **Talk**, speak, and pause:
  VAD ends the recording about 0.8 s after you stop. On a phone, **shake** to
  talk. A soft rising earcon means "listening"; speech is used only for answers.
- **Follow-ups need no wake word** for 6 s after an answer ("describe it",
  "where is it"). Background chatter in that window is ignored.
- Close the loop by voice ("stop", "mute") or with the Talk/Stop buttons.
- The app greets you by name at startup ("Hi David. Vision assist is ready.")
  and tells you the wake phrase; both can be changed or disabled in Settings.
- `Settings → Microphone` picks the input device; the recording cap is only a
  safety limit and is normally not reached.
- **What's in front?** → terse spoken summary, largest objects first.
- **Is there a cup? / where is the cup?** → answered from the current frames
  (three attempts over ~0.75 s, so holding it up works). Everyday words are
  mapped to detector labels (`mug`→cup, `sofa`→couch, `table`→dining table).
- **Anything else about the picture** ("what colour is the cup?", "how many
  people are here?") → the question is sent with the current frame to the VLM,
  which answers in one short sentence (3–5 s). Rules still handle the fixed
  commands first; questions about the world beyond the image are not answered.
- **Describe it** → the biggest object (or the named one), via the description
  model; falls back to a template sentence when the model is unavailable.
- **Find the cup** → searches the current frame for two stable hits, then
  announces position and distance.
- **Stop / mute / help** → control and discovery.
- Settings (AppBar gear) changes everything live and persists to
  `~/.config/vision-assist/settings.json`.

Alert styles: `soft` (default, recommended), `ticks` (shorter, still quiet),
`voice` (no beeps; says "Careful." when something is close), `haptic`, `off`.
Regardless of style, nothing beyond the "warn from" distance (1.0 m) makes a
sound.
In Settings, set camera height/tilt, "warn from" distance (default 1.0 m), the
close-warning distance (0.6 m) and the alert corridor (default 0.35 = middle
third); the preview annotation shows the estimate in metres. A "Reset to 1.0 m / 0.6 m" button restores these if a saved value
lingers.

**Very close objects.** Ground-plane geometry needs the object's base inside the
frame. When someone sits 40 cm away their feet are below the frame, the box is
clipped by the bottom edge, and the estimate saturates at ~1 m no matter the
calibration. The app detects that case — a box filling **≥75 % of the frame height**, or a
large box whose bottom sits in the lowest ~1.5 % of the frame — and reports
**"very close"** with a close alarm instead of a misleading metre value.

**Objects on a table.** A phone or cup rests on a surface, not the floor. The
defaults describe a laptop: camera **1.1 m** above the floor, objects on a
**0.75 m** desk. The same phone then reads ~0.65 m instead of ~2 m, while people
and chairs keep floor geometry. Settings has one-click presets —
**"Laptop on a desk"** and **"Wearable"** (1.0 m, floor) — and the geometry
sliders apply live, without pressing Save.

**Distance calibration.** For objects whose base *is* visible, the model
measures the contact point under the box. A laptop webcam looks at objects on a
*desk*, at its own height, so floor-based estimates run about 3× too large (1.0 m
reported for a 0.33 m object). Settings → *Distance
scale* corrects it: put an object at a measured distance, read "Current
estimate", type the real distance and press **Calibrate**. One correction is
usually enough; the value is persisted and shown as `dist ×0.33` in the status
panel so you can confirm it is active. For a wearable that sees the floor, set
the camera height/tilt instead (1.0 m, 20° down) and leave the scale at 1.0.
`python tools/audition_alerts.py` renders all of them to `/tmp/vision-assist-alerts`
so you can compare before a demo.

## Wake word under noise

The chime and the user share one speaker and one microphone, so a full-volume
alarm destroys the wake phrase (measured: 0 % recognition at alarm volume
0.45). The wake loop therefore ducks the chime to 12 % while speech is present
and restores it afterwards, and the phrase matcher accepts mangled greetings.
After that change the same measurement is 5/5 at volumes 0.45 and 0.8. The
acoustic trigger (openWakeWord) remains the long-term fix; see
`docs/wake-word.md`.

## Wake word and privacy

"Hey Vision" is detected by transcribing VAD-segmented utterances and matching
the phrase — accurate today, but it means speech in the room is transcribed
locally even when it does not start with the wake phrase. Nothing is written to
disk or sent over the network; wake can be disabled in Settings, and the status
panel shows the active phrase. The planned acoustic trigger (openWakeWord, ONNX)
removes the "transcribe everything" step; the design, training plan and
acceptance criteria are in `docs/wake-word.md`.

## Microphone troubleshooting

Every recording logs one line to the terminal:

```
[asr] recorded 3.10s (rms 0.040, peak 0.25), speech 1.29s
[asr] parakeet-110m 0.11s -> 'What is in front of me?'
```

- `rms` below ~0.005 means the microphone is silent: check the OS input level
  and mute state; on Android, allow the microphone permission when prompted.
- `speech 0.00s` means VAD heard no speech: speak after the ping and keep the
  pause short.
- If the wrong device is selected, the transcript is noise; use the OS default
  input device (the app does not switch devices).

## Tests

```bash
python -m pytest          # 90 unit tests, no camera/models/audio needed
```

`test_yolo.py`, `test_paradee.py` and `test_asr.py` skip themselves when their
assets are absent; `tests/test_asr.py` also exercises VAD trimming and Parakeet
on Paradee-synthesised speech (no microphone needed). Wake-phrase parsing and
the no-wake follow-up window are covered by unit tests in `test_intents.py` and
`test_assistant.py`.
There is no Flutter integration test: `cmake` is not installed on this machine.

## Layout

```
src/domain/      pure logic: bands, hazards, scene text, intents, settings
src/adapters/    capture (OpenCV), detection (YOLO+tracks), speech (Paradee,
                 earcons), models (VLM registry)
src/app/         Assistant orchestration
src/ui/          Flet AppBar, preview, panels, settings dialog, audio port
tools/           fetch_models.py, audition_alerts.py
```

## Credits

- YOLO11n — Ultralytics, AGPL-3.0.
- MobileCLIP2-B — Apple, `apple-amlr` (research licence; fine for a demo).
- Paradee-8M v1.3 — Apache-2.0; English G2P lexicons from `hexgrad/misaki`.
- Parakeet TDT 110M — NVIDIA, **CC-BY-4.0**, exported for sherpa-onnx by the
  k2-fsa team.
- Streaming zipformer (2023-06-26) — Apache-2.0, k2-fsa / icefall.
- Silero VAD — MIT.

## APK size

Flet copies `src/assets` into the app, so keep optional models outside it while
packaging:

```bash
python tools/package_models.py --profile apk   # 169 MB assets: YOLO, Paradee+g2p, VAD, Parakeet
flet build apk -v
python tools/package_models.py --profile full  # restore the VLM + zipformer afterwards
```

- `apk` (169 MB): detection, warnings, voice input and spoken answers. The
  describer falls back to the template, so “describe it” says what the detector
  saw rather than a full sentence.
- `apk-vision` (907 MB): adds Qwen3.5-0.8B for real descriptions and free-form
  questions.
- `apk-full` (1.5 GB assets): adds MobileCLIP2-B for “find my keys”, plus Qwen.
  This is the profile for a phone that can take ~1.7 GB including the runtime.
- `full` (≈1.5 GB after cleanup): restores Qwen/CLIP for local work. The
  optional streaming zipformer is not shipped; fetch it with
  `fetch_models.py --partials` if you ever want live partials.

Android audio input and output are wired (`flet-audio-recorder` +
`flet-audio`), and `pyproject.toml` declares `RECORD_AUDIO` and
`MODIFY_AUDIO_SETTINGS`. The camera is still desktop-only until a
`flet-camera` frame source replaces OpenCV on the phone.

### Android: sherpa-onnx libc++ shim

`sherpa-onnx` is not on Flet's supported wheel index. Its Android wheel is
built with NDK r29/r30 and needs `std::__ndk1::__hash_memory`, but Flet's Python
runtime ships NDK r27 libc++. Without a fix the app dies on start with:

```
ImportError: dlopen failed: cannot locate symbol
"_ZNSt6__ndk113__hash_memoryEPKvm" referenced by libsherpa_onnx-lib-_sherpa_onnx.so
```

The repo ships a tiny `libcxxshim.so` (source `tools/android/libcxxshim.cpp`) and
preloads it on Android before importing sherpa. Build and patch:

```bash
flet build apk -v
python tools/patch_android_apk.py          # adds + signs the shim
adb install -r build/flutter/build/app/outputs/flutter-apk/app-arm64-v8a-release-patched.apk
```

Uninstall the old app first: the patched APK is signed with a local debug key.
The `UserWarning: Unsupported platform (android)` from ONNX Runtime is harmless —
ONNX Runtime itself is supported by Flet and imports fine.

## Build

```bash
flet build apk -v      # Android; onnxruntime wired via [tool.flet.android]
flet build linux -v    # desktop bundle
```

Packaging bundles `src/assets/models` into the app, so copy only the models the
target needs (`--vision` is desktop-only for now).
