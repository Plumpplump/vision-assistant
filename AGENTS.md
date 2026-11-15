# AGENTS.md

Flet 1.x app: a vision assistant demo for blind users. Camera → detection →
non-verbal path alerts; spoken answers on request. `plan.md` holds the decided
model selection and the reasoning; `plan.txt` is the original brief; `model.txt`
is the model/module inventory (keep it current when models change).

## Environment
- Use the **shared venv `/home/panger/Coding/pyprojects/venv`** (Python 3.14):
  `source /home/panger/Coding/pyprojects/venv/bin/activate` or call
  `/home/panger/Coding/pyprojects/venv/bin/{python,flet,pytest}`.
- **Do not run `uv run` / `uv sync` here.** uv would create a project-local
  `.venv` and ignore the shared env.
- Installed there: `flet` 1.1.2 + `flet-cli`/`flet-desktop`/`flet-web`/
  `flet-audio`, `onnxruntime`, `opencv-python-headless`, `numpy`, `pytest`,
  `pytest-asyncio`, `platformdirs`. No torch, no transformers.
- `cmake` is **not** installed, so `flet test` / the Flutter integration-test
  harness cannot run. All tests here are pure-python unit tests.

## Commands (from the repo root)
- Desktop app (hot reload): `flet run`
- Web app: `flet run --web` (Pyodide has no onnxruntime — UI only, no models)
- Tests: `python -m pytest`; single file `python -m pytest tests/test_hazards.py`
- Models: `python tools/fetch_models.py` (≈169 MB); `--partials` adds the
  streaming zipformer; `--vision` adds MobileCLIP2-B (577 MB) + Qwen (~704 MB);
  `--verify` exercises YOLO, TTS and one Parakeet sentence
- Packaging: `python tools/package_models.py --profile apk|apk-vision|apk-full|full`
  moves optional bundles to `models-optional/` (gitignored) so `flet build` only
  ships what you choose. Never move weights by hand; restore with `full`.
- ASR benchmark: `python tools/bench_asr.py --limit 73` (needs `pyarrow` +
  `ffmpeg`, downloads the LibriSpeech dummy parquet once)
- Microphone path: `Assistant.listen()` → `SoundDeviceMicrophone` (desktop) →
  Silero VAD trim → `ParakeetAsr`; all three run in worker threads
- Render alert sounds for listening: `python tools/audition_alerts.py`
- Packaging: `flet build apk|linux|web ... -v`

## Architecture (keep these boundaries)
```
src/domain/      pure python: types, distance bands, HazardMonitor, scene text,
                 intent rules, Settings. No flet, numpy, cv2 or onnx imports.
src/adapters/    I/O: capture/ (OpenCV), detection/ (YOLO ONNX + tracker),
                 speech/ (Paradee TTS, earcons, g2p, vocab),
                 models/ (VLM registry), paths.py
src/app/         Assistant: perception loop, alert policy, intents, settings.
                 Flet-free; audio arrives through the `Speaker` port.
src/ui/          Flet only: AppBar, preview, panels, settings dialog,
                 FletSpeaker (audio implementation).
```
Rules that matter:
- Model swapping goes through `adapters/models/registry.py`
  (`VISIONASSIST_DESCRIBER`, `register()`); nothing outside `models/` names a
  model. Unknown/broken backends degrade to `UnavailableDescriber`, never crash.
- Never run ONNX inference or TTS on the Flet event loop: use
  `asyncio.to_thread` (the perception loop and `FletSpeaker` do).
- A capture thread never touches Flet controls; the UI pulls `source.latest()`.

## Camera and models
- **Camera is OpenCV only** (`adapters/capture/opencv_source.py`); `flet_camera`
  is deliberately not used (no Linux desktop support, and the demo runs on a
  laptop). `VISIONASSIST_SOURCE`: unset/index = camera, `demo` = synthetic,
  a path = video file.
- Weights live in **`src/assets/models/`** (user decision), gitignored except
  `README.md`. `tools/fetch_models.py` copies from the sibling projects
  (`~/Coding/pyprojects/schooluniform`, `~/Coding/pyprojects/SAT-Words`) and
  downloads VAD/ASR.
- Flet bundles `src/assets` into packaged builds. Use
  `tools/package_models.py --profile apk` before building (169 MB); `full`
  restores Qwen/zipformer for local use. The web build excludes `assets/models`.
- `src/assets/models/README.md` records provenance, size and licence per model.
  YOLO11n is AGPL-3.0 (Ultralytics); MobileCLIP2 is `apple-amlr` (research);
  Parakeet 110M is CC-BY-4.0 and **must be credited** in shipped builds.
- Speech input uses **`flet-audio-recorder`** (`FletAudioRecorderMicrophone`,
  16 kHz mono PCM16 streaming) and works on desktop, Android and web. The
  service needs a `Page`, so the UI constructs it and injects it into
  `Assistant`; without a page the app uses `NullMicrophone`.
  `Microphone.record` is **async** now; `_record_with_duck` creates a task for
  it and polls `endpoint.started` to duck alerts. `sounddevice` remains only as
  a dev fallback for `tools/test_wake.py`.
- Android: `flet-audio-recorder` requires `android.permission.RECORD_AUDIO`
  (declared in `[tool.flet.android.permission]`), plus optional
  `MODIFY_AUDIO_SETTINGS` for Bluetooth headsets. `min_sdk_version = 24` is
  required by the sherpa-onnx wheels. Do **not** add a CAMERA permission until a
  `flet-camera` source replaces OpenCV on the phone.
- Intent order in `domain/intents.py`: CHECK ("is there X") and LOCATE ("where
  is X") answer from live frames; FIND ("looking for X") starts a 12 s panning
  search. SCENE ("what is in front of me") must stay last, otherwise a question
  containing "in front of me" is swallowed by it. `canonical_target()` maps
  everyday words (`mug`) to COCO labels (`cup`).
- `LiveEndpoint` (streaming Silero) stops the recording ~0.8 s after speech
  ends; `Settings.listen_seconds` is only a cap. ASR/TTS are warmed in
  `Assistant._warm_models` so the first Talk press does not pay a 3 s load.
  `listen()` prints an `[asr]` line with rms/peak/speech length — use it to
  distinguish "mic silent" from "VAD found no speech" from "ASR error".
- Wake word: `Settings.wake_word` defaults to **"hey vision"**; aliases and the
  tolerant matcher live in `domain/intents.py` (`WAKE_ALIASES`,
  `strip_wake_word`). It matches anywhere in the utterance, accepts a name-first
  form ("vision, what is in front of me"), and tolerates one edit per word
  ("visions", "vison") plus mangled greetings ("day vision", "a vision").
- **Duck the alert chime while the user speaks.** `Speaker.duck_alerts` is
  implemented by `FletSpeaker` (12 % volume) and driven by
  `Assistant._record_with_duck`, used by both the wake loop and push-to-talk.
  Without it, a 0.45-volume chime removed "hey vision" from the transcript and
  the wake rate was 0/3; with it, 5/5 at 0.45 and 0.8. Do not remove this when
  touching audio.
- `tools/test_wake.py` reproduces wake behaviour acoustically: `--alarm` loops
  the chime for a human, `--say "…"` mixes chime + TTS (`--duck` simulates the
  app) for a repeatable A/B. Detection is *currently* the transcript
  fallback: the wake loop records VAD utterances and matches the phrase after
  Parakeet. That is intentional until an openWakeWord model exists — do not
  claim an acoustic KWS in docs. Wake-only plays `Earcons.wake()` (never a
  spoken "Yes?"), opens an 8 s window, and after every answer a
  `follow_up_seconds` (6 s) window accepts commands without the wake phrase
  (unknown intents are ignored, not answered). The wake loop is paused while
  `listening` or `speaking`, with a 0.6 s tail. Design/upgrade path:
  `docs/wake-word.md`.
- Accessibility: keep the Talk button, the **shake-to-talk** service
  (`ft.ShakeDetector`, mobile only) and typed commands working; wake is never
  the only way to start a request. The startup greeting says "Hi David. Vision
  assist is ready." (`greet_name`), and speech is reserved for answers.
- Silero VAD must be fed in `window_size` chunks (512 samples at 16 kHz);
  handing it a whole recording returns almost nothing. `SpeechSegmenter` does
  this; keep it that way. Trimming keeps the contiguous span from first to last
  speech window — do not concatenate segments, it removes pauses and clips
  words at the boundaries.
- Size first: the default bundle is ≈169 MB (YOLO 11, Paradee 8.5, g2p 13,
  VAD 0.6, Parakeet 136). Bigger models are opt-in: `--partials` adds the
  streaming zipformer (73 MB), `--vision` adds MobileCLIP2-B (577 MB) + Qwen
  (704 MB). Do not make CLIP/Qwen defaults.
- Speech input defaults to **Parakeet TDT 110M** (offline, 136 MB,
  `model_type=nemo_transducer`): measured 4.6 % WER / RTF 0.055 vs the streaming
  zipformer's 9.2 % / 0.129 on the same 73-clip corpus (`docs/asr-benchmark.md`).
  It cannot emit partials, so "listening" feedback comes from VAD; barge-in
  still works because VAD stops playback. The zipformer is optional
  (`--partials`) only for live word-by-word text. Re-run the benchmark before
  changing this ordering.
- Startup order matters: `_warm_models` warms **TTS first, speaks the greeting,
  then loads ASR**. The VLM is loaded lazily on first use. `_warming` freezes
  the perception loop and the wake loop while this runs; do not put the VLM
  back into startup warm-up (it delayed the greeting to ~30 s on a busy CPU).
- `Settings.describer` defaults to **"auto"**: Qwen3.5-0.8B when its bundle is
  present, the template describer otherwise. The Qwen adapter is
  `adapters/models/qwen_vlm.py` over `qwen_runner.py` (copied from the sibling
  project, extended from yes/no to greedy generation). Prompt policy lives in
  the adapter: one short question, system instruction caps the answer at 15
  words, `clean_description` trims to one sentence/20 words. Image size
  defaults to 128 px (≈12 image tokens); 96/160 are the speed/detail knobs
  (`VISIONASSIST_IMAGE_SIZE`, must be a multiple of 32). The `_lasttoken`
  decoder variant is used when present (LM head only on the final position).
  Do not run the VLM and the detector at once: `_describe_active` freezes the
  perception loop for the 3–5 s inference, and the model is warmed at startup.
- Free-form visual questions: intents win; when nothing matches and the
  utterance starts with an open question word (`looks_like_question`),
  `_answer_question` sends the text plus the crop of the mentioned/largest
  object to the describer. It never runs with the `fake` describer, so tests and
  model-less installs keep the "didn't catch that" reply. Keep the question
  short and the answer one sentence; the 0.8B model rambles otherwise.
- Open-vocabulary search uses **MobileCLIP2-B** (`adapters/models/clip.py`,
  577 MB). It is loaded lazily, embeds the query once (cached), then scores
  frames; a hit gets localised by scoring left/centre/right strips. Threshold
  default **0.22** (`Settings.clip_threshold`); measured cup photo: cup 0.305,
  spoon 0.240, phone 0.139. Encoding costs ~0.6 s, so `_describe_active` pauses
  perception during each pass and the search ticks at ~1.4 Hz. A quantised S0
  export was tried and rejected (towers misaligned); do not replace B with an
  untested export.
- `tools/fetch_models.py` checks exact byte counts: model repos that reuse
  filenames (the zipformer releases do) are refreshed instead of silently kept.

## Audio gotchas (Flet)
- Never construct `flet_audio.Audio` with `src=None`; the client service init
  throws and `play()` hangs. Construct lazily with a valid `src`.
- Assign `src`, call `page.update()`, wait for `on_loaded`, then `play()`.
- `ReleaseMode.LOOP` for alert patterns (sample-accurate), `ReleaseMode.STOP`
  for one-shots (instant replay). There is no `stop()`; use `pause()`.
- `ft.HapticFeedback` is a service: `page.services.append` + `page.update`;
  it is a no-op on desktop.

## Alert design rules (do not regress these)
- Default is a **soft two-note chime**, not a buzzer. No continuous tones, no
  square waves, no high-duty-cycle beeping: those read as alarm and mask
  traffic/footsteps.
- **Distance is metric, not apparent size.** `domain/distance.py` uses
  ground-plane pinhole geometry: `Z = h / tan(pitch + atan((y_bottom - cy) / f))`
  with `f` from `hfov_deg`. A box bottom at/above the horizon returns `None`
  and never alerts. Do not replace this with an area/size ratio: a door at 4 m
  and a mug at 0.5 m can fill the same share of the frame.
- Alerts start at `alert_near_m` (default **1.0 m**) and escalate at
  `alert_close_m` (**0.6 m**); a 0.1 m hysteresis band prevents chatter. Nothing
  beyond 1.0 m makes a sound. These thresholds apply to *calibrated* metres
  (`distance_scale`), so the two settings must be tuned together.
- **Clipping overrides geometry.** A box filling ≥75 % of the frame height is
  `very close` regardless of geometry; below that, a large box (≥50 % height)
  whose bottom sits in the lowest ~1.5 % of the frame also counts as clipped
  (`is_very_close`). The contact point is then outside the image and the
  distance saturates near 1 m. Such objects are
  forced to `Band.CLOSE` and spoken as "very close" — do not try to fix this
  with `distance_scale`, there is no data below the frame.
- Distance = rhythm, direction = stereo `balance`, height = pitch. Alerts fire
  only on band change.
- Speech is for answers on request. The only exception is the `voice` alert
  style, which says one word ("Careful.") when something is close.
- Styles: `soft` (default), `ticks`, `voice`, `haptic`, `off`, selectable in
  Settings (`Settings.alert_style`) and in `domain.types.AlertStyle`.
- Objects on a table: `surface_height_m` (default 0 = floor) moves the contact
  plane for `SURFACE_LABELS` (phone, cup, bottle, laptop, book, …). Hazards
  always use floor geometry so a table cannot make a person look close. With
  camera 1.1 m / surface 0.75 m a phone reads ~0.65 m instead of ~2 m. This is
  separate from `distance_scale`; set the surface first, then calibrate.
- Defaults describe the laptop demo: `camera_height_m = 1.1`, `surface_height_m
  = 0.75`. Settings has presets ("Laptop on a desk", "Wearable") and the
  geometry sliders use `on_change_end` to apply live, so a stale dialog cannot
  leave the app measuring against the wrong plane. Status shows `surface 0.75 m`
  when active.
- Camera calibration lives in Settings: `camera_height_m` (default 1.1),
  `camera_pitch_deg` (20°, downward), `camera_hfov_deg` (65°), and
  `distance_scale` (default 1.0). The scale is a one-number fix for the
  floor-point assumption: a laptop webcam sees desk objects at its own height
  and over-reports by roughly the same factor (~3×). `calibrate_scale()` and
  the Settings "Calibrate" button derive it from one known distance; keep the
  scale at 1.0 for a floor-facing wearable and fix height/tilt instead. With a 20° tilt
  the bottom frame edge is ~1 m away; with 0° it is ~2.2 m, so "close" alerts
  are physically impossible without tilt. Quote metres in half-metre steps
  below 3 m, whole metres above.

## Testing notes
- 191 unit tests, no camera or audio device required.
- `test_yolo.py`, `test_paradee.py` and `test_asr.py` auto-skip when their assets are missing.
- `tests/test_assistant.py` shows the pattern for new tests: stub detector,
  stub source, `RecordingSpeaker`; drive `Assistant._analyze` twice because the
  tracker confirms on the second hit.
- There is no Flutter integration test: do not add one until `cmake` exists.

## Flet 1.x API traps
- Buttons take `content=` (`ft.FilledButton(content="Save")`), not `text=`.
- `ft.Image` requires `src` (use the 1x1 `BLANK_PIXEL` in `ui/app.py`) and
  **`gapless_playback=True`**; without it Flutter blanks the image for a frame
  on every provider change, which is the flashing preview at camera rate.
- `ft.run(main)`, `ft.Icons.ADD`, `ft.Alignment.CENTER` (enum members).
- Handlers get `ft.Event[Control]` with `e.control` / `e.page`.
- Update after mutating controls (`page.update()`), and always after assigning
  a service property.
- **Never `pkill -f flet`** — it also kills OpenCode's `flet mcp` child, and
  `flet run` spawns the app in its own session, so killing the CLI process
  group leaves it running.
- A smoke-test `flet run` leaves `src/main.py` alive (parent pid 1) and it keeps
  `/dev/video0` open, so the next launch reports a busy camera. Kill that exact
  PID; `fuser -v /dev/video0` tells you which one. `OpenCVSource` retries a busy
  device every 2 s and shows the reason in the preview placeholder, so a
  transient conflict recovers without a restart.
