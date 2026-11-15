# Wake word ("Hey Vision")

Goal: a blind user can start a request without finding a button, but the app
must not feel like an always-on recorder. This note records what is implemented
today, why, and the upgrade path to a proper keyword spotter.

## What ships now (demo-grade, works today)

```
mic ─► Silero VAD (idle, ~1 ms / 32 ms chunk)
        └─ speech detected ─► LiveEndpoint (0.7 s silence ends the utterance)
                              └─► Parakeet 110M (0.1–0.4 s)
                                   └─► strip wake phrase ─► intent ─► answer
```

- Wake phrase: **"Hey Vision"** (aliases in `domain/intents.py`:
  "hi vision", "okay vision", "hey assistant", …). Settings → Wake phrase.
- The phrase and the command can be in one breath:
  *"Hey Vision, is there a cup in front of me?"* → answers immediately.
- Wake-only ("Hey Vision") plays a short rising earcon and opens a 8 s window;
  no spoken "Yes?" (speech would talk over the user).
- After any answer a **6 s follow-up window** accepts a command without the
  wake word ("describe it", "where is it"). Background chatter in that window
  is ignored unless it parses as a known intent — no "sorry, didn't catch that".
- While the assistant speaks, the wake loop is paused plus a 0.6 s tail so it
  cannot wake on its own voice.
- No audio is written to disk or sent anywhere; only the transcript of an
  utterance exists in memory.

Measured on this laptop: wake detection ends 0.7 s after you stop speaking,
Parakeet adds 0.1–0.4 s, so ~1 s from silence to answer.

## Alarm interference

The alert chime plays from the same speaker the microphone hears, so it masks
the wake phrase. Measured with `tools/test_wake.py` (chime mixed with a spoken
phrase, three trials each):

| Chime volume | Ducking | Wake matches |
| --- | --- | --- |
| none | — | 3/3 |
| 0.45 | no | **0/3** (`vision, what is in front of me?`) |
| 0.45 | 12 % | 3/3 |
| 0.80 | 12 % | 3/3 (after tolerant matching) |

Two mitigations ship:

1. **Ducking.** As soon as VAD sees speech, `duck_alerts(True)` lowers the
   looping chime to 12 % for the duration of the utterance, then restores it.
   The hazard cue stays audible, but no longer drowns the voice.
2. **Tolerant matching.** The transcript under noise loses or mangles the
   greeting ("vision, …", "day vision", "a vision"). The matcher accepts a name
   in the first two words when the remainder parses as a known command, and
   scans the whole utterance for "hey/ok + vision".

Remaining limits: loud music/TV still competes; a real acoustic keyword spotter
remains the robust answer.

## Why this is not yet best practice

| Problem | Consequence |
| --- | --- |
| Every utterance is transcribed, then discarded if it has no wake phrase | CPU/battery per utterance, and "transcribe the room" is a hard sell for privacy |
| Wake latency = end-of-utterance + ASR | ~1 s; a dedicated KWS fires 200–400 ms after the phrase |
| No acoustic pre-roll | Fine today (we keep the whole utterance), but a KWS needs a ring buffer so the command's first words are not lost |
| False accepts on conversation that starts with "Hey Vision" | Rare, but cannot be tuned without an acoustic score |
| Wake word is not authentication | Anyone can trigger it; never gate anything sensitive on it |

## Target architecture

```
mic ─► VAD + openWakeWord (Apache-2.0, ONNX, ~1 MB, ~1 % CPU, always on)
        ├─ score > threshold for 2 frames ─► 2 s pre-roll ring buffer
        └─► streaming endpoint ─► Parakeet ─► intent ─► answer
```

Why openWakeWord rather than Porcupine: both are strong, but Porcupine's Python
SDK has no Android target, and this app runs Python-on-Android through Flet.
openWakeWord is pure Python + ONNX, i.e. the stack we already ship.

"Hey Vision" is not a pretrained phrase, so it needs a custom model. Outline:

1. `pip install openwakeword` (already installed in the dev venv for
   experiments; not yet an app dependency).
2. Generate positives with synthetic TTS (Paradee's four English voices at
   varied speed/pitch, plus noise/augmentation) and negatives from LibriSpeech
   and room/TV recordings. The openWakeWord training notebook accepts this
   layout.
3. Validate: ≥95 % detection over 20 attempts per condition (quiet, TV on,
   2 m away), ≤1 false accept per hour of speech-like audio, latency <400 ms.
4. Integrate behind a `WakeDetector` port: the loop reads 80 ms frames,
   keeps a 2 s ring buffer, and only starts the ASR after a match. Keep the
   transcript fallback for platforms without a model.

Do not wire the KWS in until step 3 passes; an always-on trigger that misfires
is worse than a button.

## Accessibility decisions (and why)

- **Non-voice triggers stay first-class**: the big Talk button and
  **shake-to-talk** (`ft.ShakeDetector`, mobile). Voice fails in noise, with a
  sore throat, or when the user does not want to speak in public.
- **Earcons, not words, for state**: wake = rising pair, listening = soft ping,
  close hazard = chime. Speech is reserved for answers the user asked for.
- **No wake word for follow-ups** for 6 s after an answer, so "describe it"
  works naturally.
- **Names**: the wake phrase names the assistant ("Hey Vision"); "Hi David" is
  the greeting the assistant says at startup. Using the user's own name as the
  wake word makes every conversation about David a false trigger.
- **State is visible and audible**: status text shows the wake phrase, the Talk
  button flips to "Listening… tap to stop", Settings can disable wake entirely.
