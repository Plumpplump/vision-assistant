#!/usr/bin/env python3
"""Render every alert sound to WAV so a human can judge it.

    python tools/audition_alerts.py [output-dir]

Defaults to ``/tmp/vision-assist-alerts``. Play a few files side by side
before deciding on the default style; "loud on a laptop" and "startling in a
quiet room" are different judgements, and only ears can make them.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapters.speech import ParadeeTts  # noqa: E402
from adapters.speech.earcons import Earcons  # noqa: E402
from domain import AlertStyle, Band, Vertical  # noqa: E402


def peak_dbfs(wav: bytes) -> float:
    import numpy as np

    handle = wave.open(io.BytesIO(wav))
    data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    peak = float(np.max(np.abs(data))) / 32767.0 if data.size else 0.0
    if peak <= 0:
        return -120.0
    import math

    return 20 * math.log10(peak)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision-assist-alerts")
    out.mkdir(parents=True, exist_ok=True)
    earcons = Earcons()

    written: list[tuple[str, bytes]] = []
    for style in (AlertStyle.SOFT, AlertStyle.TICKS):
        for band in (Band.FAR, Band.NEAR, Band.CLOSE):
            wav = earcons.pattern(band, Vertical.FLOOR, style)
            if wav:
                written.append((f"{style.value}-{band.value}.wav", wav))

    written.append(("cue-blip.wav", earcons.blip()))
    written.append(("cue-found.wav", earcons.found()))
    written.append(("cue-miss.wav", earcons.miss()))

    tts = ParadeeTts()
    if tts.available():
        written.append(("voice-careful.wav", tts.synthesize("Careful.")))
        written.append(("voice-answer.wav", tts.synthesize("Chair, close, on your left.")))
    else:
        print(f"skip  voice samples: {tts.error}")

    for name, wav in written:
        (out / name).write_bytes(wav)
        print(f"{name:24s} {len(wav) / 1024:6.1f} kB  peak {peak_dbfs(wav):6.1f} dBFS")

    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
