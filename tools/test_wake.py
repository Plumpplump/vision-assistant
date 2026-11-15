#!/usr/bin/env python3
# Author: David Li, SISHK

"""Wake-word probe: log exactly what the wake loop hears.

    python tools/test_wake.py --seconds 120

Say the wake phrase (and anything else) while it runs. For every VAD-detected
utterance it prints the speech length, input level, Parakeet transcript and
whether the wake matcher accepted it. Silence with a non-trivial level means
VAD rejected speech; a transcript without a match means the phrase was heard
but not recognised.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapters.speech import (  # noqa: E402
    Earcons,
    LiveEndpoint,
    ParadeeTts,
    ParakeetAsr,
    create_microphone,
)
from domain import AlertStyle, Band, Vertical  # noqa: E402
from domain.intents import strip_wake_word  # noqa: E402

SAMPLE_RATE = 16000
PLAYBACK_RATE = 24000


def decode_wav(wav: bytes) -> np.ndarray:
    handle = wave.open(io.BytesIO(wav))
    pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


class AlarmLoop:
    """Plays the close-hazard earcon on a loop while the probe records."""

    def __init__(self, wav: bytes, volume: float) -> None:
        import sounddevice as sd

        samples = decode_wav(wav) * volume
        self._samples = samples.astype(np.float32)
        self._position = 0
        self._stream = sd.OutputStream(
            samplerate=PLAYBACK_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status) -> None:
        index = (self._position + np.arange(frames)) % self._samples.size
        outdata[:, 0] = self._samples[index]
        self._position = (self._position + frames) % self._samples.size

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


def mixed_buffer(
    alarm: bytes,
    phrase: np.ndarray,
    volume: float,
    lead: float = 1.0,
    duck: bool = False,
) -> np.ndarray:
    """Alarm loop + a spoken phrase, mixed like one laptop speaker would.

    ``duck`` reproduces the app's behaviour: when speech starts, the chime
    drops to 12 % so it does not mask the voice (and the ASR).
    """
    loop = decode_wav(alarm) * volume
    total = int((lead + len(phrase) / PLAYBACK_RATE + 1.5) * PLAYBACK_RATE)
    buffer = np.zeros(total, dtype=np.float32)
    repeats = int(np.ceil(total / loop.size))
    tiled = np.tile(loop, repeats)[:total]
    start = int(lead * PLAYBACK_RATE)
    if duck:
        tiled[start:] *= 0.12
    buffer += tiled
    buffer[start : start + phrase.size] += phrase
    return np.clip(buffer, -1.0, 1.0)


async def record_utterance(microphone, seconds: float, silence: float, threshold: float):
    endpoint = LiveEndpoint(
        max_seconds=seconds, silence=silence, min_speech=0.15, threshold=threshold
    )
    stop = threading.Event()
    started = time.monotonic()
    samples = await microphone.record(seconds, stop, endpoint, None)
    return samples, endpoint.speech(), time.monotonic() - started


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--phrase", default="hey vision")
    parser.add_argument("--silence", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument(
        "--alarm", action="store_true", help="loop the close-hazard chime while probing"
    )
    parser.add_argument("--alarm-volume", type=float, default=0.45)
    parser.add_argument(
        "--say",
        help="self-test: play this text through the speakers instead of a human",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--duck",
        action="store_true",
        help="self-test: drop the alarm to 12%% when the phrase starts",
    )
    args = parser.parse_args()

    microphone = create_microphone()
    print(f"microphone: {microphone.name} (available={microphone.available()})")
    if not microphone.available():
        print(f"error: {microphone.error}")
        return 1

    asr = ParakeetAsr()
    if asr.error:
        print(f"error: {asr.error}")
        return 1
    print("loading Parakeet…", flush=True)
    asr.load()

    alarm_wav = None
    alarm = None
    if args.alarm or args.say:
        alarm_wav = Earcons().pattern(Band.CLOSE, Vertical.FLOOR, AlertStyle.SOFT)
        if alarm_wav is None:
            raise SystemExit("alarm pattern unavailable")
    if args.alarm and not args.say:
        import sounddevice as sd

        alarm = AlarmLoop(alarm_wav, args.alarm_volume)
        alarm.start()
        print(
            f"alarm: on, looping close chime at volume {args.alarm_volume:.2f} "
            "(you speak over it)",
            flush=True,
        )

    if args.say:
        import sounddevice as sd

        tts = ParadeeTts()
        for trial in range(1, args.trials + 1):
            phrase = decode_wav(tts.synthesize(args.say))
            audio = mixed_buffer(
                alarm_wav, phrase, args.alarm_volume, duck=args.duck
            )
            player = threading.Thread(
                target=lambda data=audio: (sd.play(data, PLAYBACK_RATE), sd.wait()),
                daemon=True,
            )
            player.start()
            samples, speech, wall = await record_utterance(
                microphone, 8.0, args.silence, args.threshold
            )
            player.join()
            if speech is None:
                print(f"[self-test {trial}] no speech captured (rms {rms(samples):.3f})")
                continue
            text = asr.transcribe(speech)
            matched = strip_wake_word(text, args.phrase)
            print(
                f"[self-test {trial}] alarm {args.alarm_volume:.2f}"
                f"{' (ducked)' if args.duck else ''} -> {text!r} "
                f"| match={matched!r}",
                flush=True,
            )
        if alarm is not None:
            alarm.stop()
        return 0

    print(
        f"ready: phrase={args.phrase!r}, {args.seconds:.0f}s. "
        "Speak now (Ctrl+C to stop).",
        flush=True,
    )
    deadline = time.monotonic() + args.seconds
    attempts = matches = 0

    while time.monotonic() < deadline:
        try:
            endpoint = LiveEndpoint(
                max_seconds=8.0,
                silence=args.silence,
                min_speech=0.15,
                threshold=args.threshold,
            )
        except Exception as exc:
            print(f"error: VAD unavailable: {exc}")
            return 1

        stop = threading.Event()
        started = time.monotonic()
        samples = await microphone.record(8.0, stop, endpoint, 2.0)
        wall = time.monotonic() - started
        level = rms(samples)

        speech = endpoint.speech()
        if speech is None:
            if level > 0.01:
                print(
                    f"[probe] no VAD speech (rms {level:.3f}, "
                    f"{len(samples) / SAMPLE_RATE:.2f}s) — too quiet or not speech",
                    flush=True,
                )
            continue

        text = asr.transcribe(speech)
        matched = strip_wake_word(text, args.phrase)
        attempts += 1
        if matched is not None:
            matches += 1
        print(
            f"[probe] speech {len(speech) / SAMPLE_RATE:.2f}s rms {level:.3f} "
            f"wall {wall:.2f}s -> {text!r} | match={matched!r}",
            flush=True,
        )

    if alarm is not None:
        alarm.stop()
    print(f"done: {attempts} utterances, {matches} wake matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
