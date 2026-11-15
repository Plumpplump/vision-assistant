"""Pre-rendered earcons: quiet by design, no speech unless asked.

Design rules learned from talking to people who use audio interfaces:

* an alarm buzzer reads as danger and masks traffic, footsteps and voices —
  the very cues a blind user navigates by;
* a gentle, pitched "tap" carries the same information without the stress;
* continuous tones are out. Rhythm and loudness encode distance, and the
  closest band is still a *short* sound, not a siren;
* the default is deliberately quiet (about -12 dBFS peak), and volume is
  user-controlled.

Patterns are rendered once as gapless 2 s loops (``ReleaseMode.LOOP``) and
switched on band change; one-shots use ``ReleaseMode.STOP`` for instant
replay.
"""

from __future__ import annotations

import io
import wave
from functools import lru_cache

import numpy as np

from domain.types import AlertStyle, Band, Vertical

SAMPLE_RATE = 24_000
PATTERN_SECONDS = 2.0
AMPLITUDE = 0.30  # soft default; roughly -10 dBFS peak

# Pleasant, low-register notes (G4, C5, E5) instead of buzzy square tones.
PITCH = {
    Vertical.FLOOR: 392.0,
    Vertical.WAIST: 523.25,
    Vertical.HEAD: 659.25,
}
# Ticks stay lower and shorter, so they read as "tick" not "alarm".
TICK_PITCH = {
    Vertical.FLOOR: 240.0,
    Vertical.WAIST: 320.0,
    Vertical.HEAD: 420.0,
}


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000.0), dtype=np.float32)


def _chime(freq: float, ms: float = 260.0, amplitude: float = AMPLITUDE) -> np.ndarray:
    """A struck-bell shape: fast attack, exponential decay, soft octave."""
    count = max(1, int(SAMPLE_RATE * ms / 1000.0))
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    fundamental = np.sin(2 * np.pi * freq * t, dtype=np.float32)
    octave = 0.18 * np.sin(2 * np.pi * freq * 2 * t, dtype=np.float32)
    attack = max(1, int(SAMPLE_RATE * 0.02))
    decay = np.exp(-4.0 * t, dtype=np.float32)
    envelope = decay.copy()
    envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    return (fundamental + octave) * envelope * amplitude


def _tick(freq: float, ms: float = 45.0) -> np.ndarray:
    count = max(1, int(SAMPLE_RATE * ms / 1000.0))
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    body = np.sin(2 * np.pi * freq * t, dtype=np.float32)
    envelope = np.exp(-9.0 * t, dtype=np.float32)
    return body * envelope * AMPLITUDE * 0.8


def _sweep(start: float, end: float, ms: float) -> np.ndarray:
    count = max(1, int(SAMPLE_RATE * ms / 1000.0))
    frequencies = np.linspace(start, end, count, dtype=np.float32)
    phase = np.cumsum(2 * np.pi * frequencies / SAMPLE_RATE, dtype=np.float32)
    return (np.sin(phase, dtype=np.float32) * AMPLITUDE).astype(np.float32)


def to_wav(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _place(buffer: np.ndarray, sound: np.ndarray, at_ms: float) -> None:
    start = int(SAMPLE_RATE * at_ms / 1000.0)
    end = min(start + sound.size, buffer.size)
    if start < buffer.size:
        buffer[start:end] += sound[: end - start]


def _soft_pattern(band: Band, vertical: Vertical) -> bytes:
    """Slow chime far away, quickening pairs when close. Never continuous."""
    root = PITCH[vertical]
    buffer = np.zeros(int(SAMPLE_RATE * PATTERN_SECONDS), dtype=np.float32)
    if band is Band.FAR:
        _place(buffer, _chime(root), 0)
    elif band is Band.NEAR:
        _place(buffer, _chime(root), 0)
        _place(buffer, _chime(root), 800)
    else:  # CLOSE: an ascending pair, then a gentle pulse every 500 ms
        _place(buffer, _chime(root), 0)
        _place(buffer, _chime(root * 1.25, ms=220), 120)
        for at in (700, 1200, 1700):
            _place(buffer, _chime(root * 1.25, ms=180), at)
    return to_wav(buffer)


def _tick_pattern(band: Band, vertical: Vertical) -> bytes:
    root = TICK_PITCH[vertical]
    buffer = np.zeros(int(SAMPLE_RATE * PATTERN_SECONDS), dtype=np.float32)
    step_ms = {Band.FAR: 1000.0, Band.NEAR: 320.0, Band.CLOSE: 250.0}[band]
    for at in np.arange(0.0, PATTERN_SECONDS * 1000.0, step_ms):
        _place(buffer, _tick(root), float(at))
    return to_wav(buffer)


class Earcons:
    """Cached WAV bytes for every alert sound and style."""

    _cache: dict[tuple[AlertStyle, Band, Vertical], bytes] = {}

    def pattern(
        self, band: Band, vertical: Vertical, style: AlertStyle = AlertStyle.SOFT
    ) -> bytes | None:
        """Two-second loop for ``band``; ``None`` for styles with no sound."""
        if style in (AlertStyle.VOICE, AlertStyle.HAPTIC, AlertStyle.OFF):
            return None
        key = (style, band, vertical)
        cached = self._cache.get(key)
        if cached is None:
            cached = (
                _tick_pattern(band, vertical)
                if style is AlertStyle.TICKS
                else _soft_pattern(band, vertical)
            )
            self._cache[key] = cached
        return cached

    # Kept for compatibility with the first version of the alert layer.
    def hazard(self, band: Band, vertical: Vertical) -> bytes:
        return self.pattern(band, vertical, AlertStyle.SOFT) or b""

    @staticmethod
    @lru_cache(maxsize=1)
    def blip() -> bytes:
        """A new object entered the corridor."""
        return to_wav(
            np.concatenate([_chime(880, ms=120), _silence(40), _chime(880, ms=120)])
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def found() -> bytes:
        return to_wav(np.concatenate([_sweep(523, 784, 220), _silence(40)]))

    @staticmethod
    @lru_cache(maxsize=1)
    def miss() -> bytes:
        return to_wav(np.concatenate([_sweep(784, 523, 220), _silence(40)]))

    @staticmethod
    @lru_cache(maxsize=1)
    def wake() -> bytes:
        """Heard the wake phrase: short rising pair, then listen."""
        return to_wav(
            np.concatenate([_chime(659.25, ms=110), _silence(40), _chime(987.77, ms=140)])
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def ping() -> bytes:
        """Neutral acknowledgement (e.g. listening)."""
        return to_wav(_chime(659.25, ms=140))
