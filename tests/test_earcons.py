# Author: David Li, SISHK

import io
import wave

import numpy as np

from adapters.speech.earcons import (
    PATTERN_SECONDS,
    SAMPLE_RATE,
    Earcons,
    to_wav,
)
from domain import AlertStyle, Band, Vertical


def samples(wav: bytes) -> np.ndarray:
    handle = wave.open(io.BytesIO(wav))
    return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def duration(wav: bytes) -> float:
    handle = wave.open(io.BytesIO(wav))
    return handle.getnframes() / handle.getframerate()


def rms(wav: bytes) -> float:
    data = samples(wav).astype(np.float64)
    return float(np.sqrt(np.mean(data**2)))


def test_every_sounding_style_renders_a_full_loop():
    earcons = Earcons()
    for style in (AlertStyle.SOFT, AlertStyle.TICKS):
        for band in Band:
            for vertical in Vertical:
                wav = earcons.pattern(band, vertical, style)
                assert wav is not None
                assert wav.startswith(b"RIFF")
                assert abs(duration(wav) - PATTERN_SECONDS) < 0.01


def test_silent_styles_render_nothing():
    earcons = Earcons()
    for style in (AlertStyle.VOICE, AlertStyle.HAPTIC, AlertStyle.OFF):
        assert earcons.pattern(Band.CLOSE, Vertical.FLOOR, style) is None


def test_soft_chime_is_gentle_not_a_siren():
    earcons = Earcons()
    soft = earcons.pattern(Band.CLOSE, Vertical.FLOOR, AlertStyle.SOFT)
    assert soft is not None
    data = samples(soft).astype(np.float64) / 32767.0
    # moderate peak: loud enough to notice, quiet enough not to startle
    assert np.max(np.abs(data)) < 0.55
    # a quarter of the loop is near-silence, so ambient sound gets through
    assert float(np.mean(np.abs(data) < 0.01)) > 0.25
    # no sustained tone: energy must be decaying chimes, not a 2 s drone
    assert float(np.mean(np.abs(data) > 0.2)) < 0.5


def test_close_has_more_sound_than_far_without_being_continuous():
    earcons = Earcons()
    far = earcons.pattern(Band.FAR, Vertical.FLOOR, AlertStyle.SOFT)
    close = earcons.pattern(Band.CLOSE, Vertical.FLOOR, AlertStyle.SOFT)
    assert far is not None and close is not None
    assert rms(close) > rms(far)

    # "close" must still contain silence: a siren-like continuous tone was the
    # problem with the first design.
    data = np.abs(samples(close))
    quiet_share = float(np.mean(data < 200))
    assert quiet_share > 0.3


def test_pitch_encodes_height():
    earcons = Earcons()
    floor = samples(earcons.pattern(Band.FAR, Vertical.FLOOR, AlertStyle.SOFT))
    head = samples(earcons.pattern(Band.FAR, Vertical.HEAD, AlertStyle.SOFT))

    def peak_frequency(data: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(data.astype(np.float64)))
        index = int(np.argmax(spectrum))
        return index * SAMPLE_RATE / len(data)

    assert peak_frequency(head) > peak_frequency(floor)


def test_one_shot_cues_are_distinct_and_cached():
    earcons = Earcons()
    assert earcons.found() != earcons.miss()
    assert earcons.found() is earcons.found()
    assert earcons.blip().startswith(b"RIFF")
    assert earcons.ping().startswith(b"RIFF")


def test_default_style_is_soft():
    earcons = Earcons()
    assert earcons.hazard(Band.NEAR, Vertical.FLOOR) == earcons.pattern(
        Band.NEAR, Vertical.FLOOR, AlertStyle.SOFT
    )


def test_to_wav_clips_out_of_range_samples():
    wav = to_wav(np.array([2.0, -2.0, 0.0], dtype=np.float32))
    assert wav.startswith(b"RIFF")
    assert duration(wav) > 0
