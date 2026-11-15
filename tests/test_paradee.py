import io
import wave

import pytest

from adapters.speech import ParadeeTts

tts = ParadeeTts()
pytestmark = pytest.mark.skipif(
    not tts.available(), reason=f"Paradee assets missing: {tts.error}"
)


def duration(wav: bytes) -> float:
    handle = wave.open(io.BytesIO(wav))
    return handle.getnframes() / handle.getframerate()


def test_english_voices_are_exposed():
    voices = tts.voices
    assert "af_heart" in voices
    assert all(voice.startswith(("a", "b")) for voice in voices)


def test_synthesize_returns_24khz_wav():
    wav = tts.synthesize("Chair, close, on your left.")
    handle = wave.open(io.BytesIO(wav))
    assert handle.getnchannels() == 1
    assert handle.getframerate() == 24000
    assert 0.5 < duration(wav) < 6.0


def test_long_text_is_split_into_chunks():
    text = (
        "Chair, close, on your left. Table, a few steps away, on your right. "
        "Also a person ahead."
    )
    wav = tts.synthesize(text)
    assert duration(wav) > 1.0


def test_unknown_voice_falls_back():
    wav = tts.synthesize("Hello.", voice="not_a_voice")
    assert wav.startswith(b"RIFF")
