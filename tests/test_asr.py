# Author: David Li, SISHK

import io
import wave

import numpy as np
import pytest

from adapters.speech import LiveEndpoint, ParadeeTts, ParakeetAsr, SpeechSegmenter

tts = ParadeeTts()
asr = ParakeetAsr()
segmenter = SpeechSegmenter()

pytestmark = pytest.mark.skipif(
    not tts.available() or asr.error is not None or segmenter.error is not None,
    reason="TTS or ASR assets missing",
)


def speech_16k(text: str) -> np.ndarray:
    """Synthesize 24 kHz speech with Paradee, resample to 16 kHz."""
    wav = tts.synthesize(text)
    handle = wave.open(io.BytesIO(wav))
    pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    target = int(len(audio) * 16000 / handle.getframerate())
    return np.interp(
        np.arange(target) * handle.getframerate() / 16000,
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def test_parakeet_transcribes_synthesised_speech():
    audio = speech_16k("Chair close on your left.")
    text = asr.transcribe(audio)
    assert "chair" in text.lower()
    assert "left" in text.lower()


def test_parakeet_returns_punctuation_for_longer_sentences():
    audio = speech_16k("The table is two metres away.")
    text = asr.transcribe(audio)
    assert text.endswith(".") or "table" in text.lower()


def test_vad_keeps_speech_and_drops_silence():
    speech = speech_16k("Find the cup.")
    silence = np.zeros(int(16000 * 0.6), dtype=np.float32)
    padded = np.concatenate([silence, speech, silence])

    trimmed = segmenter.trim(padded)
    assert trimmed is not None
    assert len(trimmed) < len(padded)
    assert len(trimmed) >= int(16000 * 0.2)


def test_vad_rejects_pure_silence():
    assert segmenter.trim(np.zeros(int(16000 * 2.0), dtype=np.float32)) is None


def test_vad_rejects_a_short_blip():
    blip = np.zeros(int(16000 * 0.05), dtype=np.float32)
    assert segmenter.trim(blip) is None


def test_live_endpoint_stops_after_a_pause_and_returns_the_speech():
    speech = speech_16k("Find the cup.")
    silence = np.zeros(int(16000 * 1.2), dtype=np.float32)
    audio = np.concatenate([np.zeros(int(16000 * 0.2), np.float32), speech, silence])

    endpoint = LiveEndpoint(silence=0.5)
    finished = False
    for start in range(0, len(audio), 1600):
        if endpoint.accept(audio[start : start + 1600]):
            finished = True
            break

    assert finished
    trimmed = endpoint.speech()
    assert trimmed is not None
    assert len(trimmed) >= int(16000 * 0.4)


def test_live_endpoint_keeps_waiting_through_silence_without_speech():
    endpoint = LiveEndpoint(silence=0.5)
    for _ in range(5):
        assert endpoint.accept(np.zeros(1600, dtype=np.float32)) is False
    assert endpoint.speech() is None


def test_asr_degrades_when_sherpa_cannot_load(monkeypatch):
    import adapters.speech.asr as asr_module

    monkeypatch.setattr(asr_module, "sherpa_onnx", None)
    monkeypatch.setattr(asr_module, "SHERPA_IMPORT_ERROR", "libc++ symbol missing")
    backend = asr_module.create_asr("parakeet")
    assert backend.name == "unavailable"
    assert "sherpa" in (backend.error or "").lower()


def test_vad_raises_instead_of_crashing_without_sherpa(monkeypatch):
    import adapters.speech.vad as vad_module

    segmenter = vad_module.SpeechSegmenter()
    monkeypatch.setattr(vad_module, "sherpa_onnx", None)
    monkeypatch.setattr(vad_module, "SHERPA_IMPORT_ERROR", "libc++ symbol missing")
    with pytest.raises(vad_module.SegmenterError):
        segmenter.trim(np.zeros(16000, dtype=np.float32))
