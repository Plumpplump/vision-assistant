# Author: David Li, SISHK

"""Microphone source selection and WAV file helpers."""

import struct
import sys
import wave

import numpy as np

from adapters.speech.microphone import FletAudioRecorderMicrophone


def test_macos_defaults_to_file_mode(monkeypatch):
    monkeypatch.delenv("VISIONASSIST_MIC_MODE", raising=False)
    mic = FletAudioRecorderMicrophone(page=None)
    assert mic._file_mode is (sys.platform == "darwin")


def test_mic_mode_override(monkeypatch):
    monkeypatch.setenv("VISIONASSIST_MIC_MODE", "stream")
    assert FletAudioRecorderMicrophone(page=None)._file_mode is False
    monkeypatch.setenv("VISIONASSIST_MIC_MODE", "file")
    assert FletAudioRecorderMicrophone(page=None)._file_mode is True


def test_wav_data_offset_skips_unknown_chunks(tmp_path):
    pcm = np.arange(100, dtype=np.int16).tobytes()
    junk = b"\0" * 4093  # odd-sized chunk exercises the pad byte
    body = (
        b"WAVE"
        + b"JUNK"
        + struct.pack("<I", len(junk))
        + junk
        + b"\0"
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    path = tmp_path / "chunked.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    assert FletAudioRecorderMicrophone._wav_data_offset(path) == (
        12 + 8 + len(junk) + 1 + 8
    )


def test_wav_data_offset_handles_standard_wav(tmp_path):
    path = tmp_path / "standard.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0" * 320)
    assert FletAudioRecorderMicrophone._wav_data_offset(path) == 44


def test_pcm_conversion_keeps_odd_tail_out():
    mic = FletAudioRecorderMicrophone(page=None)
    samples = mic._to_float32(b"\0\0" + struct.pack("<h", 16384) + b"x")
    assert samples.dtype == np.float32
    assert len(samples) == 2
    assert samples[1] == 0.5
