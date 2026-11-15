"""Voice activity detection: trim recordings and find the end of an utterance.

Silero VAD (MIT, 0.6 MB) runs through ``sherpa-onnx``'s ``VoiceActivityDetector``.
Two rules that took testing to learn:

* the detector consumes fixed-size windows (512 samples at 16 kHz), not
  arbitrary buffers — feeding a whole recording at once returns almost nothing;
* trimming must keep the **contiguous span** from the first to the last speech
  window. Concatenating detected windows removes the pauses between words and
  fed the ASR clipped fragments.
"""

from __future__ import annotations

import threading

import numpy as np
import sherpa_onnx

from adapters.paths import model_path

SAMPLE_RATE = 16000
PAD_SAMPLES = 1600  # 100 ms of context on each side


class SegmenterError(Exception):
    pass


def _config(model_file, threshold: float, min_silence: float, min_speech: float, max_speech: float):
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model_file)
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = min_silence
    config.silero_vad.min_speech_duration = min_speech
    config.silero_vad.max_speech_duration = max_speech
    config.sample_rate = SAMPLE_RATE
    return config


def _segments(detector: sherpa_onnx.VoiceActivityDetector) -> list[tuple[int, int]]:
    """(start, end) sample offsets of the detected speech windows."""
    spans: list[tuple[int, int]] = []
    detector.flush()
    while not detector.empty():
        segment = detector.front
        start = int(segment.start)
        spans.append((start, start + len(segment.samples)))
        detector.pop()
    return spans


def _span_audio(audio: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray | None:
    if not spans:
        return None
    start = max(0, spans[0][0] - PAD_SAMPLES)
    end = min(audio.size, spans[-1][1] + PAD_SAMPLES)
    if end <= start:
        return None
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


class SpeechSegmenter:
    name = "silero-vad"

    def __init__(
        self,
        model_file=None,
        threshold: float = 0.5,
        min_silence: float = 0.3,
        min_speech: float = 0.15,
        max_speech: float = 15.0,
    ) -> None:
        self._model = model_file or model_path("silero-vad", "silero_vad.onnx")
        self._threshold = threshold
        self._min_silence = min_silence
        self._min_speech = min_speech
        self._max_speech = max_speech
        self._lock = threading.Lock()

    @property
    def error(self) -> str | None:
        if not self._model.is_file():
            return f"Silero VAD model missing: {self._model}"
        return None

    def trim(self, samples: np.ndarray) -> np.ndarray | None:
        """Return the spoken span, or ``None`` when nothing was said."""
        if self.error:
            raise SegmenterError(self.error or "Silero VAD unavailable")
        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < int(SAMPLE_RATE * 0.2):
            return None

        config = _config(
            self._model,
            self._threshold,
            self._min_silence,
            self._min_speech,
            self._max_speech,
        )
        window = int(config.silero_vad.window_size)
        detector = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=int(audio.size / SAMPLE_RATE) + 10
        )
        with self._lock:
            for start in range(0, audio.size, window):
                chunk = audio[start : start + window]
                if chunk.size < window:
                    chunk = np.pad(chunk, (0, window - chunk.size))
                detector.accept_waveform(chunk)
            spans = _segments(detector)

        speech = _span_audio(audio, spans)
        if speech is None or speech.size < int(SAMPLE_RATE * self._min_speech):
            return None
        return speech


class LiveEndpoint:
    """Streaming end-of-utterance detector used *while* recording.

    The Talk button becomes hands-free: the user speaks and stops, and this
    class reports ``finished`` after ``silence`` seconds without speech. It also
    keeps the raw audio so the same pass can return the spoken span.
    """

    def __init__(
        self,
        model_file=None,
        threshold: float = 0.5,
        silence: float = 0.8,
        min_speech: float = 0.3,
        max_seconds: float = 30.0,
    ) -> None:
        self._model = model_file or model_path("silero-vad", "silero_vad.onnx")
        if not self._model.is_file():
            raise SegmenterError(f"Silero VAD model missing: {self._model}")
        config = _config(self._model, threshold, 0.2, 0.1, max_seconds)
        self.window = int(config.silero_vad.window_size)
        self._detector = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=int(max_seconds) + 10
        )
        self._silence_limit = int(silence * SAMPLE_RATE)
        self._min_speech = int(min_speech * SAMPLE_RATE)
        self._pending = np.zeros(0, dtype=np.float32)
        self._raw: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self.finished = False
        self._speech: np.ndarray | None = None

    @property
    def started(self) -> bool:
        """Whether any speech has been heard yet."""
        return self._speech_samples > 0

    def accept(self, block: np.ndarray) -> bool:
        """Feed a recording block; returns True when the utterance ended."""
        if self.finished:
            return True
        audio = np.asarray(block, dtype=np.float32).reshape(-1)
        self._raw.append(audio)
        self._pending = np.concatenate([self._pending, audio])
        while self._pending.size >= self.window:
            chunk = self._pending[: self.window]
            self._pending = self._pending[self.window :]
            self._detector.accept_waveform(chunk)
            if self._detector.is_speech_detected():
                self._speech_samples += self.window
                self._silence_samples = 0
            elif self._speech_samples:
                self._silence_samples += self.window
        if (
            self._speech_samples >= self._min_speech
            and self._silence_samples >= self._silence_limit
        ):
            self.finished = True
        return self.finished

    def speech(self) -> np.ndarray | None:
        """The spoken span of everything fed so far."""
        if self._speech is not None:
            return self._speech
        if not self._raw:
            return None
        raw = np.concatenate(self._raw)
        spans = _segments(self._detector)
        speech = _span_audio(raw, spans)
        if speech is None or speech.size < int(SAMPLE_RATE * 0.2):
            return None
        self._speech = speech
        return speech


class PassthroughSegmenter:
    """Test double: accepts whatever it is given."""

    name = "passthrough"
    error = None

    def trim(self, samples: np.ndarray) -> np.ndarray | None:
        return samples if len(samples) else None
