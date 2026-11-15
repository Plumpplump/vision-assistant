"""Speech recognition adapters (sherpa-onnx).

Parakeet TDT 110M is the default: on the 73-clip LibriSpeech dummy benchmark it
scored 4.6 % WER at RTF 0.055, vs 9.2 % / 0.129 for the streaming zipformer
(`docs/asr-benchmark.md`). It is offline, so it decodes one VAD-delimited
sentence at a time; the zipformer remains available for live partials.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

try:
    import sherpa_onnx
except Exception as exc:  # Android libc++ mismatch, missing wheel, ...
    sherpa_onnx = None
    SHERPA_IMPORT_ERROR = str(exc)
else:
    SHERPA_IMPORT_ERROR = ""

from adapters.paths import model_path

SAMPLE_RATE = 16000


class AsrError(Exception):
    """Raised when recognition is unavailable or fails."""


class Asr(Protocol):
    name: str

    @property
    def error(self) -> str | None: ...

    def load(self) -> None: ...

    def transcribe(self, samples: np.ndarray) -> str: ...


def _unavailable_message() -> str:
    return (
        "Speech recognition is unavailable: sherpa-onnx failed to load "
        f"({SHERPA_IMPORT_ERROR or 'native library missing'})."
    )


class UnavailableAsr:
    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def error(self) -> str | None:
        return self._reason

    def load(self) -> None:
        raise AsrError(self._reason)

    def transcribe(self, samples: np.ndarray) -> str:
        raise AsrError(self._reason)


class ParakeetAsr:
    """Offline NeMo TDT transducer; one utterance per call."""

    name = "parakeet-110m"

    def __init__(self, directory=None, num_threads: int = 2) -> None:
        root = directory or model_path("asr-parakeet-110m")
        self._files = {
            "encoder": root / "encoder.int8.onnx",
            "decoder": root / "decoder.int8.onnx",
            "joiner": root / "joiner.int8.onnx",
            "tokens": root / "tokens.txt",
        }
        self._threads = num_threads
        self._recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self._lock = threading.Lock()

    @property
    def error(self) -> str | None:
        missing = [str(path) for path in self._files.values() if not path.is_file()]
        if missing:
            return "Parakeet ASR assets are missing: " + ", ".join(missing)
        return None

    def load(self) -> None:
        if sherpa_onnx is None:
            raise AsrError(_unavailable_message())
        with self._lock:
            if self._recognizer is not None:
                return
            if self.error:
                raise AsrError(self.error)
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(self._files["encoder"]),
                decoder=str(self._files["decoder"]),
                joiner=str(self._files["joiner"]),
                tokens=str(self._files["tokens"]),
                model_type="nemo_transducer",
                num_threads=self._threads,
                sample_rate=SAMPLE_RATE,
            )

    def transcribe(self, samples: np.ndarray) -> str:
        self.load()
        assert self._recognizer is not None
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        with self._lock:
            self._recognizer.decode_stream(stream)
        return str(stream.result.text).strip()


class ZipformerAsr:
    """Streaming zipformer used utterance-at-a-time (optional partials model)."""

    name = "zipformer-en"
    CHUNK = 1600  # 100 ms

    def __init__(self, directory=None, num_threads: int = 2) -> None:
        root = directory or model_path("asr-zipformer-en")
        self._files = {
            "encoder": root / "encoder.int8.onnx",
            "decoder": root / "decoder.int8.onnx",
            "joiner": root / "joiner.int8.onnx",
            "tokens": root / "tokens.txt",
        }
        self._threads = num_threads
        self._recognizer: sherpa_onnx.OnlineRecognizer | None = None
        self._lock = threading.Lock()

    @property
    def error(self) -> str | None:
        missing = [str(path) for path in self._files.values() if not path.is_file()]
        if missing:
            return "Zipformer ASR assets are missing: " + ", ".join(missing)
        return None

    def load(self) -> None:
        if sherpa_onnx is None:
            raise AsrError(_unavailable_message())
        with self._lock:
            if self._recognizer is not None:
                return
            if self.error:
                raise AsrError(self.error)
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(self._files["tokens"]),
                encoder=str(self._files["encoder"]),
                decoder=str(self._files["decoder"]),
                joiner=str(self._files["joiner"]),
                num_threads=self._threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
            )

    def transcribe(self, samples: np.ndarray) -> str:
        self.load()
        assert self._recognizer is not None
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        with self._lock:
            stream = self._recognizer.create_stream()
            for start in range(0, len(audio), self.CHUNK):
                stream.accept_waveform(SAMPLE_RATE, audio[start : start + self.CHUNK])
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)
            return str(self._recognizer.get_result(stream)).strip()


def create_asr(name: str | None = None) -> Asr:
    """Backend by name; falls back to a spoken explanation, never a crash."""
    import os

    if sherpa_onnx is None:
        return UnavailableAsr(_unavailable_message())
    requested = (name or os.getenv("VISIONASSIST_ASR") or "parakeet").strip().lower()
    if requested in {"parakeet", "parakeet-110m", "nemo"}:
        asr: Asr = ParakeetAsr()
    elif requested in {"zipformer", "zipformer-en", "streaming"}:
        asr = ZipformerAsr()
    else:
        return UnavailableAsr(
            f"Unknown ASR backend {requested!r}. Use 'parakeet' or 'zipformer'."
        )
    return asr
