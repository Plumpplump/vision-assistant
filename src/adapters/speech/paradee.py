# -*- coding: utf-8 -*-
"""On-device English TTS with Paradee-8M v1.3 (Kokoro-distilled) via onnxruntime.

Ported from the sibling ``SAT-Words`` project, where the same code path is
verified on desktop and Android. The model is a single 8.5 MB int8 ONNX graph
that turns phoneme ids into 24 kHz audio: no PyTorch, no transformers, no
eSpeak, no network.

Assets (bundled under ``src/assets/models``)::

    paradee/paradee_int8.onnx    phoneme ids -> waveform
    paradee/config.json          phoneme vocabulary + voice list
    g2p/{us,gb}_{gold,silver}.json   misaki English lexicons

Only the four English voices are usable, because only English lexicons are
bundled; the Hindi/French/Spanish voices need eSpeak NG.
"""

from __future__ import annotations

import io
import json
import re
import threading
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from adapters.paths import models_dir

from .g2p import Lexicon, load_lexicon, phonemize
from .vocab import SAMPLE_RATE, tokenize

MODEL_VERSION = "paradee-1.3-int8"
MAX_PHONEMES = 510

ENGLISH_VOICES: dict[str, str] = {
    "af_heart": "Heart - American, female",
    "am_puck": "Puck - American, male",
    "bf_isabella": "Isabella - British, female",
    "bm_fable": "Fable - British, male",
}
DEFAULT_VOICE = "af_heart"
VOICE_LANGUAGES = {
    "af_heart": "en-us",
    "am_puck": "en-us",
    "bf_isabella": "en-gb",
    "bm_fable": "en-gb",
}
LEXICON_PREFIX = {"en-us": "us", "en-gb": "gb"}
MIN_SPEED = 0.5
MAX_SPEED = 2.0

SENTENCE_RE = re.compile(r"[^.!?;]+[.!?;]?")


class TtsError(Exception):
    """Raised when on-device synthesis is unavailable or fails."""


def _wav_bytes(samples: np.ndarray) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1e-6:
        samples = samples * (0.9 / peak)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class ParadeeTts:
    """Lazy, thread-safe wrapper around the Paradee ONNX graph.

    ``synthesize`` blocks; call it from a worker thread (the app uses
    ``asyncio.to_thread``). Long text is split on sentence boundaries because
    the graph accepts at most 512 phoneme ids.
    """

    name = MODEL_VERSION

    def __init__(self, assets: Path | str | None = None, threads: int = 1) -> None:
        self._root = Path(assets) if assets is not None else models_dir()
        self._threads = threads
        self._lock = threading.RLock()
        self._session_lock = threading.Lock()
        self._loaded = False
        self._session: ort.InferenceSession | None = None
        self._lexicons: dict[str, Lexicon] = {}
        self._input_names: set[str] = set()
        self._voice_index: dict[str, int] = {}
        self._voice_languages: dict[str, str] = {}

    # ------------------------------------------------------------------ paths

    @property
    def model_path(self) -> Path:
        return self._root / "paradee" / "paradee_int8.onnx"

    @property
    def config_path(self) -> Path:
        return self._root / "paradee" / "config.json"

    @property
    def g2p_dir(self) -> Path:
        return self._root / "g2p"

    def _has_lexicon(self, language: str) -> bool:
        prefix = LEXICON_PREFIX.get(language.lower())
        if prefix is None:
            return False
        return all(
            (self.g2p_dir / f"{prefix}_{kind}.json").is_file()
            for kind in ("gold", "silver")
        )

    def available(self) -> bool:
        return self.model_path.is_file() and self.config_path.is_file() and (
            self._has_lexicon("en-us") or self._has_lexicon("en-gb")
        )

    @property
    def error(self) -> str | None:
        if self.available():
            return None
        return (
            f"Paradee TTS assets are missing under {self._root}. Expected "
            "paradee/paradee_int8.onnx + paradee/config.json + g2p/."
        )

    @property
    def voices(self) -> dict[str, str]:
        self.warm_up()
        return {
            voice: label
            for voice, label in ENGLISH_VOICES.items()
            if voice in self._voice_languages
        }

    # ------------------------------------------------------------- lifecycle

    def warm_up(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not self.available():
                raise TtsError(self.error or "Paradee assets are missing.")

            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self._voice_index = {
                name: index for index, name in enumerate(config.get("voices", []))
            }
            languages = config.get("languages", {})
            for voice in ENGLISH_VOICES:
                if voice not in self._voice_index:
                    continue
                language = str(
                    languages.get(voice) or VOICE_LANGUAGES.get(voice, "")
                ).lower()
                if not language.startswith("en"):
                    continue
                if not self._has_lexicon(language):
                    continue
                self._voice_languages[voice] = language
            if not self._voice_languages:
                raise TtsError(
                    "No English Paradee voices are usable; the g2p lexicons are "
                    "missing from src/assets/models/g2p."
                )

            options = ort.SessionOptions()
            # The graph is tiny: one thread keeps the other cores for the UI,
            # the camera and audio playback.
            options.intra_op_num_threads = self._threads
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), options, providers=["CPUExecutionProvider"]
            )
            self._input_names = {item.name for item in self._session.get_inputs()}
            self._loaded = True

    def _lexicon(self, language: str) -> Lexicon:
        with self._lock:
            lexicon = self._lexicons.get(language)
            if lexicon is None:
                lexicon = load_lexicon(self.g2p_dir, language)
                self._lexicons[language] = lexicon
            return lexicon

    def _resolve_voice(self, voice: str) -> tuple[str, int, str]:
        if voice not in self._voice_languages:
            voice = DEFAULT_VOICE if DEFAULT_VOICE in self._voice_languages else next(
                iter(self._voice_languages)
            )
        index = self._voice_index.get(voice)
        if index is None:
            raise TtsError(f"Voice {voice!r} is not present in the model config.")
        return voice, index, self._voice_languages[voice]

    # -------------------------------------------------------------- synthesis

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
    ) -> bytes:
        """Text -> 24 kHz mono 16-bit WAV bytes."""
        text = (text or "").strip()
        if not text:
            raise TtsError("Nothing to say.")
        self.warm_up()
        voice, voice_index, language = self._resolve_voice(voice)
        speed = max(MIN_SPEED, min(MAX_SPEED, float(speed)))
        lexicon = self._lexicon(language)

        pieces: list[np.ndarray] = []
        for chunk in self._chunks(text):
            for piece in self._split_to_fit(chunk, lexicon):
                pieces.append(
                    self._synthesize_ids(
                        self._encode(piece, lexicon), voice_index, speed
                    )
                )
        if not pieces:
            raise TtsError("Nothing pronounceable in the text.")
        return _wav_bytes(np.concatenate(pieces))

    def _chunks(self, text: str) -> list[str]:
        sentences = [match.strip() for match in SENTENCE_RE.findall(text) if match.strip()]
        return sentences or [text]

    def _encode(self, text: str, lexicon: Lexicon) -> np.ndarray | None:
        try:
            return tokenize(phonemize(text, lexicon))
        except ValueError:
            return None

    def _split_to_fit(self, text: str, lexicon: Lexicon) -> list[str]:
        if self._encode(text, lexicon) is not None:
            return [text]
        words = text.split()
        if len(words) <= 1:
            raise TtsError(f"Text is too long to pronounce in one piece: {text!r}")
        middle = len(words) // 2
        left = " ".join(words[:middle])
        right = " ".join(words[middle:])
        return self._split_to_fit(left, lexicon) + self._split_to_fit(right, lexicon)

    def _synthesize_ids(
        self, ids: np.ndarray, voice_index: int, speed: float
    ) -> np.ndarray:
        feeds: dict[str, np.ndarray] = {"input_ids": ids}
        if "speed" in self._input_names:
            feeds["speed"] = np.asarray([speed], dtype=np.float32)
        if "voice" in self._input_names:
            feeds["voice"] = np.asarray([voice_index], dtype=np.int64)
        with self._session_lock:
            assert self._session is not None
            waveform = self._session.run(None, feeds)[0]
        return np.asarray(waveform, dtype=np.float32).reshape(-1)
