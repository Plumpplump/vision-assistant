"""Speech adapters: ASR, VAD, microphone, TTS and earcons."""

from .sherpa_compat import preload as _preload_sherpa

_preload_sherpa()

from .asr import Asr, AsrError, ParakeetAsr, UnavailableAsr, ZipformerAsr, create_asr
from .earcons import Earcons
from .microphone import (
    Microphone,
    MicrophoneError,
    FletAudioRecorderMicrophone,
    NullMicrophone,
    create_microphone,
)
from .paradee import DEFAULT_VOICE, ParadeeTts, TtsError
from .vad import LiveEndpoint, PassthroughSegmenter, SpeechSegmenter

__all__ = [
    "Asr",
    "AsrError",
    "DEFAULT_VOICE",
    "Earcons",
    "LiveEndpoint",
    "Microphone",
    "MicrophoneError",
    "NullMicrophone",
    "ParadeeTts",
    "ParakeetAsr",
    "PassthroughSegmenter",
    "FletAudioRecorderMicrophone",
    "SpeechSegmenter",
    "TtsError",
    "UnavailableAsr",
    "ZipformerAsr",
    "create_asr",
    "create_microphone",
]
