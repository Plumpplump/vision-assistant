# Author: David Li, SISHK

"""Microphone input through Flet's AudioRecorder extension.

``flet-audio-recorder`` supports Windows, macOS, Linux, iOS, Android and web,
so there is one capture path for the desktop demo and the APK. It is
configured for **16 kHz mono PCM16** and delivers raw chunks through
``on_stream``; those chunks are fed to Silero VAD while recording, which keeps
the same end-of-utterance behaviour the app had with PortAudio.

The service needs a ``Page``, so it is constructed by the UI layer and injected
into :class:`app.assistant.Assistant`. Without a page (tests, headless tools) a
``NullMicrophone`` is used.

Android permission: ``android.permission.RECORD_AUDIO`` is declared in
``pyproject.toml``; ``has_permission()`` prompts on first use.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
POLL_SECONDS = 0.05


class MicrophoneError(Exception):
    pass


class Microphone(Protocol):
    name: str

    def available(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    async def record(
        self,
        max_seconds: float,
        stop_event: threading.Event,
        endpoint=None,
        idle_timeout: float | None = None,
    ) -> np.ndarray: ...


class FletAudioRecorderMicrophone:
    name = "microphone (flet-audio-recorder)"

    def __init__(self, page, sample_rate: int = SAMPLE_RATE) -> None:
        self._page = page
        self.sample_rate = sample_rate
        self._recorder = None
        self._chunks = bytearray()
        self._error: str | None = None
        self._import_error: str | None = None
        # On macOS the `record` plugin's startStream() returns valid-length
        # buffers that are all zeros, while its WAV file recording captures
        # audio normally. Default to file recording there and feed the growing
        # file to the VAD; VISIONASSIST_MIC_MODE=file|stream overrides it.
        mode = os.getenv("VISIONASSIST_MIC_MODE", "").strip().lower()
        self._file_mode = mode == "file" or (
            mode != "stream" and sys.platform == "darwin"
        )

    def _module(self):
        try:
            import flet_audio_recorder as far
        except Exception as exc:  # pragma: no cover - platform dependent
            self._import_error = str(exc)
            return None
        return far

    def available(self) -> bool:
        return self._page is not None and self._module() is not None

    @property
    def error(self) -> str | None:
        if not self.available():
            return self._import_error or "No Flet page for microphone capture."
        return self._error

    # ------------------------------------------------------------------ service

    def _ensure_recorder(self):
        if self._recorder is not None:
            return self._recorder
        far = self._module()
        if far is None or self._page is None:
            raise MicrophoneError(self._error or "Audio recorder unavailable.")
        configuration = far.AudioRecorderConfiguration(
            encoder=(
                far.AudioEncoder.WAV
                if self._file_mode
                else far.AudioEncoder.PCM16BITS
            ),
            channels=CHANNELS,
            sample_rate=self.sample_rate,
            suppress_noise=True,
            cancel_echo=True,
            auto_gain=True,
        )
        self._recorder = far.AudioRecorder(
            configuration=configuration,
            on_stream=(
                None
                if self._file_mode
                else lambda event: self._chunks.extend(event.chunk)
            ),
        )
        self._page.services.append(self._recorder)
        self._page.update()
        return self._recorder

    # ------------------------------------------------------------------ capture

    def _take_pending(self) -> np.ndarray:
        """New PCM16 bytes since the last call, as float32 mono."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        data = bytes(self._chunks)
        self._chunks.clear()
        if len(data) < 2:
            return np.zeros(0, dtype=np.float32)
        if len(data) % 2:
            data = data[:-1]
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    async def record(
        self,
        max_seconds: float,
        stop_event: threading.Event,
        endpoint=None,
        idle_timeout: float | None = None,
    ) -> np.ndarray:
        if self._file_mode:
            return await self._record_file(
                max_seconds, stop_event, endpoint, idle_timeout
            )
        return await self._record_stream(
            max_seconds, stop_event, endpoint, idle_timeout
        )

    async def _record_stream(
        self,
        max_seconds: float,
        stop_event: threading.Event,
        endpoint=None,
        idle_timeout: float | None = None,
    ) -> np.ndarray:
        recorder = self._ensure_recorder()
        self._chunks.clear()
        self._error = None

        try:
            granted = await recorder.has_permission()
        except Exception as exc:
            raise MicrophoneError(f"Microphone permission check failed: {exc}") from exc
        if not granted:
            raise MicrophoneError(
                "Microphone permission denied. Allow it in Android settings."
            )

        try:
            started = await recorder.start_recording()
        except Exception as exc:
            raise MicrophoneError(f"Could not start recording: {exc}") from exc
        if not started:
            raise MicrophoneError("Could not start recording.")

        captured: list[np.ndarray] = []
        started_at = time.monotonic()
        try:
            while True:
                await asyncio.sleep(POLL_SECONDS)
                elapsed = time.monotonic() - started_at
                block = self._take_pending()
                if block.size:
                    captured.append(block)
                    if endpoint is not None and endpoint.accept(block):
                        break
                if stop_event.is_set():
                    break
                if (
                    idle_timeout is not None
                    and endpoint is not None
                    and not endpoint.started
                    and elapsed >= idle_timeout
                ):
                    break
                if elapsed >= max_seconds:
                    break
        finally:
            try:
                await recorder.stop_recording()
            except Exception:  # pragma: no cover - already stopped
                pass
            # Let the last stream event arrive before reading the buffer.
            await asyncio.sleep(0.1)
            tail = self._take_pending()
            if tail.size:
                captured.append(tail)

        if captured:
            audio = np.concatenate(captured)
        else:
            audio = np.zeros(0, dtype=np.float32)
        if audio.size == 0:
            self._error = "The recorder produced no audio."
        return audio

    # ----------------------------------------------------------- file capture

    @staticmethod
    def _wav_data_offset(path: Path) -> int | None:
        """Offset of the PCM data chunk in a (possibly still growing) WAV."""
        try:
            with path.open("rb") as handle:
                if handle.read(4) != b"RIFF":
                    return None
                handle.seek(12)
                while True:
                    header = handle.read(8)
                    if len(header) < 8:
                        return None
                    chunk_id = header[:4]
                    chunk_size = struct.unpack("<I", header[4:])[0]
                    if chunk_id == b"data":
                        return handle.tell()
                    handle.seek(chunk_size + (chunk_size & 1), 1)
        except OSError:
            return None

    @staticmethod
    def _to_float32(data: bytes) -> np.ndarray:
        if len(data) < 2:
            return np.zeros(0, dtype=np.float32)
        usable = len(data) // 2 * 2
        return (
            np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32)
            / 32768.0
        )

    async def _record_file(
        self,
        max_seconds: float,
        stop_event: threading.Event,
        endpoint=None,
        idle_timeout: float | None = None,
    ) -> np.ndarray:
        """WAV-file recording with live polling.

        The macOS build of ``record`` returns all-zero PCM through
        ``startStream`` but writes a correct WAV file. We record to a temp WAV
        and feed its growing data chunk to the same ``LiveEndpoint``, so wake
        and push-to-talk keep their end-of-speech behaviour.
        """
        recorder = self._ensure_recorder()
        self._error = None
        try:
            granted = await recorder.has_permission()
        except Exception as exc:
            raise MicrophoneError(f"Microphone permission check failed: {exc}") from exc
        if not granted:
            raise MicrophoneError(
                "Microphone permission denied. Allow it in Android settings."
            )

        directory = Path(tempfile.mkdtemp(prefix="vision-assist-mic-"))
        path = directory / "recording.wav"
        captured: list[np.ndarray] = []
        pending = bytearray()
        offset: int | None = None
        started_at = time.monotonic()

        def consume() -> np.ndarray:
            nonlocal offset
            if offset is None:
                offset = self._wav_data_offset(path)
                if offset is None:
                    return np.zeros(0, dtype=np.float32)
            try:
                size = path.stat().st_size
            except OSError:
                return np.zeros(0, dtype=np.float32)
            if size <= offset:
                return np.zeros(0, dtype=np.float32)
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read()
            offset += len(data)
            pending.extend(data)
            usable = len(pending) // 2 * 2
            if usable == 0:
                return np.zeros(0, dtype=np.float32)
            block = self._to_float32(bytes(pending[:usable]))
            del pending[:usable]
            return block

        try:
            try:
                started = await recorder.start_recording(output_path=str(path))
            except Exception as exc:
                raise MicrophoneError(f"Could not start recording: {exc}") from exc
            if not started:
                raise MicrophoneError("Could not start recording.")
            started_at = time.monotonic()

            while True:
                await asyncio.sleep(POLL_SECONDS)
                elapsed = time.monotonic() - started_at
                block = consume()
                if block.size:
                    captured.append(block)
                    if endpoint is not None and endpoint.accept(block):
                        break
                if stop_event.is_set():
                    break
                if (
                    idle_timeout is not None
                    and endpoint is not None
                    and not endpoint.started
                    and elapsed >= idle_timeout
                ):
                    break
                if elapsed >= max_seconds:
                    break
        finally:
            try:
                await recorder.stop_recording()
            except Exception:  # pragma: no cover - already stopped
                pass
            # The recorder flushes on stop; pick up the final buffered PCM.
            await asyncio.sleep(0.05)
            tail = consume()
            if tail.size:
                captured.append(tail)
                if endpoint is not None:
                    endpoint.accept(tail)
            shutil.rmtree(directory, ignore_errors=True)

        audio = (
            np.concatenate(captured)
            if captured
            else np.zeros(0, dtype=np.float32)
        )
        if audio.size == 0:
            self._error = "The recorder produced no audio."
        return audio


class SoundDeviceMicrophone:
    """Desktop-only fallback for CLI tools and probes.

    The app itself uses :class:`FletAudioRecorderMicrophone`; this class exists
    because ``tools/test_wake.py`` runs without a Flet page. ``sounddevice`` is
    a dev dependency, never part of the Android build.
    """

    name = "microphone (sounddevice, tools)"

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._error: str | None = None

    def _sounddevice(self):
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            self._error = f"sounddevice unavailable: {exc}"
            return None
        return sd

    def available(self) -> bool:
        module = self._sounddevice()
        if module is None:
            return False
        try:
            return any(
                device.get("max_input_channels", 0) > 0
                for device in module.query_devices()
            )
        except Exception as exc:  # pragma: no cover
            self._error = f"microphone query failed: {exc}"
            return False

    @property
    def error(self) -> str | None:
        self.available()
        return self._error

    def _blocking_record(self, max_seconds, stop_event, endpoint, idle_timeout):
        module = self._sounddevice()
        if module is None:
            raise MicrophoneError(self._error or "sounddevice unavailable")
        chunks: list[np.ndarray] = []
        max_blocks = max(1, int(max_seconds * SAMPLE_RATE / 1600))
        started_at = time.monotonic()
        with module.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1600,
            device=self._device,
        ) as stream:
            for _ in range(max_blocks):
                if stop_event.is_set():
                    break
                data, _overflow = stream.read(1600)
                if not data.size:
                    continue
                block = np.asarray(data[:, 0], dtype=np.float32).copy()
                chunks.append(block)
                if endpoint is not None and endpoint.accept(block):
                    break
                if (
                    idle_timeout is not None
                    and endpoint is not None
                    and not endpoint.started
                    and time.monotonic() - started_at >= idle_timeout
                ):
                    break
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    async def record(self, max_seconds, stop_event, endpoint=None, idle_timeout=None):
        return await asyncio.to_thread(
            self._blocking_record, max_seconds, stop_event, endpoint, idle_timeout
        )


class NullMicrophone:
    name = "microphone (unavailable)"

    def __init__(self, reason: str = "Microphone unavailable.") -> None:
        self._reason = reason

    def available(self) -> bool:
        return False

    @property
    def error(self) -> str | None:
        return self._reason

    async def record(
        self,
        max_seconds: float,
        stop_event: threading.Event,
        endpoint=None,
        idle_timeout: float | None = None,
    ) -> np.ndarray:
        raise MicrophoneError(self._reason)


def create_microphone(page=None) -> Microphone:
    if page is not None:
        microphone = FletAudioRecorderMicrophone(page)
        if microphone.available():
            return microphone
    desktop = SoundDeviceMicrophone()
    if desktop.available():
        return desktop
    return NullMicrophone(
        "Microphone unavailable: no Flet page and sounddevice not installed."
    )
