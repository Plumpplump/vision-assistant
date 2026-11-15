# Author: David Li, SISHK

"""Flet audio output: the :class:`app.assistant.Speaker` implementation.

Flet rules learned the hard way (inherited from the sibling ``SAT-Words``
project):

* never construct ``fta.Audio`` with ``src=None`` — the client-side service
  init throws and the binding is never stored, so ``play()`` hangs;
* assign ``src``, call ``page.update()``, wait for ``on_loaded``, then play;
* service invokes carry no properties, so every property change needs an
  update too.

Alert patterns loop (``ReleaseMode.LOOP``) so band changes switch a gapless
pattern instead of scheduling beeps over the websocket; one-shots use
``ReleaseMode.STOP`` so replays come from memory instantly.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import traceback
from pathlib import Path

import flet as ft
import flet_audio as fta

from adapters.speech import ParadeeTts, TtsError
from app.assistant import Speaker

LOAD_TIMEOUT = 2.0


class FletSpeaker(Speaker):
    def __init__(self, page: ft.Page, tts: ParadeeTts) -> None:
        self.page = page
        self.tts = tts

        self._speech: fta.Audio | None = None
        self._speech_src: bytes | None = None
        self._speech_loaded = asyncio.Event()

        self._sfx: fta.Audio | None = None
        self._sfx_src: bytes | None = None
        self._sfx_loaded = asyncio.Event()

        self._alert: fta.Audio | None = None
        self._alert_src: str | bytes | None = None
        self._alert_playing = False
        self._alert_volume = 1.0
        self._ducked = False

        self._haptics: ft.HapticFeedback | None = None
        self._lock = asyncio.Lock()
        self._wav_cache: dict[tuple[str, str, float], bytes] = {}
        # The macOS audioplayers plugin writes byte sources into
        # ~/Library/Caches/<bundle id>/ without creating the directory, which
        # raises PathNotFoundException on a fresh Flet client. A real file path
        # takes the plugin's setSourceDeviceFile branch instead.
        self._media_dir: Path | None = None
        try:
            self._media_dir = Path(tempfile.mkdtemp(prefix="vision-assist-audio-"))
        except OSError:  # pragma: no cover - read-only temp dir
            traceback.print_exc()
        self._media_files: dict[str, str] = {}

    def _media_source(self, wav: bytes) -> str | bytes:
        """A playable source: a temp file on desktop/mobile, bytes on web."""
        if getattr(self.page, "web", False) or self._media_dir is None:
            return wav
        digest = hashlib.sha1(wav).hexdigest()
        path = self._media_files.get(digest)
        if path is None:
            file = self._media_dir / f"{digest}.wav"
            if not file.is_file() or file.stat().st_size != len(wav):
                file.write_bytes(wav)
            path = str(file)
            self._media_files[digest] = path
        return path

    # ------------------------------------------------------------------ speech

    async def speak(self, text: str, voice: str, speed: float, volume: float) -> None:
        error = self.tts.error
        if error:
            raise TtsError(error)
        key = (text, voice, round(float(speed), 2))
        wav = self._wav_cache.get(key)
        if wav is None:
            wav = await asyncio.to_thread(self.tts.synthesize, text, voice, speed)
            if len(self._wav_cache) > 32:
                self._wav_cache.clear()
            self._wav_cache[key] = wav
        async with self._lock:
            await self._play_speech(wav, volume)

    async def _play_speech(self, wav: bytes, volume: float) -> None:
        src = self._media_source(wav)
        if self._speech is None:
            self._speech_loaded.clear()
            self._speech = fta.Audio(
                src=src,
                autoplay=False,
                volume=volume,
                release_mode=fta.ReleaseMode.STOP,
                on_loaded=lambda event: self._speech_loaded.set(),
            )
            self.page.services.append(self._speech)
            self.page.update()
        elif self._speech_src != src:
            self._speech_loaded.clear()
            self._speech.src = src
            self._speech.volume = volume
            self.page.update()
        else:
            self._speech.volume = volume
            self.page.update()

        self._speech_src = src
        # Give the client a moment to receive the new source; do not wait for
        # on_loaded — a slow audio backend would otherwise block the event loop
        # and stall startup for seconds.
        await asyncio.sleep(0.12)
        await self._speech.play()

    # ------------------------------------------------------------------ earcons

    async def earcon(
        self, wav: bytes, volume: float, balance: float = 0.0, loop: bool = False
    ) -> None:
        if loop:
            await self._loop_alert(wav, volume, balance)
        else:
            await self._one_shot(wav, volume, balance)

    async def _loop_alert(self, wav: bytes, volume: float, balance: float) -> None:
        src = self._media_source(wav)
        # Remember the configured level even on first creation; otherwise
        # duck_alerts() restores to 1.0 after the first duck instead of the
        # user's alert volume. If the user is already speaking, a new alert
        # must start ducked too.
        self._alert_volume = volume
        level = volume * (0.12 if self._ducked else 1.0)
        if self._alert is None:
            self._alert = fta.Audio(
                src=src,
                autoplay=True,
                volume=level,
                balance=balance,
                release_mode=fta.ReleaseMode.LOOP,
            )
            self._alert_src = src
            self.page.services.append(self._alert)
            self.page.update()
            self._alert_playing = True
            return

        if self._alert_src != src:
            self._alert.src = src
            self._alert_src = src
        self._alert.volume = level
        self._alert.balance = balance
        self.page.update()
        if not self._alert_playing:
            await self._alert.play()
            self._alert_playing = True

    async def _one_shot(self, wav: bytes, volume: float, balance: float) -> None:
        src = self._media_source(wav)
        if self._sfx is None:
            self._sfx_loaded.clear()
            self._sfx = fta.Audio(
                src=src,
                autoplay=False,
                volume=volume,
                balance=balance,
                release_mode=fta.ReleaseMode.STOP,
                on_loaded=lambda event: self._sfx_loaded.set(),
            )
            self.page.services.append(self._sfx)
            self.page.update()
        elif self._sfx_src != src:
            self._sfx_loaded.clear()
            self._sfx.src = src
            self._sfx.volume = volume
            self._sfx.balance = balance
            self.page.update()
        else:
            self._sfx.volume = volume
            self._sfx.balance = balance
            self.page.update()

        self._sfx_src = src
        await asyncio.sleep(0.05)
        await self._sfx.play()

    async def duck_alerts(self, ducked: bool) -> None:
        """Lower the chime while the user speaks; restore it afterwards."""
        self._ducked = ducked
        if self._alert is None:
            return
        self._alert.volume = self._alert_volume * (0.12 if ducked else 1.0)
        self.page.update()

    async def silence_alerts(self) -> None:
        if self._alert is not None and self._alert_playing:
            try:
                await self._alert.pause()
            except Exception:  # pragma: no cover - player already released
                traceback.print_exc()
            self._alert_playing = False

    async def stop(self) -> None:
        for player in (self._speech, self._sfx, self._alert):
            if player is None:
                continue
            try:
                await player.pause()
            except Exception:  # pragma: no cover
                pass
        self._alert_playing = False

    # ------------------------------------------------------------------ haptics

    async def haptic(self, strength: str = "light") -> None:
        try:
            if self._haptics is None:
                self._haptics = ft.HapticFeedback()
                self.page.services.append(self._haptics)
                self.page.update()
            method = {
                "light": self._haptics.light_impact,
                "medium": self._haptics.medium_impact,
                "heavy": self._haptics.heavy_impact,
            }.get(strength, self._haptics.vibrate)
            await method()
        except Exception:  # pragma: no cover - desktop has no haptics
            pass
