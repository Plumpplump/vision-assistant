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
import traceback

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
        self._alert_src: bytes | None = None
        self._alert_playing = False
        self._alert_volume = 1.0

        self._haptics: ft.HapticFeedback | None = None
        self._lock = asyncio.Lock()
        self._wav_cache: dict[tuple[str, str, float], bytes] = {}

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
        if self._speech is None:
            self._speech_loaded.clear()
            self._speech = fta.Audio(
                src=wav,
                autoplay=False,
                volume=volume,
                release_mode=fta.ReleaseMode.STOP,
                on_loaded=lambda event: self._speech_loaded.set(),
            )
            self.page.services.append(self._speech)
            self.page.update()
        elif self._speech_src != wav:
            self._speech_loaded.clear()
            self._speech.src = wav
            self._speech.volume = volume
            self.page.update()
        else:
            self._speech.volume = volume
            self.page.update()

        self._speech_src = wav
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
        if self._alert is None:
            self._alert = fta.Audio(
                src=wav,
                autoplay=True,
                volume=volume,
                balance=balance,
                release_mode=fta.ReleaseMode.LOOP,
            )
            self._alert_src = wav
            self.page.services.append(self._alert)
            self.page.update()
            self._alert_playing = True
            return

        self._alert_volume = volume
        if self._alert_src != wav:
            self._alert.src = wav
            self._alert_src = wav
        self._alert.volume = volume
        self._alert.balance = balance
        self.page.update()
        if not self._alert_playing:
            await self._alert.play()
            self._alert_playing = True

    async def _one_shot(self, wav: bytes, volume: float, balance: float) -> None:
        if self._sfx is None:
            self._sfx_loaded.clear()
            self._sfx = fta.Audio(
                src=wav,
                autoplay=False,
                volume=volume,
                balance=balance,
                release_mode=fta.ReleaseMode.STOP,
                on_loaded=lambda event: self._sfx_loaded.set(),
            )
            self.page.services.append(self._sfx)
            self.page.update()
        elif self._sfx_src != wav:
            self._sfx_loaded.clear()
            self._sfx.src = wav
            self._sfx.volume = volume
            self._sfx.balance = balance
            self.page.update()
        else:
            self._sfx.volume = volume
            self._sfx.balance = balance
            self.page.update()

        self._sfx_src = wav
        await asyncio.sleep(0.05)
        await self._sfx.play()

    async def duck_alerts(self, ducked: bool) -> None:
        """Lower the chime while the user speaks; restore it afterwards."""
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
