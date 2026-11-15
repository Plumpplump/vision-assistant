"""Orchestration: camera -> detection -> hazards -> speech.

No Flet imports here. The UI subscribes with :meth:`Assistant.add_listener`
and supplies a :class:`Speaker` implementation for audio output.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from difflib import SequenceMatcher
from typing import Callable, Protocol

import numpy as np

from adapters.capture import Frame, FrameSource, create_source
from adapters.detection import Tracker, YoloDetector
from adapters.models import (
    DescribeRequest,
    Describer,
    DescriberError,
    create_describer,
)
from adapters.models.clip import ClipSearch
from adapters.speech import (
    DEFAULT_VOICE,
    Asr,
    AsrError,
    Earcons,
    LiveEndpoint,
    Microphone,
    MicrophoneError,
    NullMicrophone,
    ParadeeTts,
    SpeechSegmenter,
    TtsError,
    create_asr,
)
from domain import (
    AlertEvent,
    AlertStyle,
    Band,
    CameraModel,
    Detection,
    FrameScene,
    HazardMonitor,
    Intent,
    Request,
    Settings,
    describe_items,
    distance_speech,
    parse,
    scene_items,
    speakable,
    strip_wake_word,
)
from domain.distance import (
    band_from_ratio,
    is_very_close,
    pan_of,
    position_word,
    size_ratio,
)
from domain.intents import normalize
from domain.scene import canonical_target, surface_for


def _severity(band: Band) -> int:
    return {Band.FAR: 0, Band.NEAR: 1, Band.CLOSE: 2}[band]

HELP_TEXT = (
    "You can say: what is in front of me, is there a cup, where is the cup, "
    "describe it, find the cup, stop. You can also ask a question about what "
    "I can see, like what colour is the cup."
)
UNKNOWN_TEXT = "Sorry, I didn't catch that. Try: what is in front of me."
NO_MODEL_TEXT = "The description model is unavailable."


class Speaker(Protocol):
    """Audio output port. Implemented by the UI layer."""

    async def speak(self, text: str, voice: str, speed: float, volume: float) -> None: ...

    async def earcon(
        self, wav: bytes, volume: float, balance: float = 0.0, loop: bool = False
    ) -> None: ...

    async def silence_alerts(self) -> None: ...

    async def stop(self) -> None: ...

    async def haptic(self, strength: str = "light") -> None: ...

    async def duck_alerts(self, ducked: bool) -> None: ...


class NullSpeaker:
    """No-op speaker for headless tests and the web build."""

    async def speak(self, text: str, voice: str, speed: float, volume: float) -> None:
        return None

    async def earcon(
        self, wav: bytes, volume: float, balance: float = 0.0, loop: bool = False
    ) -> None:
        return None

    async def silence_alerts(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def haptic(self, strength: str = "light") -> None:
        return None

    async def duck_alerts(self, ducked: bool) -> None:
        return None


class Assistant:
    PERCEPTION_INTERVAL = 0.15
    SEARCH_TIMEOUT = 12.0
    SEARCH_TICK = 0.25
    SEARCH_HITS = 2

    def __init__(
        self,
        speaker: Speaker | None = None,
        settings: Settings | None = None,
        source: FrameSource | None = None,
        detector: YoloDetector | None = None,
        describer: Describer | None = None,
        tts: ParadeeTts | None = None,
        asr: Asr | None = None,
        microphone: Microphone | None = None,
        segmenter: SpeechSegmenter | None = None,
        clip: ClipSearch | None = None,
    ) -> None:
        self.settings = (settings or Settings.load()).normalized()
        self.source = source or create_source(self.settings.camera_index)
        self.detector = detector or YoloDetector(confidence=self.settings.confidence)
        self.describer = describer or create_describer(self.settings.describer)
        self.tts = tts or ParadeeTts()
        self.earcons = Earcons()
        self.speaker: Speaker = speaker or NullSpeaker()
        self.asr = asr or create_asr(self.settings.asr_backend)
        self.microphone = microphone or NullMicrophone(
            "Microphone is created by the UI (flet-audio-recorder needs a Page)."
        )
        self.segmenter = segmenter or SpeechSegmenter()
        self.clip = clip or ClipSearch(threshold=self.settings.clip_threshold)

        self.tracker = Tracker()
        self.camera = self._camera_from(self.settings)
        self.monitor = HazardMonitor(
            camera=self.camera,
            corridor=self.settings.corridor,
            min_size_ratio=self.settings.min_size_ratio,
            close_m=self.settings.alert_close_m,
            near_m=self.settings.alert_near_m,
        )
        self.scene = FrameScene()
        self.status = "starting"
        self.answer = ""
        self.heard = ""
        self.listening = False
        self.last_frame: Frame | None = None
        self.speaking = False
        self._record_stop = threading.Event()
        self._wake_stop = threading.Event()
        self._await_command_until = 0.0
        self._speech_cooldown_until = 0.0
        self._describe_active = False
        self._warming = False

        self._tasks: list[asyncio.Task] = []
        self._search_task: asyncio.Task | None = None
        self._alerting = False
        self._listeners: list[Callable[[], None]] = []
        self._search_generation = 0

    # ------------------------------------------------------------------ wiring

    @staticmethod
    def _camera_from(settings: Settings) -> CameraModel:
        return CameraModel(
            height_m=settings.camera_height_m,
            pitch_deg=settings.camera_pitch_deg,
            hfov_deg=settings.camera_hfov_deg,
            scale=settings.distance_scale,
        )

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:  # pragma: no cover - a broken listener must not stop us
                traceback.print_exc()

    @property
    def model_error(self) -> str | None:
        return self.detector.error or self.describer.error

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self.status = "loading detector"
        self._notify()
        try:
            await asyncio.to_thread(self.detector.load)
        except Exception as exc:
            self.status = f"detector unavailable: {exc}"
        else:
            self.status = f"detector {self.detector.name}"
        self.source.start()
        self._tasks.append(asyncio.create_task(self._perception_loop()))
        self._tasks.append(asyncio.create_task(self._warm_models()))
        if self.settings.wake_enabled and self.microphone.available():
            self._tasks.append(asyncio.create_task(self._wake_loop()))
        self._notify()

    async def _warm_models(self, greet: bool = True) -> None:
        """Warm the voice first, then ASR. The VLM loads on first use.

        Startup used to load ASR, TTS and the 700 MB VLM before speaking the
        greeting, which took ~30 s on a busy CPU. Speech is what the user
        needs first: TTS warms in about a second, the greeting plays, and ASR
        finishes behind it. The VLM is deferred because most interactions do
        not need it and its session init is the most expensive.
        """
        started = time.perf_counter()
        self._warming = True
        self.status = "loading voice…"
        self._notify()
        try:
            await asyncio.to_thread(self.tts.warm_up)
            print(f"[startup] tts {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:
            traceback.print_exc()

        if greet and self.settings.greet_enabled:
            # Fire and forget: the audio service can take seconds to load, and
            # the greeting must not delay ASR warm-up or the "ready" state.
            asyncio.create_task(
                self._say(
                    f"Hi {self.settings.greet_name}. Vision assist is ready. "
                    f"Say {self.settings.wake_word} to ask me something."
                )
            )
            print(
                f"[startup] greeting queued after "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        # Voice is usable now; let the camera run while ASR finishes loading.
        self._warming = False
        self.status = "voice ready"
        self._notify()


        def _load_asr() -> None:
            self.asr.load()

        try:
            await asyncio.to_thread(_load_asr)
        except Exception as exc:
            traceback.print_exc()
        print(f"[startup] asr {time.perf_counter() - started:.1f}s", flush=True)
        self._warming = False
        self.status = "ready"
        self._notify()

    async def shutdown(self) -> None:
        self.stop_listening()
        self._wake_stop.set()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._cancel_search()
        self.source.stop()
        await self.speaker.stop()

    # -------------------------------------------------------------- perception

    def _analyze(self, frame: Frame) -> tuple[FrameScene, list[AlertEvent]]:
        """Blocking: detector + tracker + hazard policy. Run in a thread."""
        image = frame.image
        if image is None:
            return FrameScene(timestamp=time.monotonic()), []
        self.detector.confidence = self.settings.confidence
        detections = self.detector.detect(image)
        # Only tracks seen in *this* frame may keep the alarm on. A missed
        # frame means the object is gone or out of view; waiting for the
        # tracker's miss counter (5 frames) made the chime linger.
        tracks = [
            track
            for track in self.tracker.update(detections)
            if track.confirmed and track.misses == 0
        ]
        events = self.monitor.update(tracks, frame.width, frame.height)
        items = scene_items(
            detections,
            frame.width,
            frame.height,
            camera=self.camera,
            surface_height_m=self.settings.surface_height_m,
        )
        scene = FrameScene(
            detections=detections,
            hazards=self.monitor.hazards,
            items=items,
            timestamp=time.monotonic(),
        )
        return scene, events

    async def _perception_loop(self) -> None:
        while True:
            if self._describe_active or self._warming:
                # The detector, speech models and VLM all want the same cores.
                # Freeze the camera loop while a description runs or while
                # startup loads the voice, so the greeting is not delayed.
                await asyncio.sleep(0.1)
                continue
            frame = self.source.latest()
            if (
                frame is not None
                and frame.image is not None
                and frame is not self.last_frame
            ):
                try:
                    scene, events = await asyncio.to_thread(self._analyze, frame)
                except Exception:
                    traceback.print_exc()
                else:
                    self.last_frame = frame
                    self.scene = scene
                    await self._handle_events(events)
                    self._notify()
            await asyncio.sleep(self.PERCEPTION_INTERVAL)

    # ------------------------------------------------------------------ alerts

    async def _handle_events(self, events: list[AlertEvent]) -> None:
        style = self._alert_style()
        if not self.settings.alerts_enabled or style is AlertStyle.OFF:
            if self._alerting:
                await self.speaker.silence_alerts()
                self._alerting = False
            return

        if not self.scene.hazards:
            if self._alerting:
                await self.speaker.silence_alerts()
                self._alerting = False
            return

        interesting = [event for event in events if event.kind != "clear"]
        if not interesting:
            return

        hazard = max(
            (event.hazard for event in interesting),
            key=lambda item: _severity(item.band),
        )
        close = hazard.band is Band.CLOSE
        escalation = any(
            event.previous is None
            or _severity(event.hazard.band) > _severity(event.previous)
            for event in interesting
        )

        # Haptics are private and carry urgency without noise.
        if self.settings.haptics_enabled:
            await self.speaker.haptic("heavy" if close else "light")

        if style is AlertStyle.HAPTIC:
            self._alerting = True
            return

        if style is AlertStyle.VOICE:
            # No beeps at all; one short word only when something is close.
            if close and escalation:
                await self._say("Careful.")
            self._alerting = True
            return

        wav = self.earcons.pattern(hazard.band, hazard.vertical, style)
        if wav is None:
            return
        await self.speaker.earcon(
            wav,
            volume=self.settings.alert_volume,
            balance=hazard.pan,
            loop=True,
        )
        self._alerting = True

    def _alert_style(self) -> AlertStyle:
        try:
            return AlertStyle(self.settings.alert_style)
        except ValueError:
            return AlertStyle.SOFT

    async def _record_with_duck(
        self,
        max_seconds: float,
        stop_event: threading.Event,
        endpoint: LiveEndpoint | None,
        idle_timeout: float | None = None,
    ) -> np.ndarray:
        """Record, lowering the alert chime while the user is speaking.

        The chime is generated locally and plays from the same speaker as the
        user's voice; leaving it at full level while the ASR listens measurably
        destroys the wake phrase (tested: 0.45 volume removed "hey vision" from
        the transcript). Ducking keeps a safety cue audible without masking
        speech.
        """
        task = asyncio.ensure_future(
            self.microphone.record(
                max_seconds,
                stop_event,
                endpoint,
                idle_timeout,
            )
        )
        ducked = False
        try:
            while not task.done():
                if endpoint is not None and endpoint.started and not ducked:
                    await self.speaker.duck_alerts(True)
                    ducked = True
                await asyncio.sleep(0.05)
            return task.result()
        finally:
            if ducked:
                await self.speaker.duck_alerts(False)

    # -------------------------------------------------------------------- wake

    async def _wake_loop(self) -> None:
        """Always-on VAD, one ASR pass per utterance, wake phrase required.

        Only Silero runs while the room is quiet. When speech appears, Parakeet
        transcribes the utterance and :meth:`handle_wake_text` decides whether
        it starts with the wake phrase. The microphone is paused whenever the
        assistant speaks or an explicit Talk session is recording.
        """
        while True:
            if not self.settings.wake_enabled or not self.microphone.available():
                await asyncio.sleep(0.5)
                continue
            if (
                self.listening
                or self.speaking
                or self._warming
                or time.monotonic() < self._speech_cooldown_until
            ):
                await asyncio.sleep(0.2)
                continue

            try:
                endpoint = LiveEndpoint(
                    max_seconds=8.0,
                    silence=0.6,
                    min_speech=0.15,
                    threshold=0.4,
                )
            except Exception:
                await asyncio.sleep(2.0)
                continue

            self._wake_stop.clear()
            try:
                samples = await self._record_with_duck(
                    8.0,
                    self._wake_stop,
                    endpoint,
                    2.0,
                )
            except MicrophoneError:
                await asyncio.sleep(2.0)
                continue
            except Exception:  # pragma: no cover - device specific
                traceback.print_exc()
                await asyncio.sleep(2.0)
                continue

            if self._wake_stop.is_set() or self.listening:
                await asyncio.sleep(0.1)
                continue

            speech = endpoint.speech()
            if speech is None:
                continue
            try:
                text = await asyncio.to_thread(self.asr.transcribe, speech)
            except Exception:
                traceback.print_exc()
                continue
            print(f"[wake] heard {text!r}", flush=True)
            reply = await self.handle_wake_text(text)
            if reply is None:
                print("[wake] ignored: no wake phrase", flush=True)

    async def handle_wake_text(self, text: str) -> str | None:
        """Handle one utterance; returns the reply, ``""`` for "Yes?", or None.

        None means "no wake phrase": background conversation is ignored. After a
        wake-only utterance ("Hi David") the next utterance is accepted without
        the wake phrase for a few seconds.
        """
        now = time.monotonic()
        wake_word = self.settings.wake_word
        command = strip_wake_word(text, wake_word)
        if command is None:
            if now >= self._await_command_until:
                return None
            # Follow-up window: accept a recognised command without the wake
            # word, but ignore background chatter instead of answering it.
            if parse(text).intent is Intent.UNKNOWN:
                return None
            self._await_command_until = 0.0
            return await self.ask(text)

        if not command:
            self._await_command_until = now + 8.0
            self.status = f"listening… ({wake_word})"
            self._notify()
            await self.speaker.earcon(
                self.earcons.wake(), volume=self.settings.alert_volume
            )
            return ""

        return await self.ask(command)

    # ------------------------------------------------------------------- listen

    async def listen(self) -> str:
        """Record, trim with VAD, transcribe, then answer.

        The UI calls this from the Talk button; a second call while recording
        stops it. Recording and recognition both run in worker threads so the
        camera preview keeps running.
        """
        if self.listening:
            self.stop_listening()
            return ""

        if not self.microphone.available():
            reply = (
                "The microphone is not available here. "
                "Use the text box or the quick buttons."
            )
            self.answer = reply
            self._notify()
            await self._say(reply)
            return reply

        endpoint = None
        try:
            endpoint = LiveEndpoint(max_seconds=self.settings.listen_seconds + 2)
        except Exception:
            endpoint = None  # fall back to tap-to-stop plus a VAD pass

        self._wake_stop.set()
        self._await_command_until = 0.0
        self.listening = True
        self.heard = ""
        self.status = (
            "listening… speak, then pause" if endpoint else "listening… tap Talk to stop"
        )
        self._record_stop.clear()
        self._notify()
        await self.speaker.earcon(self.earcons.ping(), volume=self.settings.alert_volume)

        samples = np.zeros(0, dtype=np.float32)
        microphone_error: str | None = None
        try:
            samples = await self._record_with_duck(
                self.settings.listen_seconds,
                self._record_stop,
                endpoint,
            )
        except MicrophoneError as exc:
            microphone_error = str(exc)
        finally:
            self.listening = False
            self._wake_stop.clear()

        if microphone_error:
            reply = f"Recording failed: {microphone_error}"
            self.answer = reply
            self.status = "ready"
            self._notify()
            await self._say(reply)
            return reply

        self.status = "recognising…"
        self._notify()

        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0

        trimmed = endpoint.speech() if endpoint is not None else None
        if trimmed is None:
            try:
                trimmed = await asyncio.to_thread(self.segmenter.trim, samples)
            except Exception:
                traceback.print_exc()
                trimmed = samples  # fall back to the raw recording

        print(
            f"[asr] recorded {samples.size / 16000:.2f}s "
            f"(rms {rms:.3f}, peak {peak:.2f}), "
            f"speech {0 if trimmed is None else trimmed.size / 16000:.2f}s",
            flush=True,
        )

        if trimmed is None or trimmed.size < int(16000 * 0.2):
            reply = (
                "I didn't hear anything. Check the microphone in Settings."
                if rms < 0.005
                else "I didn't hear any speech."
            )
            self.answer = reply
            self.status = "ready"
            self._notify()
            await self._say(reply)
            return reply

        started = time.perf_counter()
        try:
            text = await asyncio.to_thread(self.asr.transcribe, trimmed)
        except AsrError as exc:
            reply = f"Speech recognition is unavailable: {exc}"
            self.answer = reply
            self.status = "ready"
            self._notify()
            await self._say(reply)
            return reply
        except Exception:
            traceback.print_exc()
            text = ""

        self.heard = text.strip()
        print(
            f"[asr] {self.asr.name} {time.perf_counter() - started:.2f}s "
            f"-> {self.heard!r}",
            flush=True,
        )
        if not self.heard:
            reply = "I didn't catch that."
            self.answer = reply
            self.status = "ready"
            self._notify()
            await self._say(reply)
            return reply

        self.status = f"heard: {self.heard}"
        return await self.ask(self.heard)

    def stop_listening(self) -> None:
        self._record_stop.set()
        if self.listening:
            self.status = "ready"
            self._notify()

    # -------------------------------------------------------------------- talk

    async def ask(self, text: str) -> str:
        request = parse(text)
        self.status = f"heard: {text.strip() or '(nothing)'}"

        if request.intent is Intent.FIND and request.target:
            self._cancel_search()
            reply = f"Looking for the {request.target}."
            self.answer = reply
            self.status = f"searching: {request.target}"
            self._notify()
            await self._say(reply)
            self._search_task = asyncio.create_task(self._run_search(request.target))
            return reply

        reply = await self._respond(request, text)
        self.answer = reply
        self._await_command_until = time.monotonic() + self.settings.follow_up_seconds
        self._notify()
        await self._say(reply)
        return reply

    async def _respond(self, request: Request, text: str = "") -> str:
        if request.intent is Intent.SCENE:
            return describe_items(self.scene.items)
        if request.intent is Intent.CHECK:
            return await self._locate(request.target, yes_no=True)
        if request.intent is Intent.LOCATE:
            return await self._locate(request.target, yes_no=False)
        if request.intent is Intent.DESCRIBE:
            return await self._describe(request.target)
        if request.intent is Intent.STOP:
            self._cancel_search()
            await self.speaker.stop()
            return "Stopped."
        if request.intent is Intent.MUTE:
            self.settings.alerts_enabled = False
            self.settings.save()
            await self.speaker.silence_alerts()
            return "Alerts muted."
        if request.intent is Intent.HELP:
            return HELP_TEXT
        if request.intent is Intent.FIND:
            return "What are you looking for?"
        return await self._answer_question(text)

    # ---------------------------------------------------------------- describe

    def _pick_detection(self, target: str | None) -> Detection | None:
        if not self.scene.detections:
            return None
        if target is None:
            return max(
                self.scene.detections,
                key=lambda item: item.box.height,
            )
        wanted = canonical_target(target)
        best: Detection | None = None
        best_score = 0.0
        for detection in self.scene.detections:
            score = SequenceMatcher(None, speakable(detection.label), wanted).ratio()
            if score > best_score:
                best_score, best = score, detection
        return best if best_score >= 0.6 else None

    async def _answer_question(self, text: str) -> str:
        """VLM fallback for free-form questions about the current picture.

        Rules still win for the fixed intents; this only runs when nothing
        matched and the describer is a real model. With the template describer
        the app keeps the old "I didn't catch that" behaviour.
        """
        frame = self.last_frame
        if self.describer.name == "fake" or frame is None or frame.image is None:
            return UNKNOWN_TEXT

        detection = self._detection_for_question(text)
        crop = self._crop(frame, detection) if detection else frame.image
        await self.speaker.earcon(
            self.earcons.ping(), volume=self.settings.alert_volume
        )
        self._describe_active = True
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    self.describer.describe,
                    DescribeRequest(
                        image=crop,
                        question=text,
                        hint=detection.label if detection else None,
                    ),
                ),
                timeout=30.0,
            )
        except DescriberError:
            answer = ""
        except Exception:
            traceback.print_exc()
            answer = ""
        finally:
            self._describe_active = False
        self._warming = False
        return answer or UNKNOWN_TEXT

    async def _clip_position(self, target: str) -> tuple[str, float] | None:
        """Open-vocabulary lookup: (position, score) or None.

        The text embedding is cached; one image-tower pass scores the frame and
        a second pass over three strips localises it. Perception is frozen for
        the ~0.3-0.5 s each pass takes, so the two models do not fight.
        """
        frame = self.last_frame
        if frame is None or frame.image is None or not self.clip.available():
            return None
        try:
            embedding = await asyncio.to_thread(self.clip.embed_text, target)
            self._describe_active = True
            try:
                score = await asyncio.to_thread(
                    self.clip.score, frame.image, embedding
                )
                if score < self.settings.clip_threshold:
                    return None
                position, _ = await asyncio.to_thread(
                    self.clip.best_strip, frame.image, embedding
                )
                return position, score
            finally:
                self._describe_active = False
        except Exception:
            traceback.print_exc()
            return None

    async def _locate(self, target: str | None, yes_no: bool) -> str:
        """Answer "is there a cup?" / "where is the cup?" from live frames.

        Three attempts over ~0.75 s: a detector can miss an object in a single
        frame, especially while the user is holding it up.
        """
        detection = None
        for attempt in range(3):
            detection = self._match(target) if target else None
            if detection is not None:
                break
            if attempt < 2:
                await asyncio.sleep(0.25)

        frame = self.last_frame
        if detection is None and target and frame is not None:
            clip_hit = await self._clip_position(target)
            if clip_hit is not None:
                position, _score = clip_hit
                where = {
                    "left": "on your left",
                    "right": "on your right",
                    "ahead": "ahead",
                }[position]
                wanted = target
                if yes_no:
                    return f"Yes, I can see a {wanted}, {where}."
                return f"The {wanted} is {where}."

        if detection is None or frame is None:
            wanted = target or "anything"
            if wanted in {"anything", "something"}:
                return "I don't see anything in front of me."
            plural = wanted.endswith("s") and not wanted.endswith("ss")
            missing = f"any {wanted}" if plural else f"a {wanted}"
            seen = [item.label for item in self.scene.items[:2]]
            addition = (
                " I see " + " and ".join(f"a {label}" for label in seen) + "."
                if seen
                else ""
            )
            return f"I don't see {missing} in front of me.{addition}"

        label = speakable(detection.label)
        distance = self._distance_of(detection, frame)
        band = band_from_ratio(size_ratio(detection.box, frame.height))
        range_speech = (
            "very close"
            if is_very_close(detection.box, frame.height)
            else distance_speech(distance) or band.speech
        )
        where = {
            "left": "on your left",
            "right": "on your right",
            "ahead": "ahead",
        }[position_word(pan_of(detection.box, frame.width))]

        if yes_no:
            article = "an" if label[:1].lower() in "aeiou" else "a"
            return f"Yes, {article} {label}, {range_speech}, {where}."
        return f"The {label} is {range_speech}, {where}."

    def _detection_for_question(self, text: str) -> Detection | None:
        """Largest object mentioned in the question, else the largest one."""
        lowered = normalize(text)
        for detection in self.scene.detections:
            label = speakable(detection.label)
            if label.lower() in lowered:
                return detection
        return self._pick_detection(None)

    def _crop(self, frame: Frame, detection: Detection) -> np.ndarray:
        assert frame.image is not None
        height, width = frame.image.shape[:2]
        box = detection.box
        pad_x = box.width * 0.08
        pad_y = box.height * 0.08
        x1 = max(0, int(box.x1 - pad_x))
        y1 = max(0, int(box.y1 - pad_y))
        x2 = min(width, int(box.x2 + pad_x))
        y2 = min(height, int(box.y2 + pad_y))
        return frame.image[y1:y2, x1:x2].copy()

    async def _describe(self, target: str | None) -> str:
        frame = self.last_frame
        if frame is None or frame.image is None:
            return "I don't have a picture right now."
        detection = self._pick_detection(target)
        if detection is None:
            wanted = target or "anything"
            return f"I can't see {wanted} in front of me."

        crop = self._crop(frame, detection)
        question = (
            f"Describe the {target} in one short sentence."
            if target
            else "Describe what you see in one short sentence."
        )
        if self.describer.name != "fake":
            # The VLM needs a few seconds on CPU; acknowledge the request so the
            # user does not think it was ignored.
            await self.speaker.earcon(
                self.earcons.ping(), volume=self.settings.alert_volume
            )
        self._describe_active = True
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    self.describer.describe,
                    DescribeRequest(image=crop, question=question, hint=detection.label),
                ),
                timeout=30.0,
            )
        except DescriberError:
            answer = ""
        except Exception:
            traceback.print_exc()
            answer = ""
        finally:
            self._describe_active = False
        self._warming = False

        if not answer:
            answer = self._template_description(detection, frame)
        return answer

    def _distance_of(self, detection: Detection, frame: Frame) -> float | None:
        return self.camera.distance_m(
            detection.box,
            frame.width,
            frame.height,
            surface_height_m=surface_for(
                detection.label, self.settings.surface_height_m
            ),
        )

    def _template_description(self, detection: Detection, frame: Frame) -> str:
        distance = self._distance_of(detection, frame)
        band = band_from_ratio(size_ratio(detection.box, frame.height))
        pan = pan_of(detection.box, frame.width)
        position = {
            "left": "on your left",
            "right": "on your right",
            "ahead": "ahead",
        }[position_word(pan)]
        range_speech = (
            "very close"
            if is_very_close(detection.box, frame.height)
            else distance_speech(distance) or band.speech
        )
        phrase = f"{speakable(detection.label)}, {range_speech}, {position}"
        hint = ""
        if self.describer.error:
            hint = " The description model is unavailable."
        return f"{phrase}.{hint}"

    # ------------------------------------------------------------------- find

    def _match(self, target: str) -> Detection | None:
        """Fuzzy label match against the current frame (with everyday aliases)."""
        wanted = canonical_target(target)
        best: Detection | None = None
        best_score = 0.0
        for detection in self.scene.detections:
            label = speakable(detection.label)
            score = SequenceMatcher(None, label, wanted).ratio()
            if wanted in label or label in wanted:
                score = max(score, 0.9)
            if score > best_score:
                best_score, best = score, detection
        return best if best_score >= 0.6 else None

    def _cancel_search(self) -> None:
        self._search_generation += 1
        if self._search_task is not None:
            self._search_task.cancel()
            self._search_task = None

    async def _run_search(self, target: str) -> None:
        generation = self._search_generation
        deadline = time.monotonic() + self.SEARCH_TIMEOUT
        hits = 0
        found: Detection | None = None
        found_position: str | None = None
        frame_height = self.last_frame.height if self.last_frame else 480
        frame_width = self.last_frame.width if self.last_frame else 640

        clip_embedding = None
        clip_attempted = False
        clip_hits = 0
        last_clip_try = 0.0

        while time.monotonic() < deadline:
            if generation != self._search_generation:
                return
            detection = self._match(target)
            if detection is not None:
                hits += 1
                found = detection
                if hits >= self.SEARCH_HITS:
                    break
            else:
                hits = 0
                found = None

            frame = self.last_frame
            if found is None and not clip_attempted and self.clip.available():
                clip_attempted = True
                try:
                    clip_embedding = await asyncio.to_thread(
                        self.clip.embed_text, target
                    )
                except Exception:
                    traceback.print_exc()
            if (
                found is None
                and clip_embedding is not None
                and frame is not None
                and frame.image is not None
                and time.monotonic() - last_clip_try >= 0.4
            ):
                last_clip_try = time.monotonic()
                try:
                    self._describe_active = True
                    try:
                        score = await asyncio.to_thread(
                            self.clip.score, frame.image, clip_embedding
                        )
                    finally:
                        self._describe_active = False
                    if score >= self.settings.clip_threshold:
                        clip_hits += 1
                        if clip_hits >= self.SEARCH_HITS:
                            found_position, _ = await asyncio.to_thread(
                                self.clip.best_strip, frame.image, clip_embedding
                            )
                            break
                    else:
                        clip_hits = 0
                except Exception:
                    traceback.print_exc()
                    clip_hits = 0
            await asyncio.sleep(self.SEARCH_TICK)

        if generation != self._search_generation:
            return

        if found is None and found_position is not None:
            where = {
                "left": "on your left",
                "right": "on your right",
                "ahead": "ahead",
            }[found_position]
            await self.speaker.earcon(
                self.earcons.found(),
                volume=self.settings.alert_volume,
                balance={"left": -0.6, "ahead": 0.0, "right": 0.6}[found_position],
            )
            reply = f"Found the {target}, {where}."
            self.answer = reply
            self.status = "ready"
            self._notify()
            await self._say(reply)
            return

        if found is None:
            await self.speaker.earcon(
                self.earcons.miss(), volume=self.settings.alert_volume
            )
            reply = f"I don't see the {target}."
        else:
            band = band_from_ratio(size_ratio(found.box, frame_height))
            distance = None
            if self.last_frame is not None:
                distance = self._distance_of(found, self.last_frame)
            pan = pan_of(found.box, frame_width)
            position = position_word(pan)
            where = {
                "left": "on your left",
                "right": "on your right",
                "ahead": "ahead",
            }[position]
            await self.speaker.earcon(
                self.earcons.found(),
                volume=self.settings.alert_volume,
                balance=pan,
            )
            reply = (
                f"Found the {target}, "
                f"{(distance_speech(distance) or band.speech)}, {where}."
            )
        self.answer = reply
        self.status = "ready"
        self._notify()
        await self._say(reply)

    # ------------------------------------------------------------------ speech

    async def say(self, text: str) -> None:
        """Public entry point for UI actions (e.g. 'test voice')."""
        await self._say(text)

    async def _say(self, text: str) -> None:
        if not text or not self.settings.speech_enabled:
            return
        self.speaking = True
        try:
            await self.speaker.speak(
                text,
                voice=self.settings.voice or DEFAULT_VOICE,
                speed=self.settings.speed,
                volume=self.settings.speech_volume,
            )
        except TtsError as exc:
            self.status = f"speech unavailable: {exc}"
        except Exception:
            traceback.print_exc()
        finally:
            self.speaking = False
            self._speech_cooldown_until = time.monotonic() + 0.6

    # ---------------------------------------------------------------- settings

    def apply_settings(self, settings: Settings) -> None:
        previous_describer = self.settings.describer
        previous_asr = self.settings.asr_backend
        previous_wake = self.settings.wake_enabled
        self.settings = settings.normalized()
        self.camera = self._camera_from(self.settings)
        self.monitor.camera = self.camera
        self.monitor.corridor = self.settings.corridor
        self.monitor.min_size_ratio = self.settings.min_size_ratio
        self.monitor.close_m = self.settings.alert_close_m
        self.monitor.near_m = self.settings.alert_near_m
        self.clip.threshold = self.settings.clip_threshold
        self.detector.confidence = self.settings.confidence
        if self.settings.describer != previous_describer:
            self.describer = create_describer(self.settings.describer)
        if self.settings.asr_backend != previous_asr:
            self.asr = create_asr(self.settings.asr_backend)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass  # sync caller (tests); the next listen() warms it
            else:
                self._tasks.append(
                    loop.create_task(self._warm_models(greet=False))
                )
        if self.settings.wake_enabled != previous_wake:
            self._wake_stop.set()
            if self.settings.wake_enabled:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    self._tasks.append(loop.create_task(self._wake_loop()))
        if not self.settings.alerts_enabled:
            self._alerting = False
        self.settings.save()
        self._notify()

    def nearest_distance(self) -> float | None:
        """Best current estimate, used by the calibration dialog."""
        if self.scene.hazards:
            return self.scene.hazards[0].distance_m
        for item in self.scene.items:
            if item.distance_m is not None:
                return item.distance_m
        return None

    def snapshot(self) -> dict[str, object]:
        """Small status payload for the UI."""
        nearest = self.scene.hazards[0] if self.scene.hazards else None
        return {
            "status": self.status,
            "answer": self.answer,
            "heard": self.heard,
            "listening": self.listening,
            "asr": self.asr.name,
            "wake": self.settings.wake_word if self.settings.wake_enabled else "",
            "hazards": len(self.scene.hazards),
            "detections": len(self.scene.detections),
            "nearest": nearest.speech if nearest else "",
            "fps": self.last_frame.fps if self.last_frame else 0.0,
            "model_error": self.model_error,
        }
