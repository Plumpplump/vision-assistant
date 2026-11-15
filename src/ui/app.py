"""Flet UI: AppBar, camera preview, status panel, settings dialog.

The UI is deliberately thin: it owns Flet controls and a preview loop, while
:class:`app.assistant.Assistant` owns the camera, models and state.
"""

from __future__ import annotations

import asyncio
import base64
import traceback

import flet as ft

from adapters.detection.preview import annotate_jpeg
from adapters.models import available_backends
from adapters.paths import model_path
from adapters.speech import (
    DEFAULT_VOICE,
    FletAudioRecorderMicrophone,
    ParadeeTts,
    TtsError,
)
from app.assistant import Assistant
from domain import AlertStyle, Settings, calibrate_scale

from .audio import FletSpeaker

PREVIEW_INTERVAL = 0.08
CARD_RADIUS = 14
PANEL_WIDTH = 380

# 1x1 transparent PNG: ft.Image requires a src, and an empty one is rejected.
BLANK_PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBAp0Y"
    "VwAAAABJRU5ErkJggg=="
)


class VisionAssistApp:
    def __init__(self, page: ft.Page, settings: Settings | None = None) -> None:
        self.page = page
        self.settings = settings or Settings.load()
        self.tts = ParadeeTts()
        self.speaker = FletSpeaker(page, self.tts)
        self.microphone = FletAudioRecorderMicrophone(page)
        self.assistant = Assistant(
            speaker=self.speaker,
            settings=self.settings,
            tts=self.tts,
            microphone=self.microphone,
        )
        self._last_frame = None
        self._last_snapshot: dict[str, object] = {}
        self._closed = False

        self._build_controls()
        self._build_layout()
        self.assistant.add_listener(self._refresh_status)

    # ------------------------------------------------------------------ layout

    def _build_controls(self) -> None:
        self.preview_image = ft.Image(
            key="preview",
            src=BLANK_PIXEL,
            fit=ft.BoxFit.CONTAIN,
            expand=True,
            visible=False,
            # Without this Flutter shows nothing for a frame each time the
            # provider changes -> a visible flash at camera frame rate.
            gapless_playback=True,
        )
        self.placeholder_text = ft.Text(
            "Starting camera…",
            size=16,
            color=ft.Colors.BLUE_GREY_100,
            text_align=ft.TextAlign.CENTER,
        )
        self.placeholder = ft.Container(
            key="preview_placeholder",
            expand=True,
            alignment=ft.Alignment.CENTER,
            padding=24,
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.VISIBILITY, size=48, color=ft.Colors.BLUE_GREY_200),
                    self.placeholder_text,
                ],
            ),
        )
        self.preview = ft.Container(
            key="preview_card",
            expand=3,
            bgcolor=ft.Colors.BLACK,
            border_radius=CARD_RADIUS,
            content=ft.Stack(
                # EXPAND + Image(fit=CONTAIN) keeps the frame centred with equal
                # letterbox bars instead of pinning it to the top-left corner.
                fit=ft.StackFit.EXPAND,
                controls=[self.placeholder, self.preview_image],
            ),
        )

        self.status_text = ft.Text(
            "starting", key="status", size=13, color=ft.Colors.BLUE_GREY_600
        )
        self.camera_text = ft.Text(
            "camera —", key="camera", size=13, color=ft.Colors.BLUE_GREY_600
        )
        self.detection_text = ft.Text(
            "0 objects", key="detections", size=13, color=ft.Colors.BLUE_GREY_600
        )
        self.hazard_text = ft.Text(
            "path clear",
            key="hazard",
            size=15,
            weight=ft.FontWeight.W_600,
            color=ft.Colors.GREEN_700,
        )
        self.model_text = ft.Text(
            "", key="model", size=12, color=ft.Colors.BLUE_GREY_500
        )
        self.heard_text = ft.Text(
            "Heard: —", key="heard", size=13, color=ft.Colors.BLUE_GREY_600
        )
        self.answer_text = ft.Text(
            "Ask me what is in front of you.",
            key="answer",
            size=20,
            selectable=True,
            weight=ft.FontWeight.W_500,
            color=ft.Colors.BLUE_GREY_900,
        )

        self.command = ft.TextField(
            key="command",
            hint_text="Type a command: what is in front of me?",
            expand=True,
            on_submit=self._on_send,
            dense=True,
        )
        self.send_button = ft.IconButton(
            key="send",
            icon=ft.Icons.SEND,
            tooltip="Send command",
            on_click=self._on_send,
        )
        self.talk_button = ft.FilledButton(
            key="talk",
            content="Talk",
            icon=ft.Icons.MIC,
            on_click=self._on_talk,
            tooltip="Tap once, speak, then pause (or shake the phone)",
        )
        self.scene_button = ft.FilledButton(
            key="ask_scene",
            content="What's in front?",
            icon=ft.Icons.HEARING,
            on_click=self._ask_scene,
        )
        self.describe_button = ft.OutlinedButton(
            key="ask_describe",
            content="Describe it",
            icon=ft.Icons.SEARCH,
            on_click=self._ask_describe,
        )
        self.stop_button = ft.OutlinedButton(
            key="ask_stop",
            content="Stop",
            icon=ft.Icons.STOP,
            on_click=self._ask_stop,
        )

        self.mute_button = ft.IconButton(
            key="mute",
            icon=ft.Icons.VOLUME_UP if self.settings.alerts_enabled else ft.Icons.VOLUME_OFF,
            tooltip="Mute / unmute path alerts",
            on_click=self._toggle_mute,
        )
        self.settings_button = ft.IconButton(
            key="settings",
            icon=ft.Icons.SETTINGS,
            tooltip="Settings",
            on_click=self._open_settings,
        )

    def _sidebar(self) -> ft.Control:
        return ft.Container(
            width=PANEL_WIDTH,
            content=ft.Column(
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                controls=[
                    self._card(
                        "Status",
                        [
                            self.status_text,
                            ft.Row([self.camera_text, self.detection_text], spacing=16),
                            self.hazard_text,
                            self.heard_text,
                            self.model_text,
                        ],
                    ),
                    self._card(
                        "Last answer",
                        [self.answer_text],
                    ),
                    self._card(
                        "Ask",
                        [
                            self.talk_button,
                            ft.Row([self.command, self.send_button], spacing=4),
                            ft.Row(
                                [self.scene_button, self.describe_button, self.stop_button],
                                wrap=True,
                                spacing=8,
                            ),
                            ft.Text(
                                f"Say \"{self.settings.wake_word}\" to start, or "
                                "tap Talk and speak. Recording stops when you "
                                "pause; a follow-up question needs no wake word. "
                                "Typing is the fallback without a microphone.",
                                size=11,
                                color=ft.Colors.BLUE_GREY_400,
                            ),
                        ],
                    ),
                    self._card("Alert legend", [self._legend()]),
                ],
            ),
        )

    def _legend(self) -> ft.Control:
        def dot(color: str, text: str) -> ft.Control:
            return ft.Row(
                spacing=8,
                controls=[
                    ft.Container(width=12, height=12, bgcolor=color, border_radius=6),
                    ft.Text(text, size=12, color=ft.Colors.BLUE_GREY_600),
                ],
            )

        return ft.Column(
            spacing=6,
            controls=[
                dot(ft.Colors.RED_500, "under 0.6 m — quick chime + haptic"),
                dot(ft.Colors.AMBER_600, "0.6–1.0 m — one quiet chime"),
                dot(ft.Colors.GREEN_500, "beyond 1.0 m — silent"),
                ft.Text(
                    "Distances come from camera height and tilt — adjust both "
                    "in Settings until the metre readout matches your floor. "
                    "Quiet by default; short ticks, haptics or a spoken "
                    "\"Careful\" are available instead.",
                    size=11,
                    color=ft.Colors.BLUE_GREY_400,
                ),
            ],
        )

    def _card(self, title: str, controls: list[ft.Control]) -> ft.Control:
        return ft.Container(
            padding=14,
            border_radius=CARD_RADIUS,
            bgcolor=ft.Colors.WHITE,
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_100),
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text(
                        title.upper(),
                        size=11,
                        weight=ft.FontWeight.W_700,
                        color=ft.Colors.BLUE_GREY_400,
                    ),
                    *controls,
                ],
            ),
        )

    def _build_layout(self) -> None:
        page = self.page
        page.title = "VisionAssist"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 0
        page.window.width = 1180
        page.window.height = 840
        page.window.min_width = 720
        page.window.min_height = 560
        page.appbar = ft.AppBar(
            title=ft.Row(
                spacing=10,
                controls=[
                    ft.Icon(ft.Icons.ACCESSIBILITY_NEW, color=ft.Colors.BLUE_GREY_900),
                    ft.Text(
                        "VisionAssist",
                        size=20,
                        weight=ft.FontWeight.W_700,
                        color=ft.Colors.BLUE_GREY_900,
                    ),
                ],
            ),
            bgcolor=ft.Colors.WHITE,
            color=ft.Colors.BLUE_GREY_900,
            elevation=1,
            actions=[self.mute_button, self.settings_button],
        )
        page.add(
            ft.SafeArea(
                expand=True,
                content=ft.Container(
                    padding=16,
                    bgcolor=ft.Colors.BLUE_GREY_50,
                    content=ft.Row(
                        expand=True,
                        spacing=16,
                        controls=[self.preview, self._sidebar()],
                    ),
                ),
            )
        )

    # ---------------------------------------------------------------- lifecycle

    def mount(self) -> None:
        self.page.on_disconnect = self._on_disconnect
        # Hands-free fallback for phone use: shake instead of finding a button.
        self.shake = ft.ShakeDetector(on_shake=self._on_shake)
        self.page.services.append(self.shake)
        self.page.update()
        self.page.run_task(self.assistant.start)
        self.page.run_task(self._preview_loop)

    def _on_disconnect(self, event: object) -> None:
        self._closed = True
        try:
            self.assistant.source.stop()
        except Exception:  # pragma: no cover - shutdown must never raise
            pass

    async def _preview_loop(self) -> None:
        while not self._closed:
            frame = self.assistant.source.latest()
            if (
                frame is not None
                and frame.image is not None
                and frame is not self._last_frame
            ):
                self._last_frame = frame
                try:
                    jpeg = await asyncio.to_thread(
                        annotate_jpeg, frame.image, self.assistant.scene
                    )
                except Exception:  # pragma: no cover - drawing must never kill the UI
                    traceback.print_exc()
                    jpeg = b""
                if jpeg:
                    self.preview_image.src = jpeg
                    self.preview_image.visible = True
                    self.placeholder.visible = False
                self._refresh_status()
                try:
                    self.page.update()
                except RuntimeError:  # session torn down while the window closed
                    self._closed = True
                    break
            await asyncio.sleep(PREVIEW_INTERVAL)

    # ------------------------------------------------------------------ status

    def _refresh_status(self) -> None:
        snapshot = self.assistant.snapshot()
        if snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot

        self.status_text.value = str(snapshot["status"])
        nearest = str(snapshot["nearest"])
        self.hazard_text.value = nearest or "path clear"
        self.hazard_text.color = (
            ft.Colors.RED_600
            if "close" in nearest
            else ft.Colors.AMBER_700
            if nearest
            else ft.Colors.GREEN_700
        )
        self.detection_text.value = f"{snapshot['detections']} objects"
        fps = float(snapshot["fps"] or 0.0)
        source = self.assistant.source
        self.camera_text.value = f"{source.name} · {fps:.1f} fps"
        self.answer_text.value = str(snapshot["answer"] or "Ask me what is in front of you.")
        heard = str(snapshot.get("heard") or "")
        self.heard_text.value = f"Heard: {heard}" if heard else "Heard: —"
        listening = bool(snapshot.get("listening"))
        self.talk_button.content = "Listening… tap to stop" if listening else "Talk"
        self.talk_button.icon = ft.Icons.STOP if listening else ft.Icons.MIC

        parts = [f"detector: {self.assistant.detector.name}"]
        parts.append(f"describer: {self.assistant.describer.name}")
        parts.append(f"asr: {self.assistant.asr.name}")
        parts.append(f"clip: {'on' if self.assistant.clip.available() else 'missing'}")
        parts.append(f"mic: {self.assistant.microphone.name}")
        parts.append(f"dist ×{self.assistant.settings.distance_scale:.2f}")
        if self.assistant.settings.surface_height_m > 0:
            parts.append(f"surface {self.assistant.settings.surface_height_m:.2f} m")
        wake_word = str(snapshot.get("wake") or "")
        if wake_word:
            parts.append(f"wake: \"{wake_word}\"")
        if self.tts.error is None:
            parts.append("voice: paradee")
        else:
            parts.append("voice: unavailable")
        if source.error:
            parts.append(source.error)
            self.placeholder_text.value = source.error
        elif snapshot["model_error"]:
            parts.append(str(snapshot["model_error"]))
        self.model_text.value = " · ".join(parts)
        try:
            self.page.update()
        except Exception:  # pragma: no cover - page already closed
            pass

    # ----------------------------------------------------------------- actions

    async def _on_talk(self, event: ft.Event[ft.Control]) -> None:
        await self.assistant.listen()

    async def _on_shake(self, event: ft.Event[ft.Control]) -> None:
        if not self.assistant.listening:
            await self.assistant.listen()

    async def _ask_scene(self, event: ft.Event[ft.Control]) -> None:
        await self.assistant.ask("what is in front of me")

    async def _ask_describe(self, event: ft.Event[ft.Control]) -> None:
        await self.assistant.ask("describe it")

    async def _ask_stop(self, event: ft.Event[ft.Control]) -> None:
        await self.assistant.ask("stop")

    async def _on_send(self, event: ft.Event[ft.Control]) -> None:
        text = (self.command.value or "").strip()
        if not text:
            return
        self.command.value = ""
        self.page.update()
        await self.assistant.ask(text)

    def _toggle_mute(self, event: ft.Event[ft.Control]) -> None:
        self.settings.alerts_enabled = not self.settings.alerts_enabled
        self.assistant.apply_settings(self.settings)
        self.mute_button.icon = (
            ft.Icons.VOLUME_UP if self.settings.alerts_enabled else ft.Icons.VOLUME_OFF
        )
        self.page.update()

    # ----------------------------------------------------------------- settings

    def _voice_options(self) -> list[ft.DropdownOption]:
        try:
            voices = self.tts.voices
        except TtsError:
            voices = {}
        if not voices:
            voices = {DEFAULT_VOICE: "Heart - American, female"}
        return [
            ft.DropdownOption(key=key, text=label) for key, label in voices.items()
        ]

    def _open_settings(self, event: ft.Event[ft.Control]) -> None:
        self.page.show_dialog(self._build_settings_dialog())

    def _build_settings_dialog(self) -> ft.AlertDialog:
        """Constructed separately so tests can validate it without a page."""
        settings = self.settings

        confidence = ft.Slider(
            key="set_confidence",
            min=0.10,
            max=0.90,
            divisions=16,
            value=settings.confidence,
            label="{value}",
        )
        min_size = ft.Slider(
            key="set_min_size",
            min=0.01,
            max=0.30,
            divisions=29,
            value=settings.min_size_ratio,
            label="{value}",
        )
        camera_height = ft.Slider(
            key="set_camera_height",
            min=0.4,
            max=1.8,
            divisions=28,
            value=settings.camera_height_m,
            label="{value} m",
        )
        camera_pitch = ft.Slider(
            key="set_camera_pitch",
            min=0.0,
            max=45.0,
            divisions=45,
            value=settings.camera_pitch_deg,
            label="{value}°",
        )
        surface_height = ft.Slider(
            key="set_surface_height",
            min=0.0,
            max=1.2,
            divisions=24,
            value=settings.surface_height_m,
            label="{value} m",
        )
        alert_near = ft.Slider(
            key="set_alert_near",
            min=0.5,
            max=6.0,
            divisions=55,
            value=settings.alert_near_m,
            label="{value} m",
        )
        alert_close = ft.Slider(
            key="set_alert_close",
            min=0.2,
            max=3.0,
            divisions=28,
            value=settings.alert_close_m,
            label="{value} m",
        )
        def preset_laptop(event: ft.Event[ft.Control]) -> None:
            settings.camera_height_m = 1.1
            settings.surface_height_m = 0.75
            settings.distance_scale = 1.0
            camera_height.value = 1.1
            surface_height.value = 0.75
            distance_scale.value = 1.0
            self.assistant.apply_settings(settings)
            estimate_text.value = "Laptop preset: 1.1 m camera, 0.75 m desk."
            self.page.update()

        def preset_wearable(event: ft.Event[ft.Control]) -> None:
            settings.camera_height_m = 1.0
            settings.surface_height_m = 0.0
            settings.distance_scale = 1.0
            camera_height.value = 1.0
            surface_height.value = 0.0
            distance_scale.value = 1.0
            self.assistant.apply_settings(settings)
            estimate_text.value = "Wearable preset: floor geometry, scale 1.0."
            self.page.update()

        def apply_distance_settings(event: ft.Event[ft.Control]) -> None:
            """Live-apply the geometry sliders; no Save required."""
            settings.camera_height_m = float(camera_height.value or settings.camera_height_m)
            settings.camera_pitch_deg = float(camera_pitch.value or settings.camera_pitch_deg)
            settings.surface_height_m = float(
                surface_height.value
                if surface_height.value is not None
                else settings.surface_height_m
            )
            settings.distance_scale = float(distance_scale.value or settings.distance_scale)
            settings.alert_near_m = float(alert_near.value or settings.alert_near_m)
            settings.clip_threshold = float(
                clip_threshold.value or settings.clip_threshold
            )
            settings.alert_close_m = float(alert_close.value or settings.alert_close_m)
            self.assistant.apply_settings(settings)
            measured = self.assistant.nearest_distance()
            estimate_text.value = (
                f"Current estimate: {measured:.2f} m"
                if measured
                else "Current estimate: no object detected"
            )
            self.page.update()

        def reset_alert_distances(event: ft.Event[ft.Control]) -> None:
            settings.alert_close_m = 0.6
            settings.alert_near_m = 1.0
            alert_close.value = 0.6
            alert_near.value = 1.0
            self.assistant.apply_settings(settings)
            self.page.update()

        distance_scale = ft.Slider(
            key="set_distance_scale",
            min=0.20,
            max=3.0,
            divisions=56,
            value=settings.distance_scale,
            label="{value}×",
        )
        estimate = self.assistant.nearest_distance()
        estimate_text = ft.Text(
            f"Current estimate: {estimate:.2f} m" if estimate else "Current estimate: no object detected",
            size=12,
            color=ft.Colors.BLUE_GREY_600,
        )
        camera_height.on_change_end = apply_distance_settings
        camera_pitch.on_change_end = apply_distance_settings
        surface_height.on_change_end = apply_distance_settings
        distance_scale.on_change_end = apply_distance_settings
        alert_near.on_change_end = apply_distance_settings
        alert_close.on_change_end = apply_distance_settings

        true_distance = ft.TextField(
            key="set_true_distance",
            label="What is the real distance? (m)",
            dense=True,
            width=180,
        )

        def apply_calibration(event: ft.Event[ft.Control]) -> None:
            measured = self.assistant.nearest_distance()
            try:
                actual = float(true_distance.value or "")
            except ValueError:
                estimate_text.value = "Enter a number of metres, e.g. 0.33"
                self.page.update()
                return
            if not measured:
                estimate_text.value = "No object detected — point the camera at one first."
                self.page.update()
                return
            try:
                scale = calibrate_scale(measured, actual, settings.distance_scale)
            except ValueError:
                estimate_text.value = "Distances must be positive."
                self.page.update()
                return
            settings.distance_scale = scale
            self.assistant.apply_settings(settings)
            distance_scale.value = scale
            before = measured
            after = self.assistant.nearest_distance()
            estimate_text.value = (
                f"Applied ×{scale:.2f}: {before:.2f} m → "
                f"{f'{after:.2f} m' if after else '—'}"
            )
            self.page.update()
        corridor = ft.Slider(
            key="set_corridor",
            min=0.10,
            max=1.0,
            divisions=9,
            value=settings.corridor,
            label="{value}",
        )
        alert_volume = ft.Slider(
            key="set_alert_volume",
            min=0.0,
            max=1.0,
            divisions=10,
            value=settings.alert_volume,
            label="{value}",
        )
        speech_volume = ft.Slider(
            key="set_speech_volume",
            min=0.0,
            max=1.0,
            divisions=10,
            value=settings.speech_volume,
            label="{value}",
        )
        speed = ft.Slider(
            key="set_speed",
            min=0.5,
            max=2.0,
            divisions=15,
            value=settings.speed,
            label="{value}",
        )
        voice = ft.Dropdown(
            key="set_voice",
            label="Voice",
            value=settings.voice,
            options=self._voice_options(),
        )
        describer = ft.Dropdown(
            key="set_describer",
            label="Description model",
            value=settings.describer,
            options=[
                ft.DropdownOption(key="auto", text="auto (Qwen if installed)"),
                *[
                    ft.DropdownOption(key=name, text=name)
                    for name in available_backends()
                ],
            ],
        )
        wake = ft.Switch(
            key="set_wake", label="Wake word", value=settings.wake_enabled
        )
        wake_word = ft.TextField(
            key="set_wake_word",
            label="Wake phrase",
            value=settings.wake_word,
            dense=True,
        )
        greet = ft.Switch(
            key="set_greet", label="Greet me at startup", value=settings.greet_enabled
        )
        greet_name = ft.TextField(
            key="set_greet_name",
            label="My name",
            value=settings.greet_name,
            dense=True,
        )
        asr_backend = ft.Dropdown(
            key="set_asr",
            label="Speech recognition",
            value=settings.asr_backend,
            options=[
                ft.DropdownOption(key="parakeet", text="Parakeet 110M (accurate)"),
                *(
                    [ft.DropdownOption(key="zipformer", text="Zipformer (live partials)")]
                    if model_path("asr-zipformer-en", "tokens.txt").is_file()
                    else []
                ),
            ],
        )
        listen_seconds = ft.Slider(
            key="set_listen_seconds",
            min=2.0,
            max=20.0,
            divisions=18,
            value=settings.listen_seconds,
            label="{value} s",
        )
        clip_threshold = ft.Slider(
            key="set_clip_threshold",
            min=0.10,
            max=0.50,
            divisions=40,
            value=settings.clip_threshold,
            label="{value}",
        )
        style = ft.Dropdown(
            key="set_alert_style",
            label="Alert style",
            value=settings.alert_style,
            options=[
                ft.DropdownOption(key=item.value, text=item.label)
                for item in AlertStyle
            ],
        )
        alerts = ft.Switch(
            key="set_alerts", label="Path alerts", value=settings.alerts_enabled
        )
        speech = ft.Switch(
            key="set_speech", label="Speak answers", value=settings.speech_enabled
        )
        haptics = ft.Switch(
            key="set_haptics", label="Haptics", value=settings.haptics_enabled
        )

        def label(text: str) -> ft.Control:
            return ft.Text(text, size=12, color=ft.Colors.BLUE_GREY_500)

        def save(event: ft.Event[ft.Control]) -> None:
            settings.confidence = float(confidence.value or settings.confidence)
            settings.min_size_ratio = float(min_size.value or settings.min_size_ratio)
            settings.camera_height_m = float(camera_height.value or settings.camera_height_m)
            settings.camera_pitch_deg = float(camera_pitch.value or settings.camera_pitch_deg)
            settings.surface_height_m = float(
                surface_height.value
                if surface_height.value is not None
                else settings.surface_height_m
            )
            settings.distance_scale = float(distance_scale.value or settings.distance_scale)
            settings.alert_near_m = float(alert_near.value or settings.alert_near_m)
            settings.clip_threshold = float(
                clip_threshold.value or settings.clip_threshold
            )
            settings.alert_close_m = float(alert_close.value or settings.alert_close_m)
            settings.corridor = float(corridor.value or settings.corridor)
            settings.alert_volume = float(alert_volume.value or 0.0)
            settings.speech_volume = float(speech_volume.value or 0.0)
            settings.speed = float(speed.value or settings.speed)
            settings.voice = voice.value or settings.voice
            settings.describer = describer.value or settings.describer
            settings.wake_enabled = bool(wake.value)
            settings.wake_word = (wake_word.value or settings.wake_word).strip()
            settings.greet_enabled = bool(greet.value)
            settings.greet_name = (greet_name.value or settings.greet_name).strip()
            settings.asr_backend = asr_backend.value or settings.asr_backend
            settings.listen_seconds = float(listen_seconds.value or settings.listen_seconds)
            settings.clip_threshold = float(
                clip_threshold.value or settings.clip_threshold
            )
            settings.alert_style = style.value or settings.alert_style
            settings.alerts_enabled = bool(alerts.value)
            settings.speech_enabled = bool(speech.value)
            settings.haptics_enabled = bool(haptics.value)
            self.assistant.apply_settings(settings)
            self.mute_button.icon = (
                ft.Icons.VOLUME_UP
                if settings.alerts_enabled
                else ft.Icons.VOLUME_OFF
            )
            self.page.pop_dialog()

        async def test_voice(event: ft.Event[ft.Control]) -> None:
            settings.voice = voice.value or settings.voice
            settings.speed = float(speed.value or settings.speed)
            settings.speech_volume = float(speech_volume.value or 0.0)
            self.assistant.apply_settings(settings)
            await self.assistant.say("Vision assist is ready to help.")

        dialog = ft.AlertDialog(
            modal=False,
            title=ft.Text("Settings"),
            content=ft.Container(
                width=420,
                content=ft.Column(
                    tight=True,
                    scroll=ft.ScrollMode.AUTO,
                    spacing=6,
                    controls=[
                        label("Detection confidence"),
                        confidence,
                        label("Minimum object size (share of frame)"),
                        min_size,
                        label("Camera height above floor"),
                        camera_height,
                        ft.Row(
                            [
                                ft.TextButton(
                                    content="Laptop on a desk",
                                    key="preset_laptop",
                                    on_click=preset_laptop,
                                ),
                                ft.TextButton(
                                    content="Wearable",
                                    key="preset_wearable",
                                    on_click=preset_wearable,
                                ),
                            ],
                            spacing=4,
                        ),
                        label("Camera downward tilt"),
                        camera_pitch,
                        label("Objects rest on a surface at (0 = floor, 0.75 = desk)"),
                        surface_height,
                        label("Warn from this distance"),
                        alert_near,
                        label("Close warning at this distance"),
                        alert_close,
                        ft.Row(
                            [
                                ft.TextButton(
                                    content="Reset to 1.0 m / 0.6 m",
                                    on_click=reset_alert_distances,
                                )
                            ],
                            alignment=ft.MainAxisAlignment.START,
                        ),
                        label("Distance scale (calibration)"),
                        distance_scale,
                        estimate_text,
                        ft.Row(
                            [
                                true_distance,
                                ft.FilledButton(
                                    content="Calibrate",
                                    key="apply_calibration",
                                    on_click=apply_calibration,
                                ),
                            ],
                            spacing=8,
                        ),
                        ft.Text(
                            "Place an object at a measured distance, read the "
                            "estimate above, then enter the true value. The "
                            "ground-plane model assumes floor contact; a desk "
                            "webcam needs this correction.",
                            size=11,
                            color=ft.Colors.BLUE_GREY_400,
                        ),
                        label("Alert corridor (0.35 = middle third)"),
                        corridor,
                        label("Alert volume"),
                        alert_volume,
                        label("Speech volume"),
                        speech_volume,
                        label("Speech speed"),
                        speed,
                        voice,
                        describer,
                        wake,
                        wake_word,
                        greet,
                        greet_name,
                        label("Open-vocabulary match threshold"),
                        clip_threshold,
                        label("Speech recognition backend"),
                        asr_backend,
                        label("Recording cap (auto-stops on silence)"),
                        listen_seconds,
                        style,
                        alerts,
                        speech,
                        haptics,
                        ft.Row(
                            [ft.TextButton(content="Test voice", on_click=test_voice)],
                            alignment=ft.MainAxisAlignment.START,
                        ),
                    ],
                ),
            ),
            actions=[
                ft.TextButton(content="Cancel", on_click=lambda event: self.page.pop_dialog()),
                ft.FilledButton(content="Save", key="save_settings", on_click=save),
            ],
        )
        return dialog


def main(page: ft.Page) -> None:
    app = VisionAssistApp(page)
    app.mount()
