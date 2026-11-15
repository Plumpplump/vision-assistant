"""Construct the settings dialog without a Flet session.

Control kwargs are only validated when a control is built, so this catches
typos that a boot smoke test cannot reach (the dialog opens on click).
"""

from __future__ import annotations

import flet as ft
import pytest

from adapters.models.fake import FakeDescriber
from adapters.speech import ParadeeTts
from app.assistant import Assistant
from domain import Settings
from ui.app import VisionAssistApp


class StubSource:
    name = "stub"
    error = None

    def start(self):
        return None

    def stop(self):
        return None

    def latest(self):
        return None


class StubDetector:
    name = "stub"
    error = None
    confidence = 0.5

    def load(self):
        return None

    def detect(self, image):
        return []


@pytest.fixture
def app() -> VisionAssistApp:
    instance = object.__new__(VisionAssistApp)
    instance.settings = Settings()
    instance.tts = ParadeeTts()
    instance.assistant = Assistant(
        settings=instance.settings,
        source=StubSource(),
        detector=StubDetector(),
        describer=FakeDescriber(),
    )
    return instance


def test_settings_dialog_builds(app: VisionAssistApp):
    dialog = app._build_settings_dialog()
    assert isinstance(dialog, ft.AlertDialog)
    assert dialog.actions
    assert dialog.content is not None


def test_settings_dialog_offers_every_alert_style(app: VisionAssistApp):
    dialog = app._build_settings_dialog()
    controls = dialog.content.content.controls

    def walk(items):
        for item in items:
            if isinstance(item, ft.Dropdown) and item.key == "set_alert_style":
                yield item
            for attribute in ("controls", "content"):
                nested = getattr(item, attribute, None)
                if isinstance(nested, list):
                    yield from walk(nested)
                elif isinstance(nested, ft.Control):
                    yield from walk([nested])

    dropdowns = list(walk(controls))
    assert dropdowns, "alert style dropdown not found"
    keys = [option.key for option in dropdowns[0].options]
    assert keys == ["soft", "ticks", "voice", "haptic", "off"]
    assert dropdowns[0].value == "soft"
