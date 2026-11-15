"""UI layer: Flet app + audio output."""

from .app import VisionAssistApp, main
from .audio import FletSpeaker

__all__ = ["FletSpeaker", "VisionAssistApp", "main"]
