# Author: David Li, SISHK

"""Core domain types: plain data, no I/O, no Flet, no ONNX."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Box:
    """An axis-aligned pixel box."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
            score=self.score,
        )


@dataclass(frozen=True)
class Detection:
    """One labelled box from a detector."""

    box: Box
    label: str
    score: float = 1.0


@dataclass
class Track:
    """A detection with a stable identity across frames."""

    id: int
    box: Box
    label: str = ""
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    last_seen: float = 0.0

    def copy_box(self, box: Box) -> None:
        self.box = box


class Band(str, Enum):
    """Coarse distance band. Deliberately not metres."""

    FAR = "far"
    NEAR = "near"
    CLOSE = "close"

    @property
    def speech(self) -> str:
        return {
            Band.FAR: "far",
            Band.NEAR: "a few steps away",
            Band.CLOSE: "close",
        }[self]


class Vertical(str, Enum):
    """Where the object sits relative to the body."""

    FLOOR = "floor"
    WAIST = "waist"
    HEAD = "head"

    @property
    def speech(self) -> str:
        return {
            Vertical.FLOOR: "low",
            Vertical.WAIST: "at waist height",
            Vertical.HEAD: "at head height",
        }[self]


class AlertStyle(str, Enum):
    """How path warnings are delivered.

    ``SOFT`` is the default: a quiet two-note chime, never a buzzer.
    ``VOICE`` trades rhythm for one short word when something is close.
    ``HAPTIC`` is silent (phone in hand or pocket).
    """

    SOFT = "soft"
    TICKS = "ticks"
    VOICE = "voice"
    HAPTIC = "haptic"
    OFF = "off"

    @property
    def label(self) -> str:
        return {
            AlertStyle.SOFT: "Soft chime (recommended)",
            AlertStyle.TICKS: "Short ticks",
            AlertStyle.VOICE: "Quiet + short voice",
            AlertStyle.HAPTIC: "Haptics only",
            AlertStyle.OFF: "Off",
        }[self]


HALF_METRE_WORDS = {
    0.5: "half a metre",
    1.0: "one metre",
    1.5: "one and a half metres",
    2.0: "two metres",
    2.5: "two and a half metres",
    3.0: "three metres",
}


def distance_speech(distance: float | None) -> str:
    """Human phrasing that never pretends to centimetre accuracy."""
    if distance is None:
        return ""
    if distance <= 3.0:
        words = HALF_METRE_WORDS.get(round(distance * 2) / 2)
        if words:
            return f"about {words} away"
    return f"about {int(round(distance))} metres away"


POSITION_SPEECH = {
    "left": "on your left",
    "right": "on your right",
    "ahead": "ahead",
}


@dataclass(frozen=True)
class Hazard:
    """Something in the walking corridor worth warning about."""

    track_id: int
    label: str
    band: Band
    pan: float
    vertical: Vertical
    box: Box
    distance_m: float | None = None
    very_close: bool = False

    @property
    def position(self) -> str:
        if self.pan < -0.15:
            return "left"
        if self.pan > 0.15:
            return "right"
        return "ahead"

    @property
    def range_speech(self) -> str:
        if self.very_close:
            return "very close"
        return distance_speech(self.distance_m) or self.band.speech

    @property
    def speech(self) -> str:
        name = self.label or "something"
        return f"{name}, {self.range_speech}, {POSITION_SPEECH[self.position]}"


@dataclass(frozen=True)
class AlertEvent:
    """A change the alert layer may turn into a sound or a haptic pulse."""

    kind: str  # "enter" | "change" | "clear"
    hazard: Hazard
    previous: Band | None = None


@dataclass(frozen=True)
class SceneItem:
    """One described object, for the UI and for spoken summaries."""

    label: str
    band: Band
    position: str
    ratio: float
    box: Box
    distance_m: float | None = None
    very_close: bool = False

    @property
    def range_speech(self) -> str:
        if self.very_close:
            return "very close"
        return distance_speech(self.distance_m) or self.band.speech

    @property
    def speech(self) -> str:
        return f"{self.label} {self.range_speech} {POSITION_SPEECH[self.position]}"


@dataclass
class FrameScene:
    """Everything derived from one frame, handed to the UI."""

    detections: list[Detection] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    items: list[SceneItem] = field(default_factory=list)
    timestamp: float = 0.0
