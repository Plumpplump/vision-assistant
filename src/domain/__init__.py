"""Pure-python domain layer: types, distance, hazards, scene, intents, settings."""

from .distance import (
    CameraModel,
    band_from_distance,
    band_from_ratio,
    calibrate_scale,
    pan_of,
    position_word,
    size_ratio,
    vertical_of,
)
from .hazards import HazardMonitor
from .intents import Intent, Request, parse, strip_wake_word
from .scene import describe_items, describe_scene, scene_items, speakable
from .settings import Settings
from .types import (
    AlertEvent,
    AlertStyle,
    Band,
    Box,
    Detection,
    FrameScene,
    Hazard,
    SceneItem,
    Track,
    Vertical,
    distance_speech,
)

__all__ = [
    "AlertEvent",
    "AlertStyle",
    "Band",
    "Box",
    "CameraModel",
    "Detection",
    "FrameScene",
    "Hazard",
    "HazardMonitor",
    "Intent",
    "Request",
    "SceneItem",
    "Settings",
    "Track",
    "Vertical",
    "band_from_distance",
    "band_from_ratio",
    "calibrate_scale",
    "describe_items",
    "distance_speech",
    "describe_scene",
    "pan_of",
    "parse",
    "position_word",
    "scene_items",
    "size_ratio",
    "strip_wake_word",
    "speakable",
    "vertical_of",
]
