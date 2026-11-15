# Author: David Li, SISHK

"""User settings: plain data plus JSON persistence, no Flet types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

try:  # optional: only used to pick a per-user config directory
    from platformdirs import user_config_dir
except ImportError:  # pragma: no cover
    user_config_dir = None  # type: ignore[assignment]

APP_NAME = "vision-assist"
SETTINGS_FILE = "settings.json"


def default_path() -> Path:
    if user_config_dir is not None:
        return Path(user_config_dir(APP_NAME)) / SETTINGS_FILE
    return Path.home() / f".{APP_NAME}" / SETTINGS_FILE


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Settings:
    # camera
    camera_index: int = 0
    # detection
    confidence: float = 0.45
    min_size_ratio: float = 0.03
    # camera calibration (metres / degrees)
    camera_height_m: float = 1.1
    camera_pitch_deg: float = 20.0
    camera_hfov_deg: float = 65.0
    # Height of the surface that described objects rest on (0 = floor,
    # 0.75 = a typical desk). Walking hazards always use the floor.
    surface_height_m: float = 0.75
    distance_scale: float = 1.0
    # alerts
    alerts_enabled: bool = True
    alert_style: str = "soft"
    alert_volume: float = 0.45
    alert_close_m: float = 0.6
    alert_near_m: float = 1.0
    corridor: float = 0.35
    haptics_enabled: bool = True
    # speech
    wake_enabled: bool = True
    wake_word: str = "hey vision"
    greet_enabled: bool = True
    greet_name: str = "David"
    follow_up_seconds: float = 6.0
    asr_backend: str = "parakeet"
    listen_seconds: float = 8.0
    speech_enabled: bool = True
    speech_volume: float = 1.0
    voice: str = "af_heart"
    speed: float = 1.0
    # model selection
    describer: str = "auto"
    clip_threshold: float = 0.22

    def normalized(self) -> "Settings":
        self.camera_index = int(self.camera_index)
        self.confidence = _clamp(float(self.confidence), 0.05, 0.95)
        self.min_size_ratio = _clamp(float(self.min_size_ratio), 0.005, 0.5)
        self.camera_height_m = _clamp(float(self.camera_height_m), 0.3, 2.0)
        self.camera_pitch_deg = _clamp(float(self.camera_pitch_deg), 0.0, 60.0)
        self.camera_hfov_deg = _clamp(float(self.camera_hfov_deg), 30.0, 120.0)
        self.surface_height_m = _clamp(float(self.surface_height_m), 0.0, 1.5)
        self.distance_scale = _clamp(float(self.distance_scale), 0.1, 10.0)
        self.alert_close_m = _clamp(float(self.alert_close_m), 0.2, 5.0)
        self.alert_near_m = _clamp(float(self.alert_near_m), 0.4, 12.0)
        if self.alert_near_m <= self.alert_close_m:
            self.alert_near_m = self.alert_close_m + 0.5
        self.alert_volume = _clamp(float(self.alert_volume), 0.0, 1.0)
        self.corridor = _clamp(float(self.corridor), 0.1, 1.0)
        self.speech_volume = _clamp(float(self.speech_volume), 0.0, 1.0)
        self.speed = _clamp(float(self.speed), 0.5, 2.0)
        self.listen_seconds = _clamp(float(self.listen_seconds), 1.0, 30.0)
        self.follow_up_seconds = _clamp(float(self.follow_up_seconds), 0.0, 30.0)
        self.clip_threshold = _clamp(float(self.clip_threshold), 0.10, 0.60)
        return self

    def save(self, path: Path | None = None) -> Path:
        target = path or default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        target = path or default_path()
        if not target.is_file():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {field.name for field in fields(cls)}
        defaults = cls()
        # A hand-edited or older file can contain a non-numeric value. Coerce
        # what is unambiguous and drop only the fields that cannot be used,
        # rather than refusing to start with a corrupt settings file.
        values: dict[str, object] = {}
        for key, value in raw.items():
            if key not in known:
                continue
            default = getattr(defaults, key)
            try:
                if isinstance(default, bool):
                    if not isinstance(value, bool):
                        raise TypeError(key)
                elif isinstance(default, int):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
                elif isinstance(default, str) and not isinstance(value, str):
                    raise TypeError(key)
            except (TypeError, ValueError):
                continue
            values[key] = value
        try:
            return cls(**values).normalized()
        except (TypeError, ValueError):
            return cls()
