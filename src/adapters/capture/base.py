"""Frame sources: one thread per source, UI pulls the latest frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    width: int
    height: int
    timestamp: float
    fps: float
    image: np.ndarray | None = None


class FrameSource(Protocol):
    """Runs a capture thread; never touches Flet controls."""

    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest(self) -> Frame | None: ...

    @property
    def error(self) -> str | None: ...


class UnavailableSource:
    """Placeholder when the configured source does not exist."""

    def __init__(self, reason: str) -> None:
        self.name = "unavailable"
        self._reason = reason

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> Frame | None:
        return None

    @property
    def error(self) -> str | None:
        return self._reason
