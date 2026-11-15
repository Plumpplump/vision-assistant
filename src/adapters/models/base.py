# Author: David Li, SISHK

"""Vision-language backend protocol.

Nothing outside this package names a model. Adding one means writing an
adapter here and registering it in :mod:`adapters.models.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DescriberError(Exception):
    """Raised when a backend cannot answer (missing bundle, timeout, crash)."""


@dataclass(frozen=True)
class Capabilities:
    accepts_image: bool = True
    max_sentences: int = 2
    supports_follow_up: bool = False


@dataclass(frozen=True)
class DescribeRequest:
    """One question about one crop or frame (BGR, as OpenCV returns it)."""

    image: np.ndarray
    question: str
    hint: str | None = None


class Describer(Protocol):
    name: str
    capabilities: Capabilities

    @property
    def error(self) -> str | None: ...

    def load(self) -> None: ...

    def describe(self, request: DescribeRequest) -> str: ...
