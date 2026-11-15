"""Pick a frame source from configuration.

``VISIONASSIST_SOURCE`` values:
    unset / "0" / "1" ...   OpenCV camera by index
    "demo"                  synthetic scene, no camera needed
    a path to a file        video file or a folder of stills
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import Frame, FrameSource, UnavailableSource
from .demo_source import DemoSource
from .opencv_source import OpenCVSource

__all__ = ["Frame", "FrameSource", "UnavailableSource", "create_source"]


def create_source(camera_index: int = 0) -> FrameSource:
    raw = (os.getenv("VISIONASSIST_SOURCE") or "").strip()
    if not raw:
        return OpenCVSource(camera_index)
    if raw.lower() == "demo":
        return DemoSource()
    if raw.isdigit():
        return OpenCVSource(int(raw))
    path = Path(raw).expanduser()
    if path.is_dir():
        return UnavailableSource(
            f"{path} is a folder; folder replay is not enabled in this build."
        )
    if path.is_file():
        return OpenCVSource(str(path))
    return UnavailableSource(f"Source {raw!r} does not exist.")
