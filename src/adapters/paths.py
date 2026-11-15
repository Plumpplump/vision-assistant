# Author: David Li, SISHK

"""Path resolution for bundled assets and models.

Flet sets ``FLET_ASSETS_DIR`` when the app is packaged; during ``flet run`` the
assets live next to this package under ``src/assets``. Tests can point
``VISIONASSIST_MODELS_DIR`` at a fixture directory.
"""

from __future__ import annotations

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = SRC_DIR / "assets" / "models"


def assets_dir() -> Path:
    override = os.getenv("FLET_ASSETS_DIR")
    if override:
        return Path(override)
    return SRC_DIR / "assets"


def models_dir() -> Path:
    override = os.getenv("VISIONASSIST_MODELS_DIR")
    if override:
        return Path(override)
    return assets_dir() / "models"


def model_path(*parts: str) -> Path:
    return models_dir().joinpath(*parts)
