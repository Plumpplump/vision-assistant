"""Registry: model name -> adapter factory.

Swap the "brain" with ``VISIONASSIST_DESCRIBER=<name>`` or the settings
dropdown. Unknown or broken backends degrade to a spoken explanation, never a
crash.
"""

from __future__ import annotations

import os
from typing import Callable

from .base import Describer
from .unavailable import UnavailableDescriber

BACKEND_ENV = "VISIONASSIST_DESCRIBER"
DEFAULT_BACKEND = "fake"

_FACTORIES: dict[str, Callable[[], Describer]] = {}
_ALIASES: dict[str, str] = {}


def register(name: str, factory: Callable[[], Describer], *aliases: str) -> None:
    _FACTORIES[name] = factory
    for alias in aliases:
        _ALIASES[alias] = name


def available_backends() -> list[str]:
    return sorted(_FACTORIES)


def create_describer(name: str | None = None) -> Describer:
    requested = (name or os.getenv(BACKEND_ENV) or DEFAULT_BACKEND).strip().lower()
    key = _ALIASES.get(requested, requested)
    if key == "auto":
        key = _pick_auto()
    factory = _FACTORIES.get(key)
    if factory is None:
        return UnavailableDescriber(
            f"Unknown vision backend {requested!r}. "
            f"Available: {', '.join(available_backends())}"
        )
    try:
        return factory()
    except Exception as exc:  # pragma: no cover - defensive
        return UnavailableDescriber(f"Backend {requested!r} failed to start: {exc}")


def _pick_auto() -> str:
    try:
        from .qwen_vlm import bundle_available
    except ImportError:
        return DEFAULT_BACKEND
    return "qwen3.5-0.8b" if bundle_available() else DEFAULT_BACKEND


def _install() -> None:
    from .fake import FakeDescriber

    register("fake", FakeDescriber, "demo", "template")
    try:
        from .qwen_vlm import QwenVlmDescriber
    except ImportError:  # optional heavy adapter
        return
    register("qwen3.5-0.8b", QwenVlmDescriber, "qwen", "vlm")


_install()
