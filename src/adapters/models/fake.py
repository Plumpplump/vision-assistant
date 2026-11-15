"""Deterministic stand-in for a vision-language model."""

from __future__ import annotations

from .base import Capabilities, DescribeRequest, Describer

DEFAULT_ANSWER = "I cannot describe what I see in this demo build."


class FakeDescriber:
    """Answers from the detector hint, so the UI flow is testable offline."""

    name = "fake"
    capabilities = Capabilities(accepts_image=False, max_sentences=2)

    def __init__(self) -> None:
        self._loaded = False

    @property
    def error(self) -> str | None:
        return None

    def load(self) -> None:
        self._loaded = True

    def describe(self, request: DescribeRequest) -> str:
        self.load()
        hint = (request.hint or "").strip()
        if hint:
            return f"That looks like a {hint}."
        return DEFAULT_ANSWER
