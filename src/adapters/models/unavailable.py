# Author: David Li, SISHK

"""Backend that reports why it cannot answer instead of inventing text."""

from __future__ import annotations

from .base import Capabilities, DescribeRequest, Describer, DescriberError


class UnavailableDescriber:
    name = "unavailable"
    capabilities = Capabilities(accepts_image=False)

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def error(self) -> str | None:
        return self._reason

    def load(self) -> None:
        raise DescriberError(self._reason)

    def describe(self, request: DescribeRequest) -> str:
        raise DescriberError(self._reason)
