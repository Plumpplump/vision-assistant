"""Qwen3.5-0.8B vision-language adapter.

The runner does the ONNX work; this class is the ``Describer`` the rest of the
app sees. Prompt policy lives here: one short question, one short sentence
answer. The template describer remains the fallback when the bundle is absent
or generation fails.
"""

from __future__ import annotations

from .base import Capabilities, DescribeRequest, Describer, DescriberError
from .qwen_runner import (
    DEFAULT_QUESTION,
    QwenVlRunner,
    bundle_available,
    clean_description,
    find_model_dir,
)

__all__ = ["QwenVlmDescriber", "bundle_available", "find_model_dir", "clean_description"]


class QwenVlmDescriber:
    name = "qwen3.5-0.8b"
    capabilities = Capabilities(accepts_image=True, max_sentences=1)

    def __init__(self) -> None:
        self._runner = QwenVlRunner()
        self._error: str | None = None
        if self._runner.model_dir is None:
            self._error = (
                "Qwen3.5 bundle is missing under src/assets/models/qwen3.5-0.8b "
                "(fetch with `python tools/fetch_models.py --vision`)."
            )

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        if self._error:
            raise DescriberError(self._error)
        try:
            self._runner.load()
        except Exception as exc:
            self._error = f"Qwen VLM failed to load: {exc}"
            raise DescriberError(self._error) from exc

    def describe(self, request: DescribeRequest) -> str:
        if self._error:
            raise DescriberError(self._error)
        hint = (request.hint or "").strip()
        # Short prompt on purpose: a 0.8B model rambles as soon as it is asked
        # for detail, and the answer is spoken.
        asked = (request.question or "").strip()
        if asked and not asked.lower().startswith(("describe", "what is this")):
            question = asked
        else:
            question = (
                f"What is this {hint}? Describe it briefly." if hint else DEFAULT_QUESTION
            )
        try:
            description = self._runner.describe(request.image, question)
        except Exception as exc:  # generation failures must degrade, not crash
            self._error = f"Qwen VLM inference failed: {exc}"
            raise DescriberError(self._error) from exc
        text = description.text.strip()
        if not text:
            raise DescriberError("Qwen VLM returned an empty description.")
        print(
            f"[vlm] {description.tokens} tokens, "
            f"{description.latency_ms / 1000:.2f}s, "
            f"{description.image_tokens} image tokens -> {text!r}",
            flush=True,
        )
        return text
