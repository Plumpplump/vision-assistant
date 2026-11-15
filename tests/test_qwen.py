# Author: David Li, SISHK

import numpy as np
import pytest

from adapters.models import create_describer
from adapters.models.base import DescribeRequest
from adapters.models.qwen_runner import (
    bundle_available,
    build_prompt,
    clean_description,
)

pytestmark = []


def test_auto_backend_uses_qwen_when_the_bundle_exists():
    describer = create_describer("auto")
    expected = "qwen3.5-0.8b" if bundle_available() else "fake"
    assert describer.name == expected


def test_prompt_is_short_and_asks_for_one_sentence():
    prompt = build_prompt("What is this cup? Describe it briefly.")
    assert "one short sentence" in prompt
    assert prompt.count("<|image_pad|>") == 1
    assert prompt.rstrip().endswith("</think>")


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("The image shows a white cup on a table. It is empty.", "A white cup on a table."),
        ("a red chair", "A red chair."),
        ("  Cold metal    surface  ", "Cold metal surface."),
        (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
            "twentyone twentytwo",
            "One two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty.",
        ),
    ],
)
def test_clean_description(raw, cleaned):
    assert clean_description(raw) == cleaned


@pytest.mark.skipif(not bundle_available(), reason="Qwen bundle not installed")
def test_qwen_describes_an_image_briefly():
    describer = create_describer("auto")
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (60, 90, 140)
    text = describer.describe(
        DescribeRequest(image=image, question="describe", hint=None)
    )
    assert text
    assert len(text.split()) <= 22
    assert text[0].isupper()
