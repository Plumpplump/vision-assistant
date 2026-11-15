import numpy as np
import pytest

from adapters.models.clip import ClipSearch

clip = ClipSearch()
pytestmark = pytest.mark.skipif(
    not clip.available(), reason=clip.error or "MobileCLIP2 bundle missing"
)


def solid(colour: tuple[int, int, int], size: int = 256) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = colour
    return image


def test_text_embedding_is_normalised_and_cached():
    embedding = clip.embed_text("a cup")
    assert embedding.shape == (512,)
    assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5
    assert clip.embed_text("a cup") is embedding


def test_red_image_prefers_the_red_prompt():
    image = solid((0, 0, 255))  # BGR red
    red = clip.score(image, clip.embed_text("a red image"))
    blue = clip.score(image, clip.embed_text("a blue image"))
    assert red > blue
    assert red > clip.threshold


def test_best_strip_localises_a_red_left_thirds_image():
    image = np.zeros((256, 384, 3), dtype=np.uint8)
    image[:, :128] = (0, 0, 255)     # left third red
    image[:, 128:] = (255, 0, 0)     # rest blue
    position, score = clip.best_strip(image, clip.embed_text("a red image"))
    assert position == "left"
    assert score > clip.threshold
