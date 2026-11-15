import numpy as np
import pytest

from adapters.detection import YoloDetector

detector = YoloDetector()
pytestmark = pytest.mark.skipif(
    detector.error is not None, reason=detector.error or "model missing"
)


def test_labels_come_from_the_bundled_config():
    assert len(detector.labels) == 80
    assert detector.labels[0] == "person"
    assert detector.labels[41] == "cup"


def test_blank_frame_produces_no_detections():
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert detector.detect(image) == []


def test_detections_are_clipped_to_the_frame():
    # Noise cannot be asserted to yield boxes, but anything returned must be sane.
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for detection in detector.detect(image):
        assert 0 <= detection.box.x1 <= detection.box.x2 <= 320
        assert 0 <= detection.box.y1 <= detection.box.y2 <= 240
        assert detection.label in detector.labels.values()
