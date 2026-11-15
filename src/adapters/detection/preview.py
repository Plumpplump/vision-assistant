"""Annotated preview frames for the demo UI."""

from __future__ import annotations

import cv2
import numpy as np

from domain.types import Band, FrameScene

COLOR_HAZARD = (60, 60, 235)
COLOR_OTHER = (90, 190, 120)
COLOR_TEXT = (255, 255, 255)
JPEG_QUALITY = 72


def annotate(image: np.ndarray, scene: FrameScene) -> np.ndarray:
    canvas = image.copy()
    for detection in scene.detections:
        x1, y1, x2, y2 = (
            int(detection.box.x1),
            int(detection.box.y1),
            int(detection.box.x2),
            int(detection.box.y2),
        )
        color = COLOR_OTHER
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{detection.label} {detection.score:.2f}"
        cv2.putText(
            canvas,
            caption,
            (x1 + 2, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    for hazard in scene.hazards:
        x1, y1, x2, y2 = (
            int(hazard.box.x1),
            int(hazard.box.y1),
            int(hazard.box.x2),
            int(hazard.box.y2),
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_HAZARD, 3)
        if hazard.distance_m is not None:
            caption = f"{hazard.label or 'object'} {hazard.distance_m:.1f} m"
        else:
            caption = f"{hazard.label or 'object'} {hazard.band.value}"
        cv2.putText(
            canvas,
            caption,
            (x1 + 2, min(canvas.shape[0] - 6, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_HAZARD,
            1,
            cv2.LINE_AA,
        )
    return canvas


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:  # pragma: no cover - OpenCV failure is not actionable
        return b""
    return buffer.tobytes()


def annotate_jpeg(image: np.ndarray, scene: FrameScene) -> bytes:
    return encode_jpeg(annotate(image, scene))


def band_color(band: Band) -> tuple[int, int, int]:
    return {
        Band.CLOSE: (60, 60, 235),
        Band.NEAR: (40, 170, 235),
        Band.FAR: (120, 200, 120),
    }[band]
