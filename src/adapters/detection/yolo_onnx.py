# Author: David Li, SISHK

"""YOLO11n ONNX detector over all 80 COCO classes.

Adapted from the sibling ``schooluniform`` project, which only kept the person
class. Here we decode every class and let the caller filter, so the same graph
serves both "is anything in my way" and "is there a cup".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from adapters.paths import model_path
from domain.types import Box, Detection

INPUT_SIZE = 640
DEFAULT_CONFIDENCE = 0.45
DEFAULT_IOU = 0.45
PAD_VALUE = 114
MAX_DETECTIONS = 40

# Household / navigation classes. Anything else (kites, zebras) is noise for a
# walking assistant; pass allowed=None to keep all 80.
DEFAULT_ALLOWED: frozenset[str] = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "chair",
        "couch",
        "dining table",
        "bed",
        "toilet",
        "sink",
        "refrigerator",
        "oven",
        "microwave",
        "tv",
        "laptop",
        "keyboard",
        "cell phone",
        "book",
        "clock",
        "vase",
        "scissors",
        "bottle",
        "cup",
        "bowl",
        "fork",
        "knife",
        "spoon",
        "backpack",
        "handbag",
        "suitcase",
        "potted plant",
        "teddy bear",
    }
)


def find_model_path() -> Path | None:
    override = os.getenv("VISIONASSIST_YOLO_MODEL")
    candidates = [Path(override)] if override else []
    candidates.append(model_path("yolo11n", "yolo11n.onnx"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_labels(model_file: Path) -> dict[int, str]:
    config = model_file.with_name("config.json")
    if not config.is_file():
        return {}
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    id2label = raw.get("id2label", raw)
    labels: dict[int, str] = {}
    for key, value in id2label.items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return labels


def letterbox(
    image: np.ndarray, size: int = INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect, pad to a square. Returns canvas, scale, pad_x, pad_y."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = DEFAULT_IOU) -> list[int]:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class YoloDetector:
    name = "yolo11n"

    def __init__(
        self,
        model_file: str | Path | None = None,
        input_size: int = INPUT_SIZE,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        allowed: frozenset[str] | None = DEFAULT_ALLOWED,
    ) -> None:
        self.model_path = Path(model_file) if model_file else find_model_path()
        self.input_size = input_size
        self.confidence = confidence
        self.iou = iou
        self.allowed = allowed
        self.labels: dict[int, str] = (
            load_labels(self.model_path) if self.model_path else {}
        )
        self._session: ort.InferenceSession | None = None
        self._input_name = "images"

    @property
    def error(self) -> str | None:
        if self.model_path is None:
            return (
                "YOLO11n model not found. Expected "
                "src/assets/models/yolo11n/yolo11n.onnx or VISIONASSIST_YOLO_MODEL."
            )
        return None

    def load(self) -> None:
        if self._session is not None:
            return
        if self.model_path is None:
            raise RuntimeError(self.error)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.load()
        if self._session is None:  # pragma: no cover - load() raises
            return []

        height, width = image.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(image, self.input_size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        prediction = np.asarray(
            self._session.run(None, {self._input_name: blob})[0]
        )[0]
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T  # tolerate (8400, 84) exports

        class_scores = prediction[:, 4:]
        best_class = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(class_scores.shape[0]), best_class]
        mask = best_score >= self.confidence
        if not mask.any():
            return []

        raw = prediction[mask, :4]
        scores = best_score[mask]
        classes = best_class[mask]

        xyxy = np.empty_like(raw)
        xyxy[:, 0] = raw[:, 0] - raw[:, 2] / 2.0
        xyxy[:, 1] = raw[:, 1] - raw[:, 3] / 2.0
        xyxy[:, 2] = raw[:, 0] + raw[:, 2] / 2.0
        xyxy[:, 3] = raw[:, 1] + raw[:, 3] / 2.0
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

        detections: list[Detection] = []
        for class_id in np.unique(classes):
            indices = np.where(classes == class_id)[0]
            label = self.labels.get(int(class_id), f"class {int(class_id)}")
            if self.allowed is not None and label not in self.allowed:
                continue
            for keep in nms(xyxy[indices], scores[indices], self.iou):
                position = indices[keep]
                box = Box(
                    x1=float(xyxy[position, 0]),
                    y1=float(xyxy[position, 1]),
                    x2=float(xyxy[position, 2]),
                    y2=float(xyxy[position, 3]),
                    score=float(scores[position]),
                ).clipped(width, height)
                if box.width > 1 and box.height > 1:
                    detections.append(
                        Detection(box=box, label=label, score=float(scores[position]))
                    )

        detections.sort(key=lambda detection: detection.score, reverse=True)
        return detections[:MAX_DETECTIONS]
