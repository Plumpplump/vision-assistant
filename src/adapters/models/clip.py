"""Open-vocabulary search with MobileCLIP2-B ONNX.

Two static towers (image + text), no generation: embed the query text once,
then score frames with a cosine similarity. This is what lets "find my keys" work
for objects YOLO's 80 COCO classes do not contain.

Bundle: MobileCLIP2-B from the sibling ``schooluniform`` project (577 MB:
345 MB image tower + 254 MB text tower + tokenizer). Measured on a real photo:
cup image scores 0.305 for "a photo of a cup", 0.240 for "a spoon", 0.139 for
"a phone" — enough separation for a 0.22 threshold.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from adapters.paths import model_path

MODEL_NAME = "mobileclip2-b"
CONTEXT_LENGTH = 77
PAD_ID = 49407
DEFAULT_THRESHOLD = 0.22
DEFAULT_IMAGE_SIZE = 224

PROMPTS = (
    "a photo of a {target}",
    "a {target}",
    "the {target}",
    "a close-up photo of a {target}",
)


class ClipError(Exception):
    pass


def _preprocess(image: np.ndarray | bytes, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    if isinstance(image, bytes):
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ClipError("could not decode image bytes")
        image = decoded
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(
        image,
        (max(size, int(round(width * scale))), max(size, int(round(height * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    top = (resized.shape[0] - size) // 2
    left = (resized.shape[1] - size) // 2
    crop = resized[top : top + size, left : left + size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))[np.newaxis, ...]


class ClipSearch:
    name = MODEL_NAME

    def __init__(
        self,
        directory: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.directory = Path(directory) if directory else model_path(MODEL_NAME)
        self.threshold = threshold
        self._image: ort.InferenceSession | None = None
        self._text: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._lock = threading.Lock()
        self._text_cache: dict[str, np.ndarray] = {}

    @property
    def error(self) -> str | None:
        required = ("image.onnx", "text.onnx", "tokenizer.json")
        missing = [name for name in required if not (self.directory / name).is_file()]
        if missing:
            return (
                f"MobileCLIP2 bundle incomplete under {self.directory}: "
                + ", ".join(missing)
            )
        return None

    def available(self) -> bool:
        return self.error is None

    def load(self) -> None:
        with self._lock:
            if self._image is not None:
                return
            if self.error:
                raise ClipError(self.error)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = max(1, min(4, __import__("os").cpu_count() or 1))
            self._image = ort.InferenceSession(
                str(self.directory / "image.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._text = ort.InferenceSession(
                str(self.directory / "text.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))

    def _text_ids(self, prompt: str) -> np.ndarray:
        assert self._tokenizer is not None
        ids = self._tokenizer.encode(prompt).ids[:CONTEXT_LENGTH]
        ids = ids + [PAD_ID] * (CONTEXT_LENGTH - len(ids))
        return np.array([ids], dtype=np.int64)

    def embed_text(self, target: str) -> np.ndarray:
        """Average of a few prompt templates, L2-normalised."""
        key = target.strip().lower()
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        self.load()
        assert self._text is not None
        vectors = []
        for template in PROMPTS:
            output = self._text.run(
                None, {"input_ids": self._text_ids(template.format(target=key))}
            )[0][0]
            vectors.append(output)
        features = np.mean(np.stack(vectors), axis=0)
        features /= np.linalg.norm(features) or 1.0
        self._text_cache[key] = features
        return features

    def encode_image(self, image: np.ndarray | bytes) -> np.ndarray:
        self.load()
        assert self._image is not None
        tensor = _preprocess(image)
        with self._lock:
            features = np.asarray(
                self._image.run(None, {"pixel_values": tensor})[0]
            )[0]
        norm = np.linalg.norm(features)
        return features / norm if norm else features

    def score(self, image: np.ndarray | bytes, text_embedding: np.ndarray) -> float:
        return float(np.dot(self.encode_image(image), text_embedding))

    def best_strip(
        self, image: np.ndarray, text_embedding: np.ndarray
    ) -> tuple[str, float]:
        """Score left/center/right thirds and return the best (position, score)."""
        height, width = image.shape[:2]
        third = max(1, width // 3)
        strips = {
            "left": image[:, : third + third // 4],
            "ahead": image[:, third - third // 4 : 2 * third + third // 4],
            "right": image[:, 2 * third - third // 4 :],
        }
        best_position, best_score = "ahead", -1.0
        for position, crop in strips.items():
            value = self.score(crop, text_embedding)
            if value > best_score:
                best_position, best_score = position, value
        return best_position, best_score
