# Author: David Li, SISHK

"""Qwen3.5-0.8B vision-language runner (ONNX int4), short descriptions.

Adapted from the sibling ``schooluniform`` runner, which used the same graph as
a yes/no classifier. Here it generates text:

1. encode one crop with the vision encoder;
2. embed a small prompt (system + image + question) and prefill the decoder;
3. greedily decode at most ``max_new_tokens`` tokens, reusing the merged
   decoder's KV cache, stopping at EOS or ``<|im_end|>``.

The prompt is tiny and explicitly asks for one short sentence; the image is
resized to ``IMAGE_SIZE`` (a multiple of 32), keeping the image-token count low
(≈49 tokens at 224×224). Chat template and patch order follow the Qwen2-VL /
Qwen3.5 image processor.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from adapters.paths import models_dir

MODEL_NAME = "qwen3.5-0.8b"
DEFAULT_VARIANT = "q4"
DEFAULT_IMAGE_SIZE = 128
MAX_NEW_TOKENS = 24

PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
PATCH_DIM = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
GRID_FACTOR = PATCH_SIZE * MERGE_SIZE

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
ASSISTANT_PREFIX = f"{IM_START}assistant\n<think>\n\n</think>\n\n"

SYSTEM_PROMPT = (
    "You are a concise assistant for a blind person. Answer about the image in "
    "one short sentence, at most 15 words. Only describe what is clearly "
    "visible; never guess."
)
DEFAULT_QUESTION = "What is this? Describe it briefly."

_ONNX_DTYPES: dict[str, type[np.generic]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(bfloat16)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
}


def _log(message: str) -> None:
    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


def _variant() -> str:
    return os.getenv("VISIONASSIST_QWEN_VARIANT", DEFAULT_VARIANT)


def model_files(variant: str) -> tuple[str, ...]:
    return (
        "tokenizer.json",
        "config.json",
        f"onnx/decoder_model_merged_{variant}.onnx",
        f"onnx/decoder_model_merged_{variant}.onnx_data",
        f"onnx/embed_tokens_{variant}.onnx",
        f"onnx/embed_tokens_{variant}.onnx_data",
        f"onnx/vision_encoder_{variant}.onnx",
        f"onnx/vision_encoder_{variant}.onnx_data",
    )


def find_model_dir(model_dir: str | Path | None = None, variant: str | None = None) -> Path | None:
    variant = variant or _variant()
    if model_dir is not None:
        candidate = Path(model_dir)
        return candidate if _complete(candidate, variant) else None
    override = os.getenv("VISIONASSIST_QWEN_MODEL_DIR")
    candidates = [Path(override)] if override else []
    candidates += [models_dir() / f"{MODEL_NAME}-{variant}", models_dir() / MODEL_NAME]
    for candidate in candidates:
        if _complete(candidate, variant):
            return candidate
    return None


def _complete(path: Path, variant: str) -> bool:
    return all((path / name).is_file() for name in model_files(variant))


def bundle_available() -> bool:
    return find_model_dir() is not None


# ------------------------------------------------------------------- images


def smart_resize(
    height: int,
    width: int,
    factor: int = GRID_FACTOR,
    target_pixels: int = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE,
) -> tuple[int, int]:
    """Qwen2-VL smart resize: multiples of 32, area near the pixel budget."""
    if height < factor or width < factor:
        raise ValueError(f"frame is too small: {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")

    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)
    if h_bar * w_bar > target_pixels:
        beta = math.sqrt((height * width) / target_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < target_pixels:
        beta = math.sqrt(target_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


def _decode_frame(frame: bytes | np.ndarray) -> np.ndarray:
    if isinstance(frame, bytes):
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode image bytes")
        return image
    image = np.asarray(frame)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unsupported frame shape: {image.shape}")
    return image


def prepare_image(
    frame: bytes | np.ndarray,
    target_pixels: int = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """One BGR image as ``(pixel_values, image_grid_thw)``."""
    image = _decode_frame(frame)
    height, width = image.shape[:2]
    resized_h, resized_w = smart_resize(height, width, GRID_FACTOR, target_pixels)
    image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    channels_first = np.ascontiguousarray(rgb.transpose(2, 0, 1))

    grid_h = resized_h // PATCH_SIZE
    grid_w = resized_w // PATCH_SIZE
    patches = channels_first.reshape(
        3,
        grid_h // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
        grid_w // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
    )
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = patches[:, :, :, :, :, None, :, :]
    patches = np.repeat(patches, TEMPORAL_PATCH_SIZE, axis=5)
    pixel_values = np.ascontiguousarray(
        patches.reshape(grid_h * grid_w, PATCH_DIM), dtype=np.float32
    )
    image_grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, image_grid_thw


def image_token_count(image_grid_thw: np.ndarray) -> int:
    grid = np.asarray(image_grid_thw).reshape(-1)
    return int(np.prod(grid) // (MERGE_SIZE**2))


# ------------------------------------------------------------------- prompt


def build_prompt(question: str = DEFAULT_QUESTION, image_count: int = 1) -> str:
    markup = f"{VISION_START}{IMAGE_PAD}{VISION_END}" * image_count
    return (
        f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{markup}{question}{IM_END}\n"
        f"{ASSISTANT_PREFIX}"
    )


def build_input_ids(
    tokenizer: Tokenizer, prompt: str, image_tokens: int, image_token_id: int | None = None
) -> tuple[list[int], int]:
    if image_token_id is None:
        image_token_id = tokenizer.token_to_id(IMAGE_PAD)
    if image_token_id is None:
        raise RuntimeError("tokenizer has no image placeholder token")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    placeholders = sum(1 for token in ids if token == image_token_id)
    if placeholders != 1:
        raise RuntimeError(f"prompt has {placeholders} image placeholders, expected 1")
    expanded: list[int] = []
    for token in ids:
        if token == image_token_id:
            expanded.extend([image_token_id] * int(image_tokens))
        else:
            expanded.append(token)
    return expanded, image_token_id


def build_position_ids(
    input_ids: Sequence[int], image_token_id: int, image_grid: tuple[int, int]
) -> tuple[np.ndarray, int]:
    """3-row mRoPE positions plus the next text position."""
    total = len(input_ids)
    positions = np.zeros((3, 1, total), dtype=np.int64)
    grid_h, grid_w = image_grid
    merged_h = grid_h // MERGE_SIZE
    merged_w = grid_w // MERGE_SIZE
    current = 0
    index = 0
    while index < total:
        if input_ids[index] == image_token_id:
            end = index
            while end < total and input_ids[end] == image_token_id:
                end += 1
            count = end - index
            if count != merged_h * merged_w:
                raise RuntimeError(
                    f"image has {count} prompt tokens, expected {merged_h * merged_w}"
                )
            rows = np.repeat(np.arange(merged_h), merged_w)
            cols = np.tile(np.arange(merged_w), merged_h)
            positions[0, 0, index:end] = current
            positions[1, 0, index:end] = current + rows
            positions[2, 0, index:end] = current + cols
            current += max(merged_h, merged_w)
            index = end
        else:
            end = index
            while end < total and input_ids[end] != image_token_id:
                end += 1
            length = end - index
            positions[:, 0, index:end] = np.arange(current, current + length)
            current += length
            index = end
    return positions, current


# ---------------------------------------------------------------------- model


def _input_dtype(arg: ort.NodeArg) -> type[np.generic]:
    return _ONNX_DTYPES.get(arg.type, np.float32)


def _zero_caches(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for arg in session.get_inputs():
        name = arg.name
        if not (
            name.startswith("past_conv")
            or name.startswith("past_recurrent")
            or name.startswith("past_key_values")
        ):
            continue
        dims: list[int] = []
        for dim in arg.shape:
            if isinstance(dim, int):
                dims.append(dim)
            elif dim == "batch_size":
                dims.append(1)
            elif dim == "past_sequence_length":
                dims.append(0)
            else:
                dims.append(1)
        feeds[name] = np.zeros(dims, dtype=_input_dtype(arg))
    return feeds


def _cache_feed_names(output_names: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in output_names:
        if name.startswith("present_conv"):
            mapping[name] = name.replace("present_conv", "past_conv", 1)
        elif name.startswith("present_recurrent"):
            mapping[name] = name.replace("present_recurrent", "past_recurrent", 1)
        elif name.startswith("present"):
            mapping[name] = "past_key_values" + name[len("present") :]
    return mapping


def clean_description(text: str, max_words: int = 20) -> str:
    """One sentence, no role prefixes, bounded length."""
    text = " ".join(text.replace("\n", " ").split())
    for prefix in ("The image shows ", "This is an image of ", "In the image, "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :]
    for index, char in enumerate(text):
        if char in ".!?" and index > 8:
            text = text[: index + 1]
            break
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:") + "."
    if text and text[-1] not in ".!?":
        text += "."
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class Description:
    text: str
    latency_ms: float
    tokens: int
    image_size: tuple[int, int]
    image_tokens: int


class QwenVlRunner:
    """Loads the ONNX bundle once and describes one crop at a time."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        image_size: int | None = None,
        variant: str | None = None,
    ) -> None:
        self.variant = variant or _variant()
        self.model_dir = find_model_dir(model_dir, self.variant)
        resolved = image_size or int(
            os.getenv("VISIONASSIST_IMAGE_SIZE", str(DEFAULT_IMAGE_SIZE))
        )
        if resolved % GRID_FACTOR:
            raise ValueError(f"image size must be a multiple of {GRID_FACTOR}")
        self.image_size = resolved
        self._target_pixels = resolved * resolved
        self.last_token_only = True
        self._tokenizer: Tokenizer | None = None
        self._vision: ort.InferenceSession | None = None
        self._embed: ort.InferenceSession | None = None
        self._decoder: ort.InferenceSession | None = None
        self._image_token_id = -1
        self._eos: set[int] = set()
        self._cache_names: dict[str, str] = {}
        self._ready = False
        self._load_error: Exception | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Exception | None:
        return self._load_error

    def _options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        return options

    def load(self) -> None:
        with self._load_lock:
            if self._ready:
                return
            if self._load_error is not None:
                raise self._load_error
            if self.model_dir is None:
                self._load_error = RuntimeError(
                    "Qwen3.5 bundle not found under src/assets/models/qwen3.5-0.8b "
                    "(or set VISIONASSIST_QWEN_MODEL_DIR)."
                )
                raise self._load_error
            try:
                started = time.perf_counter()
                self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
                config = json.loads((self.model_dir / "config.json").read_text())
                self._image_token_id = int(config.get("image_token_id", -1))
                if self._image_token_id < 0:
                    self._image_token_id = self._tokenizer.token_to_id(IMAGE_PAD) or -1
                generation_file = self.model_dir / "generation_config.json"
                generation = (
                    json.loads(generation_file.read_text())
                    if generation_file.is_file()
                    else {}
                )
                eos_ids = generation.get("eos_token_id", [])
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                self._eos = {int(token) for token in eos_ids}
                im_end = self._tokenizer.token_to_id(IM_END)
                if im_end is not None:
                    self._eos.add(im_end)

                onnx_dir = self.model_dir / "onnx"
                self._vision = ort.InferenceSession(
                    str(onnx_dir / f"vision_encoder_{self.variant}.onnx"),
                    sess_options=self._options(),
                    providers=["CPUExecutionProvider"],
                )
                self._embed = ort.InferenceSession(
                    str(onnx_dir / f"embed_tokens_{self.variant}.onnx"),
                    sess_options=self._options(),
                    providers=["CPUExecutionProvider"],
                )
                decoder = onnx_dir / f"decoder_model_merged_{self.variant}.onnx"
                last_token = onnx_dir / (
                    f"decoder_model_merged_{self.variant}_lasttoken.onnx"
                )
                if self.last_token_only and last_token.is_file():
                    decoder = last_token
                else:
                    self.last_token_only = False
                self._decoder = ort.InferenceSession(
                    str(decoder),
                    sess_options=self._options(),
                    providers=["CPUExecutionProvider"],
                )
                self._cache_names = _cache_feed_names(
                    [output.name for output in self._decoder.get_outputs()]
                )
                _log(
                    "[vision-assist] Qwen sessions loaded in "
                    f"{time.perf_counter() - started:.1f}s at {self.image_size}px"
                    f" (last-token LM head: {self.last_token_only})"
                )
                self._ready = True
            except Exception as exc:
                self._load_error = exc
                _log(f"[vision-assist] Qwen load failed: {exc}")
                raise

    # ------------------------------------------------------------------ inference

    def _prefill(self, image: np.ndarray, question: str):
        assert self._tokenizer and self._vision and self._embed and self._decoder
        pixel_values, image_grid_thw = prepare_image(image, self._target_pixels)
        grid = (int(image_grid_thw[0][1]), int(image_grid_thw[0][2]))
        tokens = image_token_count(image_grid_thw)
        prompt = build_prompt(question)
        ids, image_token_id = build_input_ids(
            self._tokenizer, prompt, tokens, self._image_token_id
        )
        vision_dtype = _input_dtype(self._vision.get_inputs()[0])
        features = np.asarray(
            self._vision.run(
                None,
                {
                    "pixel_values": pixel_values.astype(vision_dtype, copy=False),
                    "image_grid_thw": image_grid_thw.astype(np.int64, copy=False),
                },
            )[0]
        )
        if features.shape[0] != tokens:
            raise RuntimeError(
                f"vision encoder produced {features.shape[0]} tokens, expected {tokens}"
            )
        input_ids = np.array([ids], dtype=np.int64)
        embeddings = np.asarray(self._embed.run(None, {"input_ids": input_ids})[0])
        decoder_dtype = _input_dtype(self._decoder.get_inputs()[0])
        embeddings = embeddings.astype(decoder_dtype, copy=False)
        embeddings[0, input_ids[0] == image_token_id] = features.astype(
            decoder_dtype, copy=False
        )
        positions, next_position = build_position_ids(ids, image_token_id, grid)
        feeds: dict[str, np.ndarray] = {
            "inputs_embeds": np.ascontiguousarray(embeddings),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "position_ids": positions,
        }
        feeds.update(_zero_caches(self._decoder))
        return feeds, next_position, len(ids), tokens, grid

    def describe(
        self,
        image: np.ndarray | bytes,
        question: str = DEFAULT_QUESTION,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Description:
        self.load()
        assert self._decoder and self._embed and self._tokenizer
        with self._run_lock:
            started = time.perf_counter()
            feeds, next_position, prompt_len, image_tokens, grid = self._prefill(
                image, question
            )
            outputs = self._decoder.run(None, feeds)
            generated: list[int] = []
            for _ in range(max_new_tokens):
                logits = np.asarray(outputs[0])
                token = int(np.argmax(logits[0, -1]))
                if token in self._eos:
                    break
                generated.append(token)

                if len(generated) >= max_new_tokens:
                    break
                cache = {
                    self._cache_names[output.name]: value
                    for output, value in zip(self._decoder.get_outputs()[1:], outputs[1:])
                }
                token_embedding = np.asarray(
                    self._embed.run(
                        None, {"input_ids": np.array([[token]], dtype=np.int64)}
                    )[0]
                ).astype(_input_dtype(self._decoder.get_inputs()[0]), copy=False)
                feeds = {
                    "inputs_embeds": np.ascontiguousarray(token_embedding),
                    "attention_mask": np.ones((1, prompt_len + len(generated)), dtype=np.int64),
                    "position_ids": np.array(
                        [[[next_position]], [[next_position]], [[next_position]]],
                        dtype=np.int64,
                    ),
                }
                feeds.update(cache)
                next_position += 1
                outputs = self._decoder.run(None, feeds)

            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            resized = (grid[1] * PATCH_SIZE, grid[0] * PATCH_SIZE)
            return Description(
                text=clean_description(text),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                tokens=len(generated),
                image_size=resized,
                image_tokens=image_tokens,
            )
