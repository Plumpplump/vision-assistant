"""Turn detections into short, speakable descriptions."""

from __future__ import annotations

from .distance import (
    CameraModel,
    band_from_distance,
    band_from_ratio,
    is_very_close,
    pan_of,
    position_word,
    size_ratio,
)
from .types import POSITION_SPEECH, Band, Detection, SceneItem

# COCO labels are written for datasets, not for ears.
SPEECH_LABELS: dict[str, str] = {
    "dining table": "table",
    "couch": "sofa",
    "tv": "screen",
    "potted plant": "plant",
    "cell phone": "phone",
    "bottle": "bottle",
    "cup": "cup",
    "bowl": "bowl",
    "chair": "chair",
    "person": "person",
    "backpack": "backpack",
    "handbag": "bag",
    "book": "book",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "remote": "remote",
    "refrigerator": "fridge",
    "microwave": "microwave",
    "oven": "oven",
    "sink": "sink",
    "toilet": "toilet",
    "bed": "bed",
    "door": "door",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "wine glass": "glass",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
}

# Objects that usually rest on a table/desk rather than the floor. Their
# distance is measured against the surface plane, not the floor, otherwise a
# phone 60 cm away reads as 1.5 m.
SURFACE_LABELS = frozenset(
    {
        "cell phone",
        "cup",
        "bowl",
        "bottle",
        "wine glass",
        "book",
        "laptop",
        "keyboard",
        "remote",
        "vase",
        "scissors",
        "fork",
        "knife",
        "spoon",
        "clock",
        "teddy bear",
    }
)

# Ignore things that are never a useful answer to "what is in front of me".
NOISE_LABELS = {"stop sign", "parking meter", "fire hydrant", "bench"}

# Words people use that COCO labels differently.
TARGET_ALIASES: dict[str, str] = {
    "mug": "cup",
    "glass": "wine glass",
    "sofa": "couch",
    "settee": "couch",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "screen": "tv",
    "television": "tv",
    "table": "dining table",
    "bag": "handbag",
    "rucksack": "backpack",
    "plant": "potted plant",
    "fridge": "refrigerator",
}

MAX_ITEMS = 3
MIN_ITEM_RATIO = 0.04


def speakable(label: str) -> str:
    return SPEECH_LABELS.get(label, label)


def canonical_target(target: str) -> str:
    """Map everyday words onto the COCO labels the detector emits."""
    cleaned = target.lower().strip()
    return TARGET_ALIASES.get(cleaned, cleaned)


def scene_items(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    camera: CameraModel | None = None,
    surface_height_m: float = 0.0,
    max_items: int = MAX_ITEMS,
    min_ratio: float = MIN_ITEM_RATIO,
) -> list[SceneItem]:
    """Largest objects first, deduplicated by name.

    Objects below the horizon get a metric distance. Anything at or above it
    (a shelf, a distant sign) falls back to an apparent-size band.
    """
    camera = camera or CameraModel()
    candidates: list[SceneItem] = []
    for detection in detections:
        if detection.label in NOISE_LABELS:
            continue
        ratio = size_ratio(detection.box, frame_height)
        if ratio < min_ratio:
            continue
        surface = (
            surface_height_m if detection.label in SURFACE_LABELS else 0.0
        )
        distance = camera.distance_m(
            detection.box, frame_width, frame_height, surface_height_m=surface
        )
        band = (
            band_from_distance(distance)
            if distance is not None
            else band_from_ratio(ratio)
        )
        candidates.append(
            SceneItem(
                label=speakable(detection.label),
                band=band,
                position=position_word(pan_of(detection.box, frame_width)),
                ratio=ratio,
                box=detection.box,
                distance_m=distance,
                very_close=is_very_close(detection.box, frame_height),
            )
        )
    candidates.sort(key=lambda item: item.ratio, reverse=True)

    items: list[SceneItem] = []
    seen: set[str] = set()
    for item in candidates:
        if item.label in seen:
            continue
        seen.add(item.label)
        items.append(item)
        if len(items) >= max_items:
            break
    return items


def describe_items(items: list[SceneItem]) -> str:
    """One terse sentence per object ("Chair, about one metre away, on your left.")."""
    if not items:
        return "Nothing close in front of you."
    sentences = [
        f"{item.label.capitalize()}, {item.range_speech}, "
        f"{POSITION_SPEECH[item.position]}."
        for item in items
    ]
    return " ".join(sentences)


def describe_scene(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    camera: CameraModel | None = None,
) -> str:
    return describe_items(
        scene_items(detections, frame_width, frame_height, camera=camera)
    )


def surface_for(label: str, surface_height_m: float) -> float:
    """Surface height to use for this label (0 for floor-standing objects)."""
    return surface_height_m if label in SURFACE_LABELS else 0.0


def nearest_label(detections: list[Detection], frame_height: int) -> str | None:
    """Label of the largest detection, used for 'describe it'."""
    if not detections:
        return None
    biggest = max(detections, key=lambda d: size_ratio(d.box, frame_height))
    return biggest.label


def band_of(detection: Detection, frame_height: int) -> Band:
    return band_from_ratio(size_ratio(detection.box, frame_height))
