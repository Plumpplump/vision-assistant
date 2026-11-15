# Author: David Li, SISHK

"""Minimal tracking-by-detection: stable IDs via greedy IoU matching.

Adapted from the sibling ``schooluniform`` project. Commercial pipelines
detect every frame, but assign an identity and act on it once. The
confirmation delay keeps partial boxes from triggering an alert, and the miss
counter makes a person who leaves disappear.
"""

from __future__ import annotations

import time

from domain.types import Box, Detection, Track

DEFAULT_IOU = 0.30
CONFIRM_HITS = 2
MAX_MISSES = 5


def iou(first: Box, second: Box) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


class Tracker:
    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU,
        confirm_hits: int = CONFIRM_HITS,
        max_misses: int = MAX_MISSES,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self, detections: list[Detection], now: float | None = None
    ) -> list[Track]:
        now = time.monotonic() if now is None else now
        unmatched = list(range(len(detections)))

        for track in self.tracks:
            best_index = None
            best_score = self.iou_threshold
            for index in unmatched:
                score = iou(track.box, detections[index].box)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                track.misses += 1
                continue
            detection = detections[best_index]
            track.box = detection.box
            track.label = detection.label
            track.hits += 1
            track.misses = 0
            track.last_seen = now
            track.confirmed = track.hits >= self.confirm_hits
            unmatched.remove(best_index)

        self.tracks = [
            track for track in self.tracks if track.misses <= self.max_misses
        ]

        for index in unmatched:
            detection = detections[index]
            self.tracks.append(
                Track(
                    id=self._next_id,
                    box=detection.box,
                    label=detection.label,
                    confirmed=self.confirm_hits <= 1,
                    last_seen=now,
                )
            )
            self._next_id += 1

        return sorted(self.tracks, key=lambda track: track.id)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
