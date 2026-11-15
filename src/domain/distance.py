"""Distance for a single camera, and the bands the alert policy uses.

Two different questions live here:

* **Is it a floor hazard, and how far away?** Monocular pinhole geometry. The
  bottom edge of a box is assumed to be where the object meets the floor, so
  with camera height ``h``, focal length ``f`` (from the horizontal FOV) and a
  downward tilt, ``Z = h / tan(pitch + atan((y_bottom - cy) / f))``. Distances
  are in metres, which the user can calibrate against their own pace.
* **How big does it look?** ``size_ratio`` is still used as a fallback for
  things that are not on the floor (a cup on a table) and as a noise gate.

Why not area ratio alone: a door at 4 m and a mug at 0.5 m can fill the same
share of the frame, so apparent size cannot answer "is it close?". The horizon
line can: floor points above it are metres away no matter how large the object
is, which is exactly the "don't alert for distant things" filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, radians, tan

from .types import Band, Box, Vertical

# Apparent-size fallback (share of frame height), unchanged from the first cut.
CLOSE_RATIO = 0.55
NEAR_RATIO = 0.22
RATIO_HYSTERESIS = 0.05

# Metric bands. Chosen for an adult walking pace: one step ~0.7 m.
CLOSE_M = 1.2
NEAR_M = 2.5
DISTANCE_HYSTERESIS_M = 0.10

FLOOR_BOTTOM_RATIO_VERTICAL = 0.75
HEAD_BOTTOM_RATIO = 0.55

PAN_DEADZONE = 0.15

MIN_DISTANCE_M = 0.2
MAX_DISTANCE_M = 30.0


@dataclass(frozen=True)
class CameraModel:
    """Where the camera sits and how it is aimed.

    Defaults describe a phone/chest badge: 1 m above the floor, tilted 20° down,
    65° horizontal field of view. They are demo defaults — expose them in
    Settings and let the user calibrate against a known distance.
    """

    height_m: float = 1.0
    pitch_deg: float = 20.0
    hfov_deg: float = 65.0
    # One-number correction for the demo. The model measures the floor point
    # under the box; a laptop webcam looks at objects on a desk at its own
    # height, which biases every estimate by roughly the same factor.
    scale: float = 1.0

    def focal_px(self, frame_width: int) -> float:
        if frame_width <= 0:
            return 0.0
        return (frame_width / 2.0) / tan(radians(self.hfov_deg) / 2.0)

    def distance_m(
        self,
        box: Box,
        frame_width: int,
        frame_height: int,
        surface_height_m: float = 0.0,
    ) -> float | None:
        """Distance to the floor point under the box, or ``None``.

        ``None`` means "no usable floor geometry": the contact point is at or
        above the horizon (effectively far away), or outside sane range.
        """
        focal = self.focal_px(frame_width)
        if focal <= 0 or frame_height <= 0:
            return None
        # The box bottom is the contact point. For a phone on a desk that plane
        # is ``surface_height_m`` above the floor, so the camera is that much
        # closer to it. Clamp so a surface at camera level cannot divide by ~0.
        height = max(0.05, self.height_m - max(0.0, surface_height_m))
        offset = box.y2 - frame_height / 2.0
        angle = radians(self.pitch_deg) + atan2(offset, focal)
        if degrees(angle) <= 1.5:
            return None  # at/above the horizon: not a nearby floor point
        distance = (height / tan(angle)) * self.scale
        if not MIN_DISTANCE_M <= distance <= MAX_DISTANCE_M:
            return None
        return distance


VERY_CLOSE_RATIO = 0.5
FILLS_FRAME_RATIO = 0.75
CLIP_MARGIN_RATIO = 0.015


def is_very_close(box: Box, frame_height: int) -> bool:
    """True when the object is large *and* clipped by the frame bottom.

    Ground-plane geometry needs the contact point inside the image. When a
    person or object fills the view, its feet/base are below the frame (or the
    detector insets the box a few pixels), the bottom edge saturates at the
    frame-edge distance (~1 m with the default camera), and the estimate can
    never reach 0.4 m. Treat saturation — or a box filling most of the frame —
    as "very close" instead of trusting the number.
    """
    if frame_height <= 0:
        return False
    ratio = size_ratio(box, frame_height)
    if ratio < VERY_CLOSE_RATIO:
        return False
    # A box that fills three quarters of the frame is close whatever the
    # geometry says (people at arm's length, a wall you are facing).
    if ratio >= FILLS_FRAME_RATIO:
        return True
    margin = max(3.0, CLIP_MARGIN_RATIO * frame_height)
    return (frame_height - box.y2) <= margin


def size_ratio(box: Box, frame_height: int) -> float:
    if frame_height <= 0:
        return 0.0
    return box.height / float(frame_height)


def band_from_ratio(ratio: float, previous: Band | None = None) -> Band:
    """Apparent-size fallback, with hysteresis around the boundaries."""
    if previous is Band.CLOSE:
        if ratio >= CLOSE_RATIO - RATIO_HYSTERESIS:
            return Band.CLOSE
        return Band.NEAR if ratio >= NEAR_RATIO - RATIO_HYSTERESIS else Band.FAR
    if previous is Band.NEAR:
        if ratio >= CLOSE_RATIO + RATIO_HYSTERESIS:
            return Band.CLOSE
        if ratio >= NEAR_RATIO - RATIO_HYSTERESIS:
            return Band.NEAR
        return Band.FAR
    if previous is Band.FAR:
        if ratio >= CLOSE_RATIO + RATIO_HYSTERESIS:
            return Band.CLOSE
        if ratio >= NEAR_RATIO + RATIO_HYSTERESIS:
            return Band.NEAR
        return Band.FAR
    if ratio >= CLOSE_RATIO:
        return Band.CLOSE
    if ratio >= NEAR_RATIO:
        return Band.NEAR
    return Band.FAR


def band_from_distance(
    distance: float | None,
    previous: Band | None = None,
    close_m: float = CLOSE_M,
    near_m: float = NEAR_M,
) -> Band:
    """Metric bands with hysteresis, so a threshold cannot chatter."""
    if distance is None:
        return Band.FAR
    margin = DISTANCE_HYSTERESIS_M
    if previous is Band.CLOSE:
        if distance <= close_m + margin:
            return Band.CLOSE
        return Band.NEAR if distance <= near_m + margin else Band.FAR
    if previous is Band.NEAR:
        if distance <= close_m - margin:
            return Band.CLOSE
        if distance <= near_m + margin:
            return Band.NEAR
        return Band.FAR
    if previous is Band.FAR:
        # Onset is exactly the configured threshold; hysteresis only delays
        # going quiet again (FAR is regained above near_m + margin).
        if distance <= close_m:
            return Band.CLOSE
        if distance <= near_m:
            return Band.NEAR
        return Band.FAR
    if distance <= close_m:
        return Band.CLOSE
    if distance <= near_m:
        return Band.NEAR
    return Band.FAR


def pan_of(box: Box, frame_width: int) -> float:
    """Horizontal position as -1 (hard left) .. 0 (centre) .. 1 (hard right)."""
    if frame_width <= 0:
        return 0.0
    centre = box.center_x / float(frame_width)
    return max(-1.0, min(1.0, (centre - 0.5) * 2.0))


def vertical_of(box: Box, frame_height: int) -> Vertical:
    if frame_height <= 0:
        return Vertical.WAIST
    bottom = box.y2 / float(frame_height)
    if bottom >= FLOOR_BOTTOM_RATIO_VERTICAL:
        return Vertical.FLOOR
    if bottom >= HEAD_BOTTOM_RATIO:
        return Vertical.WAIST
    return Vertical.HEAD


def position_word(pan: float) -> str:
    if pan < -PAN_DEADZONE:
        return "left"
    if pan > PAN_DEADZONE:
        return "right"
    return "ahead"


CALIBRATION_MIN = 0.1
CALIBRATION_MAX = 10.0


def calibrate_scale(measured: float, actual: float, current_scale: float = 1.0) -> float:
    """Scale factor that turns ``measured`` into ``actual``.

    ``measured`` already includes ``current_scale``, so the raw geometry is
    ``measured / current_scale`` and the new factor is ``actual / raw``.
    """
    if measured is None or measured <= 0 or actual <= 0:
        raise ValueError("measured and actual distances must be positive")
    raw = measured / max(CALIBRATION_MIN, current_scale)
    scale = actual / raw
    return max(CALIBRATION_MIN, min(CALIBRATION_MAX, scale))
