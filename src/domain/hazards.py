"""Turn tracked boxes into alert-worthy hazard events.

Policy, in the order it is applied:

1. **Below the horizon.** The bottom edge is treated as the floor contact
   point. If it sits at or above the horizon line, the object is not on the
   floor ahead (or is far beyond the alert range), which removes "there is a
   building 30 m away" from the alert stream.
2. **Metric distance.** Ground-plane geometry (`CameraModel.distance_m`) gives
   metres; anything beyond ``near_m`` is tracked silently. Alerts start at
   ``near_m`` and escalate at ``close_m``.
3. **Corridor and size.** The contact point must be in the middle of the view,
   and the box must be big enough to be real rather than a speck.
4. **Confirmation and hysteresis.** The tracker must have confirmed the object,
   and band changes use a margin so a threshold cannot chatter.

Events are plain data: deciding that two events mean one sound is the alert
layer's job, not the domain's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .distance import (
    CameraModel,
    band_from_distance,
    is_very_close,
    pan_of,
    size_ratio,
    vertical_of,
)
from .types import AlertEvent, Band, Hazard, Track, Vertical

DEFAULT_CORRIDOR = 0.35
DEFAULT_MIN_SIZE_RATIO = 0.03
DEFAULT_CLOSE_M = 0.6
DEFAULT_NEAR_M = 1.0

# A hazard already being tracked gets a small exit margin, so an object sitting
# exactly on the corridor edge cannot flap between "alarm" and "silence".
CORRIDOR_EXIT_MARGIN = 0.05


@dataclass
class _State:
    band: Band
    hazard: Hazard


class HazardMonitor:
    """Stateful, pure-python hazard detector over tracker output."""

    def __init__(
        self,
        camera: CameraModel | None = None,
        corridor: float = DEFAULT_CORRIDOR,
        min_size_ratio: float = DEFAULT_MIN_SIZE_RATIO,
        close_m: float = DEFAULT_CLOSE_M,
        near_m: float = DEFAULT_NEAR_M,
    ) -> None:
        self.camera = camera or CameraModel()
        self.corridor = corridor
        self.min_size_ratio = min_size_ratio
        self.close_m = close_m
        self.near_m = near_m
        self._states: dict[int, _State] = {}
        self.hazards: list[Hazard] = []

    def reset(self) -> None:
        self._states.clear()
        self.hazards.clear()

    # ------------------------------------------------------------------ update

    def update(
        self,
        tracks: list[Track],
        frame_width: int,
        frame_height: int,
        now: float | None = None,
    ) -> list[AlertEvent]:
        now = time.monotonic() if now is None else now
        events: list[AlertEvent] = []
        seen: set[int] = set()

        for track in tracks:
            if not track.confirmed:
                continue
            ratio = size_ratio(track.box, frame_height)
            if ratio < self.min_size_ratio:
                continue
            pan = pan_of(track.box, frame_width)
            limit = self.corridor + (
                CORRIDOR_EXIT_MARGIN if track.id in self._states else 0.0
            )
            if abs(pan) > limit:
                continue
            distance = self.camera.distance_m(track.box, frame_width, frame_height)
            if distance is None:
                continue

            seen.add(track.id)
            state = self._states.get(track.id)
            very_close = is_very_close(track.box, frame_height)
            band = (
                Band.CLOSE
                if very_close
                else band_from_distance(
                    distance,
                    state.band if state else None,
                    close_m=self.close_m,
                    near_m=self.near_m,
                )
            )
            hazard = Hazard(
                track_id=track.id,
                label=track.label,
                band=band,
                pan=pan,
                vertical=vertical_of(track.box, frame_height),
                box=track.box,
                distance_m=distance,
                very_close=very_close,
            )

            if state is None:
                self._states[track.id] = _State(band=band, hazard=hazard)
                if band is not Band.FAR:
                    events.append(AlertEvent(kind="enter", hazard=hazard))
                continue

            state.hazard = hazard
            if band is state.band:
                continue
            previous, state.band = state.band, band
            events.append(
                AlertEvent(kind="change", hazard=hazard, previous=previous)
            )

        for track_id in list(self._states):
            if track_id not in seen:
                state = self._states.pop(track_id)
                events.append(
                    AlertEvent(kind="clear", hazard=state.hazard, previous=state.band)
                )

        self.hazards = sorted(
            (state.hazard for state in self._states.values() if state.band is not Band.FAR),
            key=lambda hazard: hazard.distance_m or 1e9,
        )
        return events


def vertical_is_floor(vertical: Vertical) -> bool:
    return vertical is Vertical.FLOOR
