import pytest

from domain import Band, Box, CameraModel, Track
from domain.hazards import HazardMonitor

FRAME_W, FRAME_H = 640, 480
CAM = CameraModel(height_m=1.0, pitch_deg=20.0, hfov_deg=65.0)


def track(
    bottom: int = 470,
    height: int = 300,
    x1: int = 250,
    x2: int = 390,
    track_id: int = 1,
    label: str = "chair",
) -> Track:
    box = Box(x1=x1, y1=bottom - height, x2=x2, y2=bottom, score=0.9)
    return Track(id=track_id, box=box, label=label, hits=3, confirmed=True)


def monitor() -> HazardMonitor:
    # Explicit wide thresholds keep these tests about policy mechanics; the
    # tuned defaults (1.0 m warn / 0.6 m close) are covered separately.
    return HazardMonitor(camera=CAM, close_m=1.2, near_m=2.5, corridor=0.5)


def test_close_object_alerts_once():
    m = monitor()
    events = m.update([track()], FRAME_W, FRAME_H, now=0.0)
    assert [event.kind for event in events] == ["enter"]
    hazard = events[0].hazard
    assert hazard.band is Band.CLOSE
    assert hazard.distance_m == pytest.approx(1.01, abs=0.06)

    assert m.update([track()], FRAME_W, FRAME_H, now=0.1) == []
    assert len(m.hazards) == 1


def test_distant_object_is_tracked_but_silent():
    m = monitor()
    # feet high in the frame: 3.6 m away on this camera model
    events = m.update([track(bottom=200, height=100)], FRAME_W, FRAME_H, now=0.0)
    assert events == []
    assert m.hazards == []


def test_approach_and_retreat_change_band_both_ways():
    m = monitor()
    m.update([track(bottom=400, height=140)], FRAME_W, FRAME_H, now=0.0)  # ~1.3 m, NEAR
    assert m.hazards[0].band is Band.NEAR

    events = m.update([track(bottom=479)], FRAME_W, FRAME_H, now=0.1)
    assert [event.kind for event in events] == ["change"]
    assert events[0].hazard.band is Band.CLOSE
    assert events[0].previous is Band.NEAR

    events = m.update([track(bottom=380, height=140)], FRAME_W, FRAME_H, now=0.2)
    assert [event.kind for event in events] == ["change"]
    assert events[0].hazard.band is Band.NEAR
    assert events[0].previous is Band.CLOSE


def test_far_object_becomes_silent_again_when_it_walks_away():
    m = monitor()
    m.update([track()], FRAME_W, FRAME_H, now=0.0)
    events = m.update([track(bottom=300, height=140)], FRAME_W, FRAME_H, now=0.1)
    # 1.98 m is still inside near_m, so this is a band change, not silence
    assert events[0].hazard.band is Band.NEAR

    events = m.update([track(bottom=160, height=100)], FRAME_W, FRAME_H, now=0.2)
    assert events[0].hazard.band is Band.FAR
    assert m.hazards == []


def test_outside_corridor_clears():
    m = monitor()
    m.update([track()], FRAME_W, FRAME_H, now=0.0)
    events = m.update([track(x1=560, x2=700)], FRAME_W, FRAME_H, now=0.2)
    assert [event.kind for event in events] == ["clear"]
    assert m.hazards == []


def test_tiny_object_is_ignored():
    m = monitor()
    assert m.update([track(height=10)], FRAME_W, FRAME_H, now=0.0) == []


def test_unconfirmed_track_is_ignored():
    m = monitor()
    pending = track()
    pending.confirmed = False
    assert m.update([pending], FRAME_W, FRAME_H, now=0.0) == []


def test_hazards_are_sorted_by_distance():
    m = monitor()
    near = track(bottom=380, height=140, track_id=1, label="table")
    close = track(bottom=470, track_id=2, label="chair")
    m.update([near], FRAME_W, FRAME_H, now=0.0)
    m.update([near, close], FRAME_W, FRAME_H, now=0.1)
    assert [hazard.label for hazard in m.hazards] == ["chair", "table"]


def test_distance_is_reported_for_stereo_and_speech():
    m = monitor()
    events = m.update([track(x1=120, x2=260)], FRAME_W, FRAME_H, now=0.0)
    hazard = events[0].hazard
    assert hazard.pan < -0.15
    assert hazard.position == "left"
    assert "about one metre away" in hazard.speech
    assert "on your left" in hazard.speech


def test_default_thresholds_only_warn_within_one_metre():
    monitor_default = HazardMonitor(camera=CAM)
    assert monitor_default.near_m == 1.0
    assert monitor_default.close_m == 0.6

    # A slightly lower camera makes the visible floor start nearer, so a
    # non-clipped box can sit between close (0.6 m) and near (1.0 m).
    low = HazardMonitor(camera=CameraModel(height_m=0.7, pitch_deg=20.0, hfov_deg=65.0))
    near_events = low.update([track(bottom=402)], FRAME_W, FRAME_H, now=0.0)
    assert near_events and near_events[0].hazard.band is Band.NEAR
    monitor_default.reset()
    far_events = monitor_default.update(
        [track(bottom=300, height=140)], FRAME_W, FRAME_H, now=0.0
    )
    assert far_events == []
    assert monitor_default.hazards == []

    # after calibration the same box is 0.49 m away -> CLOSE
    calibrated = HazardMonitor(camera=CameraModel(scale=0.5))
    close_events = calibrated.update([track(bottom=479)], FRAME_W, FRAME_H, now=0.0)
    assert close_events and close_events[0].hazard.band is Band.CLOSE


def test_default_corridor_is_the_middle_third():
    monitor_default = HazardMonitor(camera=CAM)
    assert monitor_default.corridor == 0.35
    # centred object at 0.98 m warns (default warn distance is 1.0 m)
    assert monitor_default.update([track(bottom=479)], FRAME_W, FRAME_H, now=0.0)
    # object shifted to the left edge is ignored, even if it is close
    monitor_default.reset()
    assert monitor_default.update(
        [track(bottom=479, x1=120, x2=260)], FRAME_W, FRAME_H, now=0.0
    ) == []
    assert monitor_default.hazards == []


def test_corridor_has_an_exit_margin_to_stop_flapping():
    monitor_boundary = HazardMonitor(camera=CAM, close_m=1.2, near_m=2.5, corridor=0.35)
    # pan = 0.30: inside
    inside = track(x1=270, x2=410)
    assert monitor_boundary.update([inside], FRAME_W, FRAME_H, now=0.0)
    # pan = 0.38: outside the entry corridor but inside the exit margin
    edge = track(x1=290, x2=430)
    assert monitor_boundary.update([edge], FRAME_W, FRAME_H, now=0.1) == []
    assert monitor_boundary.hazards


def test_clipped_large_object_is_very_close_and_alerts():
    monitor_default = HazardMonitor(camera=CAM)
    # box runs off the bottom of the frame: geometry saturates at ~1 m, but a
    # person filling the view at 40 cm must still alarm
    events = monitor_default.update(
        [track(bottom=480, height=300)], FRAME_W, FRAME_H, now=0.0
    )
    assert events and events[0].hazard.band is Band.CLOSE
    assert events[0].hazard.very_close is True
    assert "very close" in events[0].hazard.speech
