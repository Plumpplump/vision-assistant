from math import radians, tan

import pytest

from domain import Band, Box, CameraModel, calibrate_scale, distance_speech
from domain.distance import (
    is_very_close,
    band_from_distance,
    band_from_ratio,
    pan_of,
    position_word,
    size_ratio,
    vertical_of,
)

FRAME_W, FRAME_H = 640, 480
CAM = CameraModel(height_m=1.0, pitch_deg=20.0, hfov_deg=65.0)


def box_at(bottom: int, height: int = 120, x1: int = 200, x2: int = 440) -> Box:
    return Box(x1=x1, y1=bottom - height, x2=x2, y2=bottom, score=0.9)


def test_focal_length_from_horizontal_fov():
    expected = (FRAME_W / 2) / tan(radians(65 / 2))
    assert CAM.focal_px(FRAME_W) == pytest.approx(expected, rel=1e-6)
    assert CAM.focal_px(0) == 0.0


def test_bottom_of_frame_is_about_one_metre_away():
    assert CAM.distance_m(box_at(470), FRAME_W, FRAME_H) == pytest.approx(1.01, abs=0.05)


def test_above_the_horizon_has_no_distance():
    # horizon sits at y ~= 57 for this camera; anything above is not floor ahead
    assert CAM.distance_m(box_at(40), FRAME_W, FRAME_H) is None


def test_far_floor_point_is_measured_but_beyond_alert_range():
    distance = CAM.distance_m(box_at(100), FRAME_W, FRAME_H)
    assert distance is not None and distance > 5


def test_zero_tilt_moves_the_bottom_edge_further_out():
    flat = CameraModel(height_m=1.0, pitch_deg=0.0, hfov_deg=65.0)
    assert flat.distance_m(box_at(470), FRAME_W, FRAME_H) == pytest.approx(2.18, abs=0.05)


def test_changing_the_calibration_moves_the_estimate():
    low = CameraModel(height_m=0.6, pitch_deg=20.0)
    assert low.distance_m(box_at(470), FRAME_W, FRAME_H) < CAM.distance_m(
        box_at(470), FRAME_W, FRAME_H
    )


def test_band_thresholds():
    assert band_from_distance(0.8) is Band.CLOSE
    assert band_from_distance(1.8) is Band.NEAR
    assert band_from_distance(9.0) is Band.FAR
    assert band_from_distance(None) is Band.FAR


def test_band_hysteresis_needs_a_clear_step():
    assert band_from_distance(1.25, Band.CLOSE) is Band.CLOSE
    assert band_from_distance(1.35, Band.CLOSE) is Band.NEAR
    assert band_from_distance(1.05, Band.NEAR) is Band.CLOSE  # 1.2 - 0.1 = 1.1
    assert band_from_distance(1.25, Band.NEAR) is Band.NEAR
    assert band_from_distance(2.65, Band.FAR) is Band.FAR
    assert band_from_distance(2.35, Band.FAR) is Band.NEAR
    # onset is the configured distance itself, not threshold-minus-hysteresis
    tight = {"close_m": 0.6, "near_m": 1.0}
    assert band_from_distance(1.0, Band.FAR, **tight) is Band.NEAR
    assert band_from_distance(1.01, Band.FAR, **tight) is Band.FAR
    assert band_from_distance(0.6, Band.FAR, **tight) is Band.CLOSE


def test_ratio_fallback_is_unchanged():
    assert band_from_ratio(0.62) is Band.CLOSE
    assert band_from_ratio(0.30) is Band.NEAR
    assert band_from_ratio(0.05) is Band.FAR


def test_pan_position_and_vertical():
    assert pan_of(Box(0, 0, 640, 100), 640) == 0.0
    assert position_word(-0.4) == "left"
    assert position_word(0.0) == "ahead"
    assert vertical_of(Box(0, 300, 100, 470), 480).value == "floor"
    assert size_ratio(box_at(470, height=120), 480) == 0.25


@pytest.mark.parametrize(
    ("metres", "words"),
    [
        (1.01, "about one metre away"),
        (1.29, "about one and a half metres away"),
        (2.4, "about two and a half metres away"),
        (2.9, "about three metres away"),
        (4.4, "about 4 metres away"),
    ],
)
def test_distance_speech(metres, words):
    assert distance_speech(metres) == words


def test_distance_speech_never_claims_precision_for_none():
    assert distance_speech(None) == ""


def test_scale_is_a_one_number_correction():
    box = box_at(470)
    raw = CAM.distance_m(box, FRAME_W, FRAME_H)
    scaled = CameraModel(height_m=1.0, pitch_deg=20.0, hfov_deg=65.0, scale=0.33)
    assert scaled.distance_m(box, FRAME_W, FRAME_H) == pytest.approx(raw * 0.33, rel=1e-6)


def test_calibration_scale_from_a_known_distance():
    # measured 1.01 m, actually 0.33 m -> scale 0.33
    assert calibrate_scale(1.01, 0.33, 1.0) == pytest.approx(0.3267, abs=0.001)


def test_calibration_is_idempotent_when_applied_twice():
    # after the first calibration the estimate is correct, so scaling again
    # with the same true distance must not double-correct
    first = calibrate_scale(1.01, 0.33, 1.0)
    second = calibrate_scale(1.01 * first, 0.33, first)
    assert second == pytest.approx(first, rel=1e-6)


def test_calibration_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate_scale(0, 1.0, 1.0)
    with pytest.raises(ValueError):
        calibrate_scale(1.0, -2.0, 1.0)


def test_calibration_is_clamped():
    assert calibrate_scale(1.0, 100.0, 1.0) == pytest.approx(10.0)
    assert calibrate_scale(100.0, 0.01, 1.0) == pytest.approx(0.1)


def test_very_close_requires_a_large_clipped_box():
    assert is_very_close(box_at(480, height=300), FRAME_H)
    assert is_very_close(box_at(479, height=300), FRAME_H)
    assert not is_very_close(box_at(470, height=300), FRAME_H)
    assert not is_very_close(box_at(480, height=100), FRAME_H)


def test_surface_height_moves_the_measurement_plane_closer():
    box = box_at(470)
    floor = CAM.distance_m(box, FRAME_W, FRAME_H)
    desk = CAM.distance_m(box, FRAME_W, FRAME_H, surface_height_m=0.75)
    # camera 1.0 m above the floor, surface 0.75 m -> only 0.25 m above it
    assert desk == pytest.approx(floor * 0.25, rel=1e-6)
