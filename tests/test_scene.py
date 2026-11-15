# Author: David Li, SISHK

from domain import Box, CameraModel, Detection
from domain.scene import describe_scene, scene_items, speakable

FRAME_W = 640
FRAME_H = 480
CAM = CameraModel(height_m=1.0, pitch_deg=20.0, hfov_deg=65.0)


def detection(
    label: str, bottom: int, height: int, center_x: int, width: int = 160
) -> Detection:
    x1 = center_x - width // 2
    return Detection(
        box=Box(x1=x1, y1=bottom - height, x2=x1 + width, y2=bottom, score=0.8),
        label=label,
    )


def test_description_uses_metres_for_floor_objects():
    detections = [
        detection("dining table", bottom=300, height=120, center_x=520),
        detection("chair", bottom=470, height=300, center_x=140),
        detection("cup", bottom=470, height=12, center_x=320),
    ]
    text = describe_scene(detections, FRAME_W, FRAME_H, CAM)
    assert text == (
        "Chair, about one metre away, on your left. "
        "Table, about two metres away, on your right."
    )


def test_objects_above_the_horizon_fall_back_to_size_bands():
    detections = [detection("shelf", bottom=40, height=40, center_x=320)]
    items = scene_items(detections, FRAME_W, FRAME_H, CAM)
    assert items[0].distance_m is None
    assert items[0].speech.startswith("shelf far")


def test_one_row_per_label():
    detections = [
        detection("chair", bottom=470, height=300, center_x=140),
        detection("chair", bottom=460, height=260, center_x=480),
    ]
    items = scene_items(detections, FRAME_W, FRAME_H, CAM)
    assert [item.label for item in items] == ["chair"]


def test_noise_labels_are_dropped():
    detections = [
        detection("bench", bottom=470, height=300, center_x=140),
        detection("stop sign", bottom=470, height=300, center_x=480),
    ]
    assert scene_items(detections, FRAME_W, FRAME_H, CAM) == []


def test_tiny_and_empty_scenes():
    assert describe_scene([], FRAME_W, FRAME_H, CAM) == "Nothing close in front of you."
    tiny = detection("cup", bottom=470, height=5, center_x=320)
    assert describe_scene([tiny], FRAME_W, FRAME_H, CAM) == (
        "Nothing close in front of you."
    )


def test_speech_labels_are_spoken_friendly():
    assert speakable("dining table") == "table"
    assert speakable("tv") == "screen"
    assert speakable("unmapped thing") == "unmapped thing"


def test_clipped_person_is_described_as_very_close():
    detections = [detection("person", bottom=480, height=300, center_x=320)]
    items = scene_items(detections, FRAME_W, FRAME_H, CAM)
    assert items[0].very_close is True
    assert items[0].range_speech == "very close"


def test_table_objects_use_the_surface_but_people_use_the_floor():
    from domain import CameraModel

    camera = CameraModel(height_m=1.1, pitch_deg=20.0, hfov_deg=65.0)
    detections = [
        detection("person", bottom=470, height=300, center_x=520),
        detection("cell phone", bottom=470, height=60, center_x=200),
    ]
    items = scene_items(
        detections, FRAME_W, FRAME_H, camera=camera, surface_height_m=0.75
    )
    by_label = {item.label: item for item in items}
    assert by_label["person"].distance_m > by_label["phone"].distance_m
    # phone base is 0.35 m below the camera, so it is well under a metre
    assert by_label["phone"].distance_m < 1.0
