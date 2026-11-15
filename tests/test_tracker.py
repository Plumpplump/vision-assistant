from adapters.detection import Tracker
from domain import Box, Detection


def detection(x1: float = 100, y1: float = 100, size: float = 100) -> Detection:
    return Detection(
        box=Box(x1=x1, y1=y1, x2=x1 + size, y2=y1 + size, score=0.9),
        label="chair",
    )


def test_track_needs_two_hits_before_confirmation():
    tracker = Tracker()
    tracks = tracker.update([detection()])
    assert len(tracks) == 1
    assert tracks[0].confirmed is False

    tracks = tracker.update([detection()])
    assert tracks[0].id == 1
    assert tracks[0].confirmed is True
    assert tracks[0].label == "chair"


def test_identity_survives_small_movement():
    tracker = Tracker()
    tracker.update([detection()])
    tracks = tracker.update([detection(x1=110, y1=105)])
    assert [track.id for track in tracks] == [1]


def test_track_expires_after_max_misses():
    tracker = Tracker(max_misses=2)
    tracker.update([detection()])
    tracker.update([detection()])
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == []


def test_new_object_gets_new_id():
    tracker = Tracker()
    tracker.update([detection()])
    second = detection(x1=400, y1=300, size=80)
    tracks = tracker.update([second])
    ids = {track.id for track in tracks}
    assert ids == {1, 2}
