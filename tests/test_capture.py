import time

from adapters.capture import OpenCVSource, UnavailableSource, create_source
from adapters.capture.demo_source import DemoSource


def test_missing_video_file_reports_and_stops(tmp_path):
    source = OpenCVSource(str(tmp_path / "missing.mp4"))
    source.start()
    deadline = time.monotonic() + 5.0
    while source.error is None and time.monotonic() < deadline:
        time.sleep(0.05)
    source.stop()

    assert source.error is not None
    assert "Could not open" in source.error
    assert source.latest() is None


def test_stop_is_idempotent_for_unopened_source(tmp_path):
    source = OpenCVSource(str(tmp_path / "missing.mp4"))
    source.stop()
    source.start()
    source.stop()
    source.stop()


def test_create_source_from_env(monkeypatch):
    monkeypatch.setenv("VISIONASSIST_SOURCE", "demo")
    assert isinstance(create_source(3), DemoSource)

    monkeypatch.setenv("VISIONASSIST_SOURCE", str(__file__))
    assert isinstance(create_source(3), OpenCVSource)

    monkeypatch.setenv("VISIONASSIST_SOURCE", "/definitely/not/here")
    assert isinstance(create_source(3), UnavailableSource)

    monkeypatch.delenv("VISIONASSIST_SOURCE")
    assert isinstance(create_source(3), OpenCVSource)


def test_demo_source_produces_frames():
    source = DemoSource()
    source.start()
    deadline = time.monotonic() + 5.0
    frame = None
    while frame is None and time.monotonic() < deadline:
        frame = source.latest()
        time.sleep(0.05)
    source.stop()

    assert frame is not None
    assert frame.width > 0 and frame.height > 0
    assert frame.jpeg.startswith(b"\xff\xd8")
