# Author: David Li, SISHK

"""OpenCV camera / video-file source, threaded, throttled, self-healing.

Adapted from the sibling ``schooluniform`` project. Two changes matter here:

* a busy camera is retried instead of being fatal — a leftover ``flet run``
  (or any other app) holding ``/dev/video0`` should not require a restart;
* the error message names the likely cause, because "could not open camera"
  sends people looking in the wrong place.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from .base import Frame

DEFAULT_MAX_FPS = 12.5
DEFAULT_PREVIEW_WIDTH = 640
JPEG_QUALITY = 72
RECONNECT_DELAY = 2.0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class OpenCVSource:
    def __init__(
        self,
        target: int | str = 0,
        max_fps: float = DEFAULT_MAX_FPS,
        preview_width: int = DEFAULT_PREVIEW_WIDTH,
    ) -> None:
        self.target = target
        self.max_fps = max_fps
        self.preview_width = preview_width
        self.name = (
            f"camera {target}"
            if isinstance(target, int)
            else f"video {Path(target).name}"
        )
        self._frame: Frame | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def error(self) -> str | None:
        return self._error

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        with self._lock:
            self._frame = None
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)

    # ------------------------------------------------------------------ plumbing

    def _is_file(self) -> bool:
        return isinstance(self.target, str)

    def _open_message(self) -> str:
        if self._is_file():
            return f"Could not open {self.name}. Check that the file exists."
        return (
            f"{self.name} is busy or unavailable. Another app (or a leftover "
            "`flet run`) may be holding the camera — checking `fuser -v "
            "/dev/video0` shows it. Retrying…"
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.target)
            if not cap.isOpened():
                cap.release()
                self._error = self._open_message()
                if self._is_file():
                    return  # a missing file will not appear on its own
                if self._stop.wait(RECONNECT_DELAY):
                    return
                continue

            self._error = None
            try:
                self._pump(cap)
            except Exception as exc:  # pragma: no cover - device-specific failures
                self._error = f"{self.name} failed: {exc}"
            finally:
                cap.release()

            if self._stop.wait(RECONNECT_DELAY):
                return

    def _pump(self, cap: cv2.VideoCapture) -> None:
        if isinstance(self.target, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        last_emit = 0.0
        fps = 0.0

        while not self._stop.is_set():
            ok, image = cap.read()
            if not ok:
                if self._is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(max(interval, 0.02))
                    continue
                self._error = f"{self.name} stopped returning frames; reconnecting…"
                return

            now = time.monotonic()
            elapsed = now - last_emit
            if interval and elapsed < interval:
                if self._is_file():
                    time.sleep(interval - elapsed)
                continue

            encoded = self._encode(image)
            if encoded is None:
                continue
            if last_emit and elapsed > 0:
                instant = 1.0 / elapsed
                fps = instant if fps == 0 else fps * 0.8 + instant * 0.2
            last_emit = now
            jpeg, width, height, frame_image = encoded
            with self._lock:
                self._frame = Frame(
                    jpeg=jpeg,
                    width=width,
                    height=height,
                    timestamp=now,
                    fps=fps,
                    image=frame_image,
                )

    def _encode(self, image) -> tuple[bytes, int, int, object] | None:
        height, width = image.shape[:2]
        if width > self.preview_width:
            scale = self.preview_width / width
            image = cv2.resize(
                image,
                (self.preview_width, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return None
        return buffer.tobytes(), image.shape[1], image.shape[0], image
