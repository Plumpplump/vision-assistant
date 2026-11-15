# Author: David Li, SISHK

"""Synthetic frames so the app is demoable without a camera or video file."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .base import Frame

WIDTH = 480
HEIGHT = 270
DEFAULT_MAX_FPS = 12.5
JPEG_QUALITY = 72


class DemoSource:
    """A coloured background with a moving 'person' rectangle."""

    name = "demo scene"

    def __init__(self, max_fps: float = DEFAULT_MAX_FPS) -> None:
        self.max_fps = max_fps
        self._error: str | None = None
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def latest(self) -> Frame | None:
        return self._frame

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="capture-demo", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def _run(self) -> None:
        interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        step = 0
        while not self._stop.is_set():
            image = np.full((HEIGHT, WIDTH, 3), (235, 240, 245), np.uint8)
            x = 40 + (step * 6) % (WIDTH - 120)
            cv2.rectangle(image, (x, 90), (x + 70, 230), (90, 70, 160), -1)
            cv2.circle(image, (x + 35, 70), 22, (150, 170, 200), -1)
            cv2.putText(
                image,
                "DEMO",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 130, 145),
                2,
            )
            ok, buffer = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                self._frame = Frame(
                    jpeg=buffer.tobytes(),
                    width=WIDTH,
                    height=HEIGHT,
                    timestamp=time.monotonic(),
                    fps=self.max_fps,
                    image=image,
                )
            step += 1
            time.sleep(interval)
