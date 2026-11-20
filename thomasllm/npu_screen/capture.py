"""Screen capture backend backed by dxcam (Windows).

Wraps dxcam's ScreenCaptureManager/DxCapture with a producer thread that
pushes BGRA frames + timestamps into a bounded queue (drop-oldest).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Tuple

try:
    import dxcam  # type: ignore
    _HAS_DXCAM = True
except Exception:  # pragma: no cover - dxcam optional on non-Windows
    dxcam = None  # type: ignore
    _HAS_DXCAM = False


class Frame:
    """A captured frame with metadata."""

    __slots__ = ("data", "timestamp", "frame_id")

    def __init__(self, data, timestamp: float, frame_id: int):
        self.data = data  # BGRA bytes (numpy array view from dxcam)
        self.timestamp = timestamp
        self.frame_id = frame_id


class CaptureProducer:
    """Producer thread: continuously capture frames and enqueue them.

    Args:
        fps: Target capture rate.
        region: Optional (left, top, right, bottom) region.
        window: Optional window title substring to capture.
        max_queue_size: Max frames held; oldest dropped when full.
    """

    def __init__(self, fps: float = 10.0, region: Optional[Tuple[int, int, int, int]] = None,
                 window: Optional[str] = None, max_queue_size: int = 4):
        self.fps = max(1.0, float(fps))
        self.region = region
        self.window = window
        self.max_queue_size = max(1, int(max_queue_size))
        self._queue: "deque[Frame]" = deque(maxlen=self.max_queue_size)
        self._lock = threading.Lock()
        self._frame_id = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cam = None
        self._interval = 1.0 / self.fps

    # -- queue access --------------------------------------------------------

    def get(self, timeout: Optional[float] = None) -> Optional[Frame]:
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def peek(self) -> Optional[Frame]:
        return self._queue[-1] if self._queue else None

    def size(self) -> int:
        return len(self._queue)

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not _HAS_DXCAM:
            raise RuntimeError("dxcam not installed; install with `pip install dxcam`")
        if self.window:
            self._cam = dxcam.create(self.window)
        elif self.region:
            self._cam = dxcam.create(region=self.region)
        else:
            self._cam = dxcam.create()
        if self._cam is None:
            raise RuntimeError("failed to create dxcam capture object")
        self._cam.target_height = max(1, int(self._cam.target_height))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cam = None

    # -- internals -----------------------------------------------------------

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            next_t = now + self._interval
            try:
                frame = self._cam.get_frame()
            except Exception:
                frame = None
            if frame is None:
                continue
            self._frame_id += 1
            with self._lock:
                self._queue.append(Frame(frame, time.perf_counter(), self._frame_id))


def capture_loop(capture: CaptureProducer, stop_event: threading.Event,
                 on_frame) -> None:
    """Convenience: drive a CaptureProducer until stop_event is set.

    Args:
        capture: A CaptureProducer instance.
        stop_event: threading.Event to stop the loop.
        on_frame: callable(Frame) called for each frame.
    """
    capture.start()
    try:
        while not stop_event.is_set():
            frame = capture.get(timeout=0.1)
            if frame is not None:
                on_frame(frame)
    finally:
        capture.stop()