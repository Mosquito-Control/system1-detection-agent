"""In-process shared store for the latest YOLO bboxes per camera.

This is the tap point that lets System 3's dashboard render real on-frame
detections instead of geometric position projections. The camera workers
already compute pixel bboxes inside detector.detect(); we used to throw
them away after converting the center to a bearing vector. Now we also
stash them here, and api.py exposes them to the dashboard.

Thread-safety: write/read paths are short critical sections under a single
Lock — cheap given we update at the same cadence as the existing post loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Bbox:
    """One on-frame detection. Pixel coords, origin top-left, y-down."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True)
class CamSnapshot:
    """Latest detections for one camera + the frame resolution they're in."""
    bboxes: tuple[Bbox, ...]
    frame_w: int
    frame_h: int
    ts: float            # epoch seconds when this snapshot was produced


class BboxStore:
    """Latest-snapshot-per-camera. Older snapshots are overwritten.

    Also tracks which camera the dashboard is currently viewing so the
    camera workers can prioritise inference on that one. On a single-CPU
    host with 12 RTSP threads, the X-size model otherwise round-robins at
    ~10s/cam — i.e. the box the operator is staring at lags badly. With
    the active mark, the focused cam gets ~80% of the inference budget.
    """

    # Treat the active mark as stale after this many seconds without a
    # query. Matches the dashboard's modal-close behaviour (which simply
    # stops polling) so the boost releases ~3s after the modal closes.
    _ACTIVE_TTL_S = 3.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cams: dict[str, CamSnapshot] = {}
        self._active_cam: str | None = None
        self._active_ts: float = 0.0

    def update(self, cam_id: str, bboxes: tuple[Bbox, ...], w: int, h: int) -> None:
        snap = CamSnapshot(bboxes=bboxes, frame_w=w, frame_h=h, ts=time.time())
        with self._lock:
            self._cams[cam_id] = snap

    def get(self, cam_id: str) -> CamSnapshot | None:
        with self._lock:
            return self._cams.get(cam_id)

    def snapshot(self) -> dict[str, CamSnapshot]:
        with self._lock:
            return dict(self._cams)

    def mark_active(self, cam_id: str) -> None:
        """Called from the bbox tap on every per-cam GET. Refreshes the
        TTL so the camera the dashboard is polling stays prioritised."""
        with self._lock:
            self._active_cam = cam_id
            self._active_ts = time.time()

    def is_active(self, cam_id: str) -> bool:
        """Camera workers call this to decide whether to throttle. Active
        cam runs full-tilt; everyone else adds a sleep to the loop so the
        active inference gets the CPU."""
        with self._lock:
            if self._active_cam is None:
                return False
            if time.time() - self._active_ts > self._ACTIVE_TTL_S:
                return False
            return self._active_cam == cam_id


# Process-wide singleton. Camera workers import this and call .update();
# the FastAPI handler reads via .snapshot(). One store, one process.
store = BboxStore()
