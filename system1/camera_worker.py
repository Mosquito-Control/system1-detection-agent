import logging
import time

from system1.bbox_store import Bbox, store as bbox_store
from system1.config import Settings
from system1.detector import DroneDetector
from system1.geometry import bbox_center_to_bearing
from system1.models import CameraConfig, CameraEvent, Detection
from system1.poster import post_event
from system1.rtsp_source import FrameSource

logger = logging.getLogger(__name__)


def run_camera(cam: CameraConfig, detector: DroneDetector, settings: Settings) -> None:
    """Per-camera loop for RTSP mode: capture → YOLO detect → bearing → POST.

    Intended to run in its own thread. Reconnects automatically on lost frames.
    """
    logger.info("RTSP worker started: cam=%s url=%s", cam.cam_id, cam.stream_url)
    while True:
        try:
            _camera_loop(cam, detector, settings)
        except Exception as exc:
            logger.error("cam=%s crashed: %s — restarting in 5s", cam.cam_id, exc)
            time.sleep(5.0)


def _camera_loop(cam: CameraConfig, detector: DroneDetector, settings: Settings) -> None:
    source = FrameSource(cam.stream_url, settings.capture_buffer_size)
    consecutive_failures = 0
    try:
        while True:
            t0 = time.monotonic()
            frame, ts = source.grab()

            if frame is None:
                consecutive_failures += 1
                if consecutive_failures % 10 == 1:
                    logger.warning("cam=%s no frame (#%d)", cam.cam_id, consecutive_failures)
                # OpenCV's VideoCapture doesn't auto-reconnect when the
                # publisher (Unity/MediaMTX) drops the stream — the cap
                # silently returns no frames forever. Force a teardown +
                # reconnect after ~30s of dead stream so a publisher hiccup
                # doesn't kill the worker permanently. The outer run_camera
                # loop handles the actual restart on the raised exception.
                if consecutive_failures >= 30:
                    raise RuntimeError(
                        f"cam={cam.cam_id} no frame for {consecutive_failures}s "
                        "— forcing reconnect"
                    )
                time.sleep(1.0)
                continue
            consecutive_failures = 0

            h, w = frame.shape[:2]
            raw = detector.detect(frame)

            # Tap point: stash raw pixel bboxes for the dashboard before we
            # collapse each detection to a bearing vector. The store is a
            # latest-snapshot-per-cam dict; the FastAPI server in api.py
            # exposes it to System 3. We always update (even when empty) so
            # the dashboard can clear stale boxes between frames.
            bbox_store.update(
                cam.cam_id,
                tuple(Bbox(x1=x1, y1=y1, x2=x2, y2=y2, score=score)
                      for _cx, _cy, score, x1, y1, x2, y2 in raw),
                w, h,
            )

            detections = tuple(
                Detection(
                    bearing_vector=bbox_center_to_bearing(
                        cx, cy, w, h,
                        cam.azimuth_deg, cam.elevation_deg,
                        cam.hfov_deg, cam.vfov_deg,
                    ),
                    score=score,
                )
                for cx, cy, score, *_ in raw
            )

            if detections or settings.post_empty_detections:
                event = CameraEvent(cam_id=cam.cam_id, timestamp=ts, detections=detections)
                post_event(settings.system2_url, event, settings.post_timeout_s)
                if detections:
                    logger.debug("cam=%s dets=%d", cam.cam_id, len(detections))

            elapsed = time.monotonic() - t0
            # Throttle non-focused cameras so the cam the dashboard is
            # watching gets the bulk of inference time. On CPU this is the
            # difference between the focused cam updating ~2 Hz vs ~0.1 Hz
            # (round-robin across 12 threads). Triangulation still gets a
            # bearing every ~background_post_interval_s from each cam,
            # which is plenty for S2's averaging.
            if bbox_store.is_active(cam.cam_id):
                target = settings.post_interval_s
            else:
                target = settings.background_post_interval_s
            time.sleep(max(0.0, target - elapsed))
    finally:
        source.release()
