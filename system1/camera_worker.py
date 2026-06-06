import logging
import time

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
                time.sleep(1.0)
                continue
            consecutive_failures = 0

            h, w = frame.shape[:2]
            raw = detector.detect(frame)

            detections = tuple(
                Detection(
                    bearing_vector=bbox_center_to_bearing(
                        cx, cy, w, h,
                        cam.azimuth_deg, cam.elevation_deg,
                        cam.hfov_deg, cam.vfov_deg,
                    ),
                    score=score,
                )
                for cx, cy, score in raw
            )

            if detections or settings.post_empty_detections:
                event = CameraEvent(cam_id=cam.cam_id, timestamp=ts, detections=detections)
                post_event(settings.system2_url, event, settings.post_timeout_s)
                if detections:
                    logger.debug("cam=%s dets=%d", cam.cam_id, len(detections))

            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, settings.post_interval_s - elapsed)
            time.sleep(sleep_s)
    finally:
        source.release()
