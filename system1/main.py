import logging
import sys
import threading

import uvicorn
import yaml

from system1.api import app as bbox_api
from system1.camera_worker import run_camera
from system1.config import settings
from system1.detector import DroneDetector
from system1.models import CameraConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_cameras(path: str) -> list[CameraConfig]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [CameraConfig(**c) for c in data["cameras"]]


def main() -> None:
    cameras = load_cameras(settings.cameras_file)
    if not cameras:
        logger.error("No cameras configured in %s — exiting", settings.cameras_file)
        sys.exit(1)

    detector = DroneDetector(
        settings.model_path,
        settings.conf_threshold,
        settings.target_classes or None,
        imgsz=settings.imgsz,
    )

    logger.info("System 1 starting — %d camera(s)", len(cameras))

    threads = [
        threading.Thread(
            target=run_camera,
            args=(cam, detector, settings),
            daemon=True,
            name=f"cam-{cam.cam_id}",
        )
        for cam in cameras
    ]

    for t in threads:
        t.start()

    # Bbox tap server — exposes the latest YOLO bboxes per cam to System 3.
    # Daemon thread so Ctrl-C still works on the main loop. Port 8090 is
    # mapped in docker-compose; the dashboard's /api/yolo-bboxes proxies to
    # http://localhost:8090.
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(
            bbox_api, host="0.0.0.0", port=8090, log_level="warning",
        ),
        daemon=True,
        name="bbox-api",
    )
    api_thread.start()
    logger.info("bbox tap listening on :8090 (GET /bboxes, /bboxes/{cam_id})")

    logger.info("All workers running. Ctrl-C to stop.")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
