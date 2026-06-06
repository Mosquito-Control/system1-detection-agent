import logging
import sys
import threading

import yaml

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

    logger.info("All workers running. Ctrl-C to stop.")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
