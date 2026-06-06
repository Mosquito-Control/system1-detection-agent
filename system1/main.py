import logging
import sys
import threading

import yaml

from system1.camera_worker import run_camera
from system1.config import settings
from system1.detector import DroneDetector
from system1.models import CameraConfig
from system1.udp_listener import listen

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
    dump_mode = "--dump" in sys.argv  # print UDP datagrams and exit, no POSTing

    cameras = load_cameras(settings.cameras_file)
    sim_cameras = [c for c in cameras if c.mode == "sim"]
    rtsp_cameras = [c for c in cameras if c.mode == "rtsp"]

    logger.info(
        "System 1 starting — sim cams: %d, rtsp cams: %d",
        len(sim_cameras), len(rtsp_cameras),
    )

    threads: list[threading.Thread] = []

    if sim_cameras:
        t = threading.Thread(
            target=listen,
            args=(sim_cameras, settings, dump_mode),
            daemon=True,
            name="udp-listener",
        )
        threads.append(t)

    if rtsp_cameras:
        detector = DroneDetector(
            settings.model_path,
            settings.conf_threshold,
            settings.target_classes or None,
        )
        for cam in rtsp_cameras:
            t = threading.Thread(
                target=run_camera,
                args=(cam, detector, settings),
                daemon=True,
                name=f"rtsp-{cam.cam_id}",
            )
            threads.append(t)

    if not threads:
        logger.error("No cameras configured — exiting")
        sys.exit(1)

    for t in threads:
        t.start()

    logger.info("All workers running. Ctrl-C to stop.")
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
