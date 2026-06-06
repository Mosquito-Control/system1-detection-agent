import json
import logging
import socket
from datetime import datetime, timezone

from system1.config import Settings
from system1.geometry import sim_bearing
from system1.models import CameraConfig, CameraEvent, Detection
from system1.poster import post_event

logger = logging.getLogger(__name__)

_MAX_DATAGRAM = 65535


def listen(cameras: list[CameraConfig], settings: Settings, dump: bool = False) -> None:
    """Receive Unity UDP label datagrams and forward detections to System 2.

    One datagram per sim frame covers all cameras. For each camera in the
    datagram that we recognise (by unity_name) and that has visible detections,
    we compute ENU bearing vectors and POST a CameraEvent to System 2.

    Args:
        cameras: list of CameraConfig with mode == "sim"
        settings: runtime settings
        dump: if True, print each parsed datagram to stdout and skip POSTing
    """
    cam_by_name: dict[str, CameraConfig] = {c.unity_name: c for c in cameras if c.mode == "sim"}
    if not cam_by_name:
        logger.warning("No cameras configured in sim mode — UDP listener idle")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((settings.udp_host, settings.udp_port))
    logger.info("UDP listener bound on %s:%d, watching cameras: %s",
                settings.udp_host, settings.udp_port, list(cam_by_name))

    while True:
        data, addr = sock.recvfrom(_MAX_DATAGRAM)
        try:
            msg = json.loads(data)
        except json.JSONDecodeError as exc:
            logger.warning("Bad datagram from %s: %s", addr, exc)
            continue

        if dump:
            print(json.dumps(msg, indent=2))
            continue

        ts = datetime.fromtimestamp(msg["t_unix_ms"] / 1000.0, tz=timezone.utc)

        for cam_data in msg.get("cameras", []):
            cam = cam_by_name.get(cam_data["name"])
            if cam is None:
                continue

            K = cam_data["K"]
            rot_q = cam_data["rot_q"]

            detections = tuple(
                Detection(
                    bearing_vector=sim_bearing(det["center_px"], K, rot_q),
                    score=1.0,  # ground-truth detection — full confidence
                )
                for det in cam_data.get("detections", [])
                if det.get("visible", True)
            )

            if detections or settings.post_empty_detections:
                event = CameraEvent(cam_id=cam.cam_id, timestamp=ts, detections=detections)
                post_event(settings.system2_url, event, settings.post_timeout_s)
                if detections:
                    logger.debug("cam=%s dets=%d ts=%s", cam.cam_id, len(detections), ts)
