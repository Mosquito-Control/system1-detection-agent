import logging

import requests

from system1.models import CameraEvent

logger = logging.getLogger(__name__)


def post_event(url: str, event: CameraEvent, timeout_s: float) -> bool:
    """POST a CameraEvent to System 2's /events endpoint.

    Returns True on success, False on any error. No retry — logs warning.
    """
    try:
        resp = requests.post(f"{url}/events", json=event.to_dict(), timeout=timeout_s)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("POST /events failed for %s: %s", event.cam_id, exc)
        return False
