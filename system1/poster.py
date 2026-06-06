import logging
import time

import requests

from system1.models import CameraEvent

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 0.25  # → 0.25s, 0.5s, 1.0s — total <2s, fits inside post_interval_s


def post_event(url: str, event: CameraEvent, timeout_s: float) -> bool:
    """POST a CameraEvent to System 2's /events endpoint.

    Retries up to _MAX_ATTEMPTS with exponential backoff on transient failures.
    Returns True on success, False on final failure. Detections are not queued
    to disk — bounded retry is enough for typical S2 restarts within seconds.
    """
    payload = event.to_dict()
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.post(f"{url}/events", json=payload, timeout=timeout_s)
            resp.raise_for_status()
            if attempt > 0:
                logger.info("POST /events succeeded for %s on attempt %d", event.cam_id, attempt + 1)
            return True
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
    logger.warning("POST /events failed for %s after %d attempts: %s",
                   event.cam_id, _MAX_ATTEMPTS, last_exc)
    return False
