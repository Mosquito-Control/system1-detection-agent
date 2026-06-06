import logging
import os
from datetime import datetime, timezone

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Required for RTSP-over-TCP with OpenCV/FFmpeg backend
_RTSP_TRANSPORT_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_RTSP_TRANSPORT_VAL = "rtsp_transport;tcp"


class FrameSource:
    """Wraps cv2.VideoCapture for RTSP streams or local video files."""

    def __init__(self, stream_url: str, buffer_size: int = 1) -> None:
        os.environ.setdefault(_RTSP_TRANSPORT_ENV, _RTSP_TRANSPORT_VAL)
        self._cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {stream_url}")
        logger.info("Opened stream: %s", stream_url)

    def grab(self) -> tuple[np.ndarray | None, datetime]:
        """Read one frame. Timestamp is captured at read time (UTC).

        Returns (frame, timestamp). Frame is None if the source returned no data.
        """
        ts = datetime.now(timezone.utc)
        ok, frame = self._cap.read()
        return (frame if ok else None), ts

    def release(self) -> None:
        self._cap.release()
