import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DroneDetector:
    """YOLOv8 inference wrapper for the real-camera (RTSP) path.

    The model is loaded once and shared across camera threads — Ultralytics
    is thread-safe for inference.

    Swap the model by setting MODEL_PATH env var to a fine-tuned .pt file.
    target_classes: list of COCO class IDs to keep; empty list = keep all.
    """

    def __init__(
        self,
        model_path: str,
        conf: float,
        target_classes: list[int] | None,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO  # deferred import — not needed in sim mode

        # Fail fast at startup if the model file is missing. Without this, YOLO()
        # silently tries to fetch from Ultralytics hub and crashes mid-inference
        # inside a network-isolated container.
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLO model not found at {model_path!r}. "
                "Set MODEL_PATH to a local .pt file."
            )

        self._model = YOLO(model_path)
        self._conf = conf
        self._classes = target_classes or None  # None → ultralytics detects all
        self._imgsz = imgsz
        logger.info(
            "YOLO model loaded: %s (conf=%.2f, classes=%s, imgsz=%d)",
            model_path, conf, self._classes, imgsz,
        )

    # Max bbox extent as a fraction of the frame dimension. A drone in the
    # sky at any realistic camera distance occupies a small fraction of the
    # frame. The X model occasionally fires high-confidence detections on
    # entire-frame backgrounds (it was trained on close-up drone crops).
    # Reject any box whose width or height exceeds this fraction.
    _MAX_DIM_FRAC = 0.35

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float, float, float]]:
        """Run inference on one frame.

        Returns a list of (cx_px, cy_px, score, x1, y1, x2, y2) per detection.
        cx/cy are kept for the existing bearing-vector pathway; x1..y2 are the
        full pixel bbox, exposed via bbox_store so the dashboard can render
        the actual on-frame detection instead of a position-back-projection.
        """
        results = self._model(
            frame,
            conf=self._conf,
            classes=self._classes,
            imgsz=self._imgsz,
            verbose=False,
        )
        h, w = frame.shape[:2]
        max_w = w * self._MAX_DIM_FRAC
        max_h = h * self._MAX_DIM_FRAC
        out: list[tuple[float, float, float, float, float, float, float]] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if (x2 - x1) > max_w or (y2 - y1) > max_h:
                    continue
                score = float(box.conf[0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                out.append((cx, cy, score, x1, y1, x2, y2))
        return out
