from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CameraConfig:
    cam_id: str        # sent to System 2, must match System 2's cameras.yaml
    stream_url: str    # RTSP URL (rtsp://host:port/path) or local file path for testing
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 60.0


@dataclass(frozen=True)
class Detection:
    bearing_vector: tuple[float, float, float]  # (E, N, U) unit vector in ENU frame
    score: float


@dataclass(frozen=True)
class CameraEvent:
    cam_id: str
    timestamp: datetime
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "timestamp": self.timestamp.isoformat(),
            "detections": [
                {"bearing_vector": list(d.bearing_vector), "score": d.score}
                for d in self.detections
            ],
        }
