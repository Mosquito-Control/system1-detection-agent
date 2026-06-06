from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cameras_file: str = "cameras.yaml"
    system2_url: str = "https://system2-api.agreeablesea-31719cb5.westeurope.azurecontainerapps.io"

    # YOLO / real-camera settings
    model_path: str = "yolov8s.pt"   # override MODEL_PATH env var to swap weights
    conf_threshold: float = 0.35
    target_classes: list[int] = []   # empty = detect all COCO classes

    # POST behaviour
    post_interval_s: float = 0.1     # min seconds between posts per camera (RTSP mode)
    post_timeout_s: float = 2.0
    post_empty_detections: bool = False  # if True, POST even when no drones detected

    # RTSP capture
    capture_buffer_size: int = 1     # 1 = always read freshest frame

    class Config:
        env_file = ".env"


settings = Settings()
