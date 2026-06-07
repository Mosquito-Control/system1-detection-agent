from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cameras_file: str = "cameras.yaml"
    system2_url: str = "https://system2-api.agreeablesea-31719cb5.westeurope.azurecontainerapps.io"

    # YOLO / real-camera settings
    model_path: str = "yolov8s.pt"   # override MODEL_PATH env var to swap weights
    conf_threshold: float = 0.35
    target_classes: list[int] = []   # empty = detect all COCO classes
    # Inference image size — Ultralytics letterboxes the frame to this square
    # before running the network. 640 is the native trained size. Smaller
    # values speed inference roughly linearly but shrink small drones below
    # the model's detection floor (stride 32) — 480 dropped recall noticeably
    # on this rig, so we stay at 640 and recover throughput via the
    # active-cam boost in bbox_store instead.
    imgsz: int = 640

    # POST behaviour
    post_interval_s: float = 0.1     # min seconds between posts for the dashboard-focused camera
    # Non-focused cameras sleep this long between frames so their inference
    # doesn't starve the cam the operator is watching. 1.5s ≈ 1 frame per
    # 1.5s per cam, which still gives S2 enough bearing samples to
    # triangulate without burning the CPU on cams nobody is looking at.
    background_post_interval_s: float = 1.5
    post_timeout_s: float = 2.0
    post_empty_detections: bool = False  # if True, POST even when no drones detected

    # RTSP capture
    capture_buffer_size: int = 1     # 1 = always read freshest frame

    class Config:
        env_file = ".env"


settings = Settings()
