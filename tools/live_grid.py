"""Live 4x2 grid viewer with SAHI-style sliced inference.

Each cam runs YOLOv8n-drone three ways every tick:
  - full 1280x720 frame
  - 2x2 grid (4 tiles, 20% overlap)
  - 4x4 grid (16 tiles, 20% overlap)
Boxes from all 21 tiles are projected back to full-frame coords and merged
with NMS. Tiles render progressively so the window stays responsive.

Usage:
    source .venv/bin/activate
    python tools/live_grid.py [--conf 0.10] [--imgsz 320]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from system1.geometry import bbox_center_to_bearing


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "drone-detection-ml" / "models" / "yolov8n-drone.onnx"
)
URLS = [f"rtsp://127.0.0.1:8554/cam{i}" for i in range(8)]
CAMERAS_YAML = REPO_ROOT / "cameras.yaml"


def load_cam_config():
    """Return list of dicts (one per Unity cam index 0..7) with
    cam_id, az/el/FoV — read from system1's cameras.yaml."""
    with open(CAMERAS_YAML) as f:
        data = yaml.safe_load(f)
    by_idx = {}
    for c in data["cameras"]:
        url = c["stream_url"]
        # rtsp://.../camN -> N
        idx = int(url.rsplit("cam", 1)[-1])
        by_idx[idx] = c
    return [by_idx[i] for i in range(8)]


def post_event(system2_url: str, cam_id: str, ts: datetime,
               dets_full: list, src_w: int, src_h: int, cam_cfg: dict) -> None:
    """Build CameraEvent payload from SAHI dets (full-frame coords) and POST."""
    detections = []
    for (x1, y1, x2, y2, score) in dets_full:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bearing = bbox_center_to_bearing(
            cx, cy, src_w, src_h,
            cam_cfg["azimuth_deg"], cam_cfg["elevation_deg"],
            cam_cfg["hfov_deg"], cam_cfg["vfov_deg"],
        )
        detections.append({"bearing_vector": list(bearing), "score": score})
    payload = {
        "cam_id": cam_id,
        "timestamp": ts.isoformat(),
        "detections": detections,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{system2_url}/events", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=1.0).read()
    except Exception as e:
        # don't crash the viewer on a single POST hiccup
        print(f"  POST fail cam={cam_id}: {e}", flush=True)
TILE_W, TILE_H = 480, 270   # display tile size (16:9)
GRID_COLS, GRID_ROWS = 4, 2

# SAHI slicing parameters
OVERLAP = 0.20
NMS_IOU = 0.45


def slice_grid(W: int, H: int, rows: int, cols: int, overlap: float = OVERLAP):
    """Overlapping slice rectangles (x1, y1, x2, y2)."""
    out = []
    cell_w = W / cols
    cell_h = H / rows
    ox = cell_w * overlap
    oy = cell_h * overlap
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * cell_w - ox))
            y1 = max(0, int(r * cell_h - oy))
            x2 = min(W, int((c + 1) * cell_w + ox))
            y2 = min(H, int((r + 1) * cell_h + oy))
            out.append((x1, y1, x2, y2))
    return out


def nms_keep(boxes, scores, iou_thresh: float = NMS_IOU):
    if not boxes:
        return []
    # cv2.dnn.NMSBoxes wants (x, y, w, h)
    bboxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    if len(idxs) == 0:
        return []
    return [int(i) for i in (idxs.flatten() if hasattr(idxs, "flatten") else idxs)]


class CamThread(threading.Thread):
    def __init__(self, idx: int, url: str) -> None:
        super().__init__(daemon=True, name=f"cam{idx}")
        self.idx = idx
        self.url = url
        self._frame: np.ndarray | None = None
        self._frame_ts: datetime | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.fps = 0.0

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_with_ts(self):
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), self._frame_ts

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(1.0); continue
            t0, n = time.monotonic(), 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                ts = datetime.now(timezone.utc)
                with self._lock:
                    self._frame = frame
                    self._frame_ts = ts
                n += 1
                dt = time.monotonic() - t0
                if dt >= 1.0:
                    self.fps = n / dt; n = 0; t0 = time.monotonic()
            cap.release()
            time.sleep(0.5)


def _infer_slices(model, frame, slices, conf: float, imgsz: int):
    """Run inference on each slice; project boxes back to full-frame coords."""
    all_boxes, all_scores = [], []
    for (x1, y1, x2, y2) in slices:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (imgsz, imgsz))
        r = model(crop_resized, conf=conf, classes=[0], imgsz=imgsz, verbose=False)[0]
        if len(r.boxes) == 0:
            continue
        sx = (x2 - x1) / imgsz
        sy = (y2 - y1) / imgsz
        for b in r.boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            all_boxes.append((
                x1 + bx1 * sx, y1 + by1 * sy,
                x1 + bx2 * sx, y1 + by2 * sy,
            ))
            all_scores.append(float(b.conf[0]))
    return all_boxes, all_scores


def sahi_detect(model, frame: np.ndarray, conf: float, imgsz: int):
    """Cascade: 2x2 first; escalate to 4x4 if nothing found. Returns
    (detections, stage_used) where stage is "2x2", "4x4", or "none"."""
    H, W = frame.shape[:2]

    # Stage 1 — 2x2 (4 tiles, ~640x360 each)
    boxes, scores = _infer_slices(model, frame, slice_grid(W, H, 2, 2), conf, imgsz)
    stage = "2x2"

    # Stage 2 — escalate to 4x4 (16 tiles, ~320x180 each) if nothing yet
    if not boxes:
        boxes, scores = _infer_slices(model, frame, slice_grid(W, H, 4, 4), conf, imgsz)
        stage = "4x4" if boxes else "none"

    keep = nms_keep(boxes, scores)
    dets = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]) for i in keep]
    return dets, stage


def annotate_tile(frame: np.ndarray, detections, src_w: int, src_h: int,
                  cam_idx: int, cam_fps: float, dets_full: int, tick_ms: float,
                  stage: str = "") -> np.ndarray:
    tile = cv2.resize(frame, (TILE_W, TILE_H))
    sx = TILE_W / src_w
    sy = TILE_H / src_h
    for (x1, y1, x2, y2, sc) in detections:
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(tile, p1, p2, (0, 255, 0), 2)
        cv2.putText(tile, f"{sc:.2f}", (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    label = f"cam{cam_idx}  src={cam_fps:.1f}fps  dets={dets_full} [{stage}]  {tick_ms:.0f}ms"
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Per-tile inference size. The bundled ONNX is fixed-shape 640.")
    ap.add_argument("--system2", default="http://127.0.0.1:8000",
                    help="System 2 base URL. Set to empty string to disable POSTing.")
    args = ap.parse_args()

    cam_cfgs = load_cam_config()
    post_enabled = bool(args.system2)
    print(f"posting to System 2: {args.system2 if post_enabled else 'disabled'}")

    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")
    print(f"loaded {args.model}  classes={model.names}  imgsz={args.imgsz} conf={args.conf}")
    print(f"SAHI cascade: 2x2 (4 tiles); escalate to 4x4 (16 tiles) if empty. overlap={OVERLAP}")

    threads = [CamThread(i, URLS[i]) for i in range(8)]
    for t in threads: t.start()
    print("waiting for first frame from each cam...")
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if all(t.latest() is not None for t in threads):
            break
        time.sleep(0.2)

    win = "System1 live — SAHI drone-YOLO on Unity (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TILE_W * GRID_COLS, TILE_H * GRID_ROWS)

    # Initialise grid with placeholders so we can update tile-by-tile
    blank = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    tiles_state = [blank.copy() for _ in range(8)]

    tick = 0
    while True:
        tick += 1
        tick_t0 = time.monotonic()
        per_cam_dets = [0] * 8
        per_cam_stage = ["-"] * 8
        # Snapshot all 8 cams' freshest frames + their capture timestamps NOW,
        # before sequential SAHI eats the next 5+ seconds. This keeps the
        # per-cam timestamps clustered so they fall inside System 2's window.
        snapshot = [threads[i].latest_with_ts() for i in range(8)]
        for cam_i in range(8):
            cam_t0 = time.monotonic()
            frame, ts = snapshot[cam_i]
            if frame is None:
                tiles_state[cam_i] = blank.copy()
            else:
                dets, stage = sahi_detect(model, frame, args.conf, args.imgsz)
                per_cam_dets[cam_i] = len(dets)
                per_cam_stage[cam_i] = stage
                if post_enabled and dets:
                    post_event(args.system2, cam_cfgs[cam_i]["cam_id"], ts, dets,
                               frame.shape[1], frame.shape[0], cam_cfgs[cam_i])
                cam_ms = (time.monotonic() - cam_t0) * 1000
                tiles_state[cam_i] = annotate_tile(
                    frame, dets, frame.shape[1], frame.shape[0],
                    cam_i, threads[cam_i].fps, len(dets), cam_ms, stage,
                )
            # Re-composite + show after every cam so the window animates
            rows = []
            for r_i in range(GRID_ROWS):
                rows.append(np.hstack(tiles_state[r_i * GRID_COLS:(r_i + 1) * GRID_COLS]))
            cv2.imshow(win, np.vstack(rows))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                for t in threads: t.stop()
                cv2.destroyAllWindows()
                return

        tick_ms = (time.monotonic() - tick_t0) * 1000
        print(f"tick #{tick} {tick_ms/1000:.1f}s  dets={per_cam_dets}  stage={per_cam_stage}",
              flush=True)


if __name__ == "__main__":
    main()
