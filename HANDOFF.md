# System 1 — Detection Agent: Handoff

## What This Service Does

System 1 is the bridge between cameras and System 2.

```
Camera (Unity RTSP or real)  ──RTSP stream──►  System 1 (YOLO)  ──POST /events──►  System 2  ──►  positions DB  ──►  System 3 dashboard
```

For each camera frame it:
1. Reads the frame from the RTSP stream (OpenCV)
2. Runs YOLOv8 inference to detect drones
3. Converts the pixel bbox center to an ENU bearing vector (direction from camera toward drone)
4. POSTs a structured event to System 2's `/events` endpoint

Unity cameras and real physical cameras are identical from System 1's perspective — both are
RTSP streams, both get YOLO applied. The only difference is the URL in `cameras.yaml`.

---

## Repository

`https://github.com/Tion-ping/system1-detection-agent`

---

## Related Repositories

| Repo | What it is |
|---|---|
| [`drone-detection-ml`](https://github.com/Tion-ping/drone-detection-ml) | **The ML component.** The drone model (`yolov8n-drone.onnx`), the inference wrapper, the pixel→ENU bearing geometry (canonical reference), and the full integration docs. System 1 runs this model on every RTSP camera (Unity and real). |
| [`system2-positioning-engine`](https://github.com/Tion-ping/system2-positioning-engine) | System 2 — multi-camera ray triangulation; receives our `/events` POSTs and writes GPS positions. |

> **ML note:** the bearing math in `system1/geometry.py` is mirrored in
> `drone-detection-ml/src/drone_detector/geometry.py` (the canonical reference,
> with unit tests). If you change one, change both. See that repo's
> `ARCHITECTURE.md` for the full end-to-end data flow and contracts.

---

## Unity RTSP Contract

Unity streams one RTSP feed per camera:

- URL: `rtsp://<host>:8554/cam0` … `/cam7`
- H.264, 1280×720, ~15 fps, VFR wallclock-stamped
- Transport: TCP (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`)
- HLS browser preview: `http://<host>:8888/camN`

Unity also broadcasts UDP JSON label datagrams on **port 9870** (one per sim frame).
These contain ground-truth drone positions and bounding boxes and are **not** used by
System 1's detection pipeline. Use `tools/unity_gt_listener.py` to read them for YOLO
evaluation or to look up camera `rot_q` values.

---

## Running It

### Prerequisites

```bash
pip install -r requirements.txt
```

### Locally (Unity RTSP on same machine)

```bash
python -m system1.main
```

### Docker

```bash
docker-compose up
```

`docker-compose.yml` uses `--network host` so it can reach Unity's RTSP streams on localhost.

### Ground-truth evaluation (separate terminal, optional)

```bash
python tools/unity_gt_listener.py
# prints Unity ground-truth labels per frame — useful for comparing against YOLO output
```

---

## Configuration

### `cameras.yaml` — the only file you normally need to edit

```yaml
cameras:
  - cam_id: "cam_01"            # must match System 2's cameras.yaml exactly
    stream_url: "rtsp://localhost:8554/cam0"  # Unity RTSP or real camera URL
    azimuth_deg: 0.0            # camera pan, degrees clockwise from North
    elevation_deg: -15.0        # camera tilt, degrees above horizon (negative = looking down)
    hfov_deg: 90.0              # horizontal FOV from camera spec
    vfov_deg: 60.0              # vertical FOV
```

For Unity cameras, `azimuth_deg`/`elevation_deg` must match the Unity scene camera
orientations. Run `tools/unity_gt_listener.py` to read `rot_q` from the UDP labels
if you need to derive these values.

### Environment variables / `.env`

| Variable | Default | Description |
|---|---|---|
| `SYSTEM2_URL` | Azure URL | System 2 endpoint |
| `MODEL_PATH` | `yolov8s.pt` | YOLO weights — set to a fine-tuned `.pt` path to swap the model |
| `CONF_THRESHOLD` | `0.35` | YOLO confidence cutoff |
| `POST_EMPTY_DETECTIONS` | `false` | If `true`, POST even when no drone seen |
| `POST_TIMEOUT_S` | `2.0` | HTTP timeout per POST |
| `POST_INTERVAL_S` | `0.1` | Min seconds between posts per camera |

---

## System 2 Integration Contract

**URL:** `POST https://system2-api.agreeablesea-31719cb5.westeurope.azurecontainerapps.io/events`

**Payload:**
```json
{
  "cam_id": "cam_01",
  "timestamp": "2026-06-06T14:23:00.456Z",
  "detections": [
    { "bearing_vector": [0.342, 0.876, -0.340], "score": 0.87 }
  ]
}
```

**Hard rules:**
- `cam_id` must exactly match an entry in System 2's `cameras.yaml`. Mismatch → silently dropped.
- `timestamp` is UTC, captured at frame read time.
- `bearing_vector` is a 3-element ENU unit vector `[E, N, U]`.
- `score` is YOLO detection confidence.
- Empty `detections: []` is valid — camera sees no drone this frame.

---

## File Map

```
system1/
├── config.py          # all settings (env-var backed via pydantic-settings)
├── models.py          # CameraConfig, Detection, CameraEvent dataclasses
├── geometry.py        # bbox pixel → ENU bearing vector
├── rtsp_source.py     # OpenCV VideoCapture wrapper
├── detector.py        # YOLOv8 wrapper
├── camera_worker.py   # per-camera thread: capture → detect → post
├── poster.py          # HTTP POST to System 2
└── main.py            # entry point: reads cameras.yaml, starts threads
tools/
└── unity_gt_listener.py  # standalone UDP listener for Unity ground-truth labels
```

---

## Swapping in a Fine-Tuned YOLO Model

```bash
MODEL_PATH=/path/to/drone-finetuned.pt python -m system1.main
```

No code change. The model is loaded once at startup and shared across all camera threads.

---

## What Must Be Tested First (in priority order)

### 1. RTSP stream connectivity

```bash
ffprobe -v error -show_streams rtsp://localhost:8554/cam0
# Should show: h264, 1280x720
```

### 2. YOLO detections on Unity frames

Run System 1 with `logging.DEBUG` and watch for `dets=N` log lines per camera.
If `dets=0` on every frame, either the drone isn't visible or YOLO confidence is too low —
adjust `CONF_THRESHOLD` or use a fine-tuned model.

### 3. Bearing vector plausibility

Use `tools/unity_gt_listener.py` to get ground-truth `center_px` for a visible drone, then
manually verify:

```python
from system1.geometry import bbox_center_to_bearing
# plug in the center_px from the GT listener and your camera's config values
print(bbox_center_to_bearing(640, 360, 1280, 720, azimuth_deg=0, elevation_deg=-15, hfov_deg=90, vfov_deg=60))
# compare direction with where the drone is visible in the Unity scene
```

### 4. End-to-end: positions appearing in DB

With two cameras seeing a drone simultaneously, query System 2:

```bash
psql -h drone-detection-pg.postgres.database.azure.com -U droneadmin -d dronedetection \
  -c "SELECT cam_pair, lat, lon, alt_m, inserted_at FROM positions ORDER BY inserted_at DESC LIMIT 5;"
```

A row should appear within ~1 second of the drone being visible to both cameras.
