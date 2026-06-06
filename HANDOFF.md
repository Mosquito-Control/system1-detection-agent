# System 1 — Detection Agent: Handoff

## What This Service Does

System 1 is the bridge between the cameras and System 2.

```
Unity sim  ──UDP port 9870──►  System 1  ──POST /events──►  System 2  ──►  positions DB  ──►  System 3 dashboard
Real cam   ──RTSP stream──────►  (YOLO)  ─────────────────►
```

For each camera frame it:
1. Gets the detection (from Unity labels or YOLO)
2. Converts the pixel bounding box to an ENU bearing vector (the direction from the camera toward the drone)
3. POSTs a structured event to System 2's `/events` endpoint

System 2 does the rest (triangulation, GPS computation, DB write).

---

## Repository

`https://github.com/Tion-ping/system1-detection-agent`

---

## Two Operating Modes

### `sim` mode — for the Unity simulation (current default)

Unity broadcasts one UDP JSON datagram per frame on **port 9870**, covering all cameras. It already contains the exact detection pixel coordinates, the camera's intrinsic matrix K, and the camera's world rotation quaternion. No YOLO needed.

System 1 picks the datagram apart, computes a bearing vector per detection using:

```
d_cam = normalize([(u - cx)/fx,  -(v - cy)/fy,  1])   # back-project pixel; negate y (image down, Unity cam up)
d_world = quaternion_rotate(rot_q, d_cam)              # rotate to Unity world space
E = d_world.x,  N = d_world.z,  U = d_world.y         # remap Unity (+X,+Y,+Z) → ENU (+E,+U,+N)
bearing_vector = normalize([E, N, U])
```

Timestamp sent to System 2 is `t_unix_ms / 1000` from the datagram — all cameras in the same sim frame share the same timestamp, which is essential for System 2's 1-second triangulation window.

### `rtsp` mode — for real physical cameras

Opens an RTSP stream with OpenCV, runs YOLOv8 on each frame, converts bbox centers to ENU using azimuth/elevation/FOV angles from `cameras.yaml`. Each camera runs in its own thread.

---

## Running It

### Prerequisites

```bash
pip install -r requirements.txt
```

### Locally (sim mode, Unity on same machine)

```bash
python -m system1.main
```

### Debug: print UDP datagrams without POSTing to System 2

```bash
python -m system1.main --dump
```

### Docker

```bash
docker-compose up
```

`docker-compose.yml` uses `--network host` so it can reach Unity's UDP broadcast and RTSP streams on localhost.

---

## Configuration

### `cameras.yaml` — the only file you normally need to edit

```yaml
cameras:
  - unity_name: "cam0"    # must match cameras[].name in the Unity UDP datagram
    cam_id: "cam_01"      # must match System 2's cameras.yaml exactly
    mode: sim

  - unity_name: "cam1"
    cam_id: "cam_02"
    mode: sim
```

For a real camera (RTSP + YOLO), change `mode: rtsp` and add:

```yaml
  - unity_name: ""
    cam_id: "cam_01"
    mode: rtsp
    stream_url: "rtsp://192.168.1.10:8554/stream"
    azimuth_deg: 45.0      # camera pan, degrees clockwise from North
    elevation_deg: -20.0   # camera tilt, degrees above horizon (negative = looking down)
    hfov_deg: 90.0         # horizontal FOV from camera spec
    vfov_deg: 60.0         # vertical FOV
```

### Environment variables / `.env`

| Variable | Default | Description |
|---|---|---|
| `SYSTEM2_URL` | Azure URL | System 2 endpoint |
| `UDP_PORT` | `9870` | Unity UDP label port |
| `MODEL_PATH` | `yolov8s.pt` | YOLO weights — set to a fine-tuned `.pt` path to swap the model, no code change |
| `CONF_THRESHOLD` | `0.35` | YOLO confidence cutoff |
| `POST_EMPTY_DETECTIONS` | `false` | If `true`, POST even when no drone seen (increases noise) |
| `POST_TIMEOUT_S` | `2.0` | HTTP timeout per POST |

---

## System 2 Integration Contract

**URL:** `POST https://system2-api.agreeablesea-31719cb5.westeurope.azurecontainerapps.io/events`

**Payload:**
```json
{
  "cam_id": "cam_01",
  "timestamp": "2026-06-06T14:23:00.456Z",
  "detections": [
    { "bearing_vector": [0.342, 0.876, -0.340], "score": 1.0 }
  ]
}
```

**Hard rules:**
- `cam_id` must exactly match an entry in System 2's `cameras.yaml`. Mismatch → silently dropped.
- `timestamp` must be UTC. In sim mode it comes from the UDP datagram (`t_unix_ms`), so it's automatically correct.
- `bearing_vector` is a 3-element ENU unit vector `[E, N, U]`.
- `score` is 1.0 for sim detections (ground truth). Use YOLO confidence for real cameras.
- Empty `detections: []` is valid — camera sees no drone this frame.

---

## File Map

```
system1/
├── config.py          # all settings (env-var backed via pydantic-settings)
├── models.py          # CameraConfig, Detection, CameraEvent dataclasses
├── geometry.py        # bbox pixel → ENU bearing vector (both sim and rtsp paths)
├── udp_listener.py    # sim mode: recv UDP, parse datagram, POST per camera
├── rtsp_source.py     # rtsp mode: OpenCV VideoCapture wrapper
├── detector.py        # rtsp mode: YOLOv8 wrapper
├── camera_worker.py   # rtsp mode: per-camera thread (capture → detect → post)
├── poster.py          # shared: HTTP POST to System 2
└── main.py            # entry point: reads cameras.yaml, starts threads
```

---

## What Must Be Tested First (in priority order)

### 1. Bearing vector correctness — most urgent

Wrong bearing vectors produce wrong GPS positions with no error from System 2. Use `--dump` to capture a real datagram, then manually verify:

```bash
python -m system1.main --dump
# Copy a camera entry's rot_q and a detection's center_px, then:
python3 -c "
from system1.geometry import sim_bearing
K = [623.54, 0, 640, 0, 623.54, 360, 0, 0, 1]
rot_q = [0, 0, 0, 1]          # paste real value here
center_px = [640, 360]        # paste real value here
print(sim_bearing(center_px, K, rot_q))
# For identity quaternion + center pixel: expect (0.0, 1.0, 0.0) = pure North
"
```

Compare the direction with where you can see the drone in the Unity scene relative to the camera.

### 2. Camera name mapping

Does `unity_name` in `cameras.yaml` match what Unity actually sends in `cameras[].name`? Check with `--dump` — look at the `"name"` field in each camera object. If names don't match, events are silently discarded.

### 3. End-to-end: positions appearing in DB

Once 1 and 2 are confirmed, run two cameras in the sim with a drone visible to both, then query System 2's DB:

```bash
psql -h drone-detection-pg.postgres.database.azure.com -U droneadmin -d dronedetection \
  -c "SELECT cam_pair, lat, lon, alt_m, inserted_at FROM positions ORDER BY inserted_at DESC LIMIT 5;"
```

A row should appear within ~1 second of the drone being visible to both cameras.

---

## Deploying on Individual Real Cameras

Each physical camera runs its own System 1 instance with a single-entry `cameras.yaml` in `mode: rtsp`. The threading model already supports this — one camera = one RTSP worker thread, everything else is idle.

If the sim UDP path is also needed on each machine: the Unity UDP broadcast must be routable to each camera host (port 9870 open). Each instance filters the datagram to only its own `unity_name` entry.

---

## Swapping in a Fine-Tuned YOLO Model

```bash
MODEL_PATH=/path/to/drone-finetuned.pt python -m system1.main
```

No code change. The model is loaded once at startup and shared across all RTSP camera threads.
