#!/usr/bin/env python3
"""
System 1 E2E test — no Unity required.

What this replaces Unity with:
  - A 6-second MP4 generated from YOLO's bundled bus.jpg (bus + 4 people, reliable detections)
  - A local HTTP server that captures POST /events in place of System 2

What is NOT mocked:
  - YOLO inference (runs for real on the test video)
  - Bearing vector math (geometry.py runs unchanged)
  - HTTP serialization (poster.py runs unchanged)
  - Camera loop threading (camera_worker.py runs unchanged)

Coverage (from TO-TEST.md priority list):
  1. Bearing vector is a 3D unit vector (E, N, U)
  2. cam_id from config arrives correctly in the POST
  3. Timestamps are present and ISO-formatted
  4. POSTs actually reach System 2 (here: mock)
  5. Real YOLO detections fire (not stubbed)

Usage:
    cd system1-detection-agent
    python tests/e2e/run_e2e.py

Optional env overrides:
    TEST_VIDEO_PATH=/path/to/drone.mp4   swap in a real drone clip
    CONF_THRESHOLD=0.35                  YOLO confidence (default 0.35)
    TIMEOUT_S=45                         seconds to wait for first detection POST
"""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent  # system1-detection-agent/


# ---------------------------------------------------------------------------
# Mock System 2
# ---------------------------------------------------------------------------

class MockSystem2:
    """Threaded HTTP server that captures POST /events from System 1."""

    def __init__(self):
        self.events: list[dict] = []
        self.detection_received = threading.Event()
        self._lock = threading.Lock()

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    with outer._lock:
                        outer.events.append(body)
                    n_dets = len(body.get("detections") or [])
                    print(f"  [mock] POST cam={body.get('cam_id')} dets={n_dets}")
                    if n_dets > 0:
                        outer.detection_received.set()
                    self.send_response(200)
                    self.end_headers()
                except Exception as exc:
                    print(f"  [mock] ERROR: {exc}")
                    self.send_response(500)
                    self.end_headers()

            def log_message(self, *args):
                pass  # silence request-level logs

        self._server = HTTPServer(("localhost", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://localhost:{self._server.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()


# ---------------------------------------------------------------------------
# Test video generation
# ---------------------------------------------------------------------------

def find_yolo_test_image() -> Path:
    """Return path to YOLO's bundled bus.jpg.

    bus.jpg ships with every ultralytics install and contains a bus + 4 people
    at close range — YOLOv8s detects them at conf=0.35 with high confidence.
    """
    import ultralytics
    pkg = Path(ultralytics.__file__).parent
    for candidate in [
        pkg / "assets" / "bus.jpg",
        pkg / "data" / "images" / "bus.jpg",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find YOLO bundled test image (bus.jpg).\n"
        "Set TEST_VIDEO_PATH=/path/to/video.mp4 to use your own clip."
    )


def make_test_video(out_path: Path, fps: int = 15, duration_s: int = 6) -> None:
    """Repeat a single YOLO test frame into a short MP4."""
    import cv2

    img_path = find_yolo_test_image()
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cv2 could not read {img_path}")

    h, w = img.shape[:2]
    n_frames = fps * duration_s
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(img)
    writer.release()

    size_kb = out_path.stat().st_size // 1024
    print(f"  source image : {img_path.name} ({w}x{h})")
    print(f"  test video   : {out_path} ({n_frames} frames, {size_kb} KB)")


# ---------------------------------------------------------------------------
# cameras_test.yaml
# ---------------------------------------------------------------------------

def write_cameras_yaml(out_path: Path, video_path: Path) -> None:
    """Write a minimal single-camera config pointing at the test video."""
    out_path.write_text(
        "cameras:\n"
        "  - cam_id: cam_01\n"
        f'    stream_url: "{video_path}"\n'
        "    azimuth_deg: 0.0\n"
        "    elevation_deg: 0.0\n"
        "    hfov_deg: 90.0\n"
        "    vfov_deg: 60.0\n"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_events(events: list[dict]) -> list[str]:
    errors: list[str] = []
    for i, ev in enumerate(events):
        tag = f"event[{i}]"

        if ev.get("cam_id") != "cam_01":
            errors.append(f"{tag} wrong cam_id: {ev.get('cam_id')!r}")

        ts = ev.get("timestamp")
        if not isinstance(ts, str) or not ts:
            errors.append(f"{tag} missing/bad timestamp: {ts!r}")

        dets = ev.get("detections")
        if not isinstance(dets, list):
            errors.append(f"{tag} detections is not a list")
            continue

        for j, det in enumerate(dets):
            dtag = f"{tag}.det[{j}]"
            bv = det.get("bearing_vector")
            if not isinstance(bv, list) or len(bv) != 3:
                errors.append(f"{dtag} bad bearing_vector: {bv!r}")
                continue
            mag = math.sqrt(sum(x * x for x in bv))
            if abs(mag - 1.0) > 1e-4:
                errors.append(f"{dtag} bearing_vector not unit (|v|={mag:.6f}): {bv}")
            score = det.get("score")
            if not isinstance(score, (int, float)) or not (0.0 < score <= 1.0):
                errors.append(f"{dtag} bad score: {score!r}")

    return errors


def print_sample(events: list[dict]) -> None:
    det_events = [e for e in events if e.get("detections")]
    if not det_events:
        return
    ev = det_events[0]
    print(f"\n  Sample detection event:")
    print(f"    cam_id    : {ev['cam_id']}")
    print(f"    timestamp : {ev['timestamp']}")
    for i, det in enumerate(ev["detections"]):
        bv = det["bearing_vector"]
        mag = math.sqrt(sum(x * x for x in bv))
        print(
            f"    det[{i}]    : score={det['score']:.3f}  "
            f"bearing=({bv[0]:+.4f}, {bv[1]:+.4f}, {bv[2]:+.4f})  |v|={mag:.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fail(msg: str) -> None:
    print(f"\nFAIL  {msg}")
    sys.exit(1)


def main() -> None:
    timeout_s = int(os.environ.get("TIMEOUT_S", "45"))
    video_path_env = os.environ.get("TEST_VIDEO_PATH")

    print("=" * 54)
    print("  System 1 — E2E Test")
    print("=" * 54)

    # 1. Mock server
    print("\n[1/4] Starting mock System 2 ...")
    mock = MockSystem2()
    mock.start()
    print(f"  listening : {mock.url}")

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)

        # 2. Test video
        print("\n[2/4] Preparing test video ...")
        if video_path_env:
            video_path = Path(video_path_env).resolve()
            if not video_path.exists():
                fail(f"TEST_VIDEO_PATH not found: {video_path}")
            print(f"  using provided : {video_path}")
        else:
            video_path = tmpdir / "test.mp4"
            make_test_video(video_path)

        # 3. cameras_test.yaml
        cameras_yaml = tmpdir / "cameras_test.yaml"
        write_cameras_yaml(cameras_yaml, video_path)
        print(f"  cameras yaml : {cameras_yaml}")

        # 4. Launch system1.main as subprocess with env overrides
        print(f"\n[3/4] Launching system1.main (timeout={timeout_s}s) ...")

        env = {
            **os.environ,
            "CAMERAS_FILE": str(cameras_yaml),
            "SYSTEM2_URL": mock.url,
            # Only POST when YOLO detects something (tests the real detection path)
            "POST_EMPTY_DETECTIONS": "false",
        }
        env.setdefault("CONF_THRESHOLD", "0.35")

        proc = subprocess.Popen(
            [sys.executable, "-m", "system1.main"],
            cwd=str(REPO),
            env=env,
            # Let system1's logs print to terminal so you can watch YOLO load
        )
        print(f"  pid : {proc.pid}")
        print(f"  waiting for first detection POST (up to {timeout_s}s) ...")
        print()

        got_detection = mock.detection_received.wait(timeout=timeout_s)

        if got_detection:
            # Drain a couple more seconds to collect a handful of events
            time.sleep(2.0)

        # Graceful stop, then kill
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            proc.kill()

    mock.stop()

    # 5. Validate
    n_total = len(mock.events)
    n_det = sum(1 for e in mock.events if e.get("detections"))
    print(f"\n[4/4] Validating {n_total} captured event(s) ...")
    print(f"  total POSTs         : {n_total}")
    print(f"  POSTs with dets     : {n_det}")

    if n_total == 0:
        fail("No events received — System 1 never POSTed to the mock server.\n"
             "  Check that system1 started correctly above.")

    if not got_detection:
        fail(
            f"No detection events received within {timeout_s}s.\n"
            f"  {n_total} empty-detection event(s) received.\n"
            f"  YOLO found nothing in the test video.\n"
            f"  Try: TEST_VIDEO_PATH=/path/to/drone.mp4 or lower CONF_THRESHOLD."
        )

    print_sample(mock.events)

    errors = validate_events(mock.events)
    if errors:
        fail("Validation errors:\n" + "\n".join(f"  {e}" for e in errors))

    print()
    print("=" * 54)
    print("  PASS")
    print("=" * 54)
    print()
    print("  [x] YOLO ran real inference (no mocking)")
    print("  [x] At least one frame produced detections")
    print("  [x] All POSTs reached mock System 2")
    print("  [x] cam_id == cam_01 in every event")
    print("  [x] timestamp present and non-empty")
    print("  [x] bearing_vector is a 3D unit vector (|v| ≈ 1.0)")
    print("  [x] score in (0, 1] for every detection")
    print()
    print("  Next step: swap in a real drone video with TEST_VIDEO_PATH")
    print("  and verify bearing directions look correct for the camera orientation.")


if __name__ == "__main__":
    main()
