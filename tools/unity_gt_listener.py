"""Standalone Unity ground-truth label listener.

Listens on UDP port 9870 for Unity label datagrams and prints per-frame
ground-truth drone positions and per-camera bounding boxes.

NOT part of System 1's detection pipeline. Use this for:
- Verifying YOLO detections against ground truth
- Reading camera rot_q values to derive azimuth/elevation for cameras.yaml
- Debugging Unity ↔ System 1 integration

Usage:
    python tools/unity_gt_listener.py [--port 9870]
"""

import argparse
import json
import socket

_MAX_DATAGRAM = 65535


def main() -> None:
    parser = argparse.ArgumentParser(description="Unity ground-truth label listener")
    parser.add_argument("--port", type=int, default=9870)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    print(f"Listening on UDP port {args.port} …\n")

    while True:
        data, addr = sock.recvfrom(_MAX_DATAGRAM)
        try:
            msg = json.loads(data)
        except json.JSONDecodeError as exc:
            print(f"[bad datagram from {addr}] {exc}")
            continue

        frame_id = msg.get("frame_id", "?")
        t_ms = msg.get("t_unix_ms", 0)

        drones = msg.get("drones", [])
        cameras = msg.get("cameras", [])

        print(f"frame={frame_id}  t_unix_ms={t_ms}")

        for drone in drones:
            pos = drone.get("pos_w", [])
            print(f"  drone id={drone['id']}  pos_w={pos}")

        for cam in cameras:
            name = cam.get("name", "?")
            rot_q = cam.get("rot_q", [])
            print(f"  cam={name}  rot_q={rot_q}")
            for det in cam.get("detections", []):
                if det.get("visible", True):
                    bbox = det.get("bbox_xyxy", [])
                    center = det.get("center_px", [])
                    dist = det.get("dist_m", "?")
                    print(f"    drone_id={det['drone_id']}  bbox={bbox}  center={center}  dist={dist}m")

        print()


if __name__ == "__main__":
    main()
