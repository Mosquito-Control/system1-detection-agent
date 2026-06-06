"""Derive system1+system2 camera configs from Unity's UDP ground-truth labels.

Run while DroneSim is publishing to UDP 9870. Writes:
  - system1-detection-agent/cameras.yaml   (azimuth/elevation/FoV per cam)
  - system2-positioning-engine/cameras.yaml (fake-GPS per cam, ref @ 0,0,0)

Camera convention used:
  - Unity world: left-handed, +Y up, +Z forward (Unity camera local +Z = look dir)
  - ENU (for system1 bearings & system2 triangulation):
      E = Unity x   N = Unity z   U = Unity y
  - System 2's reference origin is set to (lat=0, lon=0, alt=0), so the local
    ENU frame coincides with what we feed it. Each camera's Unity (x,y,z) is
    converted to (E,N,U) and round-tripped through enu_to_gps for the yaml.
"""
from __future__ import annotations

import json
import math
import socket
import sys
from pathlib import Path

import yaml

# --- system2's enu_to_gps reproduced here so the script has no import side-effects
import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = 2 * _F - _F ** 2


def _to_ecef(lat_r, lon_r, alt):
    N = _A / np.sqrt(1 - _E2 * np.sin(lat_r) ** 2)
    return np.array([
        (N + alt) * np.cos(lat_r) * np.cos(lon_r),
        (N + alt) * np.cos(lat_r) * np.sin(lon_r),
        (N * (1 - _E2) + alt) * np.sin(lat_r),
    ])


def _enu_rotation(lat_r, lon_r):
    sl, cl = np.sin(lat_r), np.cos(lat_r)
    sn, cn = np.sin(lon_r), np.cos(lon_r)
    return np.array([
        [-sn,       cn,      0],
        [-sl * cn, -sl * sn, cl],
        [ cl * cn,  cl * sn, sl],
    ])


def enu_to_gps(enu, ref_lat, ref_lon, ref_alt):
    rlat, rlon = np.radians(ref_lat), np.radians(ref_lon)
    R_inv = _enu_rotation(rlat, rlon).T
    ecef = _to_ecef(rlat, rlon, ref_alt) + R_inv @ np.asarray(enu)
    x, y, z = ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(5):
        N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + _E2 * N * np.sin(lat), p)
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def qrot(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def unity_to_enu(v):
    """(x_unity, y_unity, z_unity) -> (E, N, U)."""
    return v[0], v[2], v[1]


REPO = Path(__file__).resolve().parents[2]
SYS1_YAML = REPO / "system1-detection-agent" / "cameras.yaml"
SYS2_YAML = REPO / "system2-positioning-engine" / "cameras.yaml"

# Unity capture: 1280x720, vertical FOV 60°
IMG_W, IMG_H = 1280, 720
VFOV_DEG = 60.0
HFOV_DEG = math.degrees(2 * math.atan((IMG_W / IMG_H) * math.tan(math.radians(VFOV_DEG / 2))))

# Shared anchor — MUST match Orgs/system4-unity-simulation/Tools/export_camera_geojson.py
# so the cameras on the dashboard map AND the triangulated drone fixes both
# live in the same HK frame. Changing it here without updating the exporter
# (and re-running it to regenerate sim-cameras.geojson) breaks the alignment.
# Anchor: Unity scene focal point (lookAt = Unity (900, 2000)) ↔ Mong Kok area
# (22.318°N, 114.169°E), the centre of the drone flight ring in dense Kowloon.
ANCHOR_UNITY_X = 900.0
ANCHOR_UNITY_Z = 2000.0
ANCHOR_LAT = 22.318
ANCHOR_LON = 114.169


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(5)
    s.bind(("127.0.0.1", 9870))
    pkt = json.loads(s.recvfrom(16384)[0])
    s.close()

    sys1_cams = []
    sys2_cams = []
    print(f"hfov={HFOV_DEG:.2f}° vfov={VFOV_DEG}°  (Unity capture {IMG_W}x{IMG_H})")
    for c in pkt["cameras"]:
        unity_idx = int(c["name"].removeprefix("cam"))  # 0..7
        cam_id = f"cam_{unity_idx + 1:02d}"             # cam_01..cam_08

        # Orientation
        fwd_unity = qrot(c["rot_q"], (0.0, 0.0, 1.0))   # Unity cam local forward
        E, N, U = unity_to_enu(fwd_unity)
        az = (math.degrees(math.atan2(E, N)) + 360.0) % 360.0
        el = math.degrees(math.atan2(U, math.hypot(E, N)))

        # Position: Unity world -> fake-GPS around HK_CENTER, anchored at the
        # SAME Unity point as the dashboard exporter. With reference_origin set
        # to HK_CENTER (below), system2's ENU frame coincides with the
        # exporter's flat-earth frame to within sub-meter at this scale.
        pos_E, pos_N, pos_U = unity_to_enu(c["pos_w"])
        anchor_pos_E = ANCHOR_UNITY_X        # 900
        anchor_pos_N = ANCHOR_UNITY_Z        # 2000
        # ENU position relative to the anchor (which sits at HK_CENTER in WGS84):
        rel_E = pos_E - anchor_pos_E
        rel_N = pos_N - anchor_pos_N
        rel_U = pos_U                        # altitude is anchor-independent
        lat, lon, alt = enu_to_gps([rel_E, rel_N, rel_U],
                                    ANCHOR_LAT, ANCHOR_LON, 0.0)

        sys1_cams.append({
            "cam_id": cam_id,
            "stream_url": f"rtsp://localhost:8554/{c['name']}",
            "azimuth_deg": round(az, 3),
            "elevation_deg": round(el, 3),
            "hfov_deg": round(HFOV_DEG, 3),
            "vfov_deg": VFOV_DEG,
        })
        sys2_cams.append({
            "id": cam_id,
            "lat": round(lat, 8),
            "lon": round(lon, 8),
            "alt_m": round(alt, 3),
        })
        print(f"  {cam_id}: az={az:7.2f}° el={el:+6.2f}°  "
              f"unity=({c['pos_w'][0]:7.1f},{c['pos_w'][1]:6.1f},{c['pos_w'][2]:7.1f})  "
              f"fake_gps=({lat:.6f},{lon:.6f},{alt:.1f})")

    # Write system1 cameras.yaml
    with open(SYS1_YAML, "w") as f:
        f.write(
            "# Generated from Unity UDP GT — see tools/gen_camera_configs.py.\n"
            "# cam_NN <-> Unity camN with proper az/el extracted from rot_q.\n"
        )
        yaml.safe_dump({"cameras": sys1_cams}, f, sort_keys=False)
    print(f"\nwrote {SYS1_YAML.relative_to(REPO)}")

    # Write system2 cameras.yaml — anchored at HK_CENTER (same anchor as the
    # frontend's sim-cameras.geojson exporter), so triangulated drone positions
    # land in real HK and overlap visually with the sim cameras on the map.
    with open(SYS2_YAML, "w") as f:
        f.write(
            "# Generated from Unity UDP GT — see tools/gen_camera_configs.py in system1.\n"
            "# reference_origin is HK_CENTER, the SAME anchor used by\n"
            "# Orgs/system4-unity-simulation/Tools/export_camera_geojson.py and by\n"
            "# Orgs/frontend/src/components/map-canvas.tsx. Unity world axes map to ENU\n"
            "# (E=unity_x, N=unity_z, U=unity_y). Camera GPS are round-tripped from\n"
            "# Unity world coords via system2's ECEF↔ENU conversion.\n"
        )
        yaml.safe_dump({
            "reference_origin": {
                "lat": ANCHOR_LAT, "lon": ANCHOR_LON, "alt_m": 0.0,
            },
            "cameras": sys2_cams,
        }, f, sort_keys=False)
    print(f"wrote {SYS2_YAML.relative_to(REPO)}")


if __name__ == "__main__":
    main()
