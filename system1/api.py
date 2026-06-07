"""HTTP tap exposing the latest YOLO bboxes to System 3.

System 1's wire format to System 2 is bearing vectors only — the actual
pixel bboxes get thrown away after the center → bearing conversion.
That's correct for triangulation but useless for the operator console,
which wants to draw "yes that green box is on the actual drone in the
video" overlays on the live HLS feed.

This module mounts a tiny FastAPI app inside the same process as the
camera workers and serves the latest snapshot from bbox_store.store.
Same-process means no IPC, no extra serialization cost — workers do
their normal POST to S2, and the dashboard sees the bboxes as a
side-effect.

Cam id mapping note: the dashboard refers to cameras by their MediaMTX
path ("cam0".."cam11") while S1's cameras.yaml uses zero-padded
"cam_01".."cam_12". We normalise both forms in the lookup so callers
don't need to know which side they're on.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from system1.bbox_store import store

logger = logging.getLogger(__name__)

app = FastAPI(title="System 1 bbox tap", version="1.0")

# CORS open — the dashboard lives on :3000 (dev) and arbitrary prod hosts.
# This server only returns bbox metadata, never accepts writes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _normalize_id(cam_id: str) -> list[str]:
    """Return all id variants we'll accept for the same camera.

    Dashboard sends "cam4"; S1 stores "cam_05". MediaMTX path index is
    zero-based; cameras.yaml id is one-based. We compute both forms so
    a single store lookup covers either caller.
    """
    candidates = {cam_id}
    if cam_id.startswith("cam_"):
        # "cam_05" → MediaMTX "cam4"
        try:
            n = int(cam_id.split("_", 1)[1])
            candidates.add(f"cam{n - 1}")
        except (ValueError, IndexError):
            pass
    elif cam_id.startswith("cam"):
        # "cam4" → S1 "cam_05"
        try:
            n = int(cam_id[3:])
            candidates.add(f"cam_{n + 1:02d}")
        except ValueError:
            pass
    return list(candidates)


def _snap_dict(snap) -> dict:
    return {
        "frame_w": snap.frame_w,
        "frame_h": snap.frame_h,
        "ts": snap.ts,
        "bboxes": [
            {
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "score": b.score,
            }
            for b in snap.bboxes
        ],
    }


@app.get("/bboxes")
def list_bboxes() -> dict:
    """All cameras with at least one snapshot. Empty bbox list means
    "we ran YOLO and saw nothing" — distinct from "no snapshot yet"
    (camera that hasn't successfully grabbed a frame), which is absent
    from the response entirely."""
    return {cam_id: _snap_dict(snap) for cam_id, snap in store.snapshot().items()}


@app.get("/bboxes/{cam_id}")
def get_bbox(cam_id: str) -> dict:
    """Single camera. Accepts both "cam4" (dashboard) and "cam_05"
    (S1 native) — returns whichever has a snapshot.

    Returns {found: false} rather than 404 so the dashboard can
    poll cleanly without erroring during cam warmup.

    Side effect: marks this camera as actively viewed so its worker
    thread skips the throttle in camera_worker (see BboxStore.is_active).
    Without this, all 12 RTSP threads share inference round-robin and the
    cam the operator is watching updates only every ~10s."""
    cands = _normalize_id(cam_id)
    # Always mark the canonical "cam_NN" form even if no snapshot exists
    # yet — covers the warmup window where the modal opens before the
    # first frame lands.
    for cand in cands:
        if cand.startswith("cam_"):
            store.mark_active(cand)
            break
    for cand in cands:
        snap = store.get(cand)
        if snap is not None:
            return {"found": True, "cam_id": cand, **_snap_dict(snap)}
    return {"found": False, "cam_id": cam_id}
