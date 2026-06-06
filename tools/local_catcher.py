"""Local System 2 stub — prints every POST so we can watch system1's output
without writing to the team's Azure DB.

Run with the venv active:
    uvicorn tools.local_catcher:app --host 127.0.0.1 --port 9999
"""
from datetime import datetime, timezone

from fastapi import FastAPI, Request

app = FastAPI()
_count = 0


@app.post("/events")
async def events(req: Request):
    global _count
    _count += 1
    payload = await req.json()
    dets = payload.get("detections") or []
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(
        f"[{now}] #{_count:>4} cam={payload.get('cam_id'):<6} "
        f"ts={payload.get('timestamp')}  dets={len(dets)}  "
        + (
            f"first=(b={['%.3f' % x for x in dets[0]['bearing_vector']]}, "
            f"score={dets[0]['score']:.2f})"
            if dets
            else ""
        ),
        flush=True,
    )
    return {"ok": True, "received": len(dets)}


@app.get("/")
async def root():
    return {"received_events": _count}
