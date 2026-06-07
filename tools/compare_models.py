"""Side-by-side YOLO comparison: current yolov8s.pt vs FilippTrigub/yolov11x-drone-finetuned.

Runs both models on the same image(s), prints detections, writes annotated PNGs.
Usage:
    python -m tools.compare_models path/to/img1.jpg path/to/img2.jpg ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

MODELS = [
    ("yolov8s (generic COCO)",       "yolov8s.pt",        None),         # all classes
    ("yolov11x-drone (fine-tuned)",  "yolov11x-drone.pt", None),         # single class "drone"
]
CONF = 0.25
OUT = Path("compare_out")
OUT.mkdir(exist_ok=True)


def annotate(img, dets, label):
    out = img.copy()
    for x1, y1, x2, y2, conf, cls_name in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, f"{cls_name} {conf:.2f}", (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def run(model, img, name):
    res = model(img, conf=CONF, verbose=False, device="mps")[0]
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c = float(b.conf[0])
        cls = int(b.cls[0])
        cls_name = model.names.get(cls, str(cls))
        dets.append((x1, y1, x2, y2, c, cls_name))
    return dets


def main(paths):
    models = [(label, YOLO(p)) for label, p, _ in MODELS]
    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"SKIP (unreadable): {img_path}")
            continue
        print(f"\n=== {img_path}  ({img.shape[1]}x{img.shape[0]}) ===")
        for label, model in models:
            dets = run(model, img, label)
            drone_like = [d for d in dets if d[5].lower() in {"drone", "airplane", "bird", "kite"}]
            print(f"  [{label}]")
            print(f"    total detections: {len(dets)}")
            print(f"    drone-like classes: {len(drone_like)}")
            for x1, y1, x2, y2, conf, cls_name in sorted(dets, key=lambda d: -d[4])[:5]:
                print(f"      - {cls_name:10s} conf={conf:.3f}  bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            stem = Path(img_path).stem
            tag = label.split()[0].replace("(", "").replace(",", "")
            cv2.imwrite(str(OUT / f"{stem}__{tag}.png"), annotate(img, dets, label))
    print(f"\nAnnotated images written to {OUT.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.compare_models <img> [img ...]")
    main(sys.argv[1:])
