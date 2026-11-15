"""Pictures of what the detector found, for a person to look at on their own machine.

Each picture is a crop around the ball position the tracking reports, enlarged, with
  green circle   where the tracking says the ball is
  yellow boxes   what the detector calls a sports ball, with its confidence
The pictures are extracts of the match video: they stay local and are never committed.

    python -m vision preview --data-dir <bundle> --out reports_local/yolo
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from mtah.atd import load_tracking
from mtah.config import Config
from mtah.paths import tactical_assets
from viewer.data import load_homography

from .common import TILE, ball_boxes, crop_origin, load_detector, play_ranges, project

ZOOM = 2


def preview(data_root: Path, cfg: Config, cache_dir: Path, out_dir: Path, count: int = 24,
            conf: float = 0.05, tolerance_px: float = 20.0, stem: str = "BAR-ATH") -> List[str]:
    import cv2

    assets = tactical_assets(Path(data_root))
    folder = Path(data_root) / "tactical"
    tr = load_tracking(assets.atd_xml, assets.atd_txt, cfg, cache_dir)
    hom = load_homography(folder / f"{stem}_tactical_homography.csv", cfg.n_frames, cache_dir)

    wanted = np.zeros(cfg.n_frames, dtype=bool)
    for a, b in play_ranges(cfg):
        wanted[a:b + 1] = True
    seen = np.isfinite(tr.ball).all(axis=1) & np.isfinite(hom).all(axis=1)
    pool = np.flatnonzero(wanted & seen)
    frames = pool[np.linspace(0, len(pool) - 1, count).astype(int)]

    cap = cv2.VideoCapture(str(folder / f"{stem}_tactical.mp4"))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = load_detector(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        u, v = project(hom[f], float(tr.ball[f, 0]), float(tr.ball[f, 1]))
        if not (0 <= u < width and 0 <= v < height):
            continue
        x0, y0 = crop_origin(u, v, width, height)
        crop = img[y0:y0 + TILE, x0:x0 + TILE]
        boxes = ball_boxes(model, [crop], conf)[0]
        near = np.hypot(boxes[:, 0] + x0 - u, boxes[:, 1] + y0 - v) <= tolerance_px

        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
        for (cx, cy, w, h, c) in boxes:
            p1 = (int((cx - w / 2) * ZOOM) - 3, int((cy - h / 2) * ZOOM) - 3)
            p2 = (int((cx + w / 2) * ZOOM) + 3, int((cy + h / 2) * ZOOM) + 3)
            cv2.rectangle(big, p1, p2, (0, 230, 255), 2)
            cv2.putText(big, f"{c:.2f}", (p1[0], max(14, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 230, 255), 2)
        cv2.circle(big, (int((u - x0) * ZOOM), int((v - y0) * ZOOM)), 26, (60, 220, 60), 2)
        verdict = "found" if near.any() else "missed"
        cv2.putText(big, f"frame {int(f)}  {verdict}  other detections {int((~near).sum())}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        name = f"frame_{int(f):06d}_{verdict}.jpg"
        cv2.imwrite(str(out_dir / name), big, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append((name, int(f), verdict, int((~near).sum())))
    cap.release()

    page = ["<!doctype html><meta charset='utf-8'><title>YOLOv8n preview</title>",
            "<style>body{background:#111;color:#eee;font:14px sans-serif;margin:16px}"
            "figure{display:inline-block;margin:6px;vertical-align:top}"
            "img{width:620px;display:block;border-radius:6px}figcaption{padding:4px 0}</style>",
            "<h1>YOLOv8n vs tracking</h1>",
            "<p>Green circle: ball position from the tracking. Yellow box: detector says "
            "sports ball (number = confidence). Local only, do not share.</p>"]
    for name, f, verdict, extra in rows:
        page.append(f"<figure><a href='{name}'><img src='{name}' loading='lazy'></a>"
                    f"<figcaption>frame {f} &middot; {verdict} &middot; other detections {extra}"
                    "</figcaption></figure>")
    (out_dir / "index.html").write_text("\n".join(page), encoding="utf-8")
    found = sum(1 for r in rows if r[2] == "found")
    return [f"wrote {len(rows)} pictures and index.html to {out_dir}",
            f"ball found on {found} of {len(rows)}"]
