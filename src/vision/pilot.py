"""Can the detector see the ball at all? Measured against the tracking, with no human label.

On frames where the tracking reports a ball, its position is projected into the picture and
the detector is asked to find a ball there. Prints aggregates only.

    python -m vision pilot --data-dir <bundle>
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

from mtah.atd import load_tracking
from mtah.config import Config
from mtah.paths import tactical_assets
from viewer.data import load_homography

from .common import TILE, ball_boxes, crop_origin, load_detector, play_ranges, project, tile_origins


def pilot(data_root: Path, cfg: Config, cache_dir: Path, samples: int = 300,
          conf: float = 0.05, tolerance_px: float = 20.0, stem: str = "BAR-ATH") -> List[str]:
    import cv2

    assets = tactical_assets(Path(data_root))
    folder = Path(data_root) / "tactical"
    video = folder / f"{stem}_tactical.mp4"
    tr = load_tracking(assets.atd_xml, assets.atd_txt, cfg, cache_dir)
    hom = load_homography(folder / f"{stem}_tactical_homography.csv", cfg.n_frames, cache_dir)

    wanted = np.zeros(cfg.n_frames, dtype=bool)
    for a, b in play_ranges(cfg):
        wanted[a:b + 1] = True
    seen = np.isfinite(tr.ball).all(axis=1) & np.isfinite(hom).all(axis=1)
    pool = np.flatnonzero(wanted & seen)
    frames = pool[np.linspace(0, len(pool) - 1, samples).astype(int)]

    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = load_detector(cache_dir)
    origins = tile_origins(width, height)

    used = in_picture = hit_crop = hit_tiles = 0
    extra_crop = extra_tiles = 0
    sizes, confs, ms = [], [], []
    for n, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        used += 1
        u, v = project(hom[f], float(tr.ball[f, 0]), float(tr.ball[f, 1]))
        if not (0 <= u < width and 0 <= v < height):
            continue
        in_picture += 1

        x0, y0 = crop_origin(u, v, width, height)
        t0 = time.perf_counter()
        boxes = ball_boxes(model, [img[y0:y0 + TILE, x0:x0 + TILE]], conf)[0]
        ms.append(1000 * (time.perf_counter() - t0))
        near = np.hypot(boxes[:, 0] + x0 - u, boxes[:, 1] + y0 - v) <= tolerance_px
        if near.any():
            hit_crop += 1
            best = boxes[near][np.argmax(boxes[near][:, 4])]
            sizes.append(float(max(best[2], best[3])))
            confs.append(float(best[4]))
        extra_crop += int((~near).sum())

        if n % 3 == 0:      # the whole picture, tiled: what a search without a hint would cost
            tiles = ball_boxes(model, [img[y:y + TILE, x:x + TILE] for x, y in origins], conf)
            found, extra = False, 0
            for (x, y), b in zip(origins, tiles):
                d = np.hypot(b[:, 0] + x - u, b[:, 1] + y - v) <= tolerance_px
                found = found or bool(d.any())
                extra += int((~d).sum())
            hit_tiles += int(found)
            extra_tiles += extra
    cap.release()
    tiled = len(range(0, len(frames), 3))

    def pct(a, b):
        return f"{100.0 * a / b:5.1f}%" if b else "  n/a"

    out = [f"video {width}x{height}, detector {TILE}px, confidence >= {conf}",
           f"frames sampled {used} (tracking has the ball), ball inside the picture on {in_picture}",
           "",
           "crop centred on the tracked ball (the best case: we know where to look)",
           f"  ball found within {tolerance_px:.0f} px        {pct(hit_crop, in_picture)}",
           f"  other detections per crop        {extra_crop / max(in_picture, 1):.2f}"]
    if sizes:
        out.append(f"  ball box size, median            {float(np.median(sizes)):.0f} px")
        out.append(f"  confidence, median               {float(np.median(confs)):.2f}")
    out += ["",
            f"whole picture in {len(origins)} tiles (no hint)",
            f"  ball found                       {pct(hit_tiles, tiled)}",
            f"  other detections per frame       {extra_tiles / max(tiled, 1):.2f}",
            "",
            f"speed on this machine: {float(np.median(ms)):.0f} ms per {TILE}px image"]
    return out
