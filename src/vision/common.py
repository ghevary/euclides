"""Shared pieces of the vision stage: which frames to look at, the detector, the geometry."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from mtah.config import Config

WEIGHTS = "yolov8n.pt"          # Ultralytics YOLOv8-nano, pre-trained on COCO (public)
SPORTS_BALL = 32                # COCO class id
TILE = 640


def play_ranges(cfg: Config) -> List[Tuple[int, int]]:
    """Frame ranges worth decoding: the graded halves. Everything else is written as X by the
    pipeline whatever the video shows, so the detector never needs to see it."""
    return [(int(a), int(b)) for a, b in cfg.segments]


def load_detector(cache_dir: Path):
    from ultralytics import YOLO
    cache_dir.mkdir(parents=True, exist_ok=True)
    return YOLO(str(cache_dir / WEIGHTS))       # downloaded once, then read from the cache


def project(h9: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Normalised pitch position -> pixel, with the 3x3 homography of that frame (row-major)."""
    w = h9[6] * x + h9[7] * y + h9[8]
    return (h9[0] * x + h9[1] * y + h9[2]) / w, (h9[3] * x + h9[4] * y + h9[5]) / w


def crop_origin(u: float, v: float, width: int, height: int) -> Tuple[int, int]:
    """Top-left corner of a TILE x TILE crop centred on (u, v), kept inside the picture."""
    x0 = int(round(min(max(u - TILE / 2, 0), max(width - TILE, 0))))
    y0 = int(round(min(max(v - TILE / 2, 0), max(height - TILE, 0))))
    return x0, y0


def tile_origins(width: int, height: int) -> List[Tuple[int, int]]:
    """Origins of TILE x TILE tiles covering the whole picture, overlapping where needed."""
    def axis(size):
        if size <= TILE:
            return [0]
        count = int(np.ceil(size / TILE))
        return [int(round(i * (size - TILE) / (count - 1))) for i in range(count)]
    return [(x, y) for y in axis(height) for x in axis(width)]


def ball_boxes(model, images, conf: float):
    """Per image: array (k, 5) of x_centre, y_centre, width, height, confidence for sports balls."""
    results = model.predict(images, imgsz=TILE, conf=conf, classes=[SPORTS_BALL],
                            device="cpu", verbose=False)
    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 5)))
            continue
        xywh = r.boxes.xywh.cpu().numpy()
        out.append(np.concatenate([xywh, r.boxes.conf.cpu().numpy()[:, None]], axis=1))
    return out
