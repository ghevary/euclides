"""Everything the viewer reads from disk. Read-only: nothing here writes labels.

Tracking and Smart Tagging go through `mtah.atd` and `mtah.tagging`, the same loaders the
pipeline uses, so the viewer cannot disagree with the pipeline about frame alignment.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mtah.atd import Tracking, load_tracking, norm_text
from mtah.config import Config
from mtah.errors import MtahError
from mtah.features import drop_frozen, nearest_distance, team_by_frame
from mtah.submission import ALLOWED, validate_csv
from mtah.tagging import load_instances, team_sign

VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm"}


# ----------------------------------------------------------------------------- labels

def read_labels(path: Path, n_frames: int) -> np.ndarray:
    """Labels of a submission CSV as an array of single characters, after the format check."""
    problem = validate_csv(path, n_frames)
    if problem:
        raise MtahError(f"{Path(path).name}: {problem}")
    out = np.empty(n_frames, dtype="<U1")
    with open(path, "r", encoding="utf-8") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            out[i] = line.rsplit(",", 1)[1].strip().upper()
    return out


def to_segments(labels: np.ndarray) -> List[list]:
    """Run-length encoding: [[first_frame, last_frame, label], ...], both ends inclusive."""
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return []
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    ends = np.concatenate((starts[1:] - 1, [n - 1]))
    return [[int(a), int(b), str(labels[a])] for a, b in zip(starts, ends)]


# ----------------------------------------------------------------------------- homography

def load_homography(path: Path, n_frames: int, cache_dir: Optional[Path] = None) -> np.ndarray:
    """(n_frames, 9) float64, row-major 3x3 per frame, NaN where no matrix exists.

    The matrix maps normalised pitch coordinates (0-1) to image pixels (DATA.md).
    """
    path = Path(path)
    cache = None
    if cache_dir is not None:
        st = path.stat()
        cache = Path(cache_dir) / f"{path.stem}_{st.st_size}_{st.st_mtime_ns}.npz"
        if cache.is_file():
            with np.load(cache) as z:
                h = z["h"]
            if h.shape == (n_frames, 9):
                return h
    with open(path, "r", encoding="utf-8") as fh:
        header = [c.strip().lower() for c in fh.readline().split(",")]
    wanted = ["frame"] + [f"h{r}{c}" for r in range(3) for c in range(3)]
    try:
        cols = [header.index(name) for name in wanted]
    except ValueError:
        raise MtahError(f"{path.name}: expected columns {wanted}.") from None
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols, dtype=np.float64, ndmin=2)
    h = np.full((n_frames, 9), np.nan)
    frames = raw[:, 0].astype(np.int64)
    ok = (frames >= 0) & (frames < n_frames)
    h[frames[ok]] = raw[ok, 1:]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, h=h)
    return h


# ----------------------------------------------------------------------------- store

class Store:
    """All data behind the API. Tracking and homography load in a background thread."""

    def __init__(self, cfg: Config, video: Path, labels_path: Optional[Path],
                 atd_xml: Optional[Path] = None, atd_txt: Optional[Path] = None,
                 tagging: Optional[Path] = None, homography: Optional[Path] = None,
                 broadcast: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.cfg = cfg
        self.videos: Dict[str, Path] = {"tactical": Path(video)}
        if broadcast is not None and Path(broadcast).is_file():
            self.videos["broadcast"] = Path(broadcast)
        self.labels_path = Path(labels_path) if labels_path else None
        self.labels = read_labels(self.labels_path, cfg.n_frames) if self.labels_path else None
        self.atd_xml, self.atd_txt = atd_xml, atd_txt
        self.tagging_path, self.homography_path = tagging, homography
        self.cache_dir = cache_dir

        self.tracking = None
        self.homography = None
        self.d_home = self.d_away = None
        self.team = None
        self.track_state = "none" if not (atd_xml and atd_txt) else "loading"
        self.track_error = ""
        self._lock = threading.Lock()

    def start_background_load(self) -> None:
        if self.track_state == "loading":
            threading.Thread(target=self._load_tracking, daemon=True).start()

    def _load_tracking(self) -> None:
        try:
            cfg = self.cfg
            tr = load_tracking(self.atd_xml, self.atd_txt, cfg, self.cache_dir)
            scale = np.array([cfg.pitch_len_m, cfg.pitch_wid_m], dtype=np.float64)
            ball_m = tr.ball.astype(np.float64) * scale
            # The same players the model sees: frozen ghosts dropped, keepers in their team.
            players_m = drop_frozen(tr.players * scale.astype(np.float32),
                                    int(round(cfg.ghost_frozen_s * cfg.fps)))
            team = team_by_frame(players_m, tr.home, cfg)
            tr = Tracking((players_m / scale.astype(np.float32)).astype(np.float32), tr.ball, tr.home)
            big = 1e6  # no cap here: the panel shows the real distance, however large
            everyone = np.ones(len(tr.home), dtype=bool)
            d_home = nearest_distance(np.where(team[:, :, None], players_m, np.nan), ball_m, everyone, big)
            d_away = nearest_distance(np.where(team[:, :, None], np.nan, players_m), ball_m, everyone, big)
            hom = None
            if self.homography_path and Path(self.homography_path).is_file():
                hom = load_homography(self.homography_path, cfg.n_frames, self.cache_dir)
            with self._lock:
                self.tracking, self.homography, self.team = tr, hom, team
                self.d_home, self.d_away = d_home, d_away
                self.track_state = "ready"
        except Exception as error:  # shown in the page instead of killing the server
            with self._lock:
                self.track_error = str(error) or type(error).__name__
                self.track_state = "error"

    # ------------------------------------------------------------------ API payloads

    def meta(self) -> dict:
        cfg = self.cfg
        return {
            "fps": cfg.fps,
            "n_frames": cfg.n_frames,
            "segments": [[int(a), int(b)] for a, b in cfg.segments],
            "sources": sorted(self.videos, key=lambda s: s != "tactical"),
            "labels_name": self.labels_path.name if self.labels_path else None,
            "tracking": self.track_state,
            "tracking_error": self.track_error,
            "has_tagging": bool(self.tagging_path and Path(self.tagging_path).is_file()),
            "allowed": list(ALLOWED),
            # pixel space of the homography: the full-resolution tactical video (DATA.md)
            "homography_size": [1920, 1080],
        }

    def labels_payload(self) -> dict:
        if self.labels is None:
            return {"name": None, "segments": []}
        return {"name": self.labels_path.name, "segments": to_segments(self.labels)}

    def tags_payload(self) -> dict:
        """Set pieces of the Smart Tagging as frame intervals. Automatic, not ground truth."""
        if not (self.tagging_path and Path(self.tagging_path).is_file()):
            return {"set_pieces": []}
        cfg = self.cfg
        wanted = norm_text(cfg.setpiece_code)
        out = []
        for inst in load_instances(Path(self.tagging_path)):
            if norm_text(inst.code) != wanted:
                continue
            sign = team_sign(inst, cfg)
            out.append({
                "start": int(round(inst.start_s * cfg.fps)),
                "end": int(round(inst.end_s * cfg.fps)),
                "type": inst.labels.get("type", ""),
                "team": "H" if sign > 0 else "A" if sign < 0 else "",
            })
        out.sort(key=lambda item: item["start"])
        return {"set_pieces": out}

    def track_payload(self, start: int, count: int) -> dict:
        with self._lock:
            state, tr, hom, team = self.track_state, self.tracking, self.homography, self.team
            d_home, d_away = self.d_home, self.d_away
        if state != "ready":
            return {"state": state, "error": self.track_error, "start": start, "frames": []}
        n = self.cfg.n_frames
        start = max(0, min(n, start))
        stop = max(start, min(n, start + count))
        frames = []
        for i in range(start, stop):
            pts = tr.players[i]
            seen = np.isfinite(pts).all(axis=1)
            ball = tr.ball[i]
            has_ball = bool(np.isfinite(ball).all())
            item = {
                "h": _flat(pts[seen & team[i]]),
                "a": _flat(pts[seen & ~team[i]]),
                "b": _flat(ball[None, :]) if has_ball else None,
                "dh": _dist(d_home[i]) if has_ball else None,
                "da": _dist(d_away[i]) if has_ball else None,
                "H": None,
            }
            if hom is not None and np.isfinite(hom[i]).all():
                item["H"] = [float(v) for v in hom[i]]
            frames.append(item)
        return {"state": "ready", "start": start, "frames": frames}


def _flat(points: np.ndarray) -> List[float]:
    return [round(float(v), 5) for v in np.asarray(points, dtype=np.float64).reshape(-1)]


def _dist(value: float) -> Optional[float]:
    return round(float(value), 2) if np.isfinite(value) and value < 1e5 else None
