"""Per-frame features from the tracking. Missing data stays missing: nothing is zero-filled."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atd import Tracking
from .config import Config


@dataclass
class Features:
    seen: np.ndarray           # bool: the tracking reports a ball position on this frame
    usable: np.ndarray         # bool: seen, or inside a short interpolated gap
    d_home: np.ndarray         # metres from the ball to the nearest home player (capped)
    d_away: np.ndarray         # metres from the ball to the nearest away player (capped)
    oob: np.ndarray            # [0,1] share of nearby frames with the ball outside the pitch
    still_run_s: np.ndarray    # seconds: length of the motionless-ball run this frame belongs to
    missing_run_s: np.ndarray  # seconds: length of the no-ball run this frame belongs to
    speed: np.ndarray          # m/s net ball speed over about a second, NaN where unknown
    slowness: np.ndarray       # [-1,1]: +1 everybody walking, -1 somebody running, 0 unknown


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred mean over `window` frames; the window shrinks at the edges."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    half = window // 2
    csum = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """For every True element, the length of the run of True it belongs to; 0 elsewhere."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [n])))
    return np.where(mask, np.repeat(lengths, lengths), 0)


def interpolate_short_gaps(ball: np.ndarray, max_gap: int):
    """Linear interpolation across gaps of at most `max_gap` frames. Longer gaps stay NaN."""
    ball = np.asarray(ball, dtype=np.float64)
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    fill = (~seen) & (prev >= 0) & (nxt < n) & ((nxt - prev - 1) <= max_gap)
    out = np.where(seen[:, None], ball, np.nan)
    p, q = prev[fill], nxt[fill]
    w = ((idx[fill] - p) / (q - p))[:, None]
    out[fill] = ball[p] * (1.0 - w) + ball[q] * w
    return out, seen, seen | fill


def nearest_distance(players_m: np.ndarray, ball_m: np.ndarray, team: np.ndarray,
                     cap: float, chunk: int = 20000) -> np.ndarray:
    """Distance from the ball to the nearest tracked player of one team, capped at `cap`."""
    n = len(ball_m)
    out = np.full(n, cap, dtype=np.float64)
    cols = np.flatnonzero(team)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        diff = players_m[a:b][:, cols, :].astype(np.float64) - ball_m[a:b, None, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        dist = np.where(np.isfinite(dist), dist, np.inf).min(axis=1)
        out[a:b] = np.minimum(dist, cap)
    return out


def slow_ball(ball_m: np.ndarray, cfg: Config):
    """(net speed in m/s, is-motionless flag) per frame.

    Speed is the net displacement between the frames `h` before and `h` after, so tracking
    jitter of a few centimetres per frame does not look like movement. Unknown near gaps.
    """
    n, h = len(ball_m), max(1, int(cfg.speed_half_window_frames))
    speed = np.full(n, np.nan)
    if n > 2 * h:
        delta = ball_m[2 * h:] - ball_m[:-2 * h]
        speed[h:n - h] = np.hypot(delta[:, 0], delta[:, 1]) * cfg.fps / (2.0 * h)
    with np.errstate(invalid="ignore"):
        slow = np.isfinite(speed) & (speed < cfg.stationary_speed_mps)
    return speed, slow


def player_pace(players_m: np.ndarray, cfg: Config) -> np.ndarray:
    """Pace of the fastest few tracked players in m/s (cfg.player_pace_percentile of their
    speeds), smoothed over cfg.player_smooth_s.

    Speed is the net displacement of a track slot over about a second, like the ball's, so
    position jitter does not read as movement. A slot that jumps faster than a person can
    run has changed identity and is ignored. NaN where too few players are tracked.
    """
    n, h = len(players_m), max(1, int(cfg.speed_half_window_frames))
    pace = np.full(n, np.nan)
    if n <= 2 * h:
        return pace
    for a in range(h, n - h, 20000):
        b = min(n - h, a + 20000)
        delta = players_m[a + h:b + h].astype(np.float64) - players_m[a - h:b - h]
        speed = np.hypot(delta[:, :, 0], delta[:, :, 1]) * cfg.fps / (2.0 * h)
        with np.errstate(invalid="ignore"):
            speed[speed > cfg.player_max_mps] = np.nan
        enough = np.isfinite(speed).sum(axis=1) >= cfg.player_min_tracked
        block = np.full(b - a, np.nan)
        if enough.any():
            block[enough] = np.nanpercentile(speed[enough], cfg.player_pace_percentile, axis=1)
        pace[a:b] = block
    known = np.isfinite(pace)
    window = max(1, int(round(cfg.player_smooth_s * cfg.fps)))
    total = rolling_mean(np.where(known, pace, 0.0), window)
    share = rolling_mean(known.astype(np.float64), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = total / share
    return np.where(known & (share > 0.5), smooth, np.nan)


def slow_threshold(pace: np.ndarray, cfg: Config) -> float:
    """The pace below which even the fastest players are walking: a stoppage rather than play.

    Starts from the prior cfg.player_slow_mps. With cfg.player_slow_auto it is refined from
    the graded frames of this match, without any label: the pace histogram has one group of
    frames where nobody runs and one where somebody does, and the threshold is the bottom of
    the valley between them. Only a valley inside the prior range counts, and only if the
    histogram really rises on both sides of it; otherwise the prior stands.
    """
    prior = float(cfg.player_slow_mps)
    if not cfg.player_slow_auto:
        return prior
    graded = np.zeros(len(pace), dtype=bool)
    for first, last in cfg.segments:
        graded[max(0, int(first)):int(last) + 1] = True
    values = pace[graded & np.isfinite(pace)]
    if len(values) < 1000:
        return prior
    hist, edges = np.histogram(values, bins=80, range=(0.0, 8.0))      # 0.1 m/s bins
    density = rolling_mean(hist.astype(np.float64), 5)
    centres = (edges[:-1] + edges[1:]) / 2.0
    inside = np.flatnonzero((centres >= cfg.player_slow_min_mps) & (centres <= cfg.player_slow_max_mps))
    if len(inside) == 0:
        return prior
    k = int(inside[np.argmin(density[inside])])
    floor = density[k]
    if density[:k].max(initial=0.0) <= 1.1 * floor or density[k + 1:].max(initial=0.0) <= 1.1 * floor:
        return prior                                                     # no valley: one group only
    return round(float(centres[k]), 3)


def player_slowness(players_m: np.ndarray, cfg: Config) -> np.ndarray:
    """+1 where everybody walks, -1 where somebody runs, 0 where unknown or switched off."""
    if not cfg.use_player_motion:
        return np.zeros(len(players_m))
    pace = player_pace(players_m, cfg)
    threshold = slow_threshold(pace, cfg)
    z = np.clip((threshold - pace) / max(cfg.player_speed_scale_mps, 1e-9), -2.0, 2.0) / 2.0
    return np.round(np.where(np.isfinite(pace), z, 0.0), 6)


def build_features(tr: Tracking, cfg: Config) -> Features:
    scale = np.array([cfg.pitch_len_m, cfg.pitch_wid_m], dtype=np.float64)
    ball_n, seen, usable = interpolate_short_gaps(tr.ball, cfg.max_gap_frames)
    ball_m = ball_n * scale
    players_m = tr.players * scale.astype(np.float32)

    d_home = nearest_distance(players_m, ball_m, tr.home, cfg.dist_cap_m)
    d_away = nearest_distance(players_m, ball_m, ~tr.home, cfg.dist_cap_m)

    speed, slow = slow_ball(ball_m, cfg)
    still_run_s = run_lengths(slow) / float(cfg.fps)

    mx, my = cfg.oob_margin_m / cfg.pitch_len_m, cfg.oob_margin_m / cfg.pitch_wid_m
    with np.errstate(invalid="ignore"):
        out = usable & ((ball_n[:, 0] < -mx) | (ball_n[:, 0] > 1 + mx)
                        | (ball_n[:, 1] < -my) | (ball_n[:, 1] > 1 + my))
    oob = rolling_mean(out.astype(np.float64), cfg.oob_window_frames)

    missing_run_s = run_lengths(~usable) / float(cfg.fps)
    slowness = player_slowness(players_m, cfg)
    return Features(seen, usable, d_home, d_away, oob, still_run_s, missing_run_s, speed, slowness)
