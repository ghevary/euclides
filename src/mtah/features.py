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
    held: np.ndarray           # bool: ball held by a defender in their own penalty area
    near_own: np.ndarray       # bool: ball in a penalty area with a defender of that area on it
    opp_in_box: np.ndarray     # bool: ball in a penalty area with an attacker inside that area


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


PENALTY_AREA_DEPTH_M = 16.5        # Laws of the Game
PENALTY_AREA_HALF_WIDTH_M = 20.16


def home_defends_low_x(players: np.ndarray, home: np.ndarray, cfg: Config) -> np.ndarray:
    """Per frame: True where the home team defends the goal at x = 0.

    Decided once per graded half from where the two teams stand on average, so it follows
    the change of ends at halftime without any match-specific constant.
    """
    out = np.ones(len(players), dtype=bool)
    step = max(1, int(cfg.fps))
    with np.errstate(invalid="ignore"):
        for first, last in cfg.segments:
            first, last = max(0, int(first)), min(len(players) - 1, int(last))
            if last < first:
                continue
            block = players[first:last + 1:step, :, 0].astype(np.float64)
            hx, ax = block[:, home], block[:, ~home]
            if np.isfinite(hx).any() and np.isfinite(ax).any():
                out[first:last + 1] = np.nanmean(hx) <= np.nanmean(ax)
    return out


def _true_runs(mask: np.ndarray):
    """[(start, stop)] of every run of True, stop exclusive."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def held_ball(usable: np.ndarray, slow: np.ndarray, out: np.ndarray, near_own: np.ndarray,
              opp_in_box: np.ndarray, cfg: Config) -> np.ndarray:
    """Frames where a resting or unseen ball is being held, not dead.

    `near_own`: the ball is inside a penalty area with a player of the team defending it
    within cfg.held_radius_m. A resting run counts as held when a defender is that close for
    most of it; an unseen run when a defender was that close just before the ball vanished.
    Either way the ball must have arrived in play: tracked through the seconds before, and
    not from outside the pitch. That keeps goal kicks and other restarts, where the ball
    comes back from out of play, as stoppages.

    `opp_in_box`: an attacker is inside that same penalty area. A hold must start with one:
    open play that ends in the keeper's hands has attackers in the area, while at a goal
    kick or a defending free kick the Laws keep them out of it. One hold lasts at most
    cfg.held_max_s.
    """
    n = len(usable)
    held = np.zeros(n, dtype=bool)
    if not cfg.held_ball or n == 0:
        return held
    look = max(1, int(round(cfg.held_lookback_s * cfg.fps)))
    cap = max(1, int(round(cfg.held_max_s * cfg.fps)))
    just = max(1, int(cfg.fps) // 2)
    runs = [(a, b, True) for a, b in _true_runs(slow)] + [(a, b, False) for a, b in _true_runs(~usable)]
    runs.sort()
    chain_start = -1
    for a, b, resting in runs:
        lo = max(0, a - look)
        if a == 0 or out[lo:a].any():
            continue
        before = usable[lo:a] | held[lo:a]
        if before.mean() < 0.8:
            continue
        if resting:
            if near_own[a:b].mean() < 0.5:
                continue
        elif near_own[max(0, a - just):a].mean() < 0.6:
            continue
        if not held[lo:a].any():
            # a new hold, not the continuation of one: it must begin in open play
            window = opp_in_box[a:min(b, a + int(cfg.fps))] if resting else opp_in_box[max(0, a - just):a]
            if cfg.held_needs_opponent and not window.any():
                continue
            chain_start = a
        stop = min(b, chain_start + cap)
        if stop > a:
            held[a:stop] = True
    return held


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

    home_low = home_defends_low_x(tr.players, tr.home, cfg)
    with np.errstate(invalid="ignore"):
        in_width = np.abs(ball_m[:, 1] - cfg.pitch_wid_m / 2.0) <= PENALTY_AREA_HALF_WIDTH_M
        low_box = in_width & (ball_m[:, 0] >= 0.0) & (ball_m[:, 0] <= PENALTY_AREA_DEPTH_M)
        high_box = in_width & (ball_m[:, 0] <= cfg.pitch_len_m)             & (ball_m[:, 0] >= cfg.pitch_len_m - PENALTY_AREA_DEPTH_M)
    d_low = np.where(home_low, d_home, d_away)    # nearest player of the team defending x = 0
    d_high = np.where(home_low, d_away, d_home)
    near_own = usable & ((low_box & (d_low <= cfg.held_radius_m))
                         | (high_box & (d_high <= cfg.held_radius_m)))
    # Is an attacker inside the penalty area the ball is in?
    px = players_m[:, :, 0].astype(np.float64)
    py = players_m[:, :, 1].astype(np.float64)
    with np.errstate(invalid="ignore"):
        p_width = np.abs(py - cfg.pitch_wid_m / 2.0) <= PENALTY_AREA_HALF_WIDTH_M
        p_low = p_width & (px >= 0.0) & (px <= PENALTY_AREA_DEPTH_M)
        p_high = p_width & (px <= cfg.pitch_len_m) & (px >= cfg.pitch_len_m - PENALTY_AREA_DEPTH_M)
    home_cols, away_cols = tr.home, ~tr.home
    att_low = np.where(home_low, p_low[:, away_cols].any(axis=1), p_low[:, home_cols].any(axis=1))
    att_high = np.where(home_low, p_high[:, home_cols].any(axis=1), p_high[:, away_cols].any(axis=1))
    opp_in_box = usable & ((low_box & att_low) | (high_box & att_high))

    held = held_ball(usable, slow, out, near_own, opp_in_box, cfg)
    return Features(seen, usable, d_home, d_away, oob, still_run_s, missing_run_s, speed, held,
                    near_own, opp_in_box)
