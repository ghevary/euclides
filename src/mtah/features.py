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
    left: np.ndarray = None    # bool: ball unseen after it went over, or was heading over, a line
    occluded: np.ndarray = None  # bool: ball unseen, last seen at a player's feet or in a crowd


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


def drop_frozen(players_m: np.ndarray, min_frames: int) -> np.ndarray:
    """Positions that stay exactly the same for `min_frames` or more become NaN.

    The first frame of such a stretch is kept: it is where the player really was.
    """
    if min_frames <= 0 or len(players_m) < 2:
        return players_m
    same = np.zeros(players_m.shape[:2], dtype=bool)
    same[1:] = (players_m[1:] == players_m[:-1]).all(axis=2)
    out = players_m.copy()
    for slot in range(same.shape[1]):
        for a, b in _true_runs(same[:, slot]):
            if b - a >= min_frames:
                out[a:b, slot] = np.nan
    return out


def drop_keeper_doubles(players_m: np.ndarray, home: np.ndarray, cfg: Config,
                        chunk: int = 5000) -> np.ndarray:
    """Drop the second copy of a goalkeeper the tracking follows in both teams at once."""
    home = np.asarray(home, dtype=bool)
    n, s = players_m.shape[:2]
    if not cfg.dup_fix or n == 0 or s < 2:
        return players_m
    pm = players_m.astype(np.float64)
    known = np.isfinite(pm).all(axis=2)
    home_low = home_defends_low_x(pm, home, cfg)
    mark = np.zeros((n, s), dtype=bool)      # this slot is the wrong-team copy on this frame
    cols = np.arange(s)
    # Each team's goalkeeper slot: the one most often the deepest player in front of a goal.
    # The two keepers never stand side by side, so these two slots close together are one
    # person even in a crowd; any other cross-team pair has to be alone to count.
    with np.errstate(invalid="ignore"):
        depth = np.minimum(pm[:, :, 0], cfg.pitch_len_m - pm[:, :, 0])
        in_box = known & (depth <= cfg.dup_depth_m) & (
            np.abs(pm[:, :, 1] - cfg.pitch_wid_m / 2.0) <= cfg.keeper_half_width_m)
    deepest = np.argmin(np.where(in_box, depth, np.inf), axis=1)[in_box.any(axis=1)]
    counts = np.bincount(deepest, minlength=s)
    keeper = np.zeros(s, dtype=bool)
    for side in (home, ~home):
        if side.any() and counts[side].max() > 0:
            keeper[np.flatnonzero(side)[np.argmax(counts[side])]] = True
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        q = pm[a:b]
        d = np.linalg.norm(q[:, :, None, :] - q[:, None, :, :], axis=3)
        d = np.where(np.isfinite(d), d, np.inf)
        d[:, cols, cols] = np.inf
        close = (d < cfg.dup_radius_m).sum(axis=2)
        crowd = (d < cfg.dup_clear_m).sum(axis=2)
        partner = np.argmin(d, axis=2)
        rows = np.arange(b - a)[:, None]
        x, y = q[:, :, 0], q[:, :, 1]
        with np.errstate(invalid="ignore"):
            low_end = x <= cfg.dup_depth_m
            high_end = x >= cfg.pitch_len_m - cfg.dup_depth_m
            central = np.abs(y - cfg.pitch_wid_m / 2.0) <= cfg.keeper_half_width_m
        alone = (close == 1) & (crowd == 1) & (crowd[rows, partner] == 1)
        keepers = keeper[None, :] & keeper[partner] & (d[rows, cols[None, :], partner] < cfg.dup_radius_m)
        pair = (known[a:b] & (alone | keepers)
                & (home[partner] != home[None, :]) & (low_end | high_end) & central)
        defender_home = np.where(low_end, home_low[a:b, None], ~home_low[a:b, None])
        mark[a:b] = pair & (home[None, :] != defender_home)
    min_len = max(1, int(round(cfg.dup_min_s * cfg.fps)))
    out = players_m.copy()
    for slot in range(s):
        for a, b in _true_runs(mark[:, slot]):
            if b - a >= min_len:
                out[a:b, slot] = np.nan
    return out


def ball_hidden(ball_m, usable, held, cfg: Config) -> np.ndarray:
    """Unseen ball that, when last seen, was at a player's feet or among several players.

    Decided on the last frame it was seen and kept for cfg.occlude_max_s: the ball moves
    with whoever hides it, so its last position soon says nothing.
    """
    n = len(ball_m)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(usable, idx, -1))
    nearest = np.full(n, np.inf)
    crowd = np.zeros(n, dtype=np.int64)
    for a in range(0, n, 20000):
        z = min(n, a + 20000)
        d = np.linalg.norm(held[a:z].astype(np.float64) - ball_m[a:z, None, :], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        nearest[a:z] = d.min(axis=1)
        crowd[a:z] = (d < cfg.crowd_radius_m).sum(axis=1)
    covered = usable & ((nearest <= cfg.occlude_m) | (crowd >= cfg.crowd_players))
    recent = (last >= 0) & (idx - last <= int(round(cfg.occlude_max_s * cfg.fps)))
    return ~usable & recent & covered[np.maximum(last, 0)]


def ball_left_pitch(ball_m: np.ndarray, usable: np.ndarray, cfg: Config) -> np.ndarray:
    """True on unseen frames whose last seen ball was over a line or about to cross one."""
    n = len(ball_m)
    left = np.zeros(n, dtype=bool)
    if n == 0:
        return left
    with np.errstate(invalid="ignore"):
        edge = np.minimum.reduce([ball_m[:, 0], cfg.pitch_len_m - ball_m[:, 0],
                                  ball_m[:, 1], cfg.pitch_wid_m - ball_m[:, 1]])
    k = max(1, int(cfg.exit_heading_frames))
    before = np.full(n, np.nan)
    before[k:] = edge[:-k]
    with np.errstate(invalid="ignore"):
        leaving = usable & (edge <= cfg.exit_edge_m) & ((edge < 0.0) | (edge < before))
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(usable, idx, -1))
    left = ~usable & (last >= 0) & leaving[np.maximum(last, 0)]
    return left


def hold_last_position(players_m: np.ndarray, max_frames: int) -> np.ndarray:
    """Each track slot keeps its last known position for up to `max_frames` after it is lost.

    Only gaps are filled, and only forwards: nothing is invented before a track first appears.
    """
    if max_frames <= 0 or len(players_m) == 0:
        return players_m
    n = len(players_m)
    known = np.isfinite(players_m).all(axis=2)
    rows = np.arange(n)[:, None]
    last = np.maximum.accumulate(np.where(known, rows, -1), axis=0)
    fill = (~known) & (last >= 0) & ((rows - last) <= max_frames)
    out = players_m.copy()
    frame_idx, slot_idx = np.nonzero(fill)
    out[frame_idx, slot_idx] = players_m[last[frame_idx, slot_idx], slot_idx]
    return out


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


def team_by_frame(players_m: np.ndarray, home: np.ndarray, cfg: Config) -> np.ndarray:
    """(n_frames, n_slots) bool, True where the slot is a home player on that frame.

    The metadata team of each slot, except for tracklets that behave like a goalkeeper: for
    at least cfg.keeper_share of a tracklet of cfg.keeper_min_run_s or more, it is the
    deepest tracked player in front of one goal (within cfg.keeper_depth_m of the line and
    inside the width of the penalty area). Such a tracklet belongs to the team defending
    that goal. Every other slot keeps its metadata team.
    """
    home = np.asarray(home, dtype=bool)
    n, s = players_m.shape[:2]
    team = np.broadcast_to(home, (n, s)).copy()
    if not cfg.keeper_fix or n == 0:
        return team
    x = players_m[:, :, 0].astype(np.float64)
    y = players_m[:, :, 1].astype(np.float64)
    known = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore"):
        central = known & (np.abs(y - cfg.pitch_wid_m / 2.0) <= cfg.keeper_half_width_m)
    rows = np.arange(n)
    low_x = np.where(known, x, np.inf)
    high_x = np.where(known, x, -np.inf)
    low, high = np.argmin(low_x, axis=1), np.argmax(high_x, axis=1)
    deep_low = np.zeros((n, s), dtype=bool)
    deep_high = np.zeros((n, s), dtype=bool)
    deep_low[rows, low] = low_x[rows, low] <= cfg.keeper_depth_m
    deep_high[rows, high] = high_x[rows, high] >= cfg.pitch_len_m - cfg.keeper_depth_m
    deep_low &= central
    deep_high &= central

    home_low = home_defends_low_x(players_m, home, cfg)[:, None]
    keeper_home = (deep_low & home_low) | (deep_high & ~home_low)
    keeper_away = (deep_low & ~home_low) | (deep_high & home_low)
    min_len = max(1, int(round(cfg.keeper_min_run_s * cfg.fps)))
    for slot in range(s):
        for a, b in _true_runs(known[:, slot]):
            if b - a < min_len:
                continue
            if not home[slot] and keeper_home[a:b, slot].mean() >= cfg.keeper_share:
                team[a:b, slot] = True
            elif home[slot] and keeper_away[a:b, slot].mean() >= cfg.keeper_share:
                team[a:b, slot] = False
    return team


def build_features(tr: Tracking, cfg: Config) -> Features:
    scale = np.array([cfg.pitch_len_m, cfg.pitch_wid_m], dtype=np.float64)
    ball_n, seen, usable = interpolate_short_gaps(tr.ball, cfg.max_gap_frames)
    ball_m = ball_n * scale
    players_m = drop_frozen(tr.players * scale.astype(np.float32),
                            int(round(cfg.ghost_frozen_s * cfg.fps)))
    players_m = drop_keeper_doubles(players_m, tr.home, cfg)

    held = hold_last_position(players_m, int(round(cfg.player_hold_s * cfg.fps)))
    team = team_by_frame(players_m, tr.home, cfg)[:, :, None]
    everyone = np.ones(len(tr.home), dtype=bool)
    d_home = nearest_distance(np.where(team, held, np.nan), ball_m, everyone, cfg.dist_cap_m)
    d_away = nearest_distance(np.where(team, np.nan, held), ball_m, everyone, cfg.dist_cap_m)

    speed, slow = slow_ball(ball_m, cfg)
    still_run_s = run_lengths(slow) / float(cfg.fps)

    mx, my = cfg.oob_margin_m / cfg.pitch_len_m, cfg.oob_margin_m / cfg.pitch_wid_m
    with np.errstate(invalid="ignore"):
        out = usable & ((ball_n[:, 0] < -mx) | (ball_n[:, 0] > 1 + mx)
                        | (ball_n[:, 1] < -my) | (ball_n[:, 1] > 1 + my))
    oob = rolling_mean(out.astype(np.float64), cfg.oob_window_frames)

    missing_run_s = run_lengths(~usable) / float(cfg.fps)
    slowness = player_slowness(players_m, cfg)
    left = ball_left_pitch(ball_m, usable, cfg)
    occluded = ball_hidden(ball_m, usable, held, cfg) & ~left
    return Features(seen, usable, d_home, d_away, oob, still_run_s, missing_run_s, speed, slowness,
                    left, occluded)
