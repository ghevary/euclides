"""Per-frame evidence for the three states H, A, D.

The rules follow CHALLENGE.md: while the ball is alive one team always has it, a loose ball
stays with the team that last had it, and a single touch is not a change of possession. So a
frame only ever supplies EVIDENCE; the decision is taken by the temporal model in temporal.py.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .features import Features
from .tagging import TagContext

H, A, D = 0, 1, 2
STATES = ("H", "A", "D")
FORCED_PENALTY = -50.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _ramp(x, lo: float, hi: float):
    """0 up to `lo`, 1 from `hi`, linear in between."""
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def since_whistle(restarts, slowness: np.ndarray, cfg: Config) -> np.ndarray:
    """1 from the moment the players last stopped running before a restart, up to it."""
    n = len(slowness)
    out = np.zeros(n)
    if restarts is None or len(restarts) == 0:
        return out
    running = (slowness < cfg.whistle_running).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], running, [0]))))
    run_min = int(round(cfg.whistle_run_min_s * cfg.fps))
    ends = np.array([b for a, b in zip(edges[::2], edges[1::2]) if b - a >= run_min], dtype=np.int64)
    back = int(round(cfg.whistle_max_back_s * cfg.fps))
    for r in restarts:
        k = np.searchsorted(ends, r, side="right") - 1
        if k >= 0 and r - ends[k] <= back:
            out[ends[k]:r] = 1.0
    return out


def log_emissions(feat: Features, ctx: TagContext, cfg: Config) -> np.ndarray:
    """(n, 3) tempered log-likelihoods, rounded so libm differences cannot change the output."""
    # Which team: the nearer team gets the evidence, and only when somebody is near the ball.
    margin = feat.d_away - feat.d_home
    near = np.exp(-np.minimum(feat.d_home, feat.d_away) / cfg.influence_radius_m)
    team_logit = np.where(feat.usable, cfg.beta_per_m * margin * near, 0.0)
    team_logit = team_logit + cfg.phase_team_weight * ctx.team
    p_home = _sigmoid(team_logit)

    # Dead ball: out of the pitch, motionless for a long time, or unseen for longer than an
    # occlusion lasts. Each is evidence, none is a rule: a slow ball alone is not a dead ball.
    # The players say whether the game is on. A ball that rests or disappears is a stoppage
    # only if they are not running (a keeper holding it, a shielded ball: still alive), and
    # a ball that is seen and moving is still dead if everybody walks (carried to a free kick).
    # `gate` is 0 when they run, 1 when they walk, and 1 where their pace is unknown.
    # Once the ball is out of the pitch the players have no say: they are still running from
    # the last action, but the game has stopped.
    left = np.zeros(len(feat.oob)) if feat.left is None else feat.left.astype(np.float64)
    out = np.maximum(feat.oob, left)
    slow = np.where(feat.slowness < 0.0, feat.slowness * (1.0 - out), feat.slowness)
    gate = np.where(slow < 0.0, 1.0 + slow, 1.0)
    still = _ramp(feat.still_run_s, cfg.still_neutral_s, cfg.still_dead_s) * gate
    seen_logit = (_logit(cfg.dead_prior) + cfg.dead_w_out * feat.oob
                  + cfg.dead_w_stationary * still + cfg.dead_w_players * slow)
    hidden = np.zeros(len(gate)) if feat.occluded is None else feat.occluded.astype(np.float64)
    unseen = _ramp(feat.missing_run_s, cfg.missing_neutral_s, cfg.missing_dead_s) * gate * (1.0 - hidden)
    neutral = 1.0 / 3.0
    missing_logit = (_logit(neutral + (cfg.missing_dead_max - neutral) * unseen)
                     + 0.5 * cfg.dead_w_players * slow + cfg.dead_w_out * left)
    dead_logit = (np.where(feat.usable, seen_logit, missing_logit) + cfg.dead_w_setpiece * ctx.dead
                  + cfg.dead_w_whistle * since_whistle(ctx.restarts, feat.slowness, cfg))
    p_dead = _sigmoid(dead_logit)

    em = np.stack([(1.0 - p_dead) * p_home, (1.0 - p_dead) * (1.0 - p_home), p_dead], axis=1)
    log_em = np.log(np.clip(em, 1e-12, 1.0)) * cfg.emission_temperature
    log_em[ctx.forced] = (FORCED_PENALTY, FORCED_PENALTY, 0.0)
    return np.round(log_em, 6)
