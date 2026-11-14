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
    still = _ramp(feat.still_run_s, cfg.still_neutral_s, cfg.still_dead_s)
    seen_logit = (_logit(cfg.dead_prior) + cfg.dead_w_out * feat.oob
                  + cfg.dead_w_stationary * still)
    unseen = _ramp(feat.missing_run_s, cfg.missing_neutral_s, cfg.missing_dead_s)
    neutral = 1.0 / 3.0
    missing_logit = _logit(neutral + (cfg.missing_dead_max - neutral) * unseen)
    dead_logit = np.where(feat.usable, seen_logit, missing_logit) + cfg.dead_w_setpiece * ctx.dead
    p_dead = _sigmoid(dead_logit)

    em = np.stack([(1.0 - p_dead) * p_home, (1.0 - p_dead) * (1.0 - p_home), p_dead], axis=1)
    log_em = np.log(np.clip(em, 1e-12, 1.0)) * cfg.emission_temperature
    log_em[ctx.forced] = (FORCED_PENALTY, FORCED_PENALTY, 0.0)
    return np.round(log_em, 6)
