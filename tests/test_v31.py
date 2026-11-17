"""Whistle before a free kick; a ball hidden by a player's body. Synthetic only."""
import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import build_features
from mtah.model import log_emissions, since_whistle
from mtah.tagging import empty_context
from mtah.temporal import decode

FPS = 25
SCALE = np.array([105.3, 68.0])


def test_since_whistle_starts_where_the_running_stopped():
    cfg = synth.small_config()
    slow = np.full(100 * FPS, -1.0)          # running
    slow[40 * FPS:70 * FPS] = 0.8            # whistle at 40 s, everybody walks
    out = since_whistle(np.array([70 * FPS]), slow, cfg)
    assert out[40 * FPS:70 * FPS].all()
    assert not out[:40 * FPS].any() and not out[70 * FPS:].any()
    assert not since_whistle(np.array([70 * FPS]), np.full(100 * FPS, 0.8), cfg).any()


def _carry_to_the_spot(with_restart=True):
    """Home attacks at a run; foul at 20 s; a home player walks the ball 15 m to the spot
    for 25 s; free kick at 45 s; play on."""
    n = 60 * FPS
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]], use_player_motion=True)
    t = np.arange(n) / FPS
    stop = (t >= 20) & (t < 45)
    speed = np.where(stop, 0.6, 4.0)
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 6.28, 12)
    p = np.empty((n, 12, 2))
    for k in range(12):
        p[:, k, 0] = 20 + 6 * k + np.cumsum(speed * np.cos(t / 4 + phase[k])) / FPS
        p[:, k, 1] = 34 + 10 * np.sin(phase[k]) + np.cumsum(speed * np.sin(t / 4 + phase[k])) / FPS
    home = np.arange(12) < 6
    ball = p[:, 2] + (0.4, 0.0)                 # a home player has it, also while carrying it
    ctx = empty_context(n)
    if with_restart:
        ctx.restarts = np.array([45 * FPS])
    tr = Tracking((p / SCALE).astype(np.float32), (ball / SCALE).astype(np.float32), home)
    feat = build_features(tr, cfg)
    return decode(log_emissions(feat, ctx, cfg), cfg)


def test_ball_carried_to_the_free_kick_spot_is_dead_from_the_whistle():
    labels = _carry_to_the_spot()
    assert (labels[22 * FPS:44 * FPS] == "D").mean() > 0.9
    assert (labels[:19 * FPS] == "H").all()


def test_ball_hidden_by_a_body_stays_alive():
    n = 20 * FPS
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]], missing_neutral_s=1.0,
                             missing_dead_s=3.0)
    t = np.arange(n) / FPS
    p = np.empty((n, 4, 2))
    p[:, 0] = np.stack([20 + 2.0 * t, 34 + 0.1 * np.cos(t)], 1)          # home, dribbling
    p[:, 1] = np.stack([21 + 2.0 * t, 35.5 + 0.1 * np.sin(t)], 1)        # away, on the shoulder
    p[:, 2] = np.stack([20 + 0.1 * np.sin(t), 20 + 0.0 * t], 1)
    p[:, 3] = np.stack([80 + 0.1 * np.sin(t), 50 + 0.0 * t], 1)
    home = np.array([True, False, True, False])
    ball = np.full((n, 2), np.nan)
    ball[:8 * FPS] = p[:8 * FPS, 0] + (0.5, 0.0)
    ball[12 * FPS:] = p[12 * FPS:, 0] + (0.5, 0.0)     # hidden by the body for 4 s
    tr = Tracking((p / SCALE).astype(np.float32), (ball / SCALE).astype(np.float32), home)
    feat = build_features(tr, cfg)
    assert feat.occluded[8 * FPS:12 * FPS].all()
    labels = decode(log_emissions(feat, empty_context(n), cfg), cfg)
    assert (labels == "H").all()
