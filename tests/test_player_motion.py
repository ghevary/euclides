"""The players' pace as evidence of a stoppage. Synthetic tracking only."""
import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import build_features, player_pace, slow_threshold
from mtah.model import log_emissions
from mtah.tagging import empty_context
from mtah.temporal import decode

FPS = 25
SCALE = np.array([105.3, 68.0])
RUN, WALK = 3.5, 0.6            # m/s
RADIUS = 10.0                   # every player moves on a circle of this radius


def _cfg(n, **overrides):
    overrides.setdefault("use_player_motion", True)
    return synth.small_config(n_frames=n, segments=[[0, n - 1]], **overrides)


def _players(pace_per_frame):
    """22 players (11 home, then 11 away) on circles, all at the given pace per frame."""
    n = len(pace_per_frame)
    angle = np.cumsum(np.asarray(pace_per_frame, dtype=float) / RADIUS / FPS)
    centres = np.array([[15.0 + 7.5 * (i % 11), 20.0 + 28.0 * (i // 11)] for i in range(22)])
    phase = np.arange(22) * 0.7
    xy = np.empty((n, 22, 2))
    xy[:, :, 0] = centres[:, 0] + RADIUS * np.cos(angle[:, None] + phase)
    xy[:, :, 1] = centres[:, 1] + RADIUS * np.sin(angle[:, None] + phase)
    return xy, np.arange(22) < 11


def _labels(players, home, ball, cfg):
    tr = Tracking((players / SCALE).astype(np.float32), (ball / SCALE).astype(np.float32), home)
    feat = build_features(tr, cfg)
    return decode(log_emissions(feat, empty_context(cfg.n_frames), cfg), cfg), feat


def test_pace_is_measured_and_the_threshold_separates_walking_from_running():
    pace_in = np.concatenate([np.full(1500, RUN), np.full(1500, WALK)])
    players, _ = _players(pace_in)
    cfg = _cfg(len(pace_in))
    pace = player_pace(players, cfg)
    assert abs(float(np.nanmedian(pace[200:1300])) - RUN) < 0.15
    assert abs(float(np.nanmedian(pace[1700:2800])) - WALK) < 0.1
    threshold = slow_threshold(pace, cfg)
    assert WALK < threshold < RUN
    assert cfg.player_slow_min_mps <= threshold <= cfg.player_slow_max_mps


def test_threshold_is_the_valley_and_falls_back_to_the_prior_without_one():
    cfg = _cfg(6000)
    rng = np.random.default_rng(3)
    two_groups = np.concatenate([rng.normal(1.2, 0.3, 3000), rng.normal(3.6, 0.6, 3000)])
    assert 1.8 <= slow_threshold(two_groups, cfg) <= 2.6
    one_group = rng.normal(5.0, 0.3, 6000)                     # no valley inside the priors
    assert slow_threshold(one_group, cfg) == cfg.player_slow_mps
    assert slow_threshold(np.full(6000, np.nan), cfg) == cfg.player_slow_mps
    cfg.player_slow_auto = False
    assert slow_threshold(two_groups, cfg) == cfg.player_slow_mps


def test_too_few_tracked_players_say_nothing():
    players, _ = _players(np.full(400, RUN))
    players[:, 5:] = np.nan
    assert np.isnan(player_pace(players, _cfg(400))).all()


def _held_ball_match(n_hold):
    """Away has the ball; then a standing away player keeps it still while everybody runs."""
    pre, post = 12 * FPS, 12 * FPS
    n = pre + n_hold + post
    players, home = _players(np.full(n, RUN))
    keeper = 21
    sway = 0.01 * np.sin(np.arange(n_hold) / 7.0)[:, None]             # a person never freezes
    players[pre:pre + n_hold, keeper] = players[pre, keeper] + sway   # the keeper stands on it
    ball = players[:, keeper] + (0.3, 0.2)
    return players, home, ball, (pre, pre + n_hold)


def test_resting_ball_is_alive_while_the_players_keep_running():
    players, home, ball, (a, b) = _held_ball_match(12 * FPS)
    labels, feat = _labels(players, home, ball, _cfg(len(ball)))
    assert (feat.slowness[a:b] < -0.5).all()
    assert (labels == "A").all()


def test_hidden_ball_is_alive_while_the_players_keep_running():
    players, home, ball, (a, b) = _held_ball_match(12 * FPS)
    ball = ball.copy()
    ball[a + FPS:b - FPS] = np.nan
    labels, _ = _labels(players, home, ball, _cfg(len(ball)))
    assert (labels == "A").all()


def test_without_player_motion_the_same_hold_is_a_stoppage():
    players, home, ball, (a, b) = _held_ball_match(12 * FPS)
    labels, _ = _labels(players, home, ball, _cfg(len(ball), use_player_motion=False))
    assert (labels[a + 6 * FPS:b - FPS] == "D").all()


def test_moving_ball_is_dead_when_everybody_walks():
    """A stoppage in which the ball stays in sight and on the move: carried to a free kick."""
    live, stop = 14 * FPS, 16 * FPS
    pace = np.concatenate([np.full(live, RUN), np.full(stop, WALK), np.full(live, RUN)])
    players, home = _players(pace)
    carrier = 3                                                       # a home player
    ball = players[:, carrier] + (0.3, 0.2)
    wander = np.cumsum(np.full(stop, 1.5 / FPS))                      # carried at 1.5 m/s
    ball[live:live + stop, 0] += wander - wander[0]
    ball[live + stop:, 0] = players[live + stop:, carrier, 0] + 0.3
    labels, feat = _labels(players, home, ball, _cfg(len(pace)))
    assert (feat.slowness[live + 3 * FPS:live + stop - 3 * FPS] > 0.5).all()
    assert (labels[live + 4 * FPS:live + stop - 4 * FPS] == "D").all()
    assert (labels[:live - 2 * FPS] == "H").all() and (labels[live + stop + 3 * FPS:] == "H").all()
