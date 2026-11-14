import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import (build_features, interpolate_short_gaps, nearest_distance,
                           rolling_mean, run_lengths)


def test_rolling_mean_is_centred_and_shrinks_at_edges():
    x = np.array([0, 0, 3, 0, 0], dtype=float)
    np.testing.assert_allclose(rolling_mean(x, 3), [0, 1, 1, 1, 0])
    np.testing.assert_allclose(rolling_mean(np.ones(4), 9), np.ones(4))


def test_run_lengths():
    mask = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert run_lengths(mask).tolist() == [2, 2, 0, 1, 0, 0, 3, 3, 3]
    assert run_lengths(np.zeros(3, dtype=bool)).tolist() == [0, 0, 0]


def test_only_short_gaps_are_interpolated():
    ball = np.full((30, 2), np.nan)
    ball[0] = (0.0, 0.0)
    ball[4] = (0.4, 0.8)        # gap of 3 frames
    ball[25] = (1.0, 1.0)       # gap of 20 frames
    out, seen, usable = interpolate_short_gaps(ball, max_gap=5)
    np.testing.assert_allclose(out[2], (0.2, 0.4))
    assert usable[:5].all() and not usable[5:25].any() and usable[25]
    assert np.isnan(out[10]).all() and not usable[26:].any()   # nothing is extrapolated
    assert seen.sum() == 3


def test_nearest_distance_per_team_and_cap():
    players = np.full((2, 4, 2), np.nan, dtype=np.float32)
    players[:, 0] = (0.0, 0.0)      # home, 5 m from the ball
    players[:, 2] = (3.0, 0.0)      # away, 1 m from the ball
    ball = np.array([[4.0, 0.0], [np.nan, np.nan]])
    home = np.array([True, True, False, False])
    d_home = nearest_distance(players, ball, home, cap=30.0)
    d_away = nearest_distance(players, ball, ~home, cap=30.0)
    np.testing.assert_allclose([d_home[0], d_away[0]], [4.0, 1.0])
    assert d_home[1] == 30.0 and d_away[1] == 30.0     # no ball: capped, never NaN


def test_out_of_pitch_stationary_and_missing_run():
    cfg = synth.small_config(n_frames=400, segments=[[0, 399]])
    players = np.full((400, cfg.n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((400, 2), np.nan, dtype=np.float32)
    ball[:100] = (0.5, 0.5)                                  # motionless on the pitch
    ball[100:200] = (0.5, 1.2)                               # well outside the touchline
    home = np.arange(cfg.n_slots) < 24
    feat = build_features(Tracking(players, ball, home), cfg)
    assert feat.still_run_s[50] > 2.0 and feat.oob[50] == 0.0
    assert feat.oob[150] == 1.0
    assert feat.missing_run_s[300] == 200 / cfg.fps and feat.missing_run_s[50] == 0.0
    assert not feat.usable[300] and feat.usable[150]


def test_jitter_is_not_movement_but_a_pass_is():
    cfg = synth.small_config(n_frames=300, segments=[[0, 299]])
    rng = np.random.default_rng(3)
    players = np.full((300, cfg.n_slots, 2), np.nan, dtype=np.float32)
    ball = np.empty((300, 2))
    ball[:150] = (0.5, 0.5) + rng.normal(0, 0.05, (150, 2)) / (cfg.pitch_len_m, cfg.pitch_wid_m)
    ball[150:] = np.linspace((0.5, 0.5), (0.9, 0.5), 150)       # ~7 m/s
    feat = build_features(Tracking(players, ball.astype(np.float32), np.arange(cfg.n_slots) < 24), cfg)
    assert feat.still_run_s[60] > 4.0                            # 5 cm of jitter is still "at rest"
    assert feat.still_run_s[220] == 0.0 and 5.0 < feat.speed[220] < 9.0
