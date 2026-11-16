"""Frozen ghost tracks and a ball that leaves the pitch. Synthetic tracking only."""
import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import ball_left_pitch, build_features, drop_frozen, drop_keeper_doubles
from mtah.model import log_emissions
from mtah.tagging import empty_context
from mtah.temporal import decode

FPS = 25
SCALE = np.array([105.3, 68.0])


def test_a_frozen_copy_is_dropped_and_a_swaying_player_is_kept():
    n = 100
    players = np.zeros((n, 2, 2))
    players[:, 0] = (10.0, 30.0)                                    # frozen: a ghost
    players[:, 1] = (12.0, 30.0)
    players[:, 1, 0] += 0.003 * np.sin(np.arange(n))                # standing, but alive
    out = drop_frozen(players, FPS)
    assert np.isfinite(out[0, 0]).all() and np.isnan(out[1:, 0]).all()
    assert np.isfinite(out[:, 1]).all()
    short = players.copy()
    short[10:, 0, 0] += np.arange(n - 10) * 0.1                     # frozen for 9 frames only
    assert np.isfinite(drop_frozen(short, FPS)).all()


def test_ball_lost_on_its_way_over_the_line_has_left():
    cfg = synth.small_config()
    ball = np.full((20, 2), np.nan)
    ball[:10, 0] = 50.0
    ball[:10, 1] = 60.0 + np.arange(10) * 0.8                       # running towards y = 68
    usable = np.isfinite(ball).all(axis=1)
    assert ball_left_pitch(ball, usable, cfg)[10:].all()
    inward = ball.copy()
    inward[:10, 1] = 67.5 - np.arange(10) * 0.8                     # near the line, moving in
    assert not ball_left_pitch(inward, usable, cfg).any()
    middle = ball.copy()
    middle[:10, 1] = 30.0 + np.arange(10) * 0.8                     # lost in midfield
    assert not ball_left_pitch(middle, usable, cfg).any()


def _clearance(use_fix=True):
    """Away plays, kicks the ball over the touchline, then everybody keeps running 6 s."""
    play, after = 10 * FPS, 6 * FPS
    n = play + after
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]], use_player_motion=True)
    t = np.arange(n) / FPS
    players = np.empty((n, 10, 2))
    for k in range(10):
        players[:, k, 0] = 20 + 6 * k + 3.0 * t                     # everybody runs at 3 m/s
        players[:, k, 1] = 20 + 3 * (k % 4)
    home = np.arange(10) < 5
    ball = np.full((n, 2), np.nan)
    ball[:play] = players[:play, 7] + (0.4, 0.0)
    kick = play - FPS
    ball[kick:play, 1] = ball[kick, 1] + np.linspace(0, 68 - ball[kick, 1] + 1.0, FPS)
    if not use_fix:
        cfg.exit_edge_m = -1e9
    tr = Tracking((players / SCALE).astype(np.float32), (ball / SCALE).astype(np.float32), home)
    feat = build_features(tr, cfg)
    return decode(log_emissions(feat, empty_context(n), cfg), cfg), play


def test_ball_kicked_out_is_dead_at_once_while_players_still_run():
    labels, play = _clearance()
    assert (labels[play + FPS:] == "D").all()


def test_without_the_exit_the_runners_keep_it_alive():
    labels, play = _clearance(use_fix=False)
    assert (labels[play + FPS:play + 3 * FPS] != "D").any()


def test_a_keeper_followed_in_both_teams_keeps_only_the_defending_copy():
    n = 3 * FPS
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]])
    t = np.arange(n) / FPS
    players = np.empty((n, 6, 2))
    players[:, 0] = (30.0, 20.0)                                    # home defender
    players[:, 1] = np.stack([5.0 + 0.3 * np.sin(t), 34.0 + 0.2 * np.cos(t)], 1)  # home keeper
    players[:, 2] = (45.0, 40.0)
    players[:, 3] = players[:, 1] + (2.5, 1.5)                      # its copy in an away slot
    players[:, 4] = (70.0, 30.0)                                    # away field player
    players[:, 5] = np.stack([100.0 + 0.3 * np.sin(t), 34.0 + 0.0 * t], 1)  # away keeper
    players[:, 5, 1] += 0.01 * np.cos(t * 3)
    home = np.array([True, True, True, False, False, False])
    out = drop_keeper_doubles(players, home, cfg)
    assert np.isnan(out[:, 3]).all()
    assert np.isfinite(out[:, [0, 1, 2, 4, 5]]).all()
