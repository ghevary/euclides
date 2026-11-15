"""A goalkeeper the tracking puts in the other team. Synthetic tracking only."""
import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import build_features, team_by_frame
from mtah.model import log_emissions
from mtah.tagging import empty_context
from mtah.temporal import decode

FPS = 25
SCALE = np.array([105.3, 68.0])


def _match(keeper_fix=True):
    """Home defends x = 0 and its keeper is tracked in an AWAY slot. Home keeps the ball in
    its own half for 30 s, 5 s with the keeper, then 5 s with a defender, and so on."""
    n = 30 * FPS
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]], keeper_fix=keeper_fix)
    t = np.arange(n) / FPS
    players = np.empty((n, 6, 2))
    players[:, 0] = (22.0, 20.0)                                          # home defender
    players[:, 1] = (40.0, 50.0)                                          # home midfielder
    players[:, 2] = np.stack([4.0 + np.sin(t), 34.0 + 2 * np.cos(t)], axis=1)  # HOME KEEPER, away slot
    players[:, 3] = (60.0, 30.0)                                          # away forward
    players[:, 4] = (75.0, 40.0)                                          # away midfielder
    players[:, 5] = (101.0, 34.0)                                         # away keeper
    home = np.array([True, True, False, False, False, False])
    ball = np.empty((n, 2))
    phase = (np.arange(n) // (5 * FPS)) % 2
    ball[phase == 0] = players[phase == 0, 2] + (0.4, 0.0)
    ball[phase == 1] = players[phase == 1, 0] + (0.4, 0.0)
    tr = Tracking((players / SCALE).astype(np.float32), (ball / SCALE).astype(np.float32), home)
    feat = build_features(tr, cfg)
    return decode(log_emissions(feat, empty_context(n), cfg), cfg), tr, cfg


def test_the_keeper_is_given_back_to_the_team_whose_goal_it_guards():
    _, tr, cfg = _match()
    scale = np.array([cfg.pitch_len_m, cfg.pitch_wid_m], dtype=np.float32)
    team = team_by_frame(tr.players * scale, tr.home, cfg)
    assert team[:, 2].all()                                   # the keeper is home
    assert not team[:, 5].any()                               # the away keeper stays away
    assert team[:, [0, 1]].all() and not team[:, [3, 4]].any()


def test_possession_stays_home_when_the_keeper_has_the_ball():
    labels, _, _ = _match()
    assert (labels == "H").all()


def test_without_the_fix_the_keeper_hands_the_ball_to_the_away_team():
    labels, _, _ = _match(keeper_fix=False)
    assert (labels == "A").mean() > 0.3


def test_short_tracklets_are_left_alone():
    cfg = synth.small_config(n_frames=200, segments=[[0, 199]])
    players = np.full((200, 3, 2), np.nan)
    players[:, 0] = (50.0, 34.0)
    players[:, 1] = (60.0, 34.0)
    players[:50, 2] = (3.0, 34.0)                             # deepest, but only 2 s long
    team = team_by_frame(players, np.array([True, False, False]), cfg)
    assert not team[:, 2].any()
