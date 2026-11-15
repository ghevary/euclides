"""A player the tracking loses in a duel is still there. Synthetic tracking only."""
import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import build_features, hold_last_position
from mtah.model import log_emissions
from mtah.tagging import empty_context
from mtah.temporal import decode

FPS = 25
SCALE = np.array([105.3, 68.0])


def test_a_lost_track_keeps_its_position_for_a_while_only():
    players = np.full((10, 2, 2), np.nan)
    players[:3, 0] = (5.0, 5.0)                    # slot 0 seen on frames 0-2, then lost
    players[6:, 1] = (9.0, 9.0)                    # slot 1 first appears on frame 6
    out = hold_last_position(players, 4)
    assert (out[3:7, 0] == (5.0, 5.0)).all()       # held for 4 frames
    assert np.isnan(out[7:, 0]).all()              # then gone
    assert np.isnan(out[:6, 1]).all()              # nothing invented before it appears
    assert hold_last_position(players, 0) is players


def _duel(hold_s):
    """Home dribbles for 24 s with an away player on their shoulder. The tracking keeps
    dropping the home player for 1.2 s out of every 2 s, as it does when two bodies overlap."""
    n = 24 * FPS
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]], player_hold_s=hold_s)
    t = np.arange(n) / FPS
    carrier = np.stack([30.0 + 2.0 * t, np.full(n, 34.0)], axis=1)     # home, 2 m/s
    players = np.empty((n, 4, 2))
    players[:, 0] = carrier
    players[:, 1] = (10.0, 10.0)                                       # home, far away
    players[:, 2] = carrier + (0.0, 1.2)                               # away, in the duel
    players[:, 3] = (95.0, 60.0)                                       # away, far away
    ball = carrier + (0.4, 0.0)
    lost = (np.arange(n) % (2 * FPS)) < int(1.2 * FPS)
    lost[:4 * FPS] = False                                             # a clear start
    players[lost, 0] = np.nan
    home = np.array([True, True, False, False])
    tr = Tracking((players / SCALE).astype(np.float32), (ball / SCALE).astype(np.float32), home)
    feat = build_features(tr, cfg)
    return decode(log_emissions(feat, empty_context(n), cfg), cfg), feat, lost


def test_possession_survives_a_duel_in_which_the_carrier_is_lost():
    labels, feat, lost = _duel(hold_s=1.5)
    assert (feat.d_home[lost] < 3.0).all()         # the carrier is still counted
    assert (labels == "H").all()


def test_without_the_hold_the_duel_hands_the_ball_to_the_other_team():
    labels, feat, lost = _duel(hold_s=0.0)
    assert (feat.d_home[lost] > 10.0).all()
    assert (labels == "A").mean() > 0.3
