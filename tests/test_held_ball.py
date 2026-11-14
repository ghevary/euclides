"""A ball held by the goalkeeper is in play. Synthetic tracking only."""
import numpy as np

import synth
from mtah.atd import Tracking
from mtah.features import build_features, home_defends_low_x
from mtah.model import log_emissions
from mtah.tagging import empty_context
from mtah.temporal import decode

FPS = 25
L, W = 105.3, 68.0
KEEPER = np.array([100.0, 34.0])      # away keeper, inside the box at high x
MIDFIELD = np.array([60.0, 30.0])     # away midfielder


ATTACKER = np.array([92.0, 20.0])     # home forward inside the away penalty area


def _match(hold: str, arrive: str = "in_play", hold_s: int = 6, attacker_in_box: bool = True,
           **cfg_overrides):
    """Away pass in midfield, ball goes to the away keeper, who keeps it, then plays on.

    hold: "resting" (ball seen, motionless) or "unseen" (ball vanishes in the keeper's arms).
    arrive: "in_play" (tracked all the way) or "from_out" (out of the pitch and lost first,
            then placed in the box: a goal kick).
    """
    pre, travel, post = 10 * FPS, 2 * FPS, 10 * FPS
    hold_n = hold_s * FPS
    n = pre + travel + hold_n + post
    cfg = synth.small_config(n_frames=n, segments=[[0, n - 1]], **cfg_overrides)
    players = np.empty((n, 4, 2))
    players[:, 0] = (20.0, 30.0)      # home
    players[:, 1] = ATTACKER if attacker_in_box else (45.0, 40.0)   # home
    players[:, 2] = MIDFIELD          # away
    players[:, 3] = KEEPER            # away keeper
    home = np.array([True, True, False, False])

    ball = np.empty((n, 2))
    angle = np.arange(pre) / FPS                          # dribbling around the midfielder
    ball[:pre] = MIDFIELD + 2.0 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    w = np.linspace(0.0, 1.0, travel)[:, None]
    ball[pre:pre + travel] = ball[pre - 1] * (1 - w) + (KEEPER + (0.3, 0.2)) * w
    a, b = pre + travel, pre + travel + hold_n
    ball[a:b] = KEEPER + (0.3, 0.2)
    w = np.linspace(0.0, 1.0, post)[:, None]
    ball[b:] = (KEEPER + (0.3, 0.2)) * (1 - w) + (MIDFIELD + (0.3, 0.2)) * w
    if arrive == "from_out":
        ball[pre:pre + travel // 2] = (L + 4.0, 30.0)     # behind the goal line
        ball[pre + travel // 2:a + 3 * FPS] = np.nan      # lost while it is fetched
    if hold == "unseen":
        ball[a + FPS:b - FPS] = np.nan
    scale = np.array([L, W])
    tr = Tracking((players / scale).astype(np.float32), (ball / scale).astype(np.float32), home)
    feat = build_features(tr, cfg)
    labels = decode(log_emissions(feat, empty_context(n), cfg), cfg)
    return labels, feat, (a, b)


def test_sides_follow_where_the_teams_stand():
    cfg = synth.small_config(n_frames=200, segments=[[0, 99], [100, 199]])
    players = np.empty((200, 4, 2), dtype=np.float32)
    players[:100, :2, 0], players[:100, 2:, 0] = 0.3, 0.7     # home on the low side
    players[100:, :2, 0], players[100:, 2:, 0] = 0.7, 0.3     # ends changed
    players[:, :, 1] = 0.5
    low = home_defends_low_x(players, np.array([True, True, False, False]), cfg)
    assert low[:100].all() and not low[100:].any()


def test_keeper_holding_a_resting_ball_keeps_possession():
    labels, feat, (a, b) = _match("resting")
    assert feat.held[a + 2 * FPS:b - 2 * FPS].all()
    assert (labels == "A").all()


def test_keeper_hiding_the_ball_keeps_possession():
    labels, feat, (a, b) = _match("unseen")
    assert feat.held[a + 2 * FPS:b - 2 * FPS].all()
    assert (labels == "A").all()


def test_without_the_rule_the_same_hold_reads_as_a_stoppage():
    for hold in ("resting", "unseen"):
        labels, feat, (a, b) = _match(hold, hold_s=12, held_ball=False)
        assert not feat.held.any()
        assert (labels[a + 5 * FPS:b - 2 * FPS] == "D").all()


def test_ball_coming_back_from_out_of_play_is_still_a_stoppage():
    labels, feat, (a, b) = _match("resting", arrive="from_out", hold_s=12)
    assert not feat.held[a:b].any()                    # a goal kick is not a held ball
    assert (labels[a + 5 * FPS:b - 2 * FPS] == "D").all()


def test_no_attacker_in_the_area_means_a_restart_not_a_hold():
    for hold in ("resting", "unseen"):
        labels, feat, (a, b) = _match(hold, hold_s=12, attacker_in_box=False)
        assert not feat.held[a:b].any()
        assert (labels[a + 5 * FPS:b - 2 * FPS] == "D").all()


def test_a_hold_is_capped():
    labels, feat, (a, b) = _match("resting", hold_s=60)
    cfg = synth.small_config()
    assert feat.held[a + 2 * FPS:a + int(cfg.held_max_s * FPS) - FPS].all()
    assert not feat.held[a + int((cfg.held_max_s + 3) * FPS):b].any()
    assert (labels[a + 40 * FPS:b - 2 * FPS] == "D").all()
