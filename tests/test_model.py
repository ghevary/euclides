import numpy as np

import synth
from mtah.features import Features
from mtah.model import A, D, H, log_emissions
from mtah.tagging import empty_context
from mtah.temporal import decode


def _features(n, **arrays):
    base = dict(seen=np.ones(n, bool), usable=np.ones(n, bool), d_home=np.full(n, 30.0),
                d_away=np.full(n, 30.0), oob=np.zeros(n), still_run_s=np.zeros(n),
                missing_run_s=np.zeros(n), speed=np.full(n, 5.0),
                slowness=np.zeros(n))
    base.update(arrays)
    return Features(**base)


def _cfg(n):
    return synth.small_config(n_frames=n, segments=[[0, n - 1]])


def test_nearest_team_gets_the_evidence():
    n = 10
    feat = _features(n, d_home=np.full(n, 0.5), d_away=np.full(n, 6.0))
    log_em = log_emissions(feat, empty_context(n), _cfg(n))
    assert (log_em.argmax(axis=1) == H).all()
    feat = _features(n, d_home=np.full(n, 6.0), d_away=np.full(n, 0.5))
    assert (log_emissions(feat, empty_context(n), _cfg(n)).argmax(axis=1) == A).all()


def test_nobody_near_the_ball_is_no_evidence_about_the_team():
    n = 5
    feat = _features(n, d_home=np.full(n, 25.0), d_away=np.full(n, 29.0))
    log_em = log_emissions(feat, empty_context(n), _cfg(n))
    assert abs(log_em[0, H] - log_em[0, A]) < 0.05 and log_em[0, D] < log_em[0, H]


def test_short_loss_of_the_ball_keeps_the_possession():
    n = 600
    usable = np.ones(n, bool)
    usable[250:300] = False                                  # 2 s without the ball
    feat = _features(n, usable=usable, seen=usable, d_home=np.where(usable, 0.5, 30.0),
                     d_away=np.where(usable, 7.0, 30.0),
                     missing_run_s=np.where(usable, 0.0, 2.0))
    out = decode(log_emissions(feat, empty_context(n), _cfg(n)), _cfg(n))
    assert (out == "H").all()


def test_long_loss_of_the_ball_becomes_a_stoppage_between_two_possessions():
    n = 900
    usable = np.ones(n, bool)
    usable[300:600] = False                                  # 12 s without the ball
    feat = _features(n, usable=usable, seen=usable, d_home=np.where(usable, 0.5, 30.0),
                     d_away=np.where(usable, 7.0, 30.0),
                     missing_run_s=np.where(usable, 0.0, 12.0))
    out = decode(log_emissions(feat, empty_context(n), _cfg(n)), _cfg(n))
    assert (out[:300] == "H").all() and (out[300:600] == "D").all() and (out[600:] == "H").all()


def test_ball_outside_the_pitch_is_dead_then_the_restart_goes_to_the_nearer_team():
    n = 800
    oob = np.zeros(n)
    oob[300:500] = 1.0
    still = np.zeros(n)
    still[300:500] = 8.0
    d_home = np.full(n, 0.5)
    d_away = np.full(n, 7.0)
    d_home[500:], d_away[500:] = 7.0, 0.5                    # away restarts
    feat = _features(n, oob=oob, still_run_s=still, d_home=d_home, d_away=d_away)
    out = decode(log_emissions(feat, empty_context(n), _cfg(n)), _cfg(n))
    assert (out[:300] == "H").all() and (out[320:480] == "D").all() and (out[520:] == "A").all()


def test_a_ball_at_rest_is_dead_only_when_it_rests_for_long():
    n = 900
    d_home, d_away = np.full(n, 0.5), np.full(n, 7.0)
    for seconds, expected in ((2.0, "H"), (10.0, "D")):
        still = np.zeros(n)
        still[300:300 + int(seconds * 25)] = seconds
        feat = _features(n, still_run_s=still, d_home=d_home, d_away=d_away)
        out = decode(log_emissions(feat, empty_context(n), _cfg(n)), _cfg(n))
        assert out[300 + int(seconds * 25) // 2] == expected
        assert out[100] == "H" and out[800] == "H"


def test_forced_frames_are_dead():
    n = 300
    ctx = empty_context(n)
    ctx.forced[:120] = True
    feat = _features(n, d_home=np.full(n, 0.5), d_away=np.full(n, 7.0))
    out = decode(log_emissions(feat, ctx, _cfg(n)), _cfg(n))
    assert (out[:120] == "D").all() and (out[120:] == "H").all()


def test_a_team_prior_decides_only_when_the_tracking_is_silent():
    n = 400
    ctx = empty_context(n)
    ctx.team[:] = -1.0                                       # tagging says away
    silent = _features(n, usable=np.zeros(n, bool), seen=np.zeros(n, bool),
                       missing_run_s=np.full(n, 1.0))
    assert (decode(log_emissions(silent, ctx, _cfg(n)), _cfg(n)) == "A").all()
    clear = _features(n, d_home=np.full(n, 0.5), d_away=np.full(n, 7.0))
    assert (decode(log_emissions(clear, ctx, _cfg(n)), _cfg(n)) == "H").all()
