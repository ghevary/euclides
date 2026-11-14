import numpy as np

import synth
from mtah.model import A, D, H
from mtah.temporal import decode, log_transitions, merge_short_runs, viterbi


def test_transition_rows_are_probabilities():
    rows = np.exp(log_transitions(synth.small_config())).sum(axis=1)
    np.testing.assert_allclose(rows, 1.0, atol=1e-4)


def _evidence(labels, strength=1.0):
    log_em = np.zeros((len(labels), 3))
    for i, lab in enumerate(labels):
        if lab is not None:
            log_em[i] = -strength
            log_em[i, lab] = 0.0
    return log_em


def test_no_evidence_keeps_the_previous_state():
    cfg = synth.small_config()
    labels = [H] * 100 + [None] * 200 + [H] * 100
    assert (viterbi(_evidence(labels), log_transitions(cfg)) == H).all()


def test_a_brief_contradiction_does_not_split_a_possession():
    cfg = synth.small_config()
    labels = [H] * 200 + [A] * 6 + [H] * 200
    assert (viterbi(_evidence(labels), log_transitions(cfg)) == H).all()


def test_sustained_evidence_is_a_turnover_at_the_right_frame():
    cfg = synth.small_config()
    labels = [H] * 200 + [A] * 200
    path = viterbi(_evidence(labels), log_transitions(cfg))
    assert (path[:200] == H).all() and (path[200:] == A).all()


def test_merge_short_runs():
    s = np.array([H] * 100 + [A] * 10 + [H] * 100)
    assert (merge_short_runs(s, 50, 50) == H).all()
    s = np.array([H] * 100 + [D] * 10 + [A] * 200)          # short stoppage: longer neighbour
    out = merge_short_runs(s, 50, 50)
    assert (out[:100] == H).all() and (out[100:] == A).all()
    s = np.array([H] * 100 + [D] * 60 + [H] * 100)          # a real stoppage is kept
    assert np.array_equal(merge_short_runs(s, 50, 50), s)
    s = np.array([A] * 10)                                  # a lone short run has no neighbour
    assert np.array_equal(merge_short_runs(s, 50, 50), s)


def test_decode_applies_the_minimum_possession_length():
    cfg = synth.small_config(n_frames=600, segments=[[0, 599]])
    labels = [H] * 300 + [A] * 30 + [H] * 270              # 1.2 s of overwhelming evidence
    log_em = _evidence(labels, strength=5.0)
    assert (viterbi(log_em, log_transitions(cfg)) == A).sum() == 30   # Viterbi alone follows it
    assert (decode(log_em, cfg) == "H").all()                         # shorter than min_spell_s


def test_halves_are_decoded_independently_and_the_rest_is_x():
    cfg = synth.small_config(n_frames=1000, segments=[[100, 399], [600, 899]])
    labels = [None] * 1000
    for i in range(100, 400):
        labels[i] = H
    for i in range(600, 900):
        labels[i] = A
    for i in range(400, 600):                               # strong evidence in halftime
        labels[i] = D
    out = decode(_evidence(labels, strength=5.0), cfg)
    assert (out[100:400] == "H").all() and (out[600:900] == "A").all()
    assert (out[:100] == "X").all() and (out[400:600] == "X").all() and (out[900:] == "X").all()
