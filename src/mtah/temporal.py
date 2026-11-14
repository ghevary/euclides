"""Temporal model: a 3-state HMM decoded per half, then a minimum-duration clean-up.

The ranking metric matches whole possessions, so changing state has to cost something. The
transition matrix encodes how long possessions and stoppages last; Viterbi finds the label
sequence that best explains the evidence under that cost. Halves are decoded independently.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import Config
from .model import A, D, H, STATES


def log_transitions(cfg: Config) -> np.ndarray:
    leave_live = 1.0 / max(cfg.exp_spell_s_live * cfg.fps, 2.0)
    leave_dead = 1.0 / max(cfg.exp_spell_s_dead * cfg.fps, 2.0)
    turn = leave_live * cfg.turnover_share
    stop = leave_live * (1.0 - cfg.turnover_share)
    t = np.array([
        [1.0 - leave_live, turn, stop],
        [turn, 1.0 - leave_live, stop],
        [leave_dead / 2.0, leave_dead / 2.0, 1.0 - leave_dead],
    ])
    return np.round(np.log(np.clip(t, 1e-300, 1.0)), 6)


def viterbi(log_em: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    n, k = log_em.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, k), dtype=np.int8)
    cols = np.arange(k)
    delta = log_em[0].copy()
    for t in range(1, n):
        cand = delta[:, None] + log_trans
        best = cand.argmax(axis=0)
        back[t] = best
        delta = cand[best, cols] + log_em[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(states: np.ndarray):
    change = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(states)])))
    return [int(v) for v in states[starts]], [int(v) for v in lengths]


def merge_short_runs(states: np.ndarray, min_live: int, min_dead: int) -> np.ndarray:
    """Absorb runs shorter than their minimum into a neighbour, shortest first.

    A short run between two runs of the same label disappears into them. Otherwise it takes
    the label of the longer neighbour. Deterministic: ties go to the earlier run.
    """
    states = np.asarray(states)
    if len(states) == 0:
        return states.copy()
    values, lengths = _runs(states)
    while len(values) > 1:
        worst, worst_len = -1, None
        for i, (v, length) in enumerate(zip(values, lengths)):
            if length < (min_dead if v == D else min_live) and (worst_len is None or length < worst_len):
                worst, worst_len = i, length
        if worst < 0:
            break
        left = worst - 1 if worst > 0 else None
        right = worst + 1 if worst + 1 < len(values) else None
        if left is None:
            target = right
        elif right is None:
            target = left
        else:
            target = left if lengths[left] >= lengths[right] else right
        values[worst] = values[target]
        merged_v: List[int] = []
        merged_l: List[int] = []
        for v, length in zip(values, lengths):
            if merged_v and merged_v[-1] == v:
                merged_l[-1] += length
            else:
                merged_v.append(v)
                merged_l.append(length)
        values, lengths = merged_v, merged_l
    return np.repeat(np.array(values, dtype=np.int8), lengths)


def decode(log_em: np.ndarray, cfg: Config) -> np.ndarray:
    """Labels for every frame: H/A/D inside the graded halves, cfg.unlabeled elsewhere."""
    n = len(log_em)
    labels = np.full(n, cfg.unlabeled, dtype="<U1")
    names = np.array(STATES)
    trans = log_transitions(cfg)
    min_live = int(round(cfg.min_spell_s * cfg.fps))
    min_dead = int(round(cfg.min_dead_s * cfg.fps))
    for first, last in cfg.segments:
        first, last = max(0, int(first)), min(n - 1, int(last))
        if last < first:
            continue
        states = viterbi(log_em[first:last + 1], trans)
        states = merge_short_runs(states, min_live, min_dead)
        labels[first:last + 1] = names[states]
    return labels


__all__ = ["H", "A", "D", "decode", "log_transitions", "merge_short_runs", "viterbi"]
