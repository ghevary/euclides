"""Synthetic match generator for tests. Nothing here comes from the competition data."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from mtah.config import Config

HOME_ID, AWAY_ID = "team_home", "team_away"
N_HOME_SLOTS = 24  # slots 1-24 home, 25-46 away, as in a FIFA/EPTS export


def small_config(**overrides) -> Config:
    cfg = Config()
    cfg.n_frames = 3000
    cfg.segments = [[100, 1399], [1600, 2899]]
    cfg.home_name_hint = "homeville"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def write_meta(path: Path, n_slots: int = 46, swap_team_order: bool = False) -> None:
    teams = [(HOME_ID, "Homeville FC"), (AWAY_ID, "Awayton")]
    if swap_team_order:
        teams.reverse()
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<main><Metadata>",
             f'<Sessions><Session id="s1"><MatchParameters><Score idLocalTeam="{HOME_ID}" '
             f'idVisitingTeam="{AWAY_ID}"/></MatchParameters></Session></Sessions>', "<Teams>"]
    for tid, name in teams:
        parts.append(f'<Team id="{tid}"><Name>{name}</Name></Team>')
    parts.append("</Teams><Players>")
    for i in range(1, n_slots + 1):
        team = HOME_ID if i <= N_HOME_SLOTS else AWAY_ID
        parts.append(f'<Player id="{i}" teamId="{team}"><Name>Track {i}</Name></Player>')
    parts.append("</Players><PlayerChannels>")
    for i in range(1, n_slots + 1):
        for ch, suffix in (("x", "x"), ("y", "y"), ("s", "speed")):
            parts.append(f'<PlayerChannel channelId="{ch}" id="player{i}_{suffix}" playerId="{i}"/>')
    parts.append('</PlayerChannels></Metadata><DataFormatSpecifications>'
                 '<DataFormatSpecification separator=":" startFrame="1">'
                 '<StringRegister name="frameCount"/><SplitRegister separator=";">')
    for i in range(1, n_slots + 1):
        parts.append('<SplitRegister separator=",">'
                     + "".join(f'<PlayerChannelRef playerChannelId="player{i}_{s}"/>'
                               for s in ("x", "y", "speed")) + "</SplitRegister>")
    parts.append('</SplitRegister><SplitRegister separator=",">'
                 '<BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
                 '<BallChannelRef channelId="s"/></SplitRegister>'
                 "</DataFormatSpecification></DataFormatSpecifications></main>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _fmt(v: float) -> str:
    return "NaN" if not np.isfinite(v) else f"{v:.5f}"


def write_raw(path: Path, players: np.ndarray, ball: np.ndarray, first_id: int = 1) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i in range(len(ball)):
            groups = ";".join(f"{_fmt(x)},{_fmt(y)},{'NaN' if not np.isfinite(x) else '1.0'}"
                              for x, y in players[i])
            fh.write(f"{first_id + i}:{groups}:{_fmt(ball[i, 0])},{_fmt(ball[i, 1])},NaN\n")


def write_tagging(path: Path, instances: List[Tuple[str, float, float, dict]]) -> None:
    parts = ['<?xml version="1.0" ?>', "<file><ALL_INSTANCES>"]
    for n, (code, start, end, labels) in enumerate(instances, 1):
        parts.append(f"<instance><ID>{n}</ID><start>{start}</start><end>{end}</end><code>{code}</code>")
        for group, text in labels.items():
            parts.append(f"<label><group>{group}</group><text>{text}</text></label>")
        parts.append("</instance>")
    parts.append("</ALL_INSTANCES></file>")
    path.write_text("\n".join(parts), encoding="utf-8")


def formation(cfg: Config, rng: np.random.Generator):
    """11 home and 11 away players jittering around fixed spots, 5-6 m apart from each other."""
    n, s = cfg.n_frames, cfg.n_slots
    players = np.full((n, s, 2), np.nan)
    xs = np.array([20.0, 40.0, 60.0, 80.0])
    ys = np.array([14.0, 34.0, 54.0])
    spots = np.array([(x, y) for x in xs for y in ys])[:11]
    home_slots = list(range(0, 11))
    away_slots = list(range(N_HOME_SLOTS, N_HOME_SLOTS + 11))
    scale = np.array([cfg.pitch_len_m, cfg.pitch_wid_m])
    for k in range(11):
        players[:, home_slots[k], :] = (spots[k] + rng.normal(0, 0.15, (n, 2))) / scale
        players[:, away_slots[k], :] = (spots[k] + (6.0, 5.0) + rng.normal(0, 0.15, (n, 2))) / scale
    return players, spots, scale


def scenario(cfg: Config, seed: int = 7):
    """A small match with known answers.

    Per half (times relative to the first frame of the half, in seconds):
      0-14 home, 14-26 away, 26-36 dead (ball out of sight), 36-52 home.
    Inside the first home spell the ball is lost for 1.6 s; inside the away spell a home
    player's position is briefly reported on top of the ball.
    """
    rng = np.random.default_rng(seed)
    players, spots, scale = formation(cfg, rng)
    n = cfg.n_frames
    ball = np.full((n, 2), np.nan)
    truth = np.full(n, "X", dtype="<U1")
    fps = cfg.fps
    plan = [(0, 14, "H"), (14, 26, "A"), (26, 36, "D"), (36, 52, "H")]
    tags = []
    for first, last in cfg.segments:
        for a_s, b_s, lab in plan:
            a, b = first + int(a_s * fps), min(last + 1, first + int(b_s * fps))
            truth[a:b] = lab
            if lab == "D":
                restart_s = b / fps
                tags.append(("SET PIECES", restart_s - cfg.setpiece_lead_s,
                             restart_s - cfg.setpiece_lead_s + 5.0,
                             {"Team": "N/A", "Type": "Throw-in", "Half": "1st Half"}))
                continue
            offset = np.array([0.0, 0.0]) if lab == "H" else np.array([6.0, 5.0])
            carriers = rng.integers(0, 11, size=1 + (b - a) // (3 * fps) + 1)
            for f in range(a, b):
                step, phase = divmod(f - a, 3 * fps)
                src = spots[carriers[step]] + offset
                dst = spots[carriers[step + 1]] + offset
                w = max(0.0, (phase - 2 * fps) / fps)  # hold 2 s, then pass for 1 s
                ball[f] = (src * (1 - w) + dst * w + (0.4, 0.3)) / scale
        truth[min(last + 1, first + 52 * fps):last + 1] = "H"
        tail_a = first + 52 * fps
        for f in range(tail_a, last + 1):
            ball[f] = (spots[0] + (0.4, 0.3)) / scale
        # an occlusion inside the first home spell: 40 frames without the ball
        ball[first + 5 * fps:first + 5 * fps + 40] = np.nan
        # a tracking glitch inside the away spell: a home track sits on the ball for 8 frames
        g = first + 18 * fps
        players[g:g + 8, 12, :] = ball[g:g + 8]
        tags.append(("PROGRESSION", (first + 1 * fps) / fps, (first + 12 * fps) / fps,
                     {"Team": "Homeville FC", "Half": "1st Half"}))
    return players, ball, truth, tags


def write_bundle(root: Path, cfg: Config, seed: int = 7):
    """Write a synthetic data bundle under root/tactical and return the true labels."""
    folder = Path(root) / "tactical"
    folder.mkdir(parents=True, exist_ok=True)
    players, ball, truth, tags = scenario(cfg, seed)
    write_meta(folder / "BAR-ATH_tactical_atd.xml", cfg.n_slots)
    write_raw(folder / "BAR-ATH_tactical_atd.txt", players, ball)
    write_tagging(folder / "BAR-ATH_tactical_smart_tagging.xml", tags)
    return truth
