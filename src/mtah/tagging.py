"""Smart Tagging (SportsCode XML) as SOFT context. It is automatic and contains errors.

Time base (DATA.md): <start>/<end> are seconds from the first frame of the tactical video,
so round(seconds * fps) is a video frame. A set-piece tag is a window around the restart,
not its instant, and phase intervals are phases, not possessions.

Assumptions made here, stated in the technical report:
  * the Team tag of an attacking phase names the team in possession;
  * the Team tag of a set piece names the team taking the restart.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .atd import local, norm_text
from .config import Config
from .errors import MtahError


@dataclass
class Instance:
    code: str
    start_s: float
    end_s: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TagContext:
    dead: np.ndarray    # (n,) float in [0,1]: inside a pre-restart window
    team: np.ndarray    # (n,) float in [-1,1]: +1 home, -1 away, 0 no information
    forced: np.ndarray  # (n,) bool: graded frames before the kick-off of their half
    restarts: np.ndarray = None  # frames where a set piece is taken (not the kick-offs of a half)


def empty_context(n_frames: int) -> TagContext:
    return TagContext(np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames, dtype=bool),
                      np.zeros(0, dtype=np.int64))


def load_instances(path: Path) -> List[Instance]:
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        raise MtahError(f"Could not parse {path.name}.") from None
    out: List[Instance] = []
    for el in root.iter():
        if local(el.tag) != "instance":
            continue
        kids = {local(c.tag): c for c in el if local(c.tag) != "label"}
        if not {"code", "start", "end"} <= set(kids):
            continue
        try:
            start, end = float(kids["start"].text), float(kids["end"].text)
        except (TypeError, ValueError):
            continue
        labels: Dict[str, str] = {}
        for lab in el:
            if local(lab.tag) != "label":
                continue
            parts = {local(c.tag): (c.text or "") for c in lab}
            if "group" in parts:
                labels[norm_text(parts["group"])] = parts.get("text", "")
        out.append(Instance((kids["code"].text or "").strip(), start, end, labels))
    return out


def team_sign(inst: Instance, cfg: Config) -> int:
    """+1 home, -1 away, 0 unknown. 'N/A' (every throw-in, among others) is unknown."""
    text = norm_text(inst.labels.get("team"))
    if not text or text in ("n/a", "na", "none"):
        return 0
    return 1 if norm_text(cfg.home_name_hint) in text else -1


def build_context(instances: List[Instance], cfg: Config, n_frames: Optional[int] = None) -> TagContext:
    n = n_frames or cfg.n_frames
    ctx = empty_context(n)
    fps = cfg.fps
    setpiece = norm_text(cfg.setpiece_code)
    kickoff = norm_text(cfg.kickoff_type)
    phases = {norm_text(c) for c in cfg.possession_phase_codes}
    before_by_type = {norm_text(k): v for k, v in cfg.setpiece_dead_before_by_type.items()}
    after = int(round(cfg.setpiece_team_after_s * fps))
    kickoff_starts: List[int] = []
    restarts: List[int] = []

    def clip(a: int, b: int):
        return max(0, a), max(0, min(n, b))

    for inst in instances:
        code = norm_text(inst.code)
        sign = team_sign(inst, cfg)
        if code == setpiece:
            kind = norm_text(inst.labels.get("type"))
            start = int(round(inst.start_s * fps))
            restart = int(round((inst.start_s + cfg.setpiece_lead_s) * fps))
            before = int(round(before_by_type.get(kind, cfg.setpiece_dead_before_s) * fps))
            a, b = clip(restart - before, restart)
            ctx.dead[a:b] = 1.0
            if sign:
                a, b = clip(restart, restart + after)
                ctx.team[a:b] += sign
            if kind == kickoff:
                kickoff_starts.append(start)
            elif 0 <= restart < n:
                restarts.append(restart)
        elif code in phases and sign:
            a, b = clip(int(round(inst.start_s * fps)), int(round(inst.end_s * fps)))
            ctx.team[a:b] += sign
    np.clip(ctx.team, -1.0, 1.0, out=ctx.team)
    ctx.restarts = np.array(sorted(restarts), dtype=np.int64)

    if cfg.force_dead_before_kickoff:
        # A half that is graded from before its kick-off starts with a dead ball.
        for first, last in cfg.segments:
            inside = sorted(s for s in kickoff_starts if first < s <= last)
            if inside:
                ctx.forced[first:inside[0]] = True
    return ctx
