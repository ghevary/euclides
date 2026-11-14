"""Local sanity checks. Prints aggregates only, and compares them with the figures the
organisers publish in DATA.md, so a parsing or alignment mistake shows up before a submission."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .atd import load_tracking, norm_text
from .config import Config
from .paths import tactical_assets
from .tagging import build_context, load_instances

# Published in DATA.md for the tactical recording of BAR-ATH.
PUBLISHED = {"ball_seen_all": 113488, "ball_seen_graded": 94214, "graded_frames": 158857,
             "mean_objects_graded": 20.5, "tag_instances": 638, "set_pieces": 59}


def _line(name: str, value, expected=None, tol: float = 0.0) -> str:
    if expected is None:
        return f"  {name:<44} {value}"
    ok = abs(float(value) - float(expected)) <= tol
    return f"  {name:<44} {value}   published {expected}   {'OK' if ok else 'DIFFERS'}"


def audit(data_root: Path, cfg: Config, cache_dir: Optional[Path] = None) -> List[str]:
    assets = tactical_assets(Path(data_root))
    tr = load_tracking(assets.atd_xml, assets.atd_txt, cfg, cache_dir)
    default_match = cfg.n_frames == 199319
    ref = PUBLISHED if default_match else {}

    graded = np.zeros(cfg.n_frames, dtype=bool)
    for first, last in cfg.segments:
        graded[first:last + 1] = True
    seen = np.isfinite(tr.ball).all(axis=1)
    objects = np.isfinite(tr.players).all(axis=2).sum(axis=1)

    out = ["tracking"]
    out.append(_line("rows (frame ids contiguous from 1)", len(tr.ball)))
    out.append(_line("slots home / away", f"{int(tr.home.sum())} / {int((~tr.home).sum())}"))
    out.append(_line("graded frames", int(graded.sum()), ref.get("graded_frames")))
    out.append(_line("ball seen, all frames", int(seen.sum()), ref.get("ball_seen_all")))
    out.append(_line("ball seen, graded frames", int((seen & graded).sum()), ref.get("ball_seen_graded")))
    mean_objects = round(float(objects[graded].mean()), 1) if graded.any() else 0.0
    out.append(_line("mean tracked players, graded frames", mean_objects,
                     ref.get("mean_objects_graded"), tol=0.05))

    if not assets.smart_tagging.is_file():
        out.append("smart tagging: file not found, the pipeline will run without it")
        return out

    instances = load_instances(assets.smart_tagging)
    setpiece = norm_text(cfg.setpiece_code)
    pieces = [i for i in instances if norm_text(i.code) == setpiece]
    ctx = build_context(instances, cfg, cfg.n_frames)
    out.append("smart tagging")
    out.append(_line("instances", len(instances), ref.get("tag_instances")))
    out.append(_line("set pieces", len(pieces), ref.get("set_pieces")))
    out.append(_line("kick-off (start) tags", sum(
        norm_text(i.labels.get("type")) == norm_text(cfg.kickoff_type) for i in pieces)))
    out.append(_line("graded frames forced dead before a kick-off", int(ctx.forced.sum())))
    out.append(_line("graded frames with a team prior from a phase",
                     int(((ctx.team != 0) & graded).sum())))

    # Where does the restart sit inside a set-piece tag? The ball comes back into view at the
    # restart, so its visibility by offset from the tag start shows it. Use this to set
    # setpiece_lead_s; it needs no labels.
    out.append("ball visibility around the start of a set-piece tag (share of tags)")
    cells = []
    for offset in range(-10, 11, 2):
        frames = [int(round((i.start_s + offset) * cfg.fps)) for i in pieces]
        frames = [f for f in frames if 0 <= f < cfg.n_frames]
        share = float(np.mean(seen[frames])) if frames else float("nan")
        cells.append(f"{offset:+d}s {share:.2f}")
    out.append("  " + "   ".join(cells))
    return out
