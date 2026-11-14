"""Why a range of frames got its labels: the evidence the model saw there, as shares.

For debugging by a person, on their own machine. It reads the match data, so its output is
an extract of that data: do not paste it into a hosted AI tool (AGENTS.md, rule 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .atd import load_tracking
from .config import Config
from .errors import MtahError
from .features import build_features
from .model import STATES, log_emissions
from .paths import tactical_assets
from .tagging import build_context, empty_context, load_instances
from .temporal import decode


def _share(mask) -> str:
    return f"{100.0 * float(np.mean(mask)):5.1f}%"


def explain(data_root: Path, cfg: Config, first: int, last: int,
            cache_dir: Optional[Path] = None) -> List[str]:
    if not 0 <= first <= last < cfg.n_frames:
        raise MtahError(f"Frames must satisfy 0 <= first <= last < {cfg.n_frames}.")
    assets = tactical_assets(Path(data_root))
    tr = load_tracking(assets.atd_xml, assets.atd_txt, cfg, cache_dir)
    feat = build_features(tr, cfg)
    if cfg.use_tagging and assets.smart_tagging.is_file():
        ctx = build_context(load_instances(assets.smart_tagging), cfg, cfg.n_frames)
    else:
        ctx = empty_context(cfg.n_frames)
    log_em = log_emissions(feat, ctx, cfg)
    labels = decode(log_em, cfg)
    s = slice(first, last + 1)
    n = last - first + 1
    usable = feat.usable[s]

    out = [f"frames {first}..{last}  ({n} frames, {n / cfg.fps:.1f} s)", "", "labels written"]
    for lab in ("H", "A", "D", "X"):
        out.append(f"  {lab}  {_share(labels[s] == lab)}")
    out += ["", "per-frame evidence favours (before the temporal model)"]
    best = log_em[s].argmax(axis=1)
    for k, name in enumerate(STATES):
        out.append(f"  {name}  {_share(best == k)}")

    out += ["", "ball"]
    out.append(f"  seen by the tracking                 {_share(feat.seen[s])}")
    out.append(f"  usable (seen or short gap filled)    {_share(usable)}")
    out.append(f"  longest unseen run touching range    {float(feat.missing_run_s[s].max()):.1f} s")
    out.append(f"  longest resting run touching range   {float(feat.still_run_s[s].max()):.1f} s")
    out.append(f"  outside the pitch                    {_share(feat.oob[s] > 0)}")
    if usable.any():
        out.append(f"  median distance to nearest home      {float(np.median(feat.d_home[s][usable])):.1f} m")
        out.append(f"  median distance to nearest away      {float(np.median(feat.d_away[s][usable])):.1f} m")

    out += ["", "why a stoppage"]
    out.append(f"  inside a tagged pre-restart window   {_share(ctx.dead[s] > 0)}")
    out.append(f"  forced dead before a kick-off        {_share(ctx.forced[s])}")
    out.append(f"  resting long enough to count         {_share(feat.still_run_s[s] > cfg.still_neutral_s)}")
    out.append(f"  unseen long enough to count          {_share(feat.missing_run_s[s] > cfg.missing_neutral_s)}")

    out += ["", "held-ball rule (ball with a defender in their own penalty area)"]
    out.append(f"  ball in own box with a defender on it {_share(feat.near_own[s])}")
    out.append(f"  an attacker inside that box          {_share(feat.opp_in_box[s])}")
    out.append(f"  rule applied                         {_share(feat.held[s])}")
    look = int(round(cfg.held_lookback_s * cfg.fps))
    before = slice(max(0, first - look), first)
    if first > 0:
        out.append(f"  ball usable in the {cfg.held_lookback_s:.0f} s before          {_share(feat.usable[before])}")
        out.append(f"  ball outside pitch in the {cfg.held_lookback_s:.0f} s before   {_share(feat.oob[before] > 0)}")
    return out
