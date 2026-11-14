"""Configuration.

Every number here is a prior from football knowledge or from the organisers' public
documentation. None of them is tuned against the competition ground truth (the FAQ forbids
tuning on the public five minutes, and the private labels are never available).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Config:
    # --- task constants (DATA.md and games.py of the official repository) ---
    fps: int = 25
    n_frames: int = 199319
    n_slots: int = 46
    pitch_len_m: float = 105.3
    pitch_wid_m: float = 68.0
    # Graded halves as 0-based INCLUSIVE video frames (games.half_video_blocks).
    segments: List[List[int]] = field(default_factory=lambda: [[11185, 90651], [110808, 190197]])
    unlabeled: str = "X"  # written outside the graded halves, where nothing is scored

    # --- team mapping ---
    home_name_hint: str = "barcelona"   # matched case- and accent-insensitively
    home_team_id: Optional[str] = None  # override the home team id of the ATD metadata
    home_slots: Optional[str] = None    # manual override, 1-based, e.g. "1-24"
    away_slots: Optional[str] = None    # manual override, 1-based, e.g. "25-46"

    # --- ball handling ---
    max_gap_frames: int = 12            # ~0.5 s; longer gaps are never interpolated
    dist_cap_m: float = 30.0            # so one stray track cannot dominate
    oob_margin_m: float = 1.5           # ball this far outside the lines counts as out
    oob_window_frames: int = 25
    speed_half_window_frames: int = 12  # net displacement over ~1 s: robust to tracking jitter
    stationary_speed_mps: float = 0.8

    # --- ball held in the defending team's own penalty area (typically the goalkeeper) ---
    # A keeper holding the ball is in play (Laws of the Game), but to the tracking it looks
    # like a stoppage: the ball rests, or disappears in the keeper's arms.
    held_ball: bool = True
    held_radius_m: float = 1.5          # a defender this close has the ball in hands or at feet
    held_lookback_s: float = 3.0        # the ball must have arrived in play, tracked, not from out
    held_max_s: float = 8.0             # the Laws give a goalkeeper eight seconds; after that it
                                        # is treated as an ordinary stoppage again
    held_needs_opponent: bool = True    # an opponent must be inside that penalty area when the
                                        # hold starts: at a goal kick or a free kick they may not be

    # --- per-frame evidence ---
    beta_per_m: float = 0.6             # slope of the nearest-player advantage
    influence_radius_m: float = 6.0     # evidence fades when nobody is near the ball
    dead_prior: float = 0.02            # P(dead) while the ball is seen in normal play
    missing_neutral_s: float = 3.0      # ball unseen for up to this long says nothing
    missing_dead_s: float = 8.0         # ball unseen for this long is very likely a stoppage
    missing_dead_max: float = 0.85
    still_neutral_s: float = 2.0        # a ball that rests this long says nothing yet
    still_dead_s: float = 6.0           # a ball motionless this long is very likely a stoppage
    dead_w_out: float = 4.0             # logit weights
    dead_w_stationary: float = 5.0
    dead_w_setpiece: float = 1.5
    emission_temperature: float = 0.3   # frames are strongly autocorrelated

    # --- Smart Tagging as soft context ---
    use_tagging: bool = True
    setpiece_code: str = "SET PIECES"
    kickoff_type: str = "Kick Off (Start)"
    force_dead_before_kickoff: bool = True
    setpiece_lead_s: float = 2.5        # the tag is a window AROUND the restart, not its instant
    setpiece_dead_before_s: float = 8.0
    setpiece_dead_before_by_type: Dict[str, float] = field(default_factory=lambda: {
        "throw-in": 8.0, "corner kick": 20.0, "goal kick": 18.0, "free kick": 20.0,
        "kick off (after goal)": 45.0, "kick off (start)": 0.0})
    setpiece_team_after_s: float = 3.0
    possession_phase_codes: List[str] = field(default_factory=lambda: [
        "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION"])
    phase_team_weight: float = 0.4      # logit weight of a phase's team tag

    # --- temporal model ---
    exp_spell_s_live: float = 15.0      # expected possession length
    exp_spell_s_dead: float = 12.0      # expected stoppage length
    turnover_share: float = 0.5         # possessions ending in a turnover rather than a stoppage
    min_spell_s: float = 2.0            # shorter possessions are absorbed by a neighbour
    min_dead_s: float = 2.0

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            valid = {f.name for f in fields(cls)}
            unknown = sorted(set(data) - valid)
            if unknown:
                raise ValueError(f"Unknown config keys: {unknown}")
            for key, value in data.items():
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
