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
    # A ball the tracking loses just after it crossed, or was heading over, a line has left
    # the pitch: the camera stops following it. That is a stoppage at once, even though the
    # players are still running from the last action.
    exit_edge_m: float = 2.0            # last seen this close to a line (or beyond it)
    exit_heading_frames: int = 5        # and getting nearer to it over these last frames
    speed_half_window_frames: int = 12  # net displacement over ~1 s: robust to tracking jitter
    stationary_speed_mps: float = 0.8

    # --- how fast the players move: the game is alive when somebody runs, stopped when all walk ---
    use_player_motion: bool = True
    player_pace_percentile: float = 90.0  # the fastest few, not the median: in open play most
                                        # players walk too, but somebody is always running
    player_min_tracked: int = 8         # fewer tracked players than this says nothing
    player_max_mps: float = 12.0        # faster is a track swap, not a person
    player_smooth_s: float = 2.0        # a stoppage is a state of the whole pitch, not an instant
    player_slow_mps: float = 2.0        # prior: the usual walking/jogging boundary (about 7 km/h)
    player_slow_auto: bool = True       # refine it from this match, without labels (Otsu)
    player_slow_min_mps: float = 1.5    # the refined value must stay inside these priors
    player_slow_max_mps: float = 3.0
    player_speed_scale_mps: float = 0.5 # how far from the threshold before the evidence saturates
    dead_w_players: float = 4.0         # logit weight: enough to decide a long stretch, not a moment

    # --- players the tracking loses in a duel or a crowd ---
    # Two players shoulder to shoulder are often reported as one. The one that vanished is
    # still there, so a lost track keeps its last position for a moment when deciding who is
    # nearest to the ball. The duel then reads as contested, and a contested ball stays with
    # the team that had it (CHALLENGE.md), instead of jumping to whoever is still tracked.
    player_hold_s: float = 1.5          # 0 switches it off

    # --- ghost tracks ---
    # A track whose position does not change by a single millimetre is not a person: a real
    # player standing still still sways. The tracking leaves such a frozen copy behind when
    # it loses someone, typically the goalkeeper, while the real player goes on in another
    # slot. Frozen stretches this long are dropped as unknown.
    ghost_frozen_s: float = 1.0         # 0 switches it off
    # Both keepers often wear similar colours, and the tracking then follows one goalkeeper
    # twice, once in each team, a few metres apart. Two tracks of different teams that stay
    # that close in front of a goal, with nobody else around, are one person: the keeper of
    # the team defending that goal. The other track is dropped.
    dup_fix: bool = True
    dup_radius_m: float = 6.0           # the two tracks are this close
    dup_clear_m: float = 15.0           # and nobody else is this close to either
    dup_depth_m: float = 25.0           # within this distance of a goal line
    dup_min_s: float = 2.0              # for at least this long

    # --- goalkeepers put in the wrong team by the tracking ---
    # A keeper wears a different kit from their team, so automatic team assignment can put
    # them in the other team's track slots. A tracklet that is, most of the time it exists,
    # the deepest player in front of one goal is that goal's keeper, whatever its slot says.
    keeper_fix: bool = True
    keeper_depth_m: float = 30.0        # within this distance of the goal line
    keeper_half_width_m: float = 20.16  # and inside the width of the penalty area
    keeper_min_run_s: float = 5.0       # tracklets shorter than this are left alone
    keeper_share: float = 0.6           # share of the tracklet spent as that deepest player

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
