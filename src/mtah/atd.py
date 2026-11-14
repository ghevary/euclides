"""ATD loader: Metrica FIFA/EPTS metadata XML + raw text -> numpy arrays.

Frame mapping (DATA.md): the ATD frame_id runs 1..N and video_frame = frame_id - 1, so row i
of the arrays returned here is video frame i, which is row i of the submission.
Track slots are tracklets, not people: only their team membership is used.
"""
from __future__ import annotations

import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from .config import Config
from .errors import MtahError


@dataclass
class Tracking:
    players: np.ndarray  # (n_frames, n_slots, 2) float32, normalised 0-1, NaN = not seen
    ball: np.ndarray     # (n_frames, 2) float32, normalised 0-1, NaN = not seen
    home: np.ndarray     # (n_slots,) bool, True where the slot belongs to the home team


def local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def norm_text(text: Optional[str]) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def _parse_ranges(spec: str) -> Set[int]:
    out: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo) - 1, int(hi or lo)))
    return out


def _axis_index(channels: List[str], axis: str, default: int) -> int:
    for i, name in enumerate(channels):
        if name == axis or name.endswith("_" + axis):
            return i
    return default


def slot_layout(xml_path: Path, cfg: Config) -> Tuple[np.ndarray, Tuple[int, int], Optional[int]]:
    """(home mask per slot, (x column, y column), fields per slot or None if unknown)."""
    if cfg.home_slots and cfg.away_slots:
        home, away = _parse_ranges(cfg.home_slots), _parse_ranges(cfg.away_slots)
        if home & away or (home | away) != set(range(cfg.n_slots)):
            raise MtahError("home_slots and away_slots must be disjoint and cover every slot.")
        return np.array([i in home for i in range(cfg.n_slots)]), (0, 1), None

    hint = "If the metadata layout differs, set home_slots / away_slots in the config."
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        raise MtahError(f"Could not parse {xml_path.name}. {hint}") from None
    elems = list(root.iter())

    def named(name: str):
        return [e for e in elems if local(e.tag) == name]

    teams = named("Team")
    team_name = {t.attrib.get("id"): next((c.text for c in t if local(c.tag) == "Name"), "")
                 for t in teams}
    home_id = cfg.home_team_id
    if not home_id:
        home_id = next((e.attrib["idLocalTeam"] for e in named("Score")
                        if "idLocalTeam" in e.attrib), None)
    if not home_id and teams:
        home_id = teams[0].attrib.get("id")  # FIFA/EPTS convention: the first team is home
    if not home_id:
        raise MtahError(f"Could not identify the home team in {xml_path.name}. {hint}")
    wanted = norm_text(cfg.home_name_hint)
    if wanted and not cfg.home_team_id and norm_text(team_name.get(home_id)) \
            and wanted not in norm_text(team_name.get(home_id)):
        raise MtahError(f"The home team in {xml_path.name} does not match home_name_hint. "
                        "Set home_team_id or fix the hint.")

    player_team = {p.attrib["id"]: p.attrib.get("teamId") for p in named("Player") if "id" in p.attrib}
    channel = {c.attrib["id"]: (c.attrib.get("playerId"), (c.attrib.get("channelId") or "").lower())
               for c in named("PlayerChannel") if "id" in c.attrib}
    specs = named("DataFormatSpecification")
    if not specs:
        raise MtahError(f"No DataFormatSpecification in {xml_path.name}. {hint}")

    order: List[str] = []
    chans = {}
    for e in specs[0].iter():
        if local(e.tag) != "PlayerChannelRef":
            continue
        pid, cid = channel.get(e.attrib.get("playerChannelId"), (None, ""))
        if pid is None:
            raise MtahError(f"Unknown player channel in {xml_path.name}. {hint}")
        if pid not in chans:
            chans[pid] = []
            order.append(pid)
        chans[pid].append(cid)
    if len(order) != cfg.n_slots:
        raise MtahError(f"{xml_path.name} describes {len(order)} slots; expected {cfg.n_slots}. {hint}")
    first = chans[order[0]]
    if any(chans[p] != first for p in order):
        raise MtahError(f"Slots have different field layouts in {xml_path.name}. {hint}")
    slot_team = [player_team.get(p) for p in order]
    if None in slot_team or len(set(slot_team)) != 2 or home_id not in slot_team:
        raise MtahError(f"Could not map every slot to one of two teams in {xml_path.name}. {hint}")
    home = np.array([t == home_id for t in slot_team])
    return home, (_axis_index(first, "x", 0), _axis_index(first, "y", 1)), len(first)


def _cache_file(cache_dir: Optional[Path], txt_path: Path, cols: Tuple[int, int]) -> Optional[Path]:
    if cache_dir is None:
        return None
    st = txt_path.stat()
    return Path(cache_dir) / f"{txt_path.stem}_{st.st_size}_{st.st_mtime_ns}_{cols[0]}{cols[1]}.npz"


def load_tracking(xml_path: Path, txt_path: Path, cfg: Config,
                  cache_dir: Optional[Path] = None) -> Tracking:
    home, (ix, iy), k_meta = slot_layout(xml_path, cfg)
    n, s = cfg.n_frames, cfg.n_slots

    cache = _cache_file(cache_dir, txt_path, (ix, iy))
    if cache is not None and cache.is_file():
        with np.load(cache) as z:
            players, ball = z["players"], z["ball"]
        if players.shape == (n, s, 2) and ball.shape == (n, 2):
            return Tracking(players, ball, home)

    players = np.full((n, s, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    k = k_meta
    i = 0
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if i >= n:
                raise MtahError(f"{txt_path.name} has more than {n} rows.")
            parts = line.split(":")
            if len(parts) != 3:
                raise MtahError(f"{txt_path.name} row {i + 1}: expected 'frame:players:ball'.")
            try:
                fid = int(parts[0])
                vals = np.array(parts[1].strip(";").replace(";", ",").split(","), dtype=np.float64)
                bvals = np.array(parts[2].strip(";").split(","), dtype=np.float64)
            except ValueError:
                raise MtahError(f"{txt_path.name} row {i + 1}: non-numeric field.") from None
            if fid != i + 1:
                raise MtahError(f"{txt_path.name} row {i + 1}: frame id is not {i + 1}, so "
                                "video_frame = frame_id - 1 does not hold. See DATA.md.")
            if k is None:
                if vals.size % s or vals.size // s < 2:
                    raise MtahError(f"{txt_path.name} row 1: cannot split the fields into {s} slots.")
                k = vals.size // s
            if vals.size != s * k or bvals.size < 2:
                raise MtahError(f"{txt_path.name} row {i + 1}: unexpected number of fields.")
            block = vals.reshape(s, k)
            players[i, :, 0] = block[:, ix]
            players[i, :, 1] = block[:, iy]
            ball[i] = bvals[:2]
            i += 1
    if i != n:
        raise MtahError(f"{txt_path.name} has {i} rows; expected {n}.")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, players=players, ball=ball)
    return Tracking(players, ball, home)
