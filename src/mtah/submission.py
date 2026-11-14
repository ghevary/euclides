"""Write and check the submission file: `frame,label`, one row per tactical video frame."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import Config

ALLOWED = ("H", "A", "D", "X")


def write_csv(path: Path, labels: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" on every platform, so the same labels always give the same bytes.
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return path


def validate_csv(path: Path, n_frames: int) -> Optional[str]:
    """None when the file meets the SUBMISSION.md contract, otherwise the first problem found.

    A convenience copy of the contract. The official validator remains the authority.
    """
    path = Path(path)
    if not path.is_file():
        return "file not found"
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if [c.strip().lower() for c in header.strip().split(",")] != ["frame", "label"]:
            return "header must be exactly 'frame,label'"
        count = 0
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                return f"blank line at data row {count}"
            cells = line.split(",")
            if len(cells) != 2:
                return f"data row {count}: expected 2 columns"
            if cells[0].strip() != str(count):
                return f"data row {count}: frame must be {count}"
            if cells[1].strip().upper() not in ALLOWED:
                return f"data row {count}: label must be one of H, A, D, X"
            count += 1
    if count != n_frames:
        return f"{count} data rows; expected {n_frames}"
    return None


def summarize(labels: np.ndarray, cfg: Config) -> Dict[str, object]:
    """What the file claims: label counts and the possessions it reports (maximal H or A runs)."""
    labels = np.asarray(labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ALLOWED}
    spells = 0
    if len(labels):
        change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        starts = np.concatenate(([0], change))
        spells = int(np.sum(np.isin(labels[starts], ("H", "A"))))
    alive = counts["H"] + counts["A"]
    mean_s = alive / float(cfg.fps) / spells if spells else 0.0
    return {"counts": counts, "possessions": spells, "mean_possession_s": round(mean_s, 2)}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
