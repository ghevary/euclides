"""Data location. The data bundle must live OUTSIDE this repository."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import MtahError

ENV_VAR = "MTAH_DATA_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TacticalAssets:
    atd_xml: Path
    atd_txt: Path
    smart_tagging: Path


def resolve_data_root(cli_value: Optional[str] = None, allow_inside_repo: bool = False) -> Path:
    value = cli_value or os.environ.get(ENV_VAR)
    if not value:
        raise MtahError(f"Set {ENV_VAR} or pass --data-dir (the folder that contains 'tactical').")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise MtahError(f"Data directory does not exist: {root}")
    if not allow_inside_repo:
        try:
            root.relative_to(REPO_ROOT)
        except ValueError:
            return root
        raise MtahError("Keep the data bundle outside the repository so it can never be committed.")
    return root


def tactical_assets(root: Path, stem: str = "BAR-ATH_tactical") -> TacticalAssets:
    folder = root / "tactical"
    assets = TacticalAssets(
        atd_xml=folder / f"{stem}_atd.xml",
        atd_txt=folder / f"{stem}_atd.txt",
        smart_tagging=folder / f"{stem}_smart_tagging.xml",
    )
    missing = [p.name for p in (assets.atd_xml, assets.atd_txt) if not p.is_file()]
    if missing:
        raise MtahError(f"Missing tactical files in {folder}: {missing}")
    return assets
