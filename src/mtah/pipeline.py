"""Data in, labels out. One deterministic pass; no randomness anywhere."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .atd import load_tracking
from .config import Config
from .features import build_features
from .model import log_emissions
from .paths import tactical_assets
from .submission import sha256_of, summarize, validate_csv, write_csv
from .tagging import build_context, empty_context, load_instances
from .temporal import decode


def predict(data_root: Path, cfg: Config, cache_dir: Optional[Path] = None) -> np.ndarray:
    assets = tactical_assets(Path(data_root))
    tracking = load_tracking(assets.atd_xml, assets.atd_txt, cfg, cache_dir)
    features = build_features(tracking, cfg)
    if cfg.use_tagging and assets.smart_tagging.is_file():
        context = build_context(load_instances(assets.smart_tagging), cfg, cfg.n_frames)
    else:
        context = empty_context(cfg.n_frames)
    return decode(log_emissions(features, context, cfg), cfg)


def run(data_root: Path, out_path: Path, cfg: Config, cache_dir: Optional[Path] = None) -> dict:
    started = time.time()
    labels = predict(data_root, cfg, cache_dir)
    out_path = write_csv(Path(out_path), labels)
    problem = validate_csv(out_path, cfg.n_frames)
    assets = tactical_assets(Path(data_root))
    report = {
        "output": out_path.name,
        "output_sha256": sha256_of(out_path),
        "valid": problem is None,
        "problem": problem,
        "summary": summarize(labels, cfg),
        "inputs_sha256": {p.name: sha256_of(p) for p in
                          (assets.atd_xml, assets.atd_txt, assets.smart_tagging) if p.is_file()},
        "config": cfg.to_dict(),
        "versions": {"mtah": __version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "seconds": round(time.time() - started, 1),
    }
    meta = out_path.with_suffix(".meta.json")
    meta.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
