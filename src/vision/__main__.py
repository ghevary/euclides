"""Vision stage. `python -m vision pilot --data-dir <bundle>`"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mtah.config import Config
from mtah.errors import MtahError
from mtah.paths import resolve_data_root

DEFAULT_CACHE = Path.home() / ".cache" / "mtah"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot", help="measure whether the detector sees the ball, against the tracking")
    p.add_argument("--data-dir")
    p.add_argument("--config")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        cfg = Config.load(args.config)
        root = resolve_data_root(args.data_dir)
        from .pilot import pilot
        for line in pilot(root, cfg, Path(args.cache_dir), args.samples, args.conf):
            print(line)
    except MtahError as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
