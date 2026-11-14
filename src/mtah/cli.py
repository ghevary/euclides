"""Command line: `python -m mtah run|validate|audit|explain`."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .audit import audit
from .explain import explain
from .config import Config
from .errors import MtahError
from .paths import resolve_data_root
from .pipeline import run
from .submission import validate_csv

DEFAULT_CACHE = Path.home() / ".cache" / "mtah"


def _official_validate(repo: str, csv: Path) -> int:
    script = Path(repo) / "evaluation" / "validate_submission.py"
    if not script.is_file():
        print(f"official validator not found at {script}")
        return 1
    print("\n--- official validator ---")
    return subprocess.call([sys.executable, str(script), str(csv.resolve())], cwd=str(Path(repo)))


def _cache(args) -> Optional[Path]:
    return None if args.no_cache else Path(args.cache_dir)


def cmd_run(args) -> int:
    cfg = Config.load(args.config)
    root = resolve_data_root(args.data_dir, args.allow_data_in_repo)
    out = Path(args.out) if args.out else Path("submissions") / f"{args.team}.csv"
    report = run(root, out, cfg, _cache(args))
    summary = report["summary"]
    print(f"wrote {out}  ({report['seconds']} s)")
    print(f"  sha256 {report['output_sha256']}")
    print(f"  labels {summary['counts']}")
    print(f"  reports {summary['possessions']} possessions of mean {summary['mean_possession_s']} s")
    if not report["valid"]:
        print(f"INVALID: {report['problem']}")
        return 1
    print("  format check passed")
    if args.official_repo:
        return _official_validate(args.official_repo, out)
    print("  now run the official validator on this exact file (see README)")
    return 0


def cmd_validate(args) -> int:
    cfg = Config.load(args.config)
    problem = validate_csv(Path(args.csv), cfg.n_frames)
    if problem:
        print(f"INVALID: {problem}")
        return 1
    print("format check passed")
    if args.official_repo:
        return _official_validate(args.official_repo, Path(args.csv))
    return 0


def cmd_audit(args) -> int:
    cfg = Config.load(args.config)
    root = resolve_data_root(args.data_dir, args.allow_data_in_repo)
    for line in audit(root, cfg, _cache(args)):
        print(line)
    return 0


def cmd_explain(args) -> int:
    cfg = Config.load(args.config)
    root = resolve_data_root(args.data_dir, args.allow_data_in_repo)
    for line in explain(root, cfg, args.first, args.last, _cache(args)):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtah", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, data: bool):
        p.add_argument("--config", help="JSON file overriding defaults in config.py")
        if data:
            p.add_argument("--data-dir", help="folder containing 'tactical' (or set MTAH_DATA_DIR)")
            p.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                           help="parsed-tracking cache, kept outside the repository")
            p.add_argument("--no-cache", action="store_true")
            p.add_argument("--allow-data-in-repo", action="store_true", help=argparse.SUPPRESS)

    p_run = sub.add_parser("run", help="build the submission file")
    common(p_run, True)
    p_run.add_argument("--team", default="submission", help="team name; the file is <team>.csv")
    p_run.add_argument("--out", help="output path (default submissions/<team>.csv)")
    p_run.add_argument("--official-repo", help="clone of the organisers' repo, to run its validator")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="check a submission file")
    common(p_val, False)
    p_val.add_argument("csv")
    p_val.add_argument("--official-repo")
    p_val.set_defaults(func=cmd_validate)

    p_aud = sub.add_parser("audit", help="aggregate sanity checks against published figures")
    common(p_aud, True)
    p_aud.set_defaults(func=cmd_audit)

    p_exp = sub.add_parser("explain", help="why a range of frames got its labels (run locally)")
    common(p_exp, True)
    p_exp.add_argument("first", type=int, help="first video frame")
    p_exp.add_argument("last", type=int, help="last video frame, inclusive")
    p_exp.set_defaults(func=cmd_explain)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except MtahError as error:
        print(f"error: {error}")
        return 2
