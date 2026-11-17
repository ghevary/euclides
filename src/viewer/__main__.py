"""Possession Viewer.

    python -m viewer --data-dir E:\\barca --labels submissions\\Euclides.csv

Opens http://127.0.0.1:8000 with the tactical video on the left and the frame number and
label on the right. Read-only: it never edits or scores labels.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from mtah.config import Config
from mtah.errors import MtahError
from mtah.paths import resolve_data_root

from .data import VIDEO_TYPES, Store
from .server import make_server

DEFAULT_CACHE = Path.home() / ".cache" / "mtah"


def _existing(path: Path) -> Optional[Path]:
    return path if path.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="viewer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="folder containing 'tactical' (or set MTAH_DATA_DIR)")
    p.add_argument("--labels", help="submission CSV to show (frame,label); optional, "
                                    "a CSV can also be opened from the page")
    p.add_argument("--video", help="video file to play instead of <data-dir>/tactical/<stem>.mp4, "
                                   "for example a 720p proxy with the same frame count")
    p.add_argument("--stem", default="BAR-ATH", help="file name stem of the match")
    p.add_argument("--config", help="JSON file overriding defaults in mtah/config.py")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-tracking", action="store_true", help="do not load the tracking (faster start)")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                   help="parsed-tracking cache, kept outside the repository")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--allow-data-in-repo", action="store_true", help=argparse.SUPPRESS)
    return p


def build_store(args) -> Store:
    cfg = Config.load(args.config)
    root = resolve_data_root(args.data_dir, args.allow_data_in_repo)
    tactical = root / "tactical"
    stem = f"{args.stem}_tactical"

    video = Path(args.video) if args.video else tactical / f"{stem}.mp4"
    if not video.is_file():
        raise MtahError(f"Video not found: {video}")
    if video.suffix.lower() not in VIDEO_TYPES:
        raise MtahError(f"Unsupported video type {video.suffix}; use one of {sorted(VIDEO_TYPES)}.")
    labels = Path(args.labels) if args.labels else None
    if labels is not None and not labels.is_file():
        raise MtahError(f"Label file not found: {labels}")

    atd_xml = atd_txt = homography = None
    if not args.no_tracking:
        atd_xml = _existing(tactical / f"{stem}_atd.xml")
        atd_txt = _existing(tactical / f"{stem}_atd.txt")
        homography = _existing(tactical / f"{stem}_homography.csv")
        if not (atd_xml and atd_txt):
            atd_xml = atd_txt = None
    return Store(
        cfg, video, labels, atd_xml, atd_txt,
        tagging=_existing(tactical / f"{stem}_smart_tagging.xml"),
        homography=homography,
        broadcast=_existing(root / "broadcast" / f"{args.stem}_broadcast.mp4"),
        cache_dir=None if args.no_cache else Path(args.cache_dir),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = build_store(args)
    except (MtahError, ValueError) as error:
        print(f"error: {error}")
        return 2
    try:
        server = make_server(store, args.port)
    except OSError as error:
        print(f"error: cannot listen on 127.0.0.1:{args.port} ({error}). Try --port.")
        return 2
    store.start_background_load()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Possession Viewer  {url}")
    print(f"  video   {store.videos['tactical']}")
    print(f"  labels  {store.labels_path or '(none: open a CSV from the page)'}")
    print(f"  tracking {store.track_state}" + (" (first load parses the ATD; later starts use the cache)"
                                              if store.track_state == 'loading' else ""))
    print("  Ctrl+C to stop")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
