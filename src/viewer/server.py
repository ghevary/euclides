"""HTTP server for the viewer. Standard library only, bound to 127.0.0.1.

Routes
  /                     the page
  /static/<file>        its script and stylesheet
  /video/<source>       the video file, with HTTP range support (seek without reading 5 GB)
  /api/meta             frame count, fps, graded halves, what is available
  /api/labels           the label file as run-length segments
  /api/tags             set pieces from the Smart Tagging
  /api/track?start=&count=   tracking for a block of frames
  /api/frame/<n>        tracking for one frame
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .data import VIDEO_TYPES, Store

STATIC = Path(__file__).resolve().parent / "static"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
# The page may load from this server only: no CDN, no outside request of any kind.
CSP = "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; object-src 'none'; base-uri 'none'"
CHUNK = 256 * 1024
MAX_TRACK_COUNT = 512
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_range(header: str, size: int):
    """(first, last) inclusive for a single byte range, None when absent, False when unsatisfiable."""
    if not header:
        return None
    match = RANGE_RE.match(header.strip())
    if not match:
        return None  # multi-range or malformed: serve the whole file, as the RFC allows
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":  # suffix range: the last N bytes
        length = int(last)
        if length == 0:
            return False
        return max(0, size - length), size - 1
    a = int(first)
    b = int(last) if last else size - 1
    if a >= size or b < a:
        return False
    return a, min(b, size - 1)


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "PossessionViewer"

        def log_message(self, fmt, *args):  # quiet: a video player makes hundreds of requests
            return

        # ---------------------------------------------------------------- helpers
        def _head(self, status: int, ctype: str, length: int, extra=None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _bytes(self, status: int, ctype: str, body: bytes) -> None:
            self._head(status, ctype, len(body))
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self._bytes(status, "application/json; charset=utf-8", body)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        # ---------------------------------------------------------------- routes
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                self._route()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the player dropped the connection after a seek: normal

        def _route(self):
            url = urlsplit(self.path)
            path, query = url.path, parse_qs(url.query)
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path.startswith("/video/"):
                return self._video(path[len("/video/"):])
            if path == "/api/meta":
                return self._json(store.meta())
            if path == "/api/labels":
                return self._json(store.labels_payload())
            if path == "/api/tags":
                return self._json(store.tags_payload())
            if path == "/api/track":
                try:
                    start = int(query.get("start", ["0"])[0])
                    count = int(query.get("count", ["1"])[0])
                except ValueError:
                    return self._error(400, "start and count must be integers")
                return self._json(store.track_payload(start, max(1, min(MAX_TRACK_COUNT, count))))
            if path.startswith("/api/frame/"):
                try:
                    frame = int(path[len("/api/frame/"):])
                except ValueError:
                    return self._error(400, "frame must be an integer")
                if not 0 <= frame < store.cfg.n_frames:
                    return self._error(404, "frame out of range")
                payload = store.track_payload(frame, 1)
                label = None if store.labels is None else str(store.labels[frame])
                return self._json({"frame": frame, "label": label, "state": payload["state"],
                                   "tracking": payload["frames"][0] if payload["frames"] else None})
            return self._error(404, "not found")

        def _static(self, name: str):
            target = (STATIC / name).resolve()
            if STATIC not in target.parents or not target.is_file() \
                    or target.suffix not in STATIC_TYPES:
                return self._error(404, "not found")
            self._bytes(200, STATIC_TYPES[target.suffix], target.read_bytes())

        def _video(self, source: str):
            video = store.videos.get(source)
            if video is None or not video.is_file():
                return self._error(404, "no such video source")
            size = video.stat().st_size
            ctype = VIDEO_TYPES.get(video.suffix.lower(), "application/octet-stream")
            rng = parse_range(self.headers.get("Range", ""), size)
            if rng is False:
                self._head(416, ctype, 0, {"Content-Range": f"bytes */{size}"})
                return
            if rng is None:
                first, last, status, extra = 0, size - 1, 200, {}
            else:
                first, last = rng
                status, extra = 206, {"Content-Range": f"bytes {first}-{last}/{size}"}
            extra["Accept-Ranges"] = "bytes"
            self._head(status, ctype, last - first + 1, extra)
            if self.command == "HEAD":
                return
            remaining = last - first + 1
            with open(video, "rb") as fh:
                fh.seek(first)
                while remaining > 0:
                    block = fh.read(min(CHUNK, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    return Handler


def make_server(store: Store, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    server.daemon_threads = True
    return server
