"""Viewer tests on the synthetic bundle. No competition data, and a fake 'video' of plain bytes."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from mtah.pipeline import run
from viewer.data import Store, load_homography, read_labels, to_segments
from viewer.server import make_server, parse_range

VIDEO_BYTES = bytes(range(256)) * 40  # 10240 bytes


def test_parse_range():
    assert parse_range("", 100) is None
    assert parse_range("bytes=0-", 100) == (0, 99)
    assert parse_range("bytes=10-19", 100) == (10, 19)
    assert parse_range("bytes=90-500", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=100-", 100) is False
    assert parse_range("bytes=0-1,5-6", 100) is None


def test_segments_round_trip():
    labels = np.array(list("XXHHHADDX"))
    segs = to_segments(labels)
    assert segs == [[0, 1, "X"], [2, 4, "H"], [5, 5, "A"], [6, 7, "D"], [8, 8, "X"]]
    rebuilt = np.empty(len(labels), dtype="<U1")
    for a, b, lab in segs:
        rebuilt[a:b + 1] = lab
    assert (rebuilt == labels).all()


@pytest.fixture(scope="module")
def served(bundle, tmp_path_factory):
    root, cfg, _ = bundle
    work = tmp_path_factory.mktemp("viewer")
    csv = work / "labels.csv"
    run(root, csv, cfg, None)
    video = work / "fake.mp4"
    video.write_bytes(VIDEO_BYTES)
    hom = work / "hom.csv"
    with open(hom, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for i in range(cfg.n_frames):
            if i == 5:
                fh.write("5,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0\n")
            else:
                fh.write(f"{i},1920,0,0,0,1080,0,0,0,1,0.2,1\n")
    tact = root / "tactical"
    store = Store(cfg, video, csv, tact / "BAR-ATH_tactical_atd.xml", tact / "BAR-ATH_tactical_atd.txt",
                  tagging=tact / "BAR-ATH_tactical_smart_tagging.xml", homography=hom)
    server = make_server(store, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    store.start_background_load()
    for _ in range(600):
        if store.track_state != "loading":
            break
        time.sleep(0.05)
    assert store.track_state == "ready", store.track_error
    yield f"http://127.0.0.1:{server.server_address[1]}", store, csv, cfg
    server.shutdown()
    server.server_close()


def get(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, dict(response.headers), response.read()


def test_page_and_static(served):
    base = served[0]
    status, headers, body = get(base + "/")
    assert status == 200 and b"Possession Viewer" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert b"http://" not in body and b"https://" not in body  # no outside resource
    for name in ("app.js", "style.css"):
        assert get(base + "/static/" + name)[0] == 200
    with pytest.raises(urllib.error.HTTPError):
        get(base + "/static/../data.py")


def test_labels_match_csv_rows(served):
    base, store, csv, cfg = served
    payload = json.loads(get(base + "/api/labels")[2])
    labels = read_labels(csv, cfg.n_frames)
    rebuilt = np.empty(cfg.n_frames, dtype="<U1")
    covered = 0
    for a, b, lab in payload["segments"]:
        rebuilt[a:b + 1] = lab
        covered += b - a + 1
    assert covered == cfg.n_frames
    assert (rebuilt == labels).all()
    rows = csv.read_text(encoding="utf-8").split("\n")[1:]
    for frame in (0, 100, 777, 1399, 1400, cfg.n_frames - 1):
        assert rows[frame] == f"{frame},{rebuilt[frame]}"
        assert json.loads(get(f"{base}/api/frame/{frame}")[2])["label"] == rebuilt[frame]


def test_video_range(served):
    base = served[0]
    status, headers, body = get(base + "/video/tactical")
    assert status == 200 and body == VIDEO_BYTES and headers["Accept-Ranges"] == "bytes"
    status, headers, body = get(base + "/video/tactical", {"Range": "bytes=1000-1999"})
    assert status == 206 and body == VIDEO_BYTES[1000:2000]
    assert headers["Content-Range"] == f"bytes 1000-1999/{len(VIDEO_BYTES)}"
    status, headers, body = get(base + "/video/tactical", {"Range": "bytes=10000-"})
    assert status == 206 and body == VIDEO_BYTES[10000:]
    with pytest.raises(urllib.error.HTTPError) as info:
        get(base + "/video/tactical", {"Range": "bytes=999999-"})
    assert info.value.code == 416
    with pytest.raises(urllib.error.HTTPError):
        get(base + "/video/broadcast")


def test_track_block(served):
    base, store, _, cfg = served
    payload = json.loads(get(base + "/api/track?start=200&count=16")[2])
    assert payload["state"] == "ready" and payload["start"] == 200 and len(payload["frames"]) == 16
    tr = store.tracking
    for k, item in enumerate(payload["frames"]):
        i = 200 + k
        seen = np.isfinite(tr.players[i]).all(axis=1)
        assert len(item["h"]) == 2 * int((seen & tr.home).sum())
        assert len(item["a"]) == 2 * int((seen & ~tr.home).sum())
        assert (item["b"] is not None) == bool(np.isfinite(tr.ball[i]).all())
        if item["b"] is None:
            assert item["dh"] is None and item["da"] is None
        assert item["H"] == [1920, 0, 0, 0, 1080, 0, 0, 0, 1]
    assert json.loads(get(base + "/api/track?start=5&count=1")[2])["frames"][0]["H"] is None
    tail = json.loads(get(f"{base}/api/track?start={cfg.n_frames - 3}&count=50")[2])
    assert len(tail["frames"]) == 3


def test_tags_and_meta(served):
    base, _, _, cfg = served
    meta = json.loads(get(base + "/api/meta")[2])
    assert meta["n_frames"] == cfg.n_frames and meta["fps"] == cfg.fps
    assert meta["sources"] == ["tactical"] and meta["tracking"] == "ready"
    tags = json.loads(get(base + "/api/tags")[2])["set_pieces"]
    assert all(t["start"] <= t["end"] for t in tags)


def test_homography_cache(served, tmp_path):
    cfg = served[3]
    hom = served[1].homography_path
    first = load_homography(hom, cfg.n_frames, tmp_path)
    again = load_homography(hom, cfg.n_frames, tmp_path)
    assert np.array_equal(first, again, equal_nan=True)
    assert np.isnan(first[5]).all()
