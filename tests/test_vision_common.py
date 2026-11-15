"""Geometry of the vision stage. No video, no detector, no torch needed."""
import numpy as np

import synth
from vision.common import TILE, crop_origin, play_ranges, project, tile_origins


def test_only_graded_frames_are_decoded():
    cfg = synth.small_config()
    assert play_ranges(cfg) == [(100, 1399), (1600, 2899)]


def test_projection_follows_the_homography():
    h = np.array([1920, 0, 0, 0, 1080, 0, 0, 0, 1], dtype=float)
    assert project(h, 0.5, 0.5) == (960.0, 540.0)
    h[8] = 2.0
    assert project(h, 0.5, 0.5) == (480.0, 270.0)


def test_crop_stays_inside_the_picture():
    assert crop_origin(960, 540, 1920, 1080) == (640, 220)
    assert crop_origin(5, 5, 1920, 1080) == (0, 0)
    assert crop_origin(1915, 1075, 1920, 1080) == (1920 - TILE, 1080 - TILE)


def test_tiles_cover_the_picture():
    origins = tile_origins(1920, 1080)
    assert len(origins) == 6
    covered = np.zeros((1080, 1920), dtype=bool)
    for x, y in origins:
        assert 0 <= x <= 1920 - TILE and 0 <= y <= 1080 - TILE
        covered[y:y + TILE, x:x + TILE] = True
    assert covered.all()
