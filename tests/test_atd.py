import numpy as np
import pytest

import synth
from mtah.atd import load_tracking, slot_layout
from mtah.errors import MtahError
from mtah.paths import tactical_assets


def _write(tmp_path, cfg, **raw_kwargs):
    players, ball, _, _ = synth.scenario(cfg)
    xml, txt = tmp_path / "m.xml", tmp_path / "m.txt"
    synth.write_meta(xml, cfg.n_slots)
    synth.write_raw(txt, players, ball, **raw_kwargs)
    return xml, txt, players, ball


def test_row_i_is_video_frame_i(tmp_path):
    cfg = synth.small_config()
    xml, txt, players, ball = _write(tmp_path, cfg)
    tr = load_tracking(xml, txt, cfg)
    assert tr.players.shape == (cfg.n_frames, cfg.n_slots, 2)
    assert tr.ball.shape == (cfg.n_frames, 2)
    # frame_id 1 is row 0 and frame_id N is row N-1: no shift anywhere in between
    for row in (0, 100, 1234, cfg.n_frames - 1):
        np.testing.assert_allclose(tr.ball[row], ball[row], atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(tr.players[row], players[row], atol=1e-5, equal_nan=True)


def test_missing_values_stay_missing(tmp_path):
    cfg = synth.small_config()
    xml, txt, players, ball = _write(tmp_path, cfg)
    tr = load_tracking(xml, txt, cfg)
    assert np.array_equal(np.isnan(tr.ball), np.isnan(ball))
    assert np.array_equal(np.isnan(tr.players), np.isnan(players))


def test_team_comes_from_metadata_not_slot_order(tmp_path):
    cfg = synth.small_config()
    for swapped in (False, True):
        xml = tmp_path / f"meta_{swapped}.xml"
        synth.write_meta(xml, cfg.n_slots, swap_team_order=swapped)
        home, cols, k = slot_layout(xml, cfg)
        assert home[:24].all() and not home[24:].any()
        assert cols == (0, 1) and k == 3


def test_home_hint_mismatch_is_an_error(tmp_path):
    cfg = synth.small_config(home_name_hint="awayton")
    xml = tmp_path / "meta.xml"
    synth.write_meta(xml, cfg.n_slots)
    with pytest.raises(MtahError):
        slot_layout(xml, cfg)


def test_manual_slot_override(tmp_path):
    cfg = synth.small_config(home_slots="25-46", away_slots="1-24")
    home, _, _ = slot_layout(tmp_path / "unused.xml", cfg)
    assert not home[:24].any() and home[24:].all()


def test_frame_ids_not_starting_at_one_are_rejected(tmp_path):
    cfg = synth.small_config()
    xml, txt, _, _ = _write(tmp_path, cfg, first_id=0)
    with pytest.raises(MtahError):
        load_tracking(xml, txt, cfg)


def test_row_count_mismatch_is_rejected(tmp_path):
    cfg = synth.small_config()
    xml, txt, _, _ = _write(tmp_path, cfg)
    for wrong in (cfg.n_frames - 1, cfg.n_frames + 1):
        with pytest.raises(MtahError):
            load_tracking(xml, txt, synth.small_config(n_frames=wrong))


def test_cache_round_trip(bundle, tmp_path):
    root, cfg, _ = bundle
    assets = tactical_assets(root)
    first = load_tracking(assets.atd_xml, assets.atd_txt, cfg, tmp_path)
    assert list(tmp_path.glob("*.npz"))
    second = load_tracking(assets.atd_xml, assets.atd_txt, cfg, tmp_path)
    np.testing.assert_array_equal(first.ball, second.ball)
    np.testing.assert_array_equal(first.players, second.players)
