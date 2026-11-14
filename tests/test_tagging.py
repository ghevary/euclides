import numpy as np

import synth
from mtah.tagging import build_context, load_instances, team_sign


def _ctx(tmp_path, cfg, instances):
    path = tmp_path / "tags.xml"
    synth.write_tagging(path, instances)
    loaded = load_instances(path)
    return loaded, build_context(loaded, cfg, cfg.n_frames)


def test_instances_and_team_signs(tmp_path):
    cfg = synth.small_config()
    loaded, _ = _ctx(tmp_path, cfg, [
        ("PROGRESSION", 10.0, 20.0, {"Team": "Homeville FC"}),
        ("BUILD UP", 30.0, 40.0, {"Team": "Awayton"}),
        ("SET PIECES", 50.0, 55.0, {"Team": "N/A", "Type": "Throw-in"}),
    ])
    assert [i.code for i in loaded] == ["PROGRESSION", "BUILD UP", "SET PIECES"]
    assert [team_sign(i, cfg) for i in loaded] == [1, -1, 0]


def test_seconds_times_fps_is_a_video_frame(tmp_path):
    cfg = synth.small_config()
    _, ctx = _ctx(tmp_path, cfg, [("PROGRESSION", 10.0, 20.0, {"Team": "Homeville FC"})])
    assert ctx.team[249] == 0 and ctx.team[250] == 1 and ctx.team[499] == 1 and ctx.team[500] == 0


def test_defending_and_untagged_phases_add_no_team_prior(tmp_path):
    cfg = synth.small_config()
    _, ctx = _ctx(tmp_path, cfg, [
        ("DEFENDING IN MIDDLE THIRD", 10.0, 20.0, {"Team": "Awayton"}),
        ("PROGRESSION", 30.0, 40.0, {"Team": "N/A"}),
    ])
    assert not ctx.team.any()


def test_set_piece_marks_the_stretch_before_the_restart(tmp_path):
    cfg = synth.small_config(setpiece_lead_s=2.0)
    _, ctx = _ctx(tmp_path, cfg, [("SET PIECES", 60.0, 65.0, {"Team": "Awayton", "Type": "Throw-in"})])
    restart = int(62.0 * cfg.fps)
    before = int(cfg.setpiece_dead_before_by_type["throw-in"] * cfg.fps)
    assert ctx.dead[restart - before:restart].all()
    assert ctx.dead[restart] == 0 and ctx.dead[restart - before - 1] == 0
    after = int(cfg.setpiece_team_after_s * cfg.fps)
    assert (ctx.team[restart:restart + after] == -1).all() and ctx.team[restart + after] == 0


def test_frames_before_a_late_kick_off_are_forced_dead(tmp_path):
    cfg = synth.small_config()
    first, _ = cfg.segments[1]
    kick_s = (first + 200) / cfg.fps
    _, ctx = _ctx(tmp_path, cfg, [
        ("SET PIECES", (cfg.segments[0][0] - 50) / cfg.fps, 5.0, {"Type": "Kick Off (Start)"}),
        ("SET PIECES", kick_s, kick_s + 5.0, {"Type": "Kick Off (Start)"}),
    ])
    assert ctx.forced[first:first + 200].all()
    assert int(ctx.forced.sum()) == 200  # the first half is graded from its kick-off: nothing forced


def test_overlapping_team_priors_are_clipped(tmp_path):
    cfg = synth.small_config()
    _, ctx = _ctx(tmp_path, cfg, [
        ("PROGRESSION", 10.0, 20.0, {"Team": "Homeville FC"}),
        ("CREATING CHANCES", 12.0, 18.0, {"Team": "Homeville FC"}),
    ])
    assert np.max(ctx.team) == 1.0
