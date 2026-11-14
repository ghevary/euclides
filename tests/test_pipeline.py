import json

import numpy as np

import synth
from mtah.cli import main
from mtah.pipeline import predict, run
from mtah.submission import validate_csv


def _spells(labels):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    return [(labels[s], s) for s in starts if labels[s] in ("H", "A")]


def test_end_to_end_recovers_the_synthetic_match(bundle):
    root, cfg, truth = bundle
    labels = predict(root, cfg)
    graded = truth != "X"
    assert (labels[~graded] == "X").all() and not (labels[graded] == "X").any()
    assert float(np.mean(labels[graded] == truth[graded])) > 0.95
    # the same possessions, not just mostly the same frames
    assert [lab for lab, _ in _spells(labels)] == [lab for lab, _ in _spells(truth)]
    for (_, got), (_, want) in zip(_spells(labels), _spells(truth)):
        assert abs(int(got) - int(want)) <= cfg.fps        # every boundary within one second


def test_two_runs_give_the_same_bytes(bundle, tmp_path):
    root, cfg, _ = bundle
    a = run(root, tmp_path / "a" / "team.csv", cfg)
    b = run(root, tmp_path / "b" / "team.csv", cfg, cache_dir=tmp_path / "cache")
    c = run(root, tmp_path / "c" / "team.csv", cfg, cache_dir=tmp_path / "cache")  # from the cache
    assert a["valid"] and a["output_sha256"] == b["output_sha256"] == c["output_sha256"]
    assert (tmp_path / "a" / "team.csv").read_bytes() == (tmp_path / "c" / "team.csv").read_bytes()


def test_runs_without_smart_tagging(bundle, tmp_path):
    root, cfg, truth = bundle
    cfg_off = synth.small_config(use_tagging=False)
    labels = predict(root, cfg_off)
    graded = truth != "X"
    assert float(np.mean(labels[graded] == truth[graded])) > 0.9


def test_cli_run_validate_and_audit(bundle, tmp_path, capsys):
    root, cfg, _ = bundle
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"n_frames": cfg.n_frames, "segments": cfg.segments,
                                    "home_name_hint": cfg.home_name_hint}), encoding="utf-8")
    out = tmp_path / "out" / "team.csv"
    args = ["--config", str(cfg_path), "--data-dir", str(root), "--no-cache"]
    assert main(["run", *args, "--out", str(out)]) == 0
    assert validate_csv(out, cfg.n_frames) is None
    assert out.with_suffix(".meta.json").is_file()
    assert main(["validate", "--config", str(cfg_path), str(out)]) == 0
    assert main(["audit", *args]) == 0
    printed = capsys.readouterr().out
    assert "format check passed" in printed and "slots home / away" in printed


def test_cli_reports_a_missing_data_directory(tmp_path, capsys):
    assert main(["run", "--data-dir", str(tmp_path / "nope"), "--no-cache"]) == 2
    assert "does not exist" in capsys.readouterr().out
