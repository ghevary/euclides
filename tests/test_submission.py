import numpy as np

import synth
from mtah.submission import summarize, validate_csv, write_csv


def test_written_file_meets_the_contract(tmp_path):
    labels = np.array(list("XXHHHDDAAX"))
    path = write_csv(tmp_path / "team.csv", labels)
    raw = path.read_bytes()
    assert raw.startswith(b"frame,label\n0,X\n1,X\n2,H\n") and raw.endswith(b"9,X\n")
    assert b"\r" not in raw and raw.count(b"\n") == 11
    assert validate_csv(path, 10) is None


def test_validator_rejects_the_usual_mistakes(tmp_path):
    def check(text, n=3):
        path = tmp_path / "bad.csv"
        path.write_text(text, encoding="utf-8")
        return validate_csv(path, n)

    assert check("frame,label\n0,H\n1,A\n2,D\n") is None
    assert check("Frame , Label\n0,h\n1,a\n2,x\n") is None        # case and spaces are tolerated
    assert check("0,H\n1,A\n2,D\n") is not None                    # no header
    assert check("frame,label\n1,H\n2,A\n3,D\n") is not None       # off by one
    assert check("frame,label\n0,H\n1,A\n") is not None            # short
    assert check("frame,label\n0,H\n1,A\n2,D\n3,D\n") is not None  # long
    assert check("frame,label\n0,H\n1,\n2,D\n") is not None        # blank cell
    assert check("frame,label\n0,H\n1,Z\n2,D\n") is not None       # unknown label
    assert check("frame,label\n0,H\n2,A\n1,D\n") is not None       # out of order
    assert check(",frame,label\n0,0,H\n1,1,A\n2,2,D\n") is not None  # index column
    assert validate_csv(tmp_path / "nope.csv", 3) is not None


def test_summary_counts_possessions_as_maximal_live_runs():
    cfg = synth.small_config()
    labels = np.array(list("XXHHHDDHHAAAX"))
    info = summarize(labels, cfg)
    assert info["counts"] == {"H": 5, "A": 3, "D": 2, "X": 3}
    assert info["possessions"] == 3                          # H, H after the stoppage, A
