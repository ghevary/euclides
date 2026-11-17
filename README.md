# mtah-possession

Ball possession for every frame of a football match, from automatic tracking data.
An entry for [More than a Hack 2027, Phase 1](https://github.com/metrica-sports/more-than-a-hack-2027).

The output is the competition's submission file: `frame,label` with one row per tactical
video frame and the labels `H` (home in possession), `A` (away), `D` (dead ball) and `X`
(outside the graded halves).

**Status.** The code is tested on synthetic data and its output passes the official
validator at full match size. It has not been scored: there is no number yet to compare
with the published baseline (`spellF1` 0.2433). Technical document: `docs/Euclides.md`.

## Setup

Python 3.9 or newer. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .                  # makes `python -m mtah` and `python -m viewer` importable
git config core.hooksPath .githooks        # refuses commits that contain data
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

`requirements.txt` pins numpy. If that version has no build for your Python, change the pin
and commit it: the pin is what lets someone else regenerate the same file.

## Data

The match data is not in this repository and must never be. Keep the bundle outside the
clone and point at the folder that contains `tactical/`:

```powershell
$env:MTAH_DATA_DIR = "D:\mtah2027"          # PowerShell
export MTAH_DATA_DIR=/path/to/mtah2027      # macOS / Linux
```

Only three files are read: `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt` and
`BAR-ATH_tactical_smart_tagging.xml`.

## Run

```bash
python -m pytest                                  # 51 tests on synthetic data
python -m mtah audit                              # compares aggregates with DATA.md
python -m mtah run --team <team_name>             # writes submissions/<team_name>.csv
```

Then check the exact file you will upload with the organisers' validator. Clone their
repository next to this one and pass its path:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../more-than-a-hack-2027
python -m mtah run --team <team_name> --official-repo ../more-than-a-hack-2027
```

`run` also writes `submissions/<team_name>.meta.json` with the configuration, the package
versions and SHA-256 hashes of the inputs and of the output. Two runs on the same data give
the same bytes.

`audit` prints counts such as the number of frames with a ball and marks each one `OK` or
`DIFFERS` against the figure published in `DATA.md`. Run it first: a `DIFFERS` means the
parser or the frame alignment is wrong, and nothing after it can be trusted.

## How it works

1. **Load** (`atd.py`). Row *i* of the tracking is video frame *i* (`frame_id - 1`). The
   loader refuses a file whose frame ids are not exactly `1..N`. Team membership comes from
   the metadata; track slots are never treated as players.
2. **Clean the tracking** (`features.py`). Positions frozen to the millimetre for 1 s are
   ghost copies and are dropped; a goalkeeper followed in both teams at once keeps only the
   copy of the team defending that goal; a goalkeeper tracked in the other team's slots is
   given back to the team whose goal they guard; a player the tracking loses in a duel
   keeps their last position for 1.5 s.
3. **Evidence per frame** (`features.py`, `model.py`).
   - *Which team:* distance from the ball to the nearest player of each side. The nearer
     side gets the evidence, weighted down when nobody is close to the ball.
   - *Dead ball:* the ball outside the pitch or lost on its way over a line, at rest for
     several seconds, or unseen for longer than an occlusion lasts; and how fast the
     players move (somebody always runs in open play). A ball lost at a player's feet or
     in a crowd is hidden, not stopped.
   - *No ball:* gaps up to half a second are interpolated. Longer gaps give no evidence
     about the team, so the previous possession carries on.
4. **Smart Tagging as a prior** (`tagging.py`). Attacking phases add a small push towards
   their tagged team. A set piece is a stoppage back to the moment the players last stopped
   running before it (the whistle). A half that is graded from before its kick-off starts
   with a dead ball.
5. **Temporal model** (`temporal.py`). A three-state hidden Markov model (H, A, D) decoded
   with Viterbi, one half at a time. Changing state costs something, so a deflection or a
   tracking glitch does not split a possession. Runs shorter than two seconds are absorbed
   by a neighbour.
6. **Write and check** (`submission.py`). `X` outside the graded halves, never inside.

Every number is in `src/mtah/config.py` with a one-line reason. To try a variant, write a
JSON file with the keys you want to change and pass `--config variant.json`.

## Possession Viewer

A local web page to check the output by eye: the tactical video on the left, the frame
number and its label on the right, the whole match as a coloured timeline underneath.

```powershell
python -m viewer --data-dir E:\barca --labels submissions\Euclides.csv
```

On Windows, `run_viewer.bat` does the same without installing the package. The page opens
at `http://127.0.0.1:8000/`. It needs no extra dependency (Python standard library and
numpy) and makes no outside request: the server listens on `127.0.0.1` only and the page
carries a content security policy that allows nothing but itself.

| What | How |
|---|---|
| Play, pause, speed 0.25x to 4x | buttons, `Space` |
| One frame back or forward | `Left` / `Right`; with `Shift` one second, with `Ctrl` ten |
| Go to a frame | type the number, or `G` |
| Next or previous change of label | `[` / `]` |
| Whole match at a glance | click or drag on the timeline; yellow marks are Smart Tagging set pieces |
| Frame-level detail | the strip below shows 10 s each side, one column per frame |
| Tracking | nearest player of each team to the ball, ball seen or not, overlay on the video (`O`), mini pitch |
| Compare two submissions | "Compare with CSV…" adds a second row and marks where they differ |
| Full screen | `F` or double click; the frame number and label stay on screen |
| Other video input | the source menu switches to the broadcast video; "Open video…" plays any local file; `--video` serves a proxy |

**Frame accuracy.** A jump to frame N seeks to the middle of that frame, `(N + 0.5) / 25`
seconds, and the number shown comes from `requestVideoFrameCallback`, the timestamp of the
frame actually on screen. Pausing snaps to a whole frame. This was checked with a
synthetic video that carries its frame number in the picture.

**Video inputs.** The tactical video is the reference: row N of the CSV is frame N. A
video with a different frame count (the broadcast recording has 164,487 frames and its own
timeline) plays with its own frame counter and **without labels**, because no mapping
between the two timelines exists yet. A proxy made with ffmpeg at a lower resolution keeps
the labels as long as it has the same 199,319 frames.

**What it deliberately does not do.** It cannot edit a label, and it computes no score and
no agreement percentage. The rules forbid hand labels on this match and tuning on the
public ground truth; watching the video to understand and debug is allowed, turning what
was seen into labels, corrections or frame-specific constants is not (RULES.md).

## What is assumed and not yet verified

- The `Team` tag of an attacking phase names the team in possession, and the `Team` tag of
  a set piece names the team that restarts.
- The restart sits about 2.5 s into a set-piece tag (`setpiece_lead_s`). `audit` prints the
  ball's visibility around tag starts so this can be set from the data.
- The priors in `config.py` come from football knowledge and the organisers' documents.
  They are not tuned on any labels, so they may be far from the best values.

## Rules this repository follows

- Labels are produced by the algorithm. Nothing is labelled or corrected by hand.
- The public five minutes of ground truth are not used for tuning.
- No video, tracking or label file of the match is uploaded to hosted AI tools. See
  `AGENTS.md` and section 7 of the technical document.
- No data in git: see `.gitignore` and `.githooks/pre-commit`.

## Layout

```
src/mtah/        config, loaders, features, model, temporal, submission, pipeline, audit, cli
src/viewer/      read-only web viewer: server.py, data.py, static/ (page, script, style)
src/vision/      dropped experiment: YOLOv8n ball detection (needs ultralytics; not used)
tests/           synthetic match generator and tests
docs/            technical document (Euclides.md)
.githooks/       pre-commit hook that refuses data files
```
