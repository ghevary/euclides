# Technical document

Team: `Euclides` · More than a Hack 2027, Phase 1 · repository: <https://github.com/ghevary/euclides>

## 1. Approach

Possession is treated as a hidden state that persists, not as a per-frame classification.
Each frame supplies evidence for three states (home, away, dead ball), and a three-state
hidden Markov model decoded with Viterbi, one half at a time, turns that evidence into
intervals.

We chose this over a per-frame rule with smoothing, which is what the reference baseline
does, for two reasons. The ranking metric matches whole possessions, so the cost of a
change of state should be explicit and should depend on how strong the evidence is, which a
majority vote cannot express. And the ball is unseen on about 41% of graded frames, where a
per-frame rule has nothing to say while a state model carries the last state forward.

We chose it over a learned model because no labels are allowed for this match beyond five
minutes that may not be used for tuning, and we had no labelled external matches in the
same tracking format.

Most of the work went into the evidence: cleaning the tracking (goalkeepers in the wrong
team, ghost tracks, players lost in duels) and telling a stoppage from slow play (how fast
the players move, the whistle before a set piece, a ball hidden by a body).

## 2. Assets used

| Asset | Used | How |
|---|---|---|
| Tactical ATD (tracking) | yes | ball and player positions, team of each track slot, player pace |
| Smart Tagging | yes, as a prior | team of attacking phases; set pieces as restarts and as the end of a stoppage |
| Homography | only in the viewer | to draw the tracking on the video; the pipeline works in pitch coordinates |
| Tactical video | not by the pipeline | watched in the read-only viewer (`src/viewer`) to debug, never to label |
| Broadcast ATD and video | tried, not in the submission | see below |

**Tried and dropped.**

- *Ball detection on the video with YOLOv8n* (`src/vision`, not used by the pipeline). On the
  tactical video the ball is about 9 pixels wide; a pilot found false detections outnumbering
  real ones, at 116 ms per 640-pixel tile on a CPU. Not worth it against a ball track that is
  already there.
- *Broadcast recording.* We matched every broadcast frame to the tactical frame it shows,
  from the tracking alone (both trackings are in the same pitch coordinates), found replays
  and close-ups, and filled tactical frames without a ball from live broadcast frames. It
  filled about 4,500 graded frames and did not change the five public minutes. We kept it out
  of the submission for lack of time to check it by eye.
- *Joining tracklets into numbered players* to fill players lost between two tracklets. It
  joined 1,540 of 2,234 tracklets but left far more "players" than a match has, so we kept
  only the simpler short hold described below.

Without Smart Tagging (`use_tagging: false`) the output changes on 12,110 of the 158,857
graded frames (7.6%) and reports 297 possessions of mean 13.0 s instead of 253 of mean
14.4 s. We kept the tagging because the kick-off and restart priors encode football
knowledge the tracking alone does not carry.

## 3. Possession rules implemented

**Which team has the ball**

- The side whose nearest player is closer to the ball gets the evidence, in proportion to
  the difference in distance and discounted when nobody is within a few metres. A ball in
  flight therefore says nothing, and the previous team keeps possession: a loose ball stays
  with the last team.
- Only sustained evidence changes the state, and a change that lasts under two seconds is
  merged back. A single touch, a deflection or a won header without control does not last
  that long.
- **Duels.** Two players shoulder to shoulder are often tracked as one. A player the tracking
  loses keeps their last position for 1.5 s, so a duel reads as contested and the ball stays
  with the team that had it instead of jumping to whoever is still tracked.
- **Goalkeepers in the wrong team.** A keeper wears a different kit, and the tracking
  sometimes puts them in the other team's slots. A tracklet of 5 s or more that is, for at
  least 60% of it, the deepest central player within 30 m of a goal belongs to the team
  defending that goal. Which goal a team defends is read from where the teams stand in each
  half.
- **Ghost tracks.** A track whose position does not change by a single millimetre for 1 s is
  a copy left behind by the tracking, not a person, and is dropped. When both teams' keeper
  slots, or any two players of different teams alone in front of a goal, stay within 6 m of
  each other for 2 s, they are one goalkeeper followed twice; the copy in the team not
  defending that goal is dropped.

**Dead ball.** No single cue decides; each adds evidence.

- The ball seen more than 1.5 m outside the lines, or lost while crossing or heading over a
  line within 2 m of it.
- The ball at rest for 2 to 6 s, or unseen for 3 to 8 s.
- **How fast the players move.** In open play somebody is always running. The 90th
  percentile of player speed, smoothed over 2 s, is compared with a walking threshold found
  in this match's own speed distribution (2.25 m/s, between priors of 1.5 and 3 m/s). When
  everybody walks the game is probably stopped, even with the ball moving; when somebody
  runs, a resting or unseen ball (a keeper holding it, a shielded ball) is still alive.
  Once the ball is out of the pitch the players' pace no longer counts.
- **A ball hidden by a body.** A ball lost at a player's feet (within 2 m) or among three or
  more players is hidden by them; for up to 5 s its unseen time does not count towards a
  stoppage.
- **The whistle.** A tagged set piece is a stoppage back to the moment the players last
  stopped running before it, up to 90 s. A ball carried to the free-kick spot is therefore
  dead from the whistle, not still in possession.
- **Restarts.** The seconds before a tagged restart lean towards dead ball; a team tag on
  the set piece gives a short prior to that team afterwards.
- **Kick-off.** Graded frames before the tagged kick-off of a half are dead ball.

**Output.** Possessions and stoppages shorter than two seconds are merged into a neighbour.
`X` is written only outside the graded halves.

## 4. Limitations

Watching the full match in the viewer, we still see:

- a goalkeeper holding the ball sometimes labelled as a dead ball (about five or six times);
- a goalkeeper still counted in the wrong team (about four or five times);
- a ball in a crowd of players sometimes taken for a stoppage;
- a ball that has crossed a line still counted as possession for a moment (three or four
  times).

More generally:

- No parameter is fitted. All values are priors, so the model is unlikely to be at its best
  setting.
- With the ball unseen and no tag, the model cannot tell which team has it; it assumes the
  previous one.
- Smart Tagging is automatic, and its errors are imported in proportion to its weight.
- Nothing is known about control beyond proximity: who is nearest is taken as who has it.

With more time: add the broadcast ball where tactical loses it (the matching is written),
recognise a keeper holding the ball from a ball that stops with the keeper in their own
area after a save rather than after the ball went out, and fit the priors on labelled
external matches.

## 5. Training data

None. No model is trained and no parameter is fitted. No labelled dataset of any match was
used, and nothing on the provided game was labelled or corrected by hand. No frame number of
this match appears in the code.

## 6. Results

The submitted file `Euclides.csv` passes the organisers' `validate_submission.py`.

| Label | Frames | Share |
|---|---|---|
| H | 57,450 | 28.8% |
| A | 33,697 | 16.9% |
| D | 67,710 | 34.0% |
| X | 40,462 | 20.3% |

It reports 253 possessions of mean 14.4 s, and home holds 63.0% of the live-ball time.

On the five public minutes it scores spellF1 0.4875 and macroF1 0.7657, against 0.3256 and
0.6207 for the organisers' `baseline_smooth50` on the same window. Those five minutes hold
17 possessions, so this is a check that the pipeline works, not a measure to rank on. We
scored each version once as such a check. The public window chose no parameter; the
submitted version is not the one with the highest public spellF1 (an earlier one had
0.5125): it was chosen for its behaviour on fouls and hidden balls seen in the viewer.

## 7. Tools

- **Possession Viewer** (`src/viewer`). A read-only local web page that plays the tactical
  video next to the labels, the tracking and the Smart Tagging set pieces. It was used to
  understand the data and debug the pipeline, as RULES.md allows. It cannot edit a label
  and computes no score.
- **AI coding assistants** were used to write and review code and documentation. They are
  not part of the pipeline: no model is called when the submission is generated. They were
  shown aggregate statistics computed locally (counts and quantiles over the match) and a
  few viewer screenshots that team members took to describe errors; no video file, tracking
  file or label file was uploaded to them. See `AGENTS.md`.

## 8. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
export MTAH_DATA_DIR=/path/to/bundle                   # the folder containing tactical/
python -m mtah run --team Euclides
```

Output SHA-256: `3242c0bf2d7cf8cb5cad7383d6135595b710fa6f069bbb325f497e890e30209a`, with
numpy 2.5.3 on Python 3.14. A run without the parsing cache gives the same bytes. The
pipeline has no randomness.
