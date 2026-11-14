# Technical document

Team: `EUCLIDES` · More than a Hack 2027, Phase 1 · repository: <https://github.com/ghevary/euclides>

## 1. Approach

Possession is treated as a hidden state that persists, not as a per-frame classification.
Each frame supplies evidence for three states (home, away, dead ball), and a three-state
hidden Markov model decoded with Viterbi turns that evidence into intervals.

We chose this over a per-frame rule with smoothing, which is what the reference baseline
does, for two reasons. The ranking metric matches whole possessions, so the cost of a
change of state should be explicit and should depend on how strong the evidence is, which a
majority vote cannot express. And the ball is unseen on 41% of graded frames, where a
per-frame rule has nothing to say while a state model simply carries the last state forward.

We chose it over a learned model because no labels are allowed for this match beyond five
minutes that may not be used for tuning, and we had no labelled external matches in the
same tracking format.

## 2. Assets used

| Asset | Used | How |
|---|---|---|
| Tactical ATD (tracking) | yes | ball and player positions; team of each track slot |
| Smart Tagging | yes, as a prior | team of attacking phases; set pieces as restarts |
| Homography | no | the tracking is already in pitch coordinates |
| Tactical video | not by the pipeline | not decoded; watched in the read-only viewer (`src/viewer`) to debug, never to label |
| Broadcast video and ATD | no | not synchronised with the tactical timeline |

Without Smart Tagging (`use_tagging: false`) the output changes on 8,589 of the 158,857 graded
frames (5.4%) and reports 314 possessions of mean 11.6 s instead of 291 of mean 12.2 s. We
cannot say which is better: neither file was scored. We kept the tagging because the
kick-off and restart priors encode football knowledge the tracking alone does not carry.

## 3. Possession rules implemented

- **Who has the ball.** The side whose nearest player is closer to the ball gets the
  evidence, in proportion to the difference in distance and discounted when nobody is
  within a few metres. A ball in flight therefore says nothing, and the previous team keeps
  possession, which matches the rule that a loose ball stays with the last team.
- **Change of possession.** Only sustained evidence changes the state, and a change that
  lasts under two seconds is merged back. A single touch, a deflected pass or a won header
  without control does not last that long.
- **Missing ball.** Gaps up to 0.5 s are interpolated. Longer gaps give no evidence about
  the team. A gap longer than about four seconds becomes evidence of a stoppage.
- **Dead ball.** Evidence comes from the ball being outside the pitch, at rest for more
  than a few seconds, unseen for a long time, or inside the window before a tagged restart.
  No single cue decides; a slow ball alone is not a dead ball.
- **Restarts.** After a stoppage the first team with clear evidence gets the possession;
  a team tag on the set piece adds a short prior.
- **Kick-off.** Graded frames before the tagged kick-off of a half are dead ball.
- **Ball held by the goalkeeper.** A keeper holding the ball is in play, but in the tracking
  the ball rests or disappears, which reads as a stoppage. So a resting or unseen ball does
  not count as evidence of a dead ball while all of this holds: the ball is inside a
  penalty area with a player of the defending team within 1.5 m; it arrived in play
  (tracked through the three seconds before, and not from outside the pitch); an attacker
  was inside that penalty area when the hold began; and no more than eight seconds have
  passed. The last two come from the Laws of the Game: at a goal kick or a defending free
  kick opponents must be outside the area, and a keeper may hold the ball for eight
  seconds. Which goal each team defends is read from where the teams stand in each half.
  The rule changes six stretches of the match, 1,161 frames in total.
- **Minimum length.** Possessions and stoppages shorter than two seconds are merged into a
  neighbour.
- **Abstention.** `X` is written only outside the graded halves.

## 4. Limitations

- No parameter is fitted. All values are priors, so the model is unlikely to be at its best
  setting.
- With the ball unseen and no tag, the model cannot tell which team has it; it assumes the
  previous one.
- A stoppage during which the ball stays visible and moving, such as a player carrying it
  to a free kick, is not recognised as dead.
- The tracking has no ball height and no player pose, so a ball in the keeper's hands, a
  goal kick and a free kick look alike. The held-ball rule separates them by context only.
  It misses a hold that follows a lost ball (a shot caught after the tracking dropped it),
  a hold with no attacker in the area, and anything past eight seconds; it can also keep
  alive a stoppage inside the area that happens to meet its conditions.
- Smart Tagging is automatic and its errors are imported in proportion to its weight.
- Track identities are not used, so nothing is known about control beyond proximity.

With more time: fit the priors on labelled external matches, use player and team movement
to decide possession while the ball is unseen, and detect restarts from formations.

## 5. Training data

None. No model is trained and no parameter is fitted. No labelled dataset of any match was
used, and nothing on the provided game was labelled or corrected by hand. The five public
minutes of ground truth were not used to choose any value.

## 6. Results

The submitted file `EUCLIDES.csv` passes the organisers' `validate_submission.py`.

| Label | Frames | Share |
|---|---|---|
| H | 55,083 | 27.6% |
| A | 33,515 | 16.8% |
| D | 70,259 | 35.2% |
| X | 40,462 | 20.3% |

It reports 291 possessions of mean 12.2 s. `X` is written only on the 40,462 ungraded
frames. The file has not been scored: we did not use the public window to compare versions,
so we have no `spellF1` to report against the baseline of 0.2433.

## 7. Tools

- **Possession Viewer** (`src/viewer`). A read-only local web page that plays the tactical
  video next to the labels, the tracking and the Smart Tagging set pieces. It was used to
  understand the data and debug the pipeline, as RULES.md allows. It cannot edit a label
  and computes no score.
- **AI coding assistants** were used to write and review code and documentation. They are
  not part of the pipeline: no model is called when the submission is generated. See
  `AGENTS.md` for the rules they worked under.

## 8. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
export MTAH_DATA_DIR=/path/to/bundle                   # the folder containing tactical/
python -m mtah run --team EUCLIDES
```

Output SHA-256: `8fc784644a103c044eb0a834a34cd1d44f27e806c6efa19bf6824bf074263ae9`, with
numpy 2.5.3 on Python 3.14. A second run on the same data gave the same bytes. The pipeline has no randomness.
