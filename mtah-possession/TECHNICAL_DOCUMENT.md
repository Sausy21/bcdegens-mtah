# Barca Club Degens — Technical Document

## Approach

The system converts per-frame tracking-derived features into raw team-control evidence, then
decodes that evidence with a deterministic temporal state machine. It distinguishes possession
from brief touches by requiring sustained control before changing the current team.

## Assets used

- Tactical automatic tracking data: parsed directly from the FIFA/EPTS metadata and raw text.
- Smart Tagging: set-piece intervals provide approximate restart windows. They are combined with
  tracking-based ball movement and nearby-player evidence rather than copied directly as labels.
- Homography and video: not used by the current fallback pipeline. They remain possible future
  inputs for recovering missing ball observations.
- No hand-produced labels from the provided match are used.

## Feature extraction

The generator streams the tactical ATD without a kloppy dependency. Normalized player and ball
coordinates are converted to metres using the supplied pitch dimensions. For every frame it
computes the number of tracked players, nearest home/away distance to the ball, and players from
each team within five metres. Ball speed is derived from consecutive pitch positions because the
raw ball-speed channel does not behave like metres per second on this export. A centred rolling
median supplies the smoothed speed, and speeds above 40 m/s are flagged as suspect rather than
deleted. Missing observations remain missing rather than being interpolated. Numeric features are
rounded to three decimals before decoding for route-independent reproducibility.

## Possession rules

- A team produces control evidence when its nearest player is within the configured control
  radius and closer than the opposing team's nearest player by the configured margin.
- Evidence must persist for a configured number of frames before possession changes.
- Missing ball locations, tracking jumps, and inconclusive frames retain the current state.
- Published kickoff anchors keep pre-kickoff frames in each graded half at `D`.
- Frames outside the official half blocks are labelled `X`.
- A set-piece tag is treated as a search window, not a dead-ball label. The restart is the first
  configurable run of non-jump ball movement with a player close enough to control it. Frames
  preceding that restart are marked `D`, capped at 15 seconds for unusually long tag windows. A
  tag without sustained restart evidence does not invent a dead-ball interval.

On the supplied Smart Tagging file, the method parsed 638 events and 59 set pieces and found
tracking-supported restarts in 58. All corners, free kicks, goal kicks, throw-ins and starting
kickoffs were detected. The one conservative miss is a kickoff after a goal whose tag window has
no ball observations. The resulting full-match output contains 356 possession spells of mean
16.2 seconds and labels 9.4% of graded frames as dead ball.

The feature generator was verified against the independently supplied feature table over all
199,319 frames. Frame alignment, missing-value patterns, boolean features, and count features
matched exactly; numeric differences were at most 0.001 after CSV rounding. Running from the raw
ATD and running from the supplied feature table produced byte-identical submission files with
SHA-256 `7b0e832079926b7d7f86c09a41d358330668beb9925e6983d8336638a3df12d8`.

## Reproduction

Install the pinned dependencies, then run:

```text
python run_pipeline.py --data-dir <data-root> --output <team-name>.csv
python release_check.py <team-name>.csv
```

See `SOLUTION.md` for full installation, feature-inspection, comparison, validation, and test
commands.

## Training data

No labelled training data is used. The published public ground-truth window was used only as a
pipeline sanity check and was not used to select or tune thresholds.

## Limitations

- Smart Tagging begins its interval only shortly before a restart, so the conservative method can
  miss the earlier part of a stoppage and can miss events where ball tracking is absent.
- Goalkeeper control is inferred through the same proximity rules as outfield-player control;
  there is no goalkeeper-specific state.
- Inconclusive loose-ball periods retain the last established team without a fixed timeout, which
  can overextend a possession when ball tracking is absent for a long interval.
- The fixed thresholds are general tracking-based heuristics and were not tuned against the
  provided match's human labels.
