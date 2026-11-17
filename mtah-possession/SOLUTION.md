# BCDegens solution pipeline

This directory contains deterministic ATD feature generation and a testable possession decoder.
Competition data and derived frame-level feature files are local inputs and must never be committed.

## Environment

Python 3.8–3.11 is supported.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Expected feature file

The local CSV must contain exactly 199,319 ordered rows, frames `0..199318`, and these columns:

```text
frame, graded, ball_x_m, ball_y_m, ball_speed_ms, ball_speed_smooth_ms,
ball_jump, n_tracked, d_nearest_home_m, d_nearest_away_m,
n_home_within_5m, n_away_within_5m
```

Optional `dead_ball`, `smart_set_piece`, and `smart_restart` columns are added when Smart Tagging
is available. Numeric tracking features are canonicalized to three decimal places before decoding
so raw-data and saved-CSV runs make identical threshold decisions.

## Generate and validate a submission

Generate directly from the official tactical ATD without persisting derived features:

```powershell
python run_pipeline.py --data-dir C:\local\mtah-data --output outputs\BarcaClubDegens.csv
```

Or generate an inspectable local feature file first:

```powershell
python build_features.py --data-dir C:\local\mtah-data --output C:\local\features.csv
python inspect_features.py C:\local\features.csv
python run_pipeline.py --features C:\local\features.csv `
  --tagging-xml C:\local\mtah-data\tactical\BAR-ATH_tactical_smart_tagging.xml `
  --output outputs\BarcaClubDegens.csv `
  --debug-transitions outputs\transitions.csv `
  --diagnostics outputs\diagnostics.json
python evaluation\validate_submission.py outputs\BarcaClubDegens.csv
python release_check.py outputs\BarcaClubDegens.csv
```

The pipeline validates the input schema and frame sequence, generates all labels, validates the
submission in memory, prints its SHA-256 hash, and reports possession/dead-ball structure. The
transition log is local derived match data and must not be committed.

To compare two feature implementations without using ground truth:

```powershell
python compare_features.py C:\local\fallback_features.csv C:\local\team_features.csv
```

The comparison reports missingness disagreements and numerical differences for every feature.
`inspect_features.py` reports ball coverage, jump frequency, distance availability, graded-frame
coverage, and tracked-player quantiles before any labels are generated.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests use synthetic data only. They cover raw XML/ATD parsing, metre conversion, position-based
ball speed, canonical precision, team-distance features, SportsCode parsing, restart detection,
sustained switches, transition reasons, missing-ball carry, tracking jumps, dead-ball evidence,
outside-match `X`, structural diagnostics, frame alignment, and deterministic CSV output.

## Configuration

`config/feature_generation.json` controls speed smoothing and jump detection.
`config/dead_ball_rules.json` controls conservative restart detection inside Smart Tagging
set-piece windows. `config/possession_rules.json` contains the possession-state parameters. The defaults encode
the team's provisional 2 m control radius, 0.5 m opposing-player margin, and 10-frame persistence
rule. Unresolved rules remain visible configuration fields rather than hidden assumptions.

## Verified release state

The full official tactical ATD was processed successfully for all 199,319 frames. Its generated
features match the independently supplied table apart from harmless 0.001 CSV-rounding
differences, and both input routes produce the same final CSV byte for byte. The conservative
default intentionally leaves the single kickoff-after-goal event with no ball observations
unresolved instead of inventing a restart.

Before uploading, run the validator and release check on the exact team-named CSV. The remaining
steps are repository review/merge and submission-form upload, not pipeline dependencies.

GitHub Actions runs the synthetic test suite on every push and pull request. It receives code and
synthetic fixtures only; restricted competition data is never uploaded.
