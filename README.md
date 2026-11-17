# bcdegens-mtah

Barca Club Degens' possession-inference entry for More Than A Hack 2027.

## Reproduce the submission

Put the provided match files in `mtah-possession/data/tactical/` with their original names
(match data is gitignored and never committed). Then, from `mtah-possession/`:

```bash
pip install numpy pandas
python3 tools/explore_data.py data/tactical/BAR-ATH_tactical_atd.txt data/tactical/BAR-ATH_tactical_atd.xml data/tactical/BAR-ATH_tactical_smart_tagging.xml --out outputs
python3 run_josh_pipeline.py --data-dir data --features outputs/features.csv --output outputs/BarcaClubDegens.csv
```

The output is identical to the submitted `BarcaClubDegens.csv`
(SHA-256 `ec8960fcbc76c1a3efb7754b82ae9c7a61de4a5cd80f0e4f59d94934d88f810e`).
Tested with Python 3.13, numpy 2.5, pandas 3.0.

## What's where

- `tools/explore_data.py`: per-frame feature table from the tactical tracking
- `game_logic/rules.py`: the possession rules (control, dead ball, restarts)
- `game_logic/RULEBOOK.md`: every rule and threshold with its reason
- `game_logic/test_rules.py`: synthetic football scenes (41 pass, 1 known gap); `game_logic/audit_scenes.py`: robustness audit
- `run_josh_pipeline.py`: runs the rules and writes the submission

## Other versions (not used for the submitted file)

- `run_pipeline.py`: version A, tag-based dead ball (Rishit). Reproduce with
  `python3 run_pipeline.py --data-dir data --output outputs/tag_based.csv`
  (SHA-256 begins `7b0e832079926b7d`). Compared with version B in the technical document.
- `game_logic/make_submission.py`: development command with its own feature table.

The technical document `BarcaClubDegens.pdf` is submitted with the form.
