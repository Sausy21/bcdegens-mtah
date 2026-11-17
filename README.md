# bcdegens-mtah

Barca Club Degens' reproducible possession-inference entry for More than a Hack 2027.

- Competition starter and solution code: [`mtah-possession/`](mtah-possession/)
- Team pipeline instructions: [`mtah-possession/SOLUTION.md`](mtah-possession/SOLUTION.md)
- Technical document: [`mtah-possession/TECHNICAL_DOCUMENT.md`](mtah-possession/TECHNICAL_DOCUMENT.md)
- Aggregate experiment log: [`mtah-possession/EXPERIMENTS.md`](mtah-possession/EXPERIMENTS.md)

Restricted match data, videos, and derived frame-level features are intentionally excluded.

From `mtah-possession/`, the pipeline runs directly from the official local data bundle:

```powershell
python run_pipeline.py --data-dir C:\local\mtah-data --output outputs\BarcaClubDegens.csv
```
