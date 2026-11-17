# Experiment log

The public five-minute labels are used only to verify that the pipeline is aligned and functional,
not to select thresholds. Full-match structural diagnostics do not use ground truth.

| Date | Variant | Feature SHA-256 | Submission SHA-256 | Public macroF1 | Public spellF1 | Notes |
|---|---|---|---|---:|---:|---|
| 2026-11-17 | provisional control state machine | `9E8ECA619AF4B74B573D94A3C3654C79F09F6FF7805D3167E0843A4EC8261D20` | `0907681B92F70CFB20440DBF8B3B04E1BF6AA1731E9D77EB09A594670FD9DED8` | 0.5027 | 0.3273 | Pipeline sanity check before Smart Tagging dead-ball inference; 17 public possessions make this score too noisy for tuning. |
| 2026-11-17 | conservative Smart Tagging restarts | `9E8ECA619AF4B74B573D94A3C3654C79F09F6FF7805D3167E0843A4EC8261D20` | `236AC8BA13B25FA4B1B6350AB0DA0549848FF4B0A5E8C0C330D34B18DA9C4EEC` | 0.6191 | 0.4000 | Detected 56/59 set-piece restarts; public check improved without threshold tuning, but its bootstrap noise band remains much larger than the score difference. |
| 2026-11-17 | full-window restart search with capped dead interval | `9E8ECA619AF4B74B573D94A3C3654C79F09F6FF7805D3167E0843A4EC8261D20` | `7B0E832079926B7D7F86C09A41D358330668BEB9925E6983D8336638A3DF12D8` | 0.6191 | 0.4000 | Detected 58/59 restarts by searching full long tag windows while capping dead-ball output at 15 seconds. The only miss has no ball observations. Public score is unchanged because the recovered events are outside that window. |

For each future run, record the Git commit, configuration files, aggregate structural diagnostics,
and hashes. Never add the feature CSV, transition log, or other frame-level derived data.
