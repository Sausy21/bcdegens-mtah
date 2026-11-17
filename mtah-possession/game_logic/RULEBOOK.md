# Possession rulebook

The football rules that turn tracking data into per-frame possession evidence. Each rule states the
football principle, the reason for it, how it is implemented, and the tests that check it. Section 8
records the results of every label-free check run on the match.

| File | Content |
|---|---|
| `game_logic/rules.py` | Implementation. Pure functions: feature table in, per-frame arrays out. |
| `game_logic/test_rules.py` | Synthetic football scenes (no match data). Run: `python game_logic/test_rules.py` |
| `game_logic/make_submission.py` | One command from the provided data to the submission CSV. Run: `python game_logic/make_submission.py --out TEAM_NAME.csv` |
| `game_logic/requirements.txt` | Pinned versions used for the submission (Python 3.11.9, numpy 2.4.6, pandas 3.1.1) |
| `game_logic/audit_scenes.py` | Audit of the scenes: mutation, threshold sensitivity, noise (section 11). Run: `python game_logic/audit_scenes.py` |

The diagnostics in section 8 were computed in a local notebook that is not part of the repository,
because its outputs are derived from the match data. The results are reproduced here as aggregates.

---

## 1. Scope and definitions

**Labels.** One label per video frame: `H` home team has the ball, `A` away team has the ball, `D` dead
ball, `X` abstain. Only the two graded halves are scored.

**Possession** (CHALLENGE.md). Possession changes only when the opponent gains controlled possession. A
touch without control, a deflection that does not intercept, and an aerial duel won without subsequent
control are not changes. While the ball is loose or contested, the last team in possession keeps it.
After every stoppage there is a clear possessor.

**Team codes** in the code: `0` = H (first `<Team>` in the tracking XML), `1` = A, `-1` = nobody or unknown.

**Units.** Metres, metres per second, video frames at 25 per second.

---

## 2. Inputs

One row per video frame. `rules.from_feature_table` maps the team feature table (`features.csv`) onto
these names and refuses to run if a required column is missing.

| Rules column | `features.csv` column | Meaning |
|---|---|---|
| `ball_x_m`, `ball_y_m` | same | Ball position (m); blank when the tracker does not see the ball |
| `ball_present` | derived | Ball position given |
| `ball_out` | derived | Ball more than 0.5 m past a touchline or goal line |
| `ball_speed` | `ball_speed_ms` (missing where `ball_jump`) | Ball speed (m/s) |
| `d_H`, `d_A` | `d_nearest_home_m`, `d_nearest_away_m` | Distance from the ball to the nearest home / away player (m) |
| `s_near_H`, `s_near_A` | `s_nearest_home_ms`, `s_nearest_away_ms` | Speed of that nearest player (m/s) |
| `mean_speed` | `mean_player_speed_ms` | Mean speed of all tracked players (m/s) |

Optional columns passed through for diagnostics: `graded`, `n_tracked`, `n_home_within_5m`,
`n_away_within_5m`, `cx_home_m`, `cx_away_m`, `ball_speed_smooth_ms`.

```python
t = rules.from_feature_table(pd.read_csv("data/derived/features.csv"), pitch=(105.3, 68.0))
labels, info = rules.run_pipeline(t, tag_team, graded, pitch=(105.3, 68.0))
```

`tag_team` is the Smart Tagging team per frame (rule 11), or all `-1` to run without it. `info` contains
every intermediate per-frame array and a log of every possession switch.

---

## 3. Pipeline order

`run_pipeline` applies the rules in this order:

1. Clean false ball detections (rule 1).
2. Control evidence (rule 2).
3. Ball events: kicks and receptions (rule 6).
4. Received-ball guard (rule 7).
5. Dead-ball evidence: ball out (rule 8), play stopped (rule 9).
6. Drop stoppages under 4 s; merge play under 1 s (rule 10).
7. Stoppages inside tracking gaps take their edges from events (rule 12).
8. Restart team (rule 14).
9. Per-frame evidence: control, completed passes (rule 5), Smart Tagging in long gaps (rule 13).
10. Decoder: hysteresis, possession held through loose and missing balls, switch placement (rules 3, 4, 15).

`decode_simple` / `run_pipeline` are a reference decoder used to evaluate the evidence end to end. The
evidence functions are designed to be consumed by any decoder.

---

## 4. Rules

Each parameter is listed as `NAME = value` and is defined in `rules.PARAMS`. Its origin is recorded in
`rules.PROVENANCE` (section 6).

### Rule 1. False ball detections are treated as missing
- **Principle.** A ball far beyond the touchline and far from every player is a tracking error, not a ball
  out of play. A real ball out of play stays within a few metres of the line.
- **Implementation.** `clean_ball`: a ball more than `JUNK_MARGIN_M = 5.0` m outside the pitch is set to
  missing; ball speeds computed from it are discarded.
- **Tests.** `false_ball_detection`.

### Rule 2. Control
- **Principle.** A player has the ball when it is within reach, clearly closer to him than to any opponent,
  and moving with him.
- **Implementation.** `ctrl_evidence`. All three conditions:
  - nearest player within `CTRL_RADIUS_M = 2.0` m: reach and a stride, plus about 0.5 m of tracking noise;
  - every opponent at least `CTRL_MARGIN_M = 1.0` m further away. Two noisy distances differ by about
    ±0.7 m by chance, so a smaller lead is not reliable;
  - |ball speed − player speed| below `CTRL_SPEED_MATCH = 2.0` m/s. Without this condition, a slow pass
    rolling past an opponent would count as his control. Unknown speed does not veto.
- If both teams are equally close, nobody has control.
- **Output.** `ctrl_team` per frame, and `ctrl_strength` (0–1) for decoders that weight evidence.
- **Tests.** `dribble`, `ball_flies_past_opponent`, `jittery_duel`.

### Rule 3. Possession is held until the opponent takes control
- **Principle.** Loose, contested and untracked balls stay with the last team in possession
  (CHALLENGE.md).
- **Implementation.** `decode_simple`. The current team keeps possession until the other team shows
  evidence for `TAKEOVER_FRAMES = 10` frames (0.4 s).
- **Tests.** `brief_contested_touch`, `missing_but_playing`.

### Rule 4. A takeover requires control in a row; it is dated from the first touch
- **Principle.** Two separate touches are not control. A tracking dropout of a few frames does not
  interrupt control.
- **Implementation.** If the challenger's evidence breaks off for more than `CTRL_BREAK_FRAMES = 12` frames
  (0.5 s), its count restarts. When a takeover is confirmed, it is backdated to the challenger's first
  touch.
- **Tests.** `pass_to_opponent`, `two_pokes_no_control`, `interception_with_blip`.

### Rule 5. A completed pass is control for the passing team
- **Principle.** CHALLENGE.md lists completing consecutive passes as a sign of control. A pass kicked and
  received by the same team shows that team had the ball for the whole flight.
- **Implementation.** In `run_pipeline`, frames of a completed pass that have no other evidence count as
  evidence for the passing team. A team that wins the ball and passes immediately takes over from the
  winning touch, not from the reception. Switch: `disable=("pass_control",)`.
- **Tests.** `win_and_pass_at_once`.

### Rule 6. Kicks and receptions
- **Principle.** Play is a sequence of ball events. A kick is the ball leaving a player at pass speed. A
  reception is a fast ball slowing to a player's pace and staying with him.
- **Implementation.** `ball_events`.
  - A **flight** is the ball at `PASS_SPEED = 8.0` m/s or more for at least `MIN_FLIGHT_FRAMES = 3`
    frames, mostly without control. Shorter runs are speed jitter; a fast ball under control is a sprint
    dribble.
  - The **kick** is credited to the team in control, or failing that the nearest team within reach, in
    the `KICK_LOOKBACK = 12` frames before the flight.
  - The **reception** is the first frame within `RECEPTION_LOOKAHEAD = 25` frames after the flight from
    which one team controls the ball for `RECEPTION_HOLD = 5` frames. A ball that passes a player without
    slowing is not received.
  - **Pass result:** completed (same team), intercepted (other team), or not received (out, shot, lost
    by the tracker).
  - **Placed kick** (diagnostic only): ball still below `STILL_SPEED = 0.5` m/s for `PLACED_FRAMES = 25`
    frames before the kick. Not used by any rule (section 8.2).
- **Tests.** `event_pass_intercepted`, `event_flies_past`, `event_sprint_dribble`, `event_placed_kick`,
  `event_speed_spike`.

### Rule 7. A received ball is live
- **Principle.** A pass that arrives and is controlled means the ball is in play. The receiver may hold
  it while other players walk. A goalkeeper may hold it for up to 6 s (Law 12).
- **Implementation.** `live_hold_guard`. From a reception, rule 9 cannot fire until the earliest of: the
  next kick; `LIVE_HOLD_FRAMES = 150` frames (6 s); or the receiver's team losing the ball for
  `GUARD_LOSS_FRAMES = 12` frames. A foul normally knocks the ball loose, which ends the guard. The guard
  never overrides rule 8. Switch: `disable=("reception_guard",)`.
- **Tests.** `keeper_holds`, `keeper_holds_6s`, `received_then_hold_6s`, `received_then_fouled`.

### Rule 8. Ball out of play
- **Principle.** The ball is dead from the moment it crosses a line until the restart puts it back into
  play.
- **Implementation.** `dead_evidence` (value 1.0).
  - **Start.** The ball is more than 0.5 m past a line.
  - **End.** The restart: the ball, inside the pitch, is kicked or thrown at `RESTART_SPEED = 8.0` m/s
    or faster, and stays inside for the next second.
  - **Not a restart:** a ball carried back at walking pace, or passed back to a thrower standing outside
    the pitch.
  - **Earliest restart:** `MIN_DEAD_FRAMES = 100` frames (4 s) after the ball went out. The clock starts
    when the ball goes out, so a thrower holding the ball behind the line does not extend the stoppage
    past the throw.
  - **Safety cap:** `MAX_DEAD_FRAMES = 2250` frames (90 s).
- **Tests.** `throw_in`, `quick_throw_in`, `pass_back_to_thrower`, `thrower_holds_ball_out`,
  `ball_jogged_back`.

### Rule 9. Play stopped without the ball going out
- **Principle.** Fouls, offsides and injuries stop play inside the pitch. All players slow down and nobody
  plays the ball.
- **Implementation.** `dead_evidence` (value 0.7). Both conditions:
  - the median speed of all players over `STOP_WINDOW = 100` frames is below `SLOW_TEAM = 1.0` m/s;
  - no ball at `PASS_SPEED` within the surrounding 2 s.
- The ball need not be still; during stoppages it is usually carried, placed or thrown back.
- **Tests.** `free_kick_no_out`, `foul_ball_carried`, `missing_and_stopped`, `slow_buildup_quick_passes`,
  `slow_buildup_long_holds`.

### Rule 10. Minimum stoppage length
- **Principle.** A restart takes time. Even a quick throw-in takes about 4 s from the ball going dead.
  Shorter "stoppages" are pauses in live play.
- **Implementation.** Dead stretches shorter than `MIN_DEAD_FRAMES = 100` frames (4 s) are discarded.
  Live gaps shorter than `MIN_ALIVE_GAP = 25` frames (1 s) between two stoppages are merged into them.

### Rule 11. Smart Tagging team per frame
- **Principle.** Automatic phase tags indicate which team is attacking. Defending-phase tags name the
  team **without** the ball (OVERVIEW.md).
- **Implementation.** `tag_team_from_events`.
  - Every phase event adds one vote to its frames, so overlapping tags never overwrite each other.
  - Codes containing "defen" give their vote to the other team.
  - Set pieces are excluded, because they are not phases. Tags whose team is `N/A`, which includes every
    throw-in, are ignored.
  - Club names are matched to the tracking XML by fuzzy string matching (`side_of_team_name`).

### Rule 12. Stoppages inside tracking gaps take their edges from events
- **Principle.** Player speed shows **whether** play stopped; ball events show **when**. Set pieces and
  throw-ins are usually taken out of a tracking gap, and they put the ball into play at pass speed.
- **Implementation.**
  - A **restart flight** (`restart_flights`) is a flight that starts after the ball was missing for at
    least `RESTART_GAP_MISSING = 50` of the previous `RESTART_GAP_WINDOW = 75` frames, and that leaves the
    ball inside the pitch.
  - If a gap already contains at least 4 s of dead evidence and ends in a restart flight, the stoppage
    runs from the moment the ball vanished to that flight (`snap_gap_stoppages`).
  - This never creates a stoppage, and never clears a frame with the ball out of play. Switch:
    `disable=("gap_snap",)`.
- **Tests.** `stoppage_in_gap`, `open_play_gap_lull`.

### Rule 13. Long tracking gaps use Smart Tagging
- **Principle.** Beyond a few seconds without the ball, holding the last team is a guess. Smart Tagging
  covers time when the ball is not tracked.
- **Implementation.** In gaps of `GAP_TAG_FRAMES = 125` frames (5 s) or more, the Smart Tagging team is
  evidence. It needs `TAG_TAKEOVER_FRAMES = 50` frames (2 s) to switch, because tag phases are coarse.

### Rule 14. Restart team
- **Principle.** After every stoppage one team restarts and has the ball (CHALLENGE.md).
- **Implementation.** `restart_team`: the first team in control within `RESTART_LOOKAHEAD = 125` frames
  (5 s); otherwise the Smart Tagging team.

### Rule 15. An unobserved switch is placed at the middle of the blind stretch
- **Principle.** When the ball is not tracked, the moment it changed hands is unknown. Nothing in the
  game makes the moment the tracker finds the ball again more likely than any other. Switching at
  reappearance places every unobserved turnover at the latest possible moment, which is a systematic
  bias.
- **Decision rule.** With no information on the timing, the midpoint of the blind stretch minimises the
  largest possible timing error (halving it) and the expected error. It is a decision rule, not a value
  fitted to data.
- **Implementation.** `decode_simple`. If the ball was missing for at least `SWITCH_GAP_MIN_FRAMES = 25`
  frames (1 s) between the last evidence for the old team and the first evidence for the new team, the
  switch is placed at the middle of the missing stretch. It is never later than the new team's first
  evidence. Below 1 s the error is small, and the switch stays at the first touch. Switch:
  `disable=("gap_midpoint",)`.
- **Tests.** `turnover_hidden_in_gap`, `turnover_in_gap_with_tag`, `short_blip_does_not_move_switch`.

### Diagnostic: anchors
Not used for labels. `anchors` marks frames where possession is nearly certain:
- **Firm control:** ball within `FIRM_RADIUS_M = 1.0` m, every opponent `FIRM_MARGIN_M = 2.0` m further
  away, moving with the player, for `FIRM_FRAMES = 25` frames.
- **Received passes.**
- **Completed passes.**
- **Ball out:** a dead anchor.

Results are in section 8.3.

- **Tests.** `anchor_dribble`, `anchor_duel_is_not_anchor`, `anchor_short_touch_is_not_anchor`,
  `anchor_completed_pass`, `anchor_intercepted_pass`, `anchor_ball_out`.

---

## 5. Scenario tests

Each scene is a synthetic feature table describing a situation with an unambiguous football answer, run
through the full pipeline. The scenes are written from football knowledge, not from match data. Current
result: 41 pass, 1 known gap, 0 fail. Section 11 audits how much the scenes prove.

| Scene | Situation | Expected result |
|---|---|---|
| dribble | H dribbles 10 s | H throughout |
| pass_to_opponent | H passes, A receives | Switch at A's first touch (±1 s); 2 possessions |
| ball_flies_past_opponent | 15 m/s pass passes 0.8 m from an A player | H throughout |
| brief_contested_touch | A controls the ball for 0.3 s in a duel | No change |
| two_pokes_no_control | A touches the ball twice, 1 s apart | No change |
| interception_with_blip | A intercepts; 3-frame tracking dropout | Switch at the interception |
| win_and_pass_at_once | A wins the ball and passes at once to a teammate | Switch at the winning touch (±0.5 s) |
| turnover_hidden_in_gap | H before an 8 s tracking gap, A after | Switch at the middle of the gap (±1 s) |
| turnover_in_gap_with_tag | 12 s gap; Smart Tagging shows A from 3 s | Switch no later than the tag change |
| short_blip_does_not_move_switch | Visible interception; 0.4 s dropout mid-pass | Switch at A's first touch |
| jittery_duel | Nearest team alternates every 3 frames for 3 s | No flicker |
| throw_in | Ball out, carried back 8 s, thrown in | D from the ball going out to the throw |
| quick_throw_in | Throw-in 4.5 s after the ball goes out | ≥ 80% D, then A |
| pass_back_to_thrower | Ball reappears as a pass to the thrower outside | D until the throw (≥ 90%) |
| thrower_holds_ball_out | Thrower holds the ball behind the line 8 s | D ends at the throw (±1 s) |
| ball_jogged_back | Ball out; jogged back at 5 m/s for 6 s, then thrown | D until the throw (±1 s) |
| foul_ball_carried | Foul; ball carried to the spot at 1.5 m/s for 12 s | ≥ 70% D |
| free_kick_no_out | Foul, ball still 15 s, players walking | ≥ 70% D, then H |
| missing_and_stopped | Ball missing 20 s, players walking | ≥ 80% D, then A |
| missing_but_playing | Ball missing 10 s, players running | H throughout |
| stoppage_in_gap | Ball missing; players move 6 s, then walk 14 s; kick out of the gap | D from the ball vanishing to the kick (±1 s) |
| open_play_gap_lull | Ball missing 9 s including a 3.5 s lull | No D |
| slow_buildup_quick_passes | Slow circulation, 1.5 s holds | No D |
| slow_buildup_long_holds | Slow circulation, 3.5 s holds | No D |
| keeper_holds, keeper_holds_6s | Shot caught; keeper holds 5 s / 6 s | No D; A from the catch |
| received_then_hold_6s | Each centre-back receives and holds 6 s | No D |
| received_then_fouled | Receive, dribble 2 s, fouled, 15 s stoppage | ≥ 70% D |
| false_ball_detection | Ball reported 20 m beyond the touchline for 1 s | Ignored; H throughout |
| event_*, anchor_* | Detector checks | See rule 6 and the anchor diagnostic |
| feature_table_roundtrip | Table in `features.csv` column names | Labels identical to the native table |
| **slow_buildup_5s_hold** (known gap) | Dribble; all players slow; ball held at one player's feet 5 s | Expected live; current rules give 4 s of D |

**Known gap.** A player who dribbles and then holds the ball while all players slow down is
indistinguishable, in the current inputs, from a foul where the fouled player stays on the ball. Its
frequency on the match and candidate fixes are in sections 8.8 and 9.

---

## 6. Parameter provenance

Every parameter and logic choice has an entry in `rules.PROVENANCE` with a kind and a reason.

| Kind | Meaning |
|---|---|
| PRIOR | Set from football or physics, without looking at the match. |
| INFORMED | Set after looking at aggregate statistics of the match; must pass the robustness tests. |
| GT-MEANING | Informed by the public 5 minutes, used only to understand what a label means. |
| REJECTED | Tried, failed its pre-written test, not used. |

INFORMED items: `JUNK_MARGIN_M`, `CTRL_SPEED_MATCH`, `SLOW_TEAM`, `STOP_WINDOW`, restart at pass speed,
gap snap (rule 12).

---

## 7. Validation method

No threshold or version is chosen by scoring against labels.

1. **Scene first.** Each new rule gets a synthetic scene that fails before the change and passes after
   it. Existing scenes must keep passing.
2. **Pre-registration.** Before each run on the match, the expected label-free results are written down.
   Misses are recorded (section 8), not explained away.
3. **Label-free checks.**
   - Possession count and length against football norms.
   - Dead-ball share and stoppage lengths.
   - Both halves alike.
   - Event counts against football norms.
   - Consistency with anchors.
   - Implausible patterns (section 8.8).
4. **Robustness.**
   - Each parameter is changed by ±20%, measuring the change in possessions and dead-ball share.
   - spellF1 of each variant is measured using the current output as the reference ("self-spellF1"). A
     parameter whose variants score below 0.85 is fragile.
5. **Public window.** Scored once, for correctness only, on one version. It is not used to compare
   versions or choose values. The organisers' baselines show that the public window can rank methods in
   the wrong order.

---

## 8. Results on the match (label-free)

All figures are aggregates over the graded frames of the provided match.

### 8.1 Data
| Fact | Value |
|---|---|
| Ball tracked | 58.7% of graded frames |
| False ball detections (> 5 m outside the pitch) | 0.79% of frames |
| Tracking gaps | long rather than frequent: median ≈ 11 s |
| Mean player speed | 1.18 m/s while the ball is missing, 2.17 m/s while it is tracked |
| Team feature table vs independent build | identical on all shared columns (rows, graded mask, ball presence, distances, nearest team); ball speeds correlate at r = 0.999 at lag 0/+1 |

### 8.2 Ball events (rule 6)
| Measure | Half 1 | Half 2 | Pre-registered range | Result |
|---|---|---|---|---|
| Kicks | 359 | 368 | 150–500 | Pass |
| Received share of kicks | 60% | 60% | 50–85% | Pass |
| Completed (same-team) share of received | 80% | 82% | 65–90% (≈ 50% = random) | Pass |
| Home share of kicks | 68% | 58% | near home share of possession (≈ 62%) | Pass |
| Median flight time | 1.6 s | 1.5 s | 0.5–1.5 s | Missed narrowly: the 8 m/s threshold excludes soft short passes |
| Placed kicks | 0 | 0 | 5–35 | Failed: the tracker rarely reports a stationary ball. Not used. |

### 8.3 Anchors
| Measure | Result | Pre-registered | |
|---|---|---|---|
| Coverage | 17.2% / 17.1% per half | 20–40% | Missed (sparse) |
| H/A conflicts | 0.00% | < 1% | Pass |
| Median unknown stretch between anchors | 3.2 s | < 5 s | Pass |
| Unknown time in stretches over 5 s / 15 s | 84% / 59% | — | The difficulty is concentrated in long stretches |
| v7 agreement on team-anchor frames | 93.4%; 6.3% D during stoppages; 0.2% other team | > 90% | Pass |
| Ball-out frames labelled D | 100% | — | Pass |

### 8.4 Stoppage signatures (rules 9 and 12)
| Test | Result | Decision |
|---|---|---|
| Collective slowdown ("whistle"): count of slowdowns per half at ratio thresholds 0.5 → 0.6 | 41 → 99 and 43 → 79 (+141%, +84%); pre-registered plateau < 25% change; 11–17% near a ball out | Rejected |
| Flights out of a ≥ 2 s tracking gap | 41 / 52 per half (expected 25–80); median player speed before them 1.12 m/s vs 2.10 m/s for other flights (required ≥ 0.3 m/s apart) | Adopted as the restart flight (rule 12) |
| Rule 12 effect | Dead-ball share of missing-ball frames 59.5% → 67.0%; median stoppage 13.5 s → 17.0 s; `SLOW_TEAM` ±20% swing in dead-ball share 12.3 → 12.1 points (target < 8: missed) | Kept for its edges; does not reduce `SLOW_TEAM` sensitivity |
| Rule 7 effect | Guard active on 10.8% of frames; changed 0.03% of frames from D to live | Kept; rare on this match |

### 8.5 Output shape by version
| Version | Possessions | Mean (s) | Under 2 s | Dead ball | Notes |
|---|---|---|---|---|---|
| `baseline_smooth50` (published floor) | 594 | 7.2 | 31.3% | 32.3% | Reference |
| v2 | 282 | 15.1 | 6.7% | 32.9% | |
| v3 | 282 | 15.1 | 6.7% | 32.8% | |
| v4 | 275 | 15.2 | 6.9% | 33.8% | |
| v5 | 270 | 15.5 | 7.0% | 33.8% | Pre-registered: 255–275 possessions, flicker not rising. Pass. |
| v6 | 273 | 14.7 | 8.8% | 36.5% | Anchors "other team" 4.9% → 0.2%, ball-out frames live 16.4% → 0.0% (pass); flicker ≤ 7.5% and dead ball ±2 points (missed) |
| v7 | 273 | 14.7 | 8.8% | 36.5% | 12 switches moved, all earlier, median 2.3 s; 0.54% of frames changed, none involving D. Direction as pre-registered; magnitude smaller than predicted (1–4%). |

v7 stoppages: 94, median 17.5 s, 90th percentile 51.9 s. Missing-ball frames labelled D: 71.4%; tracked-ball
frames labelled D: 11.9%. Home share of live play: 61.4%.

### 8.6 Robustness
Each parameter changed by −20% / +20%:

| Parameter | Possessions | Dead ball | Self-spellF1 |
|---|---|---|---|
| `CTRL_RADIUS_M` | 252 / 277 | 33.9 / 33.8% | 0.932 / 0.978 |
| `CTRL_MARGIN_M` | 270 / 256 | 33.8 / 33.8% | 1.000 / 0.962 |
| `CTRL_SPEED_MATCH` | 255 / 280 | 33.8 / 33.8% | 0.949 / 0.972 |
| `TAKEOVER_FRAMES` | 281 / 247 | 33.8 / 33.8% | 0.970 / 0.946 |
| `CTRL_BREAK_FRAMES` | 268 / 272 | 33.8 / 33.8% | 0.995 / 0.993 |
| `GAP_TAG_FRAMES` | 270 / 267 | 33.8 / 33.8% | 1.000 / 0.983 |
| `SLOW_TEAM` | 269 / 271 | 27.0 / 39.1% | 0.940 / 0.913 |
| `STOP_WINDOW` | 270 / 268 | 33.9 / 34.0% | 0.996 / 0.996 |
| `PASS_SPEED` | 271 / 273 | 33.8 / 34.0% | 0.990 / 0.977 |
| `RESTART_SPEED` | 271 / 269 | 33.5 / 34.1% | 0.985 / 1.000 |
| `MIN_DEAD_FRAMES` | 273 / 268 | 34.0 / 33.5% | 0.984 / 0.989 |

Possessions and dead ball measured on v5; self-spellF1 on v6. All parameters changed together at random
within ±20% (10 draws): self-spellF1 minimum 0.825, median 0.896, maximum 0.942. Pre-registered: every
parameter ≥ 0.85 (`SLOW_TEAM` ≥ 0.75), joint median ≥ 0.80. Pass. `SLOW_TEAM` moves the dead-ball share
by 12 points but the possession structure much less.

### 8.7 Where rule 15 applies
| Measure | Result |
|---|---|
| H↔A switches | 182 |
| Switches with ≥ 1 s of missing ball between the two teams' evidence (moved by rule 15) | 12 (new team's first evidence: Smart Tagging 8, control 3, completed pass 1) |
| Switches not moved | 170 (first evidence: control 145, Smart Tagging 13, completed pass 12) |

Tracking gaps of ≥ 1 s by what happens across them:

| Class | Gaps | Share of missing time | Median length |
|---|---|---|---|
| Stoppage in or at the edge of the gap | 89 | 81.4% | 16.3 s |
| Same team throughout | 25 | 8.0% | 6.8 s |
| Switch, Smart Tagging evidence inside | 23 | 10.5% | 10.7 s |
| Switch, no evidence inside | 0 | 0% | — |

Conclusions:
- Most possession switches are observed in visible play.
- Most missing-ball time belongs to stoppages, not to hidden turnovers.
- Every switch inside a tracking gap already has Smart Tagging evidence.

Of 96 unknown stretches between anchors of opposite teams, 20 contain ≥ 1 s of missing ball. Rule 15
corrects a real bias but applies rarely on this match.

### 8.8 Implausible patterns (v7)
**Possessions under 2 s:** 24 (8.8%).

| Context | Count | Median | Ball missing |
|---|---|---|---|
| Other team → short → other team (flicker) | 11 | 1.56 s | 0% |
| Other team → short → D | 5 | 1.28 s | 0% |
| D → short → D | 5 | 1.64 s | 20% |
| D → short → other team | 3 | 1.44 s | 26% |

**Stoppages under 6 s:** 14.

| Cause | Count | Median | Same team either side | Contact in the previous second | Ball missing |
|---|---|---|---|---|---|
| Players stopped (rule 9) | 5 | 5.3 s | 60% | 0% | 40% |
| Ball out (rule 8) | 9 | 4.0 s | 44% | 0% | 0% |

**Stoppages over 60 s:** 6.

| Cause | Lengths | Ball missing |
|---|---|---|
| Ball out | 67, 69, 119, 143 s | 91% |
| Players stopped | 65, 143 s | 98% |

The graded period before the second-half kick-off (117 s) is a correct long stoppage.

Reading:
- The remaining flicker is short visible-ball takeovers that are reversed within 2 s. They may be real
  (ball won and lost again).
- Short "players stopped" stoppages with the same team on both sides and no contact match the known gap.
  At most 3 occur on the match.
- The contact signature (both teams within 1.5 m) was never observed before a short stoppage. Its
  usefulness is unverified.

### 8.9 Public window (scored once, v6)
| Submission | spellF1 | macroF1 | Possessions (truth: 17) | Mean length (truth: 14.2 s) |
|---|---|---|---|---|
| v6 | 0.459 | 0.793 | 17 | 15.4 s |
| `baseline_smooth125` | 0.424 | — | 16 | 18.5 s |
| `baseline_smooth50` (published floor) | 0.326 | — | 26 | 10.8 s |

The published floor scores 0.2433 on the full match. With 17 possessions, the window's noise is about
±0.1. The window is better tracked than the full match, so the full-match score is expected to be
lower.

### 8.10 Independent cross-check, null tests, halves (v7, run once)
**Set-piece tags.** No rule uses them. Share of tag starts with a D label within ±5 s:

| Type | Tags | Near D |
|---|---|---|
| Corner kick | 12 | 100% |
| Free kick | 5 | 100% |
| Goal kick | 10 | 60% |
| Kick-off after a goal | 4 | 100% |
| Kick-off (start) | 1 | 100% |
| Throw-in | 26 | 81% |
| **All set pieces** | **58** | **84%** |
| Random graded frames (200 draws) | 58 | 49% (95th percentile 60%) |

Pre-registered: overall ≥ random + 25 points (**pass**, +35). Every type with ≥ 5 tags above the random
95th percentile (**miss**: goal kicks at 60%, equal to the 95th percentile). The likely cause is that a
shot or cross leaving over the goal line is often not tracked, and the goal kick is taken while players
are still moving, so neither rule 8 nor rule 9 fires. Not fixed: changing a rule because of this check
would end its independence.

**Null tests.** Self-spellF1 against the v7 output:

| Input broken | Self-spellF1 | Possessions | Dead ball |
|---|---|---|---|
| Player speed shifted by 60 s | 0.865 | 272 | 28.2% |
| Smart Tagging shifted by 5 minutes | 0.837 | 279 | 36.5% |
| Smart Tagging removed | 0.891 | 248 | 36.5% |

- **Player speed.** Pre-registered: below 0.825 (**miss**). Breaking the signal moves the dead-ball share
  by 8 points. It changes the possession structure more than any single ±20% parameter change (lowest
  0.913), but less than the worst joint change (0.825). The possession structure is driven mainly by
  control evidence and the ball-out rule; player speed mainly decides how much time is dead. This is
  consistent with section 8.6.
- **Smart Tagging.** Pre-registered: shifted no higher than removed (**pass**: 0.837 vs 0.891). Wrong
  tags do more harm than no tags, so the tags carry real information. Removing them lowers possessions
  from 273 to 248: they create the switches inside long gaps.

**Halves.** Consistency only; not a hold-out.

| Half | Possessions | Per minute | Mean | Under 2 s | Dead ball | Home share of live play |
|---|---|---|---|---|---|---|
| 1 | 133 | 2.51 | 15.6 s | 6.0% | 34.3% | 60.7% |
| 2 | 140 | 2.65 | 13.8 s | 11.4% | 38.6% | 62.1% |

Pre-registered: possessions per minute, mean length and dead-ball share within ±15% (**pass**: +5.6%,
−11.5%, +12.5%).
- The graded pre-kick-off period of the second half (117 s, all D) accounts for about 3.7 points of the
  dead-ball difference.
- Short possessions are almost twice as frequent in the second half. This was not pre-registered and is
  unexplained.

---

## 9. Known limitations and open work

- **Rule 9 depends on a speed level with no natural cut.** It moves the dead-ball share strongly but
  possession structure weakly (section 8.6).
- **Known gap** (section 5). It affects at most 3 stoppages on the match (section 8.8). Candidate
  discriminators, not implemented:
  1. Contact before the stop (a foul requires contact). Not observed in the data so far.
  2. Duration over 6 s while one player holds the ball (Law 12 limit for keepers; free-kick set-ups take
     longer).
  3. Opponents retreating 9.15 m (Law 13).
  4. Goalkeeper identity. Requires a role column.
- **Goal kicks.** Only 60% of tagged goal kicks have a predicted stoppage nearby (section 8.10). A
  candidate rule from Law 16: a restart flight that starts inside a goal area is a goal kick, so the ball
  was dead from the moment it vanished. Implementing it would end the independence of the set-piece
  check.
- **Short possessions in the second half** are almost twice as frequent as in the first. Unexplained.
- **Long tracking gaps.** 81% of missing-ball time is in or around stoppages (section 8.7). Stoppage edges
  inside gaps (rule 12) therefore matter more than switch timing. Team shape (the team in possession
  advances and spreads; the defending team retreats and compacts) is a candidate source of evidence that
  does not need the ball. It requires team centroid columns.

---

## 10. Version history

| Version | Change |
|---|---|
| v2 | Rules moved to `rules.py`; minimum stoppage 2 s → 4 s (keeper and long-hold scenes). |
| v3 | Ball events and the received-ball guard (rules 6–7). |
| v4 | Stoppage edges inside tracking gaps (rule 12). |
| v5 | Takeover requires control in a row (rule 4). |
| v6 | Completed pass = control (rule 5); a restart must put the ball into play; ball-out clock starts when the ball goes out (rule 8). |
| v7 | Unobserved switches at the middle of the blind stretch (rule 15). |

---

## 11. Methodology audit

What the evidence for this rulebook does and does not show.

### 11.1 Scenario tests (`audit_scenes.py`, no match data)
**Mutation.** Each rule switched off in turn:

| Rule switched off | Scenes that fail |
|---|---|
| False ball detections (rule 1) | 1 |
| Speed match (rule 2) | 2 |
| Ball-out latch (rule 8) | 4 |
| Play stopped (rule 9) | 5 |
| Restart at pass speed (rule 8) | 1 (`ball_jogged_back`) |
| Ball need not be still (rule 9) | 1 (`foul_ball_carried`) |
| Smart Tagging (rule 13) | 1 |
| Received-ball guard (rule 7) | 2 |
| Gap edges (rule 12) | 1 |
| Completed pass = control (rule 5) | 1 |
| Gap midpoint (rule 15) | 1 |

Every rule that can be switched off is caught by at least one scene. The last two logic choices (restart
at pass speed, ball need not be still) had no scene until this audit; `ball_jogged_back` and
`foul_ball_carried` were added for them. No scene fails under every ablation.

**Threshold sensitivity.** Each parameter changed by ±20%: 26 of 30 parameters flip no scene. Most scenes
therefore test the logic, not the exact values. Four parameters sit at a scene boundary:

| Parameter | −20% | +20% | Reading |
|---|---|---|---|
| `TAKEOVER_FRAMES` | `brief_contested_touch` fails | — | The scene's 0.3 s touch is just under the 0.4 s takeover. |
| `MIN_DEAD_FRAMES` | `open_play_gap_lull` fails | known gap passes | The 4 s value sits between two scenes. It was raised from 2 s because scenes failed, so it is partly set by the scenes. |
| `SLOW_TEAM` | known gap passes | — | The known gap is a threshold effect: 0.9 m/s walking vs a 1.0 m/s cut. |
| `MIN_FLIGHT_FRAMES` | `event_speed_spike` fails | — | The scene's 2-frame spike is just under the 3-frame minimum. |

**Noise.** Each scene run 20 times with tracking-like noise: distances ±0.3 m, player speed ±0.5 m/s,
ball speed ±1.0 m/s, team speed ±0.15 m/s (1 sd), and 3% of frames dropped. The noise level is an
assumption, not a measurement.

| Group | Pass rate |
|---|---|
| All 30 label-level scenes (possession and dead ball, including the known gap) | 95–100% |
| `event_sprint_dribble`, `event_speed_spike`, `anchor_duel_is_not_anchor`, `anchor_short_touch_is_not_anchor`, `anchor_completed_pass` | 100% |
| `event_pass_intercepted`, `event_flies_past` | 70%, 45% |
| `anchor_intercepted_pass`, `anchor_ball_out` | 60%, 65% |
| `anchor_dribble`, `event_placed_kick` | 0%, 0% |

Reading:
- **The label-level rules are robust to noise.**
- **The event and anchor detectors are not.** They need exact runs of consecutive frames, and one noisy
  frame breaks a run.
- Consequences for section 8:
  - The anchor coverage of 17% is depressed by noise.
  - Anchor agreement is a weak check.
  - The placed-kick detector cannot work on noisy speeds, which matches its zero count on the match.

### 11.2 Sources of contamination
| Source | Effect | Mitigation | Remaining risk |
|---|---|---|---|
| Rules and scenes written by the same people | Scenes may encode the rules' own assumptions | Mutation, sensitivity and noise audit (11.1); scenes written before the code | Scenes cannot show a rule is *correct* football, only that it does what it says |
| Idealised scenes | Clean inputs flatter detectors | Noise audit | Detector-level scenes fail under noise (11.1) |
| `MIN_DEAD_FRAMES` set from scene failures | Labelled PRIOR, but partly fitted to scenes | Disclosed here and in `PROVENANCE` | Sits between two scenes |
| Iterating on full-match aggregates (v2–v7) | Each change was kept after seeing match statistics; repeated looking can fit the match | Pre-registered expectations; misses recorded; changes justified by football and scenes | Not eliminated: the choice of *which* problems to work on came from the match |
| Pre-registered ranges written by the same people, sometimes from the previous version's numbers | Ranges can be easy to meet | Misses kept in the record (8.2, 8.3, 8.4, 8.5) | Moderate |
| Shape norms partly from the public window | The mean possession of 14.2 s and the count of 17 come from the public truth | Used only as a reference for possession length | Small: also consistent with general football norms |
| Anchors built from the same detectors as the rules | Agreement with anchors is partly by construction (e.g. completed passes in rule 5 and in the anchors) | Treated as a consistency check, not as validation | Not independent evidence |
| Self-spellF1 robustness | Measures stability, not correctness | Interpreted only as sensitivity | — |
| Public window | Its result (v6) preceded the work on switch timing (v7) | Scored once; not used to choose values or versions; rule 15 uses no information from it | The priority of the switch-timing work was influenced by it |
| Smart Tagging used in rules 11, 13 and 14 | Phase tags are not an independent check | Set-piece tags are excluded from every rule and kept as the only independent check (section 8.10) | — |
| No hold-out data | Both halves were seen before every decision | Per-half consistency only (section 8.10) | A true hold-out does not exist for this match |

### 11.3 What the evidence supports
- The rules implement their stated football principles. Each is caught by a scene; the label-level
  rules are robust to noise.
- Predicted stoppages coincide with independently tagged set pieces: 84% within ±5 s against 49% for
  random moments. This is strong for corners, free kicks, kick-offs and throw-ins, and weak for goal
  kicks (section 8.10).
- The Smart Tagging input carries real information (null test, section 8.10).
- The output has a football-like structure (possession count and length, stoppage lengths) and is stable
  under ±20% parameter changes.
- One public-window score above the published floor.

### 11.4 What the evidence does not support
- How accurate the output is on the full match. No label-free check measures accuracy.
- That any single parameter value is optimal.
- That the event and anchor detectors are reliable on noisy data.
