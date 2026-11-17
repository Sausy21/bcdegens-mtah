# Josh's possession rules: handoff to Sausthava

**What this is:** the football logic that turns the feature table into per-frame evidence (who controls
the ball, is the ball dead, who restarts), how each rule was chosen and tested, and the five columns
`features.csv` needs so the rules can run on it.

**Files**
| File | What it is |
|---|---|
| `game_logic/rules.py` | the rules. Pure functions: feature table in, per-frame arrays out. Reads no files. |
| `game_logic/test_rules.py` | 40 synthetic football scenes, no match data. `python game_logic/test_rules.py` |
| `game_logic/explore.ipynb` | exploration and label-free diagnostics (sections 11â€“16) |
| this file | the guide |

**Status:** 39 scenes pass and 1 is a known gap (section 6). On the shared columns, your `features.csv`
matches the notebook's own table exactly (same rows, `graded`, ball presence, distances, nearest team;
ball speeds line up at lag 0/+1 as expected from forward vs centred differences). The only thing missing
is three player-speed columns.

---

## 1. What you need to add to `features.csv`

| Column | Meaning | Needed for |
|---|---|---|
| `s_nearest_home_ms`, `s_nearest_away_ms` | speed (m/s) of the player counted in `d_nearest_home_m` / `d_nearest_away_m`; blank when the ball is missing | **required.** Control test: the ball moves *with* the player (dribble), not past him (pass) |
| `mean_player_speed_ms` | mean speed of all tracked players | **required.** "Play has stopped" |
| `cx_home_m`, `cx_away_m` | mean x (m) of each team's tracked players | optional. Which goal each team defends per half, kick-off shape (future rules) |

Speeds are computed **from positions**, like your ball speed, not from the file's speed channel (you
found that channel unreliable for the ball). A centred difference over Â±2 frames (0.16 s) stops the
frame-to-frame jitter of point tracking from looking like running. Anything over 12 m/s is a tracking
jump (top sprint speed is ~10 m/s).

### Proposed function (drop-in for `explore_data.py`)

```python
import warnings

PLAYER_JUMP_MS = 12.0  # faster than any sprint (~10 m/s): a tracking jump, not a player


def rule_columns(x, y, team, dist_h, dist_a):
    """The five extra columns Josh's rules need, as a dict of arrays (one value per video frame).

    x, y           normalised player positions (frames x slots), NaN = untracked   (read_atd_arrays)
    team           0 home / 1 away per slot                                         (slot_teams)
    dist_h, dist_a ball-to-player distances (m), NaN for the other team's slots     (main())
    """
    px, py = x * PITCH[0], y * PITCH[1]
    speed = np.full(px.shape, np.nan, dtype=np.float32)          # m/s, centred over +/-2 frames
    speed[2:-2] = np.hypot(px[4:] - px[:-4], py[4:] - py[:-4]) * FPS / 4.0
    speed[speed > PLAYER_JUMP_MS] = np.nan
    rows = np.arange(len(px))

    def nearest_speed(dist_team):
        has = np.isfinite(dist_team).any(axis=1)                 # False when the ball is missing
        idx = np.argmin(np.where(np.isfinite(dist_team), dist_team, np.inf), axis=1)
        return np.where(has, speed[rows, idx], np.nan)

    with warnings.catch_warnings():                              # frames with nobody tracked -> NaN, quietly
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "s_nearest_home_ms": nearest_speed(dist_h),
            "s_nearest_away_ms": nearest_speed(dist_a),
            "mean_player_speed_ms": np.nanmean(speed, axis=1),
            "cx_home_m": np.nanmean(np.where(team[None, :] == 0, px, np.nan), axis=1),
            "cx_away_m": np.nanmean(np.where(team[None, :] == 1, px, np.nan), axis=1),
        }
```

In `main()`, after `d_home` / `d_away`, add the columns to `feats`:

```python
    feats = feats.assign(**rule_columns(x, y, team, dist_h, dist_a))   # or merge into the dict
```

**One question:** does a `<Player>` element in the ATD XML carry a position/role or shirt number? If it
does, a column `nearest_home_is_gk` / `nearest_away_is_gk` (the nearest player is that team's keeper)
would let the rules tell a keeper holding the ball (live) from a free kick being set up (dead) almost
directly. Only the attribute names are needed, no values.

### Running the rules on your table

```python
import numpy as np, pandas as pd
import rules                                   # game_logic/rules.py

ft = pd.read_csv("data/derived/features.csv")
t = rules.from_feature_table(ft, pitch=(105.3, 68.0))     # renames, derives ball_present / ball_out,
                                                          # ball_jump -> speed missing
graded = ft["graded"].to_numpy(bool)
labels, info = rules.run_pipeline(t, tag_team, graded, (105.3, 68.0))
```

- `from_feature_table` stops with a clear error if a required column is missing. It never guesses.
- `tag_team` is the Smart Tagging team per frame (`rules.tag_team_from_events`, section 4.4). Pass
  `np.full(len(t), -1)` to run without it.
- `info` holds every intermediate per-frame array: `ctrl`, `strength`, `dead_evidence`, `dead`,
  `restart`, `events`, `anchors`, `live_guard`, `restart_flights`.
- `run_pipeline`'s decoder (`decode_simple`) is a **stand-in** so the evidence can be judged end to end.
  Rishit's decoder is meant to consume the evidence functions directly.
- Test `scene_feature_table_roundtrip` checks that a table in your column names gives exactly the same
  labels as the notebook's table.

---

## 2. Conventions

- One row per **video frame** = submission row. Metres and m/s. `FPS = 25`; windows are in frames.
- Team codes: `0` = H (home, first `<Team>` in the XML), `1` = A, `-1` = nobody/unknown.
- Every threshold is in `rules.PARAMS` and has an entry in `rules.PROVENANCE`:
  - **PRIOR:** football or physics, chosen without looking at the match
  - **INFORMED:** chosen after looking at aggregates of this match, so it must pass the robustness tests
  - **GT-MEANING:** informed by the public 5 minutes, used only to understand what a label means
  - **REJECTED:** tried, failed its pre-written test, not used
- Nothing refers to a specific frame or moment of the match.

---

## 3. The rules, in the order a match unfolds

Each rule gives its football meaning, the test as implemented, the parameters (`PARAMS` name = value),
and the scenes in `test_rules.py` that cover it.

### 3.1 Clean the tracker's mistakes (`clean_ball`)
- **Football:** a ball 20â€“30 m beyond the touchline, far from every player, is a false detection. A real
  ball out of play stays within a few metres of the line.
- **Test:** ball more than `JUNK_MARGIN_M = 5.0` m outside the pitch â†’ treated as missing (not "out").
- **Scenes:** `false_ball_detection`.

### 3.2 Control: who is on the ball (`ctrl_evidence` â†’ `ctrl_team`, `ctrl_strength`)
- **Football:** a player has the ball when it is within reach, clearly closer to him than to any
  opponent, and moving *with* him.
- **Test:** all three of:
  - nearest player within `CTRL_RADIUS_M = 2.0` m (reach + stride + ~0.5 m tracking noise)
  - every opponent at least `CTRL_MARGIN_M = 1.0` m further away. Two noisy distances differ by about
    Â±0.7 m by chance, so a 0.5 m lead is often noise.
  - |ball speed âˆ’ that player's speed| < `CTRL_SPEED_MATCH = 2.0` m/s (INFORMED). Without this, a slow
    pass rolling past an opponent would count as his control. Unknown speed does not veto.
- Both teams close â†’ 50/50, nobody gets credit.
- **Scenes:** `dribble`, `ball_flies_past_opponent`, `jittery_duel`.

### 3.3 Keeping and losing the ball (decoder hysteresis, `decode_simple`)
- **Loose, contested or untracked ball â†’ the last team keeps it** (CHALLENGE.md definition).
- **Takeover needs control in a row:** the other team must control the ball for
  `TAKEOVER_FRAMES = 10` frames (0.4 s) without the current team touching it.
  - If their control breaks off for more than `CTRL_BREAK_FRAMES = 12` frames (0.5 s), the count starts
    again: two separate pokes are not control.
  - A tracking blip of a few frames doesn't break it.
- **The switch is backdated** to the new team's first touch.
- **A switch nobody saw** goes to the middle of the blind stretch. If the ball was missing for at least
  `SWITCH_GAP_MIN_FRAMES = 25` frames (1 s) between the last evidence for the old team and the first for
  the new one, the switch is placed at the middle of that missing stretch. That halves the worst-case
  timing error, and it is a decision rule, not a fitted value. It is never later than the new team's
  first evidence. Switch off with `disable=("gap_midpoint",)`.
- **A completed pass is control** (CHALLENGE.md: "a team completes â€¦ consecutive passes"). Frames of a pass
  kicked and received by the same team count as that team's control. So a team that wins the ball and
  passes at once takes over from the winning touch, not at the reception. Switch off with
  `disable=("pass_control",)`.
- **Scenes:** `pass_to_opponent`, `brief_contested_touch`, `two_pokes_no_control`,
  `interception_with_blip`, `win_and_pass_at_once`, `jittery_duel`, `turnover_hidden_in_gap`,
  `turnover_in_gap_with_tag`, `short_blip_does_not_move_switch`.

### 3.4 Ball events: kicks and receptions (`ball_events`)
- **Kick:** the ball reaches pass speed (`PASS_SPEED = 8.0` m/s) as it leaves a player.
  - A flight must stay fast for at least `MIN_FLIGHT_FRAMES = 3` frames; shorter is speed jitter.
  - A fast ball that stays at a player's feet is a sprint dribble, not a kick.
  - The kick is credited to the team in control (or nearest within reach) in the `KICK_LOOKBACK = 12`
    frames before it.
- **Reception:** within `RECEPTION_LOOKAHEAD = 25` frames after a flight, the first frame from which one
  team controls the ball for `RECEPTION_HOLD = 5` frames. A ball that flies past without slowing to the
  player's speed is not received.
- **Pass result:** received by the kicker's team (completed), by the opponent (intercepted), or not
  received (out, shot, lost by the tracker).
- **Placed kick:** a ball still (< `STILL_SPEED = 0.5` m/s) for `PLACED_FRAMES = 25` frames before the
  kick. **Not used by any rule:** the tracker almost never reports a still ball, so it never fires.
- **Scenes:** `event_pass_intercepted`, `event_flies_past`, `event_sprint_dribble`, `event_placed_kick`,
  `event_speed_spike`.

### 3.5 A received ball is live (`live_hold_guard`)
- **Football:** a pass that arrives and is controlled means the ball is in play. The receiver may hold it
  while everyone else walks: a keeper for up to 6 s (Law 12), or an unpressed centre-back.
- **Test:** from a reception, "play has stopped" (3.6b) can't fire until the earliest of:
  - the next kick
  - `LIVE_HOLD_FRAMES = 150` frames (6 s)
  - the receiver's team losing control for `GUARD_LOSS_FRAMES = 12` frames. A foul usually knocks the
    ball loose, so real stoppages still get through.
- It never overrides a ball out of play.
- **Scenes:** `keeper_holds`, `keeper_holds_6s`, `received_then_hold_6s`, `received_then_fouled`.

### 3.6 Dead ball (`dead_evidence`, 0â€“1)
**a) The ball went out (value 1.0, the out-latch).**
- **Starts:** the ball is more than 0.5 m past a line.
- **Ends:** at the **restart**: the ball, inside the pitch, is kicked or thrown at `RESTART_SPEED = 8.0`
  m/s **and stays inside for the next second**. A pass back to a thrower standing outside is not a
  restart.
- **The restart can't come before** `MIN_DEAD_FRAMES = 100` frames (4 s) after the ball **went** out. The
  clock runs from the ball going out, not from the last frame it was seen out, so a thrower holding the
  ball behind the line doesn't push the end past the throw.
- **Safety cap:** `MAX_DEAD_FRAMES = 2250` frames (90 s).
- A ball carried back at walking pace doesn't end the stoppage (INFORMED).

**b) Play has stopped (value 0.7)**, for fouls, offsides and injuries:
- the median speed of all players over `STOP_WINDOW = 100` frames is below `SLOW_TEAM = 1.0` m/s
  (INFORMED), **and**
- no ball at pass speed in the surrounding 2 s.
- The ball need not be still: during a stoppage it is usually carried, placed or thrown back
  (GT-MEANING).

**c) Tidying** (in `run_pipeline`):
- dead stretches shorter than `MIN_DEAD_FRAMES` (4 s) are dropped: even a quick throw-in takes that long
- live gaps shorter than `MIN_ALIVE_GAP = 25` frames (1 s) between two stoppages are merged

- **Scenes:** `throw_in`, `quick_throw_in`, `pass_back_to_thrower`, `thrower_holds_ball_out`,
  `free_kick_no_out`, `slow_buildup_quick_passes`, `slow_buildup_long_holds`.

### 3.7 When the tracker loses the ball (`restart_flights`, `snap_gap_stoppages`, Smart Tagging)
- **Players moving normally:** play goes on, and the last team keeps the ball.
- **Players slow:** a stoppage via 3.6b.
- **Stoppage edges come from events** (INFORMED): player speed decides *whether* a gap is a stoppage, ball
  events decide *when*.
  - A **restart flight** is a ball at pass speed after the ball was missing for at least
    `RESTART_GAP_MISSING = 50` of the previous `RESTART_GAP_WINDOW = 75` frames, which stays in the pitch.
  - If a gap already has at least 4 s of dead evidence and ends in a restart flight, the stoppage runs
    from the ball vanishing to that flight.
  - This never creates a stoppage, and never clears a frame with the ball out.
- **Long gaps use Smart Tagging:** in gaps of `GAP_TAG_FRAMES = 125` frames (5 s) or more, the Smart
  Tagging team counts as evidence, needing `TAG_TAKEOVER_FRAMES = 50` frames to switch, because tag
  phases are coarse.
- **Scenes:** `missing_and_stopped`, `missing_but_playing`, `stoppage_in_gap`, `open_play_gap_lull`.

### 3.8 Who restarts (`restart_team`)
- The first team in control within `RESTART_LOOKAHEAD = 125` frames (5 s) after play restarts; otherwise
  the Smart Tagging team.
- CHALLENGE.md: after every stoppage there is a clear possessor.

### 3.9 Smart Tagging team per frame (`tag_team_from_events`, `side_of_team_name`)
- Every phase event adds one vote to its frames, so overlapping tags never overwrite each other.
- "Defending â€¦" / "Defensive transition" tags name the team **without** the ball, so their vote goes to
  the other side (OVERVIEW.md).
- Set pieces are skipped (they aren't phases). Tags with team `N/A` (including every throw-in) are
  ignored.
- Club names are matched fuzzily to the XML team names, so nothing is hard-coded.

### 3.10 Anchors (`anchors`, diagnostic, not used for labels yet)
Moments we're nearly sure of, for bridging the unknown stretches between them:
- **Firm control:** ball within `FIRM_RADIUS_M = 1.0` m, every opponent `FIRM_MARGIN_M = 2.0` m further,
  moving with the player, for `FIRM_FRAMES = 25` frames (1 s).
- **Received pass:** a reception, for `RECEPTION_HOLD` frames.
- **Completed pass:** the whole flight.
- **Ball out:** a dead anchor.

H/A anchors never conflicted on this match. The current output agrees with nearly all team-anchor
frames, and the remainder is mostly D during a stoppage (a player carrying the ball back), which is correct.
- **Scenes:** `anchor_dribble`, `anchor_duel_is_not_anchor`, `anchor_short_touch_is_not_anchor`,
  `anchor_completed_pass`, `anchor_intercepted_pass`, `anchor_ball_out`.

---

## 4. Scenario catalogue (`test_rules.py`)

Each scene is a small synthetic table describing a situation a coach would agree on, run through the
real pipeline. Because the scenes come from football, not from the match, passing them is evidence
that a rule encodes football rather than this game.

| Scene | Situation | Expected |
|---|---|---|
| dribble | H dribbles 10 s | H throughout |
| pass_to_opponent | H pass, 0.6 s flight, A receives | switch at A's first touch (Â±1 s), 2 possessions |
| ball_flies_past_opponent | 15 m/s pass passes 0.8 m from an A player | stays H |
| brief_contested_touch | A has 0.3 s of control in a duel | no change |
| two_pokes_no_control | A pokes twice, 1 s apart, ball loose between | no change |
| interception_with_blip | A intercepts, 3-frame tracking blip after | switch at the interception |
| win_and_pass_at_once | A wins the ball, passes at once to a teammate | switch at the winning touch (Â±0.5 s) |
| jittery_duel | nearest team flips every 3 frames for 3 s | no flicker |
| turnover_hidden_in_gap | H before an 8 s tracking gap, A after it | switch at the middle of the gap (Â±1 s) |
| turnover_in_gap_with_tag | same, 12 s gap, Smart Tagging says A from 3 s in | switch no later than the tag change |
| short_blip_does_not_move_switch | visible interception with a 0.4 s blip mid-pass | switch at A's first touch |
| throw_in | ball out, carried back 8 s, thrown in | D from out (Â±0.2 s) to throw (Â±1 s), then A |
| quick_throw_in | throw-in 4.5 s after the ball went out | â‰¥ 80% D, then A |
| pass_back_to_thrower | ball reappears as a pass to the thrower outside | D until the throw (â‰¥ 90%) |
| thrower_holds_ball_out | thrower holds ball behind the line 8 s, throws | D ends at the throw (Â±1 s) |
| free_kick_no_out | foul, ball still 15 s, everyone walking | â‰¥ 70% D, then H |
| missing_and_stopped | ball missing 20 s, players walking | â‰¥ 80% D, then A |
| missing_but_playing | ball missing 10 s, players running | H throughout, no D |
| stoppage_in_gap | ball missing; players move 6 s then walk 14 s; kick out of the gap | D from the ball vanishing to the kick (Â±1 s) |
| open_play_gap_lull | ball missing 9 s with a 3.5 s lull, then a pass | no D |
| slow_buildup_quick_passes | slow circulation, 1.5 s holds | no D |
| slow_buildup_long_holds | slow circulation, 3.5 s holds | no D |
| keeper_holds / keeper_holds_6s | shot caught, keeper holds 5 s / 6 s | no D, A from the catch |
| received_then_hold_6s | each centre-back receives and holds 6 s | no D |
| received_then_fouled | receive, dribble 2 s, fouled, 15 s stoppage | â‰¥ 70% D |
| false_ball_detection | ball "seen" 20 m beyond the touchline for 1 s | ignored, H throughout |
| event_* | kick/reception detector checks | see 3.4 |
| anchor_* | anchor checks | see 3.10 |
| feature_table_roundtrip | a table in your column names | identical labels |
| **slow_buildup_5s_hold (KNOWN GAP)** | dribble, then everyone slows with the ball at one player's feet for 5 s | should be live; we give 4 s of D |

The known gap is honest: "dribble, everyone slows, ball stays at his feet" is also exactly what a foul
looks like. A keeper flag or similar identity information is the most likely fix.

---

## 5. Tried and rejected
- **Placed-ball restarts** ("ball still, then kicked"): the tracker almost never reports a still ball,
  so the detector found no set pieces. Kept only as a diagnostic.
- **"The whistle"** (every player slowing at once, as a stoppage start): the count of slowdowns kept
  rising with the threshold (no natural cut) and rarely matched balls going out. `REJECTED` in the
  ledger.

## 6. Known limits
- **`SLOW_TEAM` is the shakiest rule.** Player speed has no natural break between "walking in play" and
  "stopped", so a Â±20% nudge moves the dead-ball share by several points. The number of possessions
  barely moves. Notebook section 15 measures how much it moves spellF1 itself.
- The dribble-then-stand case (section 4's known gap).
- The rules see events only when the ball is tracked. Long tracking gaps are still mostly "hold the last
  team, or Smart Tagging".

## 7. How we keep it from overfitting the match
1. **Provenance ledger:** every value has a written football reason and a kind (section 2).
2. **Scenes first:** every new rule gets a synthetic scene that fails before the change and passes after.
3. **Pre-registration:** before each run on the match, the expected result is written in the notebook.
   Misses are recorded, not explained away.
4. **Label-free checks only** (shape, stability, both halves alike, event counts against football norms).
   The public 5 minutes are scored once, for correctness only (notebook section 16), never to pick a
   threshold or a version.

## 8. Checklist for you
- [ ] Add `rule_columns(...)` to `explore_data.py` and regenerate `features.csv` (keep it out of git).
- [ ] Answer the `<Player>` role/shirt-number question.
- [ ] Run `python game_logic/test_rules.py`: expect 39 PASS, 1 KNOWN GAP, 0 FAIL.
- [ ] Run notebook section 14 with the new `features.csv`: the label agreement should be â‰¥ 95%. Any gap
      comes only from position-based vs file-based player speeds.
