"""Josh's football logic: per-frame evidence functions for the possession decoder.

Pure functions: a feature table in (pandas, one row per video frame, metres and m/s), per-frame
arrays out. Nothing here reads files, and nothing refers to a specific frame or a specific match:
every threshold is in PARAMS with a football reason in PROVENANCE. The guide, written for the team,
is game_logic/RULES_HANDOFF.md; the football scenario tests are game_logic/test_rules.py.

Quick start with the team's features.csv:
    t = from_feature_table(pd.read_csv("data/derived/features.csv"), pitch=(105.3, 68.0))
    labels, info = run_pipeline(t, tag_team, graded, pitch=(105.3, 68.0))

Columns the rules use (from_feature_table maps features.csv onto these names):
    ball_present, ball_x_m, ball_y_m, ball_speed   ball seen?, position (m), speed (m/s)
    d_H, d_A                                        distance ball -> nearest home / away player (m)
    s_near_H, s_near_A                              speed of that nearest player (m/s)
    mean_speed                                      mean speed of all tracked players (m/s)
    ball_out                                        ball more than 0.5 m past a line

Contents, in the order a match unfolds (team codes: 0 H, 1 A, -1 nobody / unknown):
    clean_ball          false ball detections far off the pitch -> missing
    ctrl_evidence       who controls the ball -> ctrl_team, ctrl_strength (0-1)
    ball_events         kicks and receptions (team, completed or intercepted pass)
    live_hold_guard     a player who just received the ball and still holds it -> play is live
    dead_evidence       0-1, how sure we are the ball is dead (ball out, or play stopped)
    restart_flights     kicks/throws out of a tracking gap that put the ball into play
    snap_gap_stoppages  stoppages hidden in a tracking gap get their football edges
    restart_team        who restarts after a stoppage
    anchors             moments we are nearly sure of (diagnostic, not used for labels)
    from_feature_table  the team's features.csv -> the columns above
    tag_team_from_events  Smart Tagging team per frame
    decode_simple, run_pipeline  a stand-in decoder so the evidence can be judged end to end;
                        the team's real decoder is Rishit's
"""

import difflib
import unicodedata

import numpy as np
import pandas as pd

FPS = 25                     # video frame rate of the provided match (games.py); windows below are in frames
HOME, AWAY, NOBODY = 0, 1, -1
LABELS = np.array(["H", "A", "D", "X"])

# Every threshold the rules use. Units in the name: _M metres, _SPEED m/s, _FRAMES video frames (25 = 1 s).
# The football reason for each value is in PROVENANCE below.
PARAMS = dict(
    # cleaning (clean_ball)
    JUNK_MARGIN_M=5.0,              # ball this far outside the pitch = false detection, not "out"
    STILL_SPEED=0.5,                # m/s; slower = ball effectively stationary
    # control (ctrl_evidence) and changing hands (decode_simple)
    CTRL_RADIUS_M=2.0,              # nearest player within reach
    CTRL_MARGIN_M=1.0,              # ... and clearly closer than any opponent
    CTRL_SPEED_MATCH=2.0,           # m/s; |ball speed - player speed| below this = ball moves with him
    TAKEOVER_FRAMES=10,             # 0.4 s of the other team's control before the ball changes hands
    CTRL_BREAK_FRAMES=int(0.5 * FPS),   # their control broken longer than this -> the count restarts
    TAG_TAKEOVER_FRAMES=50,         # 2 s of Smart Tagging evidence to switch inside a long gap
    GAP_TAG_FRAMES=125,             # gaps of 5 s or more use the Smart Tagging team
    SWITCH_GAP_MIN_FRAMES=FPS,      # a switch hidden in >= 1 s of missing ball goes to the gap's middle
    # dead ball (dead_evidence) and tidying (run_pipeline)
    SLOW_TEAM=1.0,                  # m/s; median player speed below this = play has stopped
    STOP_WINDOW=4 * FPS,            # window for that median
    PASS_SPEED=8.0,                 # m/s; a ball this fast is being played
    RESTART_SPEED=8.0,              # m/s; a restart is a kick or throw at least this fast
    MIN_DEAD_FRAMES=100,            # 4 s: shortest stoppage, and earliest restart after the ball goes out
    MIN_ALIVE_GAP=25,               # 1 s: shorter play between two stoppages is merged into them
    MAX_DEAD_FRAMES=90 * FPS,       # 90 s safety cap on a ball-out stoppage
    RESTART_LOOKAHEAD=125,          # restart team = first team in control within 5 s
    # ball events (ball_events / live_hold_guard)
    MIN_FLIGHT_FRAMES=3,            # a kicked ball stays at pass speed at least this long
    KICK_LOOKBACK=int(0.5 * FPS),   # kicker = who had the ball in the 0.5 s before the flight
    RECEPTION_LOOKAHEAD=FPS,        # reception within 1 s after the flight ends
    RECEPTION_HOLD=5,               # 0.2 s of control = a receiving touch
    PLACED_FRAMES=FPS,              # ball still 1 s before a kick = placed (diagnostic only)
    LIVE_HOLD_FRAMES=6 * FPS,       # a received ball held up to 6 s (Law 12) keeps play live
    GUARD_LOSS_FRAMES=int(0.5 * FPS),   # receiver losing the ball this long ends that
    # restarts out of tracking gaps (restart_flights / snap_gap_stoppages)
    RESTART_GAP_WINDOW=3 * FPS,     # restart flight = ball missing >= RESTART_GAP_MISSING
    RESTART_GAP_MISSING=2 * FPS,    #   of the RESTART_GAP_WINDOW frames before it
    # anchors (diagnostic)
    FIRM_RADIUS_M=1.0,              # ball at the feet
    FIRM_MARGIN_M=2.0,              # every opponent this much further away
    FIRM_FRAMES=FPS,                # for at least 1 s
)

# Where every value and logic choice came from. Kinds:
#   PRIOR       football or physics, would be chosen without looking at this match
#   INFORMED    chosen after looking at aggregates of this match (needs the robustness tests)
#   GT-MEANING  informed by the public 5 minutes, used only to understand what a label means
#   REJECTED    tried, failed its pre-written test, not used (kept so nobody re-adds it unknowingly)
PROVENANCE = {
    "JUNK_MARGIN_M":       ("INFORMED", "ball detections up to 33 m outside the pitch were seen; a real ball out of play stays within a few metres of the line"),
    "STILL_SPEED":         ("PRIOR", "below 0.5 m/s a ball is effectively stationary"),
    "CTRL_RADIUS_M":       ("PRIOR", "a player's reach plus a stride, plus ~0.5 m point-tracking noise; team proposal"),
    "CTRL_MARGIN_M":       ("PRIOR", "two noisy distances differ by about +/-0.7 m by chance, so 'clearly closer' needs ~1 m"),
    "CTRL_SPEED_MATCH":    ("INFORMED", "ball moving with the player; the ball-minus-player speed histogram of this match peaks within +/-1 m/s"),
    "TAKEOVER_FRAMES":     ("PRIOR", "team proposal: 0.4 s of control before the ball changes hands"),
    "CTRL_BREAK_FRAMES":   ("PRIOR", "team proposal: takeover needs control 'in a row'; a break over 0.5 s means two separate "
                                     "touches (CHALLENGE.md: a touch without control is no change), shorter = tracking blip"),
    "SWITCH_GAP_MIN_FRAMES": ("PRIOR", "when nobody saw the ball change hands, the middle of the blind stretch halves the "
                                       "worst-case timing error (a decision rule, not a fit); under 1 s the error is small "
                                       "anyway, so short blips keep the switch at the first touch"),
    "TAG_TAKEOVER_FRAMES":("PRIOR", "Smart Tagging phases are coarse; 2 s before they can switch the team"),
    "GAP_TAG_FRAMES":      ("PRIOR", "beyond ~5 s without the ball, 'hold the last team' is a guess; use the tagged team"),
    "SLOW_TEAM":           ("INFORMED", "play stopped: match median player speed 1.5 m/s, ~0.75 m/s around tagged corners (this match)"),
    "STOP_WINDOW":         ("INFORMED", "stoppages last several seconds, lulls are brief; added after SLOW_TEAM proved fragile"),
    "PASS_SPEED":          ("PRIOR", "a short pass is ~8 m/s or faster; no ball that fast for 2 s = nobody is playing it"),
    "RESTART_SPEED":       ("PRIOR", "a restart is a kick or throw at pass speed, not a ball carried back at jogging pace"),
    "MIN_DEAD_FRAMES":     ("PRIOR", "even a quick throw-in or free kick takes ~4 s from the ball going dead; shorter 'stoppages' "
                                     "are live holds (keeper, centre-back). Raised from 2 s after the test_rules.py "
                                     "keeper / long-hold scenes failed at 2 s and passed at 4 s"),
    "MIN_ALIVE_GAP":       ("PRIOR", "under 1 s of play between two stoppages is noise"),
    "MAX_DEAD_FRAMES":     ("PRIOR", "safety cap: no stoppage after the ball goes out lasts more than 90 s"),
    "RESTART_LOOKAHEAD":   ("PRIOR", "the restarting team touches the ball within ~5 s of the restart"),
    "MIN_FLIGHT_FRAMES":   ("PRIOR", "a kicked ball stays at pass speed for more than 2 frames (0.08 s); shorter = speed jitter of point tracking"),
    "KICK_LOOKBACK":       ("PRIOR", "foot contact is instantaneous; the centred speed estimate puts the first fast frame within ~0.5 s of the last touch"),
    "RECEPTION_LOOKAHEAD": ("PRIOR", "a first touch takes the ball from pass speed to the receiver's speed within ~1 s"),
    "RECEPTION_HOLD":      ("PRIOR", "0.2 s of the ball moving with the player = a controlling touch, not a brush or deflection"),
    "PLACED_FRAMES":       ("PRIOR", "a set-piece ball sits still for at least ~1 s before it is kicked"),
    "LIVE_HOLD_FRAMES":    ("PRIOR", "Law 12: a keeper may hold the ball 6 s; outfield players rarely hold longer unpressed"),
    "GUARD_LOSS_FRAMES":   ("PRIOR", "control lapses shorter than 0.5 s are tracking noise or a touch, not losing the ball"),
    "logic: reception guard":  ("PRIOR", "a pass arriving and being controlled means play is live: CHALLENGE.md control proxies; "
                                         "added because the keeper 6 s scene failed (test_rules.py)"),
    "RESTART_GAP_WINDOW":  ("PRIOR", "restart flight = ball at pass speed after being missing >= 2 s of the previous 3 s: "
                                     "restarts are taken out of tracking gaps. Definition written before measuring; "
                                     "passed the pre-registered 12c test (players ~1 m/s slower before these flights)"),
    "RESTART_GAP_MISSING": ("PRIOR", "see RESTART_GAP_WINDOW"),
    "logic: gap snap":     ("INFORMED", "player speed decides WHETHER a gap is a stoppage, events decide WHEN: it runs from "
                                        "the ball vanishing to the restart flight. Added because SLOW_TEAM moved D% by 12 "
                                        "points under a 20% nudge (11b) and speed has no natural cut (5b, 12c)"),
    "logic: completed pass = control": ("PRIOR", "CHALLENGE.md proxy 'a team completes consecutive passes'; added when "
                                                 "13b showed v5 contradicting 7.9% of completed-pass frames"),
    "logic: restart stays in play":    ("PRIOR", "a restart puts the ball into play; a pass back to a thrower outside is not one. "
                                                 "Bug found via 13b (16% of out frames labelled live), reproduced in test_rules.py"),
    "logic: whistle detector":("REJECTED", "a sudden collective slowdown as the stoppage start: the 12c drop count kept "
                                            "climbing with the threshold (no plateau) and rarely matched balls going out"),
    "FIRM_RADIUS_M":       ("PRIOR", "anchor: ball at the player's feet, half the everyday control radius"),
    "FIRM_MARGIN_M":       ("PRIOR", "anchor: opponent clearly further, ~3x the +/-0.7 m chance difference of two noisy distances"),
    "FIRM_FRAMES":         ("PRIOR", "anchor: CHALLENGE.md's control proxy is 'keeps the ball for a few seconds'; 1 s is the cautious end"),
    "logic: tag votes":         ("PRIOR", "overlapping Smart Tagging phases each vote; none overwrites another"),
    "logic: flip defensive tags":  ("PRIOR", "OVERVIEW.md: Defending/Defensive Transition name the team without the ball"),
    "logic: ball need not be still": ("GT-MEANING", "public 5 min: the ball is 'still' in only 1.5% of D frames; it is carried, placed or thrown back"),
    "logic: restart = pass speed":   ("INFORMED", "dead stretches were too short for football; carried balls were ending them"),
}

ABLATIONS = ("junk_filter", "speed_match", "out_latch", "stopped", "pass_restart", "ball_still_required", "tags",
             "reception_guard", "gap_snap", "pass_control", "gap_midpoint")


# -- helpers -------------------------------------------------------------------------------------

def runs(flag):
    """[(start, length)] of consecutive True runs."""
    flag = np.asarray(flag, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], flag, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def clean_mask(mask, min_on, min_off):
    """Close gaps shorter than min_off between True runs, then drop True runs shorter than min_on."""
    m = np.asarray(mask, dtype=bool).copy()
    for s, l in runs(~m):
        if l < min_off and s > 0 and s + l < len(m):
            m[s:s + l] = True
    for s, l in runs(m):
        if l < min_on:
            m[s:s + l] = False
    return m


# -- evidence ------------------------------------------------------------------------------------

def clean_ball(t, pitch, p=PARAMS):
    """A ball far outside the pitch is a false detection, not a ball out of play: make it missing.

    Returns (cleaned copy of t, boolean mask of the frames changed). Idempotent.
    """
    length, width = pitch
    t = t.copy()
    bx, by = t["ball_x_m"], t["ball_y_m"]
    m = p["JUNK_MARGIN_M"]
    junk = t["ball_present"] & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    next_to_junk = junk.shift(1, fill_value=False) | junk.shift(-1, fill_value=False)
    t.loc[junk, ["ball_x_m", "ball_y_m", "ball_speed", "d_H", "d_A", "s_near_H", "s_near_A"]] = np.nan
    t.loc[junk, ["ball_present", "ball_out"]] = False
    t.loc[next_to_junk, "ball_speed"] = np.nan        # computed from the junk position
    return t, junk.to_numpy()


def ctrl_evidence(t, p=PARAMS, disable=()):
    """Who controls the ball this frame -> (ctrl_team, ctrl_strength).

    Football: a player has the ball when it is (1) within reach, (2) clearly closer to him than to
    any opponent, and (3) moving with him, i.e. a dribble or a held ball, not a pass flying past.
    """
    dH, dA = t["d_H"].to_numpy(float), t["d_A"].to_numpy(float)
    team = np.where(dA < dH, AWAY, HOME)
    team = np.where(np.isnan(dH) & np.isfinite(dA), AWAY, team)
    d_own = np.where(team == HOME, dH, dA)
    d_opp = np.where(team == HOME, dA, dH)
    d_opp = np.where(np.isfinite(d_opp), d_opp, np.inf)          # no opponent tracked: nobody contests
    s_own = np.where(team == HOME, t["s_near_H"].to_numpy(float), t["s_near_A"].to_numpy(float))
    diff = np.abs(t["ball_speed"].to_numpy(float) - s_own)
    moving_with = ~(diff >= p["CTRL_SPEED_MATCH"])               # speed unknown (NaN): don't veto
    if "speed_match" in disable:
        moving_with = np.ones(len(t), dtype=bool)
    ok = (t["ball_present"].to_numpy(bool) & (d_own <= p["CTRL_RADIUS_M"])
          & (d_opp - d_own >= p["CTRL_MARGIN_M"]) & moving_with)
    strength = (np.clip(1 - d_own / p["CTRL_RADIUS_M"], 0, 1)
                * np.clip((d_opp - d_own) / (2 * p["CTRL_MARGIN_M"]), 0, 1))
    return np.where(ok, team, NOBODY).astype(np.int8), np.where(ok, strength, 0.0)


def dead_evidence(t, p=PARAMS, disable=(), live_guard=None):
    """How sure we are the ball is dead, 0-1.

    1.0  the ball went out and the restart has not been taken yet. Restart = the ball, inside the
         pitch, is kicked or thrown at RESTART_SPEED and stays inside for the next second, at least
         MIN_DEAD_FRAMES after the ball WENT out. A ball carried back or placed at jogging pace, or
         passed back to a thrower standing outside, does not end the stoppage.
    0.7  play has stopped: all players slow over STOP_WINDOW, and nobody has passed the ball in the
         last ~2 s (ball missing, or never faster than PASS_SPEED). The ball need not be still:
         during a stoppage it is usually being carried, placed or thrown back.
         Not while live_guard is True (a player who has just received a pass is holding it: see
         live_hold_guard). The guard never overrides a ball that went out.
    """
    n = len(t)
    present = t["ball_present"].to_numpy(bool)
    out = t["ball_out"].to_numpy(bool)
    bs = t["ball_speed"].to_numpy(float)

    restart_speed = 3.0 if "pass_restart" in disable else p["RESTART_SPEED"]
    # A restart puts the ball INTO play: it must stay inside the pitch for the next second
    # (a pass back to a thrower standing outside is not a restart).
    out_cum = np.concatenate(([0], np.cumsum(out)))
    out_next_s = (out_cum[np.minimum(np.arange(n) + FPS, n)] - out_cum[:n]) > 0
    restart = present & ~out & (bs >= restart_speed) & ~out_next_s
    latch = np.zeros(n, dtype=bool)
    if "out_latch" not in disable:
        # 'since' counts from the moment the ball WENT out (the latch switching on), not from the last
        # frame it was seen out: a thrower holding the ball behind the line until the throw must not
        # push the restart 4 s past the throw.
        on, since = False, 0
        for i, (o, r) in enumerate(zip(out.tolist(), restart.tolist())):
            if o:
                if not on:
                    on, since = True, 0
            elif on and ((r and since >= p["MIN_DEAD_FRAMES"]) or since >= p["MAX_DEAD_FRAMES"]):
                on = False
            latch[i] = on
            since += 1

    stopped = np.zeros(n, dtype=bool)
    if "stopped" not in disable:
        w = int(p["STOP_WINDOW"])
        team_speed = (pd.Series(t["mean_speed"].to_numpy(float))
                      .rolling(w, center=True, min_periods=max(1, w // 2)).median().to_numpy())
        if "ball_still_required" in disable:
            quiet_ball = ~present | (bs < p["STILL_SPEED"])
        else:
            fastest = pd.Series(bs).rolling(2 * FPS, center=True, min_periods=1).max().to_numpy()
            quiet_ball = ~(fastest >= p["PASS_SPEED"])               # NaN (ball missing) -> no pass
        stopped = (team_speed < p["SLOW_TEAM"]) & quiet_ball
        if live_guard is not None:
            stopped &= ~np.asarray(live_guard, dtype=bool)
    return np.maximum(latch * 1.0, stopped * 0.7)


def ball_events(t, ctrl, p=PARAMS):
    """Kicks and receptions, from the ball's speed and who is close to it.

    Football:
    - A flight is the ball at pass speed (>= PASS_SPEED) for at least MIN_FLIGHT_FRAMES, not moving
      with a player (a fast ball under control most of the time is a sprint dribble, not a pass).
    - Kick: the start of a flight, credited to the team that controlled the ball (or, failing that,
      was nearest within reach) in the KICK_LOOKBACK before it. 'placed' marks a kick of a ball that
      sat still for PLACED_FRAMES just before: the set-piece signature.
    - Reception: within RECEPTION_LOOKAHEAD after a flight, the first frame from which one team
      controls the ball (ctrl_evidence) for RECEPTION_HOLD frames. A ball that flies past a player
      without slowing to his speed is not received (ctrl_evidence's speed test).

    Returns a dict of per-frame arrays; events sit on single frames, NOBODY / False elsewhere:
        kick        team of the kicker at the kick frame
        placed      the kick was of a placed ball
        reception   team of the receiver at the reception frame
        pass_result at the kick frame: 1 received by the kicker's team, 0 by the opponent,
                    -1 not received (out, shot, lost by the tracker) or kicker unknown
        flight_frames at the kick frame: frames from the kick to the reception (0 if none)
    """
    n = len(t)
    ctrl = np.asarray(ctrl)
    present = t["ball_present"].to_numpy(bool)
    bs = t["ball_speed"].to_numpy(float)
    dH, dA = t["d_H"].to_numpy(float), t["d_A"].to_numpy(float)
    near_team = np.where(dA < dH, AWAY, HOME)
    with np.errstate(invalid="ignore"):
        near_ok = np.fmin(dH, dA) <= p["CTRL_RADIUS_M"]               # NaN -> False
        fast = bs >= p["PASS_SPEED"]
        still = present & (bs < p["STILL_SPEED"])
    still_len_at_end = np.zeros(n, dtype=np.int32)                    # length of a still run, at its last frame
    for s, l in runs(still):
        still_len_at_end[s + l - 1] = l

    kick = np.full(n, NOBODY, dtype=np.int8)
    placed = np.zeros(n, dtype=bool)
    reception = np.full(n, NOBODY, dtype=np.int8)
    pass_result = np.full(n, -1, dtype=np.int8)
    flight_frames = np.zeros(n, dtype=np.int32)
    hold, look = int(p["RECEPTION_HOLD"]), int(p["RECEPTION_LOOKAHEAD"])
    for s, l in runs(fast):
        if l < p["MIN_FLIGHT_FRAMES"] or np.mean(ctrl[s:s + l] >= 0) >= 0.5:
            continue
        lo = max(0, s - int(p["KICK_LOOKBACK"]))
        had = np.flatnonzero(ctrl[lo:s] >= 0)
        close = np.flatnonzero(near_ok[lo:s])
        if len(had):
            kick[s] = ctrl[lo + had[-1]]
        elif len(close):
            kick[s] = near_team[lo + close[-1]]
        placed[s] = kick[s] >= 0 and still_len_at_end[lo:s].max(initial=0) >= p["PLACED_FRAMES"]

        e = s + l
        for f in range(e, min(n - hold + 1, e + look)):
            c = ctrl[f]
            if c >= 0 and np.all(ctrl[f:f + hold] == c):
                reception[f] = c
                flight_frames[s] = f - s
                if kick[s] >= 0:
                    pass_result[s] = int(c == kick[s])
                break
    return dict(kick=kick, placed=placed, reception=reception, pass_result=pass_result,
                flight_frames=flight_frames)


def live_hold_guard(events, ctrl, p=PARAMS):
    """Frames where play must be live because a player has just received the ball and still has it.

    Football: a pass that arrives and is controlled means the ball is in play. The receiver may then
    hold it (a keeper up to 6 s by Law 12, an unpressed centre-back a few seconds) while everyone
    else walks, which looks like a stoppage on player speed alone. The guard runs from the reception
    until the earliest of: the next kick, LIVE_HOLD_FRAMES, or the receiver's team losing control for
    GUARD_LOSS_FRAMES (a foul usually knocks the ball loose, which ends the guard).
    """
    n = len(ctrl)
    ctrl = np.asarray(ctrl)
    guard = np.zeros(n, dtype=bool)
    kicks = np.flatnonzero(events["kick"] >= 0)
    for f in np.flatnonzero(events["reception"] >= 0):
        team = events["reception"][f]
        end = min(n, f + int(p["LIVE_HOLD_FRAMES"]))
        k = np.searchsorted(kicks, f + 1)
        if k < len(kicks):
            end = min(end, kicks[k])
        lost, i = 0, f
        while i < end:
            lost = 0 if ctrl[i] == team else lost + 1
            if lost >= p["GUARD_LOSS_FRAMES"]:
                i -= lost - 1
                break
            i += 1
        guard[f:i] = True
    return guard


def restart_flights(t, p=PARAMS):
    """Frames where a flight (ball at PASS_SPEED for MIN_FLIGHT_FRAMES) starts after the ball was missing
    for at least RESTART_GAP_MISSING of the previous RESTART_GAP_WINDOW frames.

    Football: set pieces and throw-ins are usually taken out of a tracking gap (the ball was out, in
    a player's hands, or hidden in a crowd), and they start with a kick or throw at pass speed. No
    kicker is required: nobody was visible near a ball that was not visible.
    A restart puts the ball INTO play: the ball must be inside the pitch when the fast run ends and
    for the second after (a throw-in may start behind the line; a ball passed back to the thrower
    standing outside is not a restart).
    """
    n = len(t)
    out_ = t["ball_out"].to_numpy(bool)
    present = t["ball_present"].to_numpy(bool)
    missing = ~present
    with np.errstate(invalid="ignore"):
        fast = t["ball_speed"].to_numpy(float) >= p["PASS_SPEED"]
    w = int(p["RESTART_GAP_WINDOW"])
    miss_before = np.concatenate(([0], np.cumsum(missing)))           # miss_before[i] = missing frames in [0, i)
    out = np.zeros(n, dtype=bool)
    for s, l in runs(fast):
        if l >= p["MIN_FLIGHT_FRAMES"] and miss_before[s] - miss_before[max(0, s - w)] >= p["RESTART_GAP_MISSING"]:
            e = s + l
            if e < n and present[e] and not out_[e:e + FPS].any():
                out[s] = True
    return out


def snap_gap_stoppages(dead, t, flights, p=PARAMS):
    """Give a stoppage hidden in a tracking gap its football edges.

    If the dead-ball evidence already covers at least MIN_DEAD_FRAMES of a gap, and the gap ends in a
    restart flight, the stoppage runs from the moment the ball vanished to that restart (capped at
    MAX_DEAD_FRAMES), and any dead frames carried past the restart by the slow-moving speed average
    are cleared. Gaps without dead evidence are left alone: this never creates a stoppage, it only
    moves the edges of one that player speed already found. A frame with the ball out of play is
    never cleared: the clearing stops there.
    Returns (new dead mask, number of gaps snapped).
    """
    out_ = t["ball_out"].to_numpy(bool)
    dead = np.asarray(dead, dtype=bool).copy()
    n = len(dead)
    reach = int(p["RESTART_GAP_WINDOW"] - p["RESTART_GAP_MISSING"])   # a restart flight can start this late after the gap
    flight_idx = np.flatnonzero(flights)
    snapped = 0
    for s, l in runs(~t["ball_present"].to_numpy(bool)):
        e = s + l
        if dead[s:e].sum() < p["MIN_DEAD_FRAMES"]:
            continue
        k = np.searchsorted(flight_idx, e)
        if k == len(flight_idx) or flight_idx[k] > e + reach:
            continue
        f = flight_idx[k]
        dead[max(s, f - int(p["MAX_DEAD_FRAMES"])):f] = True
        i = f
        while i < n and dead[i] and not out_[i]:
            dead[i] = False
            i += 1
        snapped += 1
    return dead, snapped


def anchors(t, events, p=PARAMS):
    """Moments we are nearly sure of, for bridging the unknown stretches between them.

    Team anchors (who has the ball):
      firm      ball within FIRM_RADIUS_M of a player, every opponent FIRM_MARGIN_M further, moving
                with him (CTRL_SPEED_MATCH), for at least FIRM_FRAMES in a row
      reception a received pass, from the reception for RECEPTION_HOLD frames
      pass      a completed pass (kicked and received by the same team): that team had the ball
                for the whole flight
    Dead anchor:
      out       ball clearly past a line (after clean_ball)
    Frames where H and A anchors disagree are dropped from both (counted in 'conflict').
    Returns dict: team (int8, NOBODY where none), dead (bool), and the masks firm_H, firm_A,
    recv, pass_, out, conflict for diagnostics.
    """
    n = len(t)
    dH, dA = t["d_H"].to_numpy(float), t["d_A"].to_numpy(float)
    sH, sA = t["s_near_H"].to_numpy(float), t["s_near_A"].to_numpy(float)
    bs = t["ball_speed"].to_numpy(float)
    present = t["ball_present"].to_numpy(bool)
    with np.errstate(invalid="ignore"):
        def firm(d_own, d_opp, s_own):
            d_opp = np.where(np.isfinite(d_opp), d_opp, np.inf)
            ok = (present & (d_own <= p["FIRM_RADIUS_M"]) & (d_opp - d_own >= p["FIRM_MARGIN_M"])
                  & ~(np.abs(bs - s_own) >= p["CTRL_SPEED_MATCH"]))
            out = np.zeros(n, dtype=bool)
            for s, l in runs(ok):
                if l >= p["FIRM_FRAMES"]:
                    out[s:s + l] = True
            return out
        firm_H, firm_A = firm(dH, dA, sH), firm(dA, dH, sA)

    vote = np.zeros((2, n), dtype=bool)
    vote[HOME] |= firm_H
    vote[AWAY] |= firm_A
    recv = np.zeros(n, dtype=bool)
    hold = int(p["RECEPTION_HOLD"])
    for f in np.flatnonzero(events["reception"] >= 0):
        vote[events["reception"][f], f:f + hold] = True
        recv[f:f + hold] = True
    pass_ = np.zeros(n, dtype=bool)
    for s in np.flatnonzero(events["pass_result"] == 1):
        e = s + int(events["flight_frames"][s])
        vote[events["kick"][s], s:e] = True
        pass_[s:e] = True

    conflict = vote[HOME] & vote[AWAY]
    team = np.where(vote[HOME] & ~conflict, HOME, np.where(vote[AWAY] & ~conflict, AWAY, NOBODY)).astype(np.int8)
    out = t["ball_out"].to_numpy(bool)
    team[out] = NOBODY                      # a ball past the line belongs to nobody
    return dict(team=team, dead=out, firm_H=firm_H, firm_A=firm_A, recv=recv, pass_=pass_, out=out,
                conflict=conflict)


def restart_team(ctrl_team, tag_team, p=PARAMS):
    """Team that takes the restart: the next team in control within RESTART_LOOKAHEAD frames,
    otherwise the Smart Tagging team."""
    nxt = (pd.Series(np.where(ctrl_team >= 0, ctrl_team, np.nan))
           .bfill(limit=int(p["RESTART_LOOKAHEAD"])).fillna(NOBODY).to_numpy().astype(np.int8))
    return np.where(nxt >= 0, nxt, tag_team).astype(np.int8)


# -- the team's feature table (Sausthava's features.csv) -----------------------------------------

# his column -> the name the rules use. Speeds and centroids are the columns requested in
# game_logic/RULES_HANDOFF.md (section 1); the rules cannot run without the three speed columns.
FT_COLUMNS = {
    "ball_x_m": "ball_x_m", "ball_y_m": "ball_y_m", "ball_speed_ms": "ball_speed",
    "d_nearest_home_m": "d_H", "d_nearest_away_m": "d_A",
    "s_nearest_home_ms": "s_near_H", "s_nearest_away_ms": "s_near_A", "mean_player_speed_ms": "mean_speed",
}
FT_SPEED_COLUMNS = ("s_nearest_home_ms", "s_nearest_away_ms", "mean_player_speed_ms")
FT_OPTIONAL = {"graded": "graded", "n_tracked": "n_tracked", "n_home_within_5m": "n_H_5m",
               "n_away_within_5m": "n_A_5m", "cx_home_m": "cx_H", "cx_away_m": "cx_A",
               "ball_speed_smooth_ms": "ball_speed_smooth"}


def from_feature_table(ft, pitch, out_margin=0.5, require_speeds=True):
    """The team's feature table -> the table the rules use (one row per video frame, metres, m/s).

    - ball_present: ball position given; ball_out: more than out_margin past a line
    - ball_speed: his ball_speed_ms, set missing where ball_jump (> 40 m/s: tracking jump or a ball
      in the air projected onto the ground), as the notebook's own table does
    - every other column is renamed (FT_COLUMNS, FT_OPTIONAL)
    With require_speeds=False, missing speed columns become NaN (for comparisons only: the rules
    need them, so run_pipeline on such a table is not meaningful).
    """
    absent = [c for c in FT_COLUMNS if c not in ft.columns and not (c in FT_SPEED_COLUMNS and not require_speeds)]
    if absent:
        raise ValueError("feature table lacks {}; see game_logic/RULES_HANDOFF.md (section 1)".format(absent))
    ft = ft.sort_values("frame") if "frame" in ft.columns else ft
    if "frame" in ft.columns and not np.array_equal(ft["frame"].to_numpy(), np.arange(len(ft))):
        raise ValueError("feature table frames must be 0..n-1, one row per video frame")
    t = pd.DataFrame({ours: (ft[his].to_numpy(float) if his in ft.columns else np.full(len(ft), np.nan))
                      for his, ours in FT_COLUMNS.items()})
    for his, ours in FT_OPTIONAL.items():
        if his in ft.columns:
            t[ours] = ft[his].to_numpy()
    if "ball_jump" in ft.columns:
        jump = ft["ball_jump"].astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
        t.loc[jump, "ball_speed"] = np.nan
    length, width = pitch
    bx, by = t["ball_x_m"], t["ball_y_m"]
    t["ball_present"] = bx.notna() & by.notna()
    t["ball_out"] = t["ball_present"] & ((bx < -out_margin) | (bx > length + out_margin)
                                         | (by < -out_margin) | (by > width + out_margin))
    t.index.name = "video_frame"
    return t


# -- Smart Tagging -------------------------------------------------------------------------------

def _norm(name):
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return " ".join(name.lower().split())


def side_of_team_name(name, home_name, away_name):
    """Match a tag's team name to the home or away club name from the tracking metadata.

    Fuzzy, because the two sources need not spell a club the same way. NOBODY when it is unclear.
    """
    a = difflib.SequenceMatcher(None, _norm(name), _norm(home_name)).ratio()
    b = difflib.SequenceMatcher(None, _norm(name), _norm(away_name)).ratio()
    if max(a, b) < 0.4 or abs(a - b) < 0.1:
        return NOBODY
    return HOME if a > b else AWAY


def tag_team_from_events(events, n_frames, home_name, away_name, exclude=(), flip_defending=True):
    """Per-frame team with the ball according to Smart Tagging, by votes.

    Every phase event adds one vote to its own frames, so overlapping tags never overwrite each
    other. Defending phases ('DEFENDING IN ...', 'DEFENSIVE TRANSITION') name the team WITHOUT the
    ball, so their vote goes to the other side. Set pieces are left out (they are not phases).
    Returns (tag_team int8 array, {code: [(start, end, side)]}, {tag team name: side}).
    """
    votes = np.zeros((2, n_frames), dtype=np.int16)
    code_events, mapping = {}, {}
    for e in events:
        code = (e["code"] or "").strip()
        if code.lower() == "set pieces" or code in exclude:
            continue
        name = e["tags"].get("Team") or ""
        if name not in mapping:
            mapping[name] = side_of_team_name(name, home_name, away_name)
        side = mapping[name]
        if side < 0:
            continue
        if flip_defending and "defen" in code.lower():
            side = 1 - side
        votes[side, e["start_frame"]:e["end_frame"]] += 1
        code_events.setdefault(code, []).append((e["start_frame"], e["end_frame"], side))
    tag_team = np.where(votes[HOME] > votes[AWAY], HOME,
                        np.where(votes[AWAY] > votes[HOME], AWAY, NOBODY)).astype(np.int8)
    return tag_team, code_events, mapping


# -- stand-in decoder ----------------------------------------------------------------------------

def decode_simple(ev_team, ev_need, dead, restart, break_frames=None, missing=None, gap_min_frames=None):
    """Codes 0 H, 1 A, 2 D, -1 X.

    - Dead ball -> D, and nobody owns the ball until play restarts.
    - Restart -> the restarting team owns it.
    - The other team takes over only after ev_need frames of its evidence IN A ROW: without our team
      touching it in between, and without its own evidence breaking off for more than break_frames
      (two separate pokes are not control; a tracking blip of a few frames does not break it).
      The change is backdated to their first touch.
    - If the ball was MISSING for at least gap_min_frames between the last evidence for the old team
      and the first for the new one, nobody saw when it changed hands: the switch goes to the middle
      of that blind stretch (it halves the worst-case error), never later than the new team's first
      evidence. Needs `missing` (bool per frame); without it, switches stay at the first touch.
    - Loose ball, contested ball, ball missing -> the current team keeps it.
    """
    brk = int(PARAMS["CTRL_BREAK_FRAMES"] if break_frames is None else break_frames)
    gmin = int(PARAMS["SWITCH_GAP_MIN_FRAMES"] if gap_min_frames is None else gap_min_frames)
    miss = None if missing is None else np.asarray(missing, dtype=bool)
    ev_team, ev_need, dead, restart = (np.asarray(a).tolist() for a in (ev_team, ev_need, dead, restart))
    out = np.full(len(ev_team), -1, dtype=np.int8)
    cur, cand, cnt, cand_start, last_cand, cur_seen = -1, -1, 0, 0, -10 ** 9, 0
    for i, (e, need, d) in enumerate(zip(ev_team, ev_need, dead)):
        if d:
            out[i] = 2
            cur, cand, cnt = -1, -1, 0
            continue
        if cur < 0:
            cur, cur_seen = (e if e >= 0 else restart[i]), i
        elif e >= 0:
            if e == cur:
                cand, cnt, cur_seen = -1, 0, i
            else:
                if e != cand or i - last_cand > brk:
                    cand, cnt, cand_start = e, 0, i
                last_cand = i
                cnt += 1
                if cnt >= need:
                    start = cand_start
                    if miss is not None:
                        blind = np.flatnonzero(miss[cur_seen + 1:cand_start])
                        if len(blind) >= gmin:
                            start = min(start, cur_seen + 1 + (blind[0] + blind[-1] + 1) // 2)
                    out[start:i] = e
                    cur, cand, cnt, cur_seen = e, -1, 0, i
        out[i] = cur
    return out


def run_pipeline(t, tag_team, graded, pitch, p=None, disable=(), **overrides):
    """Feature table -> (labels array of 'H'/'A'/'D'/'X', dict of intermediate evidence).

    Steps: clean_ball -> ctrl_evidence -> ball_events -> live_hold_guard -> dead_evidence ->
    drop stoppages under 4 s / merge play under 1 s -> snap_gap_stoppages -> restart_team ->
    evidence per frame (control, completed passes, Smart Tagging in long gaps) -> decode_simple.
    tag_team: Smart Tagging team per frame (tag_team_from_events), or all -1 to run without it.
    graded: True on scored frames; everything else is labelled X.
    p / overrides replace entries of PARAMS. disable switches off named parts (see ABLATIONS),
    for the ablation tests only.
    Returned evidence: ctrl, strength, dead_evidence, dead, restart, used_tag, events, live_guard,
    restart_flights, snapped_gaps, anchors.
    """
    p = {**PARAMS, **(p or {}), **overrides}
    unknown = set(disable) - set(ABLATIONS)
    if unknown:
        raise ValueError("unknown ablation(s): {}".format(sorted(unknown)))
    if "junk_filter" not in disable:
        t, _ = clean_ball(t, pitch, p)
    tag_team = np.asarray(tag_team, dtype=np.int8)
    if "tags" in disable:
        tag_team = np.full(len(t), NOBODY, dtype=np.int8)

    ctrl, strength = ctrl_evidence(t, p, disable)
    events = ball_events(t, ctrl, p)
    guard = live_hold_guard(events, ctrl, p)
    dead_ev = dead_evidence(t, p, disable, live_guard=None if "reception_guard" in disable else guard)
    dead = clean_mask(dead_ev > 0.5, int(p["MIN_DEAD_FRAMES"]), int(p["MIN_ALIVE_GAP"]))
    flights = restart_flights(t, p)
    snapped = 0
    if "gap_snap" not in disable:
        dead, snapped = snap_gap_stoppages(dead, t, flights, p)
        dead = clean_mask(dead, int(p["MIN_DEAD_FRAMES"]), int(p["MIN_ALIVE_GAP"]))
    restart = restart_team(ctrl, tag_team, p)

    gap_long = np.zeros(len(t), dtype=bool)
    for s, l in runs(~t["ball_present"].to_numpy(bool)):
        if l >= p["GAP_TAG_FRAMES"]:
            gap_long[s:s + l] = True
    ev_team = ctrl.copy()
    ev_need = np.full(len(t), int(p["TAKEOVER_FRAMES"]), dtype=np.int32)
    if "pass_control" not in disable:
        # A completed pass is control for the passing team for the whole flight (CHALLENGE.md: a team
        # completing passes has control), so a team that wins the ball and passes at once takes over
        # from the winning touch, not from the reception. Only fills frames with no control evidence.
        for s in np.flatnonzero(events["pass_result"] == 1):
            e = s + int(events["flight_frames"][s])
            seg_ = ev_team[s:e]
            seg_[seg_ < 0] = events["kick"][s]
    use_tag = gap_long & (tag_team >= 0)
    ev_team[use_tag] = tag_team[use_tag]
    ev_need[use_tag] = int(p["TAG_TAKEOVER_FRAMES"])

    missing = None if "gap_midpoint" in disable else ~t["ball_present"].to_numpy(bool)
    codes = decode_simple(ev_team, ev_need, dead, restart, p["CTRL_BREAK_FRAMES"], missing,
                          p["SWITCH_GAP_MIN_FRAMES"])
    labels = LABELS[np.where(codes < 0, 3, codes)]
    labels[~np.asarray(graded, dtype=bool)] = "X"      # outside the graded halves nothing is scored
    return labels, dict(ctrl=ctrl, strength=strength, dead_evidence=dead_ev, dead=dead,
                        restart=restart, used_tag=use_tag, events=events, live_guard=guard,
                        restart_flights=flights, snapped_gaps=snapped, anchors=anchors(t, events, p))
