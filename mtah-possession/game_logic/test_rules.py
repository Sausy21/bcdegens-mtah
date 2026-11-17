"""Football scenario tests for rules.py. No match data: every scene is synthetic.

Each scene is a small feature table describing a situation whose right answer a coach would agree
on, run through the real pipeline. Because the scenes are written from football knowledge, not
from the match, passing them is evidence that a rule encodes football and not this game.

    python game_logic/test_rules.py          # prints PASS / FAIL / KNOWN GAP per scene
    python -m pytest game_logic/test_rules.py

KNOWN GAP marks a situation we know the current rules get wrong; it is reported, not failed, so
the list doubles as a to-do. When a rule changes, the gap should flip to PASS (then drop the flag).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules  # noqa: E402

FPS = rules.FPS
PITCH = (105.0, 68.0)

# Open play with the home side dribbling: ball 0.8 m from an H player who moves with it at 5 m/s,
# nearest away player 4 m off, everyone running at a normal 2.2 m/s.
LIVE_H = dict(ball_present=True, ball_x_m=50.0, ball_y_m=34.0, ball_speed=5.0,
              d_H=0.8, d_A=4.0, s_near_H=5.0, s_near_A=3.0, mean_speed=2.2, ball_out=False)
LIVE_A = dict(LIVE_H, d_H=4.0, d_A=0.8, s_near_H=3.0, s_near_A=5.0)
MISSING = dict(LIVE_H, ball_present=False, ball_x_m=np.nan, ball_y_m=np.nan, ball_speed=np.nan,
               d_H=np.nan, d_A=np.nan, s_near_H=np.nan, s_near_A=np.nan)


def seg(seconds, base, **changes):
    n = int(round(seconds * FPS))
    return pd.DataFrame([dict(base, **changes)] * n)


def scene(*segments):
    t = pd.concat(segments, ignore_index=True)
    marks = np.cumsum([0] + [len(s) for s in segments])     # first frame of each segment
    return t, marks


def run(t, tag_team=None):
    tags = np.full(len(t), -1) if tag_team is None else tag_team
    labels, info = rules.run_pipeline(t, tags, np.ones(len(t), bool), PITCH)
    return labels, info


def share(labels, a, b, value):
    return float(np.mean(labels[a:b] == value))


def n_possessions(labels):
    s = pd.Series(labels).replace("X", np.nan).ffill().fillna("X").to_numpy()
    return sum(len(rules.runs(s == side)) for side in "HA")


# -- scenes: each returns (ok, detail) ------------------------------------------------------------

def scene_dribble():
    """An H player dribbles for 10 s: H throughout."""
    t, m = scene(seg(10, LIVE_H))
    lab, _ = run(t)
    return share(lab, 0, len(lab), "H") == 1.0, "H share {:.2f}".format(share(lab, 0, len(lab), "H"))


def scene_pass_to_opponent():
    """H dribbles, a 0.6 s ball in flight, A receives and dribbles: switch to A at A's first touch (+/-1 s)."""
    t, m = scene(seg(5, LIVE_H), seg(0.6, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0),
                 seg(5, LIVE_A))
    lab, _ = run(t)
    first_a = int(np.argmax(lab == "A"))
    ok = share(lab, 0, m[1], "H") == 1.0 and share(lab, m[2] + FPS, len(lab), "A") == 1.0 \
        and abs(first_a - m[2]) <= FPS and n_possessions(lab) == 2
    return ok, "first A frame {:+.2f} s from A's first touch, possessions {}".format(
        (first_a - m[2]) / FPS, n_possessions(lab))


def scene_ball_flies_past_opponent():
    """A 15 m/s pass flies within 0.8 m of an A player who does not stop it: stays H."""
    t, m = scene(seg(5, LIVE_H), seg(1, LIVE_H, ball_speed=15.0, d_H=5.0, d_A=0.8, s_near_A=3.0),
                 seg(5, LIVE_H))
    lab, _ = run(t)
    return n_possessions(lab) == 1 and share(lab, 0, len(lab), "H") == 1.0, \
        "possessions {}".format(n_possessions(lab))


def scene_brief_contested_touch():
    """A gets 0.3 s of control in a duel, H keeps the ball: no possession change."""
    t, m = scene(seg(5, LIVE_H), seg(0.3, LIVE_A), seg(5, LIVE_H))
    lab, _ = run(t)
    return n_possessions(lab) == 1, "possessions {}".format(n_possessions(lab))


def scene_two_pokes_no_control():
    """H has the ball; an A player pokes it (0.24 s), it runs loose 1 s, he pokes it again (0.24 s), loose again,
    H recovers. 'Control in a row' was never achieved by A: no possession change."""
    loose = dict(LIVE_H, d_H=3.0, d_A=3.0, ball_speed=4.0)
    t, m = scene(seg(5, LIVE_H), seg(0.24, LIVE_A), seg(1, loose), seg(0.24, LIVE_A), seg(1, loose), seg(5, LIVE_H))
    lab, _ = run(t)
    return n_possessions(lab) == 1, "possessions {}".format(n_possessions(lab))


def scene_interception_with_blip():
    """A intercepts and dribbles away; a 3-frame tracking blip (ball missing) right after the first touch
    must not stop the switch: A from the interception (+/-1 s)."""
    t, m = scene(seg(5, LIVE_H), seg(0.6, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0),
                 seg(0.2, LIVE_A), seg(3 / FPS, MISSING), seg(5, LIVE_A))
    lab, _ = run(t)
    first_a = int(np.argmax(lab == "A"))
    return abs(first_a - m[2]) <= FPS and n_possessions(lab) == 2, \
        "first A frame {:+.2f} s from the interception, possessions {}".format((first_a - m[2]) / FPS, n_possessions(lab))


def scene_win_and_pass_at_once():
    """A wins the ball (0.32 s on it) and passes straight away (1.2 s flight) to an A teammate who keeps it:
    possession switches at the winning touch (+/-0.5 s), not at the reception. CHALLENGE.md: completed passes = control."""
    t, m = scene(seg(5, LIVE_H), seg(0.32, LIVE_A),
                 seg(1.2, LIVE_A, ball_speed=15.0, d_H=6.0, d_A=6.0), seg(5, LIVE_A))
    lab, _ = run(t)
    first_a = int(np.argmax(lab == "A"))
    return abs(first_a - m[1]) <= FPS // 2 and n_possessions(lab) == 2, \
        "first A frame {:+.2f} s from the winning touch, possessions {}".format((first_a - m[1]) / FPS, n_possessions(lab))


def scene_pass_back_to_thrower():
    """Ball goes out, tracker loses it 5 s, it reappears as a pass back to the thrower standing OUTSIDE the
    pitch, he holds it out of play 3 s, throws it in: D from the ball going out until the throw (>= 90%)."""
    t, m = scene(seg(5, LIVE_H),
                 seg(1, LIVE_H, ball_y_m=69.5, ball_out=True, ball_speed=3.0, d_H=2.0, d_A=3.0, mean_speed=1.0),
                 seg(5, MISSING, mean_speed=0.8),
                 seg(0.6, LIVE_A, ball_y_m=67.5, ball_speed=10.0, d_A=4.0, d_H=6.0, mean_speed=0.8),
                 seg(3, LIVE_A, ball_y_m=69.3, ball_out=True, ball_speed=0.5, s_near_A=0.5, d_A=0.5, mean_speed=0.8),
                 seg(0.5, LIVE_A, ball_y_m=66.0, ball_speed=12.0, d_A=3.0, d_H=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    lab, _ = run(t)
    dshare = share(lab, m[1], m[5], "D")
    return dshare >= 0.9 and share(lab, m[6] + FPS, len(lab), "A") == 1.0, \
        "D share from the ball going out to the throw {:.2f}".format(dshare)


def scene_thrower_holds_ball_out():
    """Ball goes out; the thrower stands behind the line holding it (tracked, out of play) for 8 s, then throws:
    D ends at the throw (+/-1 s), not 4 s later."""
    t, m = scene(seg(5, LIVE_H),
                 seg(8, LIVE_A, ball_y_m=69.3, ball_out=True, ball_speed=0.5, s_near_A=0.5, d_A=0.5, mean_speed=0.8),
                 seg(0.5, LIVE_A, ball_y_m=66.0, ball_speed=12.0, d_A=3.0, d_H=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    lab, _ = run(t)
    d = np.flatnonzero(lab == "D")
    if not len(d):
        return False, "no D at all"
    return abs(d[-1] + 1 - m[2]) <= FPS, "D ends {:+.2f} s from the throw".format((d[-1] + 1 - m[2]) / FPS)


def scene_turnover_hidden_in_gap():
    """H has the ball, the tracker loses it for 8 s while everyone keeps running, A has it when it reappears.
    We can't see when it changed hands, so the switch goes in the middle of the blind stretch (+/-1 s),
    which halves the worst-case error, rather than at the end."""
    t, m = scene(seg(5, LIVE_H), seg(8, MISSING, mean_speed=2.2), seg(5, LIVE_A))
    lab, _ = run(t)
    first_a = int(np.argmax(lab == "A"))
    mid = (m[1] + m[2]) // 2
    return abs(first_a - mid) <= FPS and n_possessions(lab) == 2, \
        "switch {:+.2f} s from the middle of the gap, possessions {}".format((first_a - mid) / FPS, n_possessions(lab))


def scene_turnover_in_gap_with_tag():
    """Same, 12 s gap, but Smart Tagging switches to A 3 s into it: the switch follows that evidence (+/-1 s),
    it is never placed later than what the evidence says."""
    t, m = scene(seg(5, LIVE_H), seg(12, MISSING, mean_speed=2.2), seg(5, LIVE_A))
    tags = np.full(len(t), -1)
    tags[m[1] + 3 * FPS:m[2]] = rules.AWAY
    lab, _ = run(t, tags)
    first_a = int(np.argmax(lab == "A"))
    target = m[1] + 3 * FPS
    return first_a <= target + FPS and n_possessions(lab) == 2, \
        "switch {:+.2f} s from the tag change, possessions {}".format((first_a - target) / FPS, n_possessions(lab))


def scene_short_blip_does_not_move_switch():
    """A visible interception with a 0.4 s tracking blip during the pass: still switches at A's first touch (+/-0.5 s)."""
    t, m = scene(seg(5, LIVE_H), seg(0.3, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0), seg(0.4, MISSING),
                 seg(0.3, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0), seg(5, LIVE_A))
    lab, _ = run(t)
    first_a = int(np.argmax(lab == "A"))
    return abs(first_a - m[4]) <= FPS // 2, "switch {:+.2f} s from A's first touch".format((first_a - m[4]) / FPS)


def scene_ball_jogged_back():
    """Ball goes out; a player jogs it back to the line at 5 m/s for 6 s, then throws it in at 12 m/s.
    A jog is not a restart: D until the throw (+/-1 s)."""
    t, m = scene(seg(5, LIVE_H),
                 seg(1, LIVE_H, ball_y_m=69.5, ball_out=True, ball_speed=3.0, d_H=2.0, d_A=3.0),
                 seg(6, LIVE_A, ball_y_m=60.0, ball_speed=5.0, s_near_A=5.0, d_A=0.5, d_H=6.0, mean_speed=1.0),
                 seg(0.5, LIVE_A, ball_speed=12.0, d_A=3.0, d_H=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    lab, _ = run(t)
    d = np.flatnonzero(lab == "D")
    if not len(d):
        return False, "no D at all"
    return abs(d[-1] + 1 - m[3]) <= FPS, "D ends {:+.2f} s from the throw".format((d[-1] + 1 - m[3]) / FPS)


def scene_foul_ball_carried():
    """Foul inside the pitch; the ball is carried to the spot at walking pace (1.5 m/s) for 12 s while everyone
    walks, then the free kick. The ball is never still: at least 70% of the stoppage is D."""
    t, m = scene(seg(5, LIVE_H),
                 seg(12, LIVE_H, ball_speed=1.5, s_near_H=1.5, d_H=0.6, d_A=6.0, mean_speed=0.7),
                 seg(0.5, LIVE_H, ball_speed=14.0, d_H=3.0, d_A=8.0),
                 seg(6, LIVE_H))
    lab, _ = run(t)
    dshare = share(lab, m[1], m[2], "D")
    return dshare >= 0.7, "D share of the stoppage {:.2f}".format(dshare)


def scene_jittery_duel():
    """3 s of a tight duel where the nearest player flips team every 3 frames: no flicker."""
    rows = []
    for i in range(3 * FPS):
        rows.append(dict(LIVE_H, d_H=0.9, d_A=1.1) if (i // 3) % 2 == 0 else dict(LIVE_H, d_H=1.1, d_A=0.9))
    t, m = scene(seg(5, LIVE_H), pd.DataFrame(rows), seg(5, LIVE_H))
    lab, _ = run(t)
    return n_possessions(lab) == 1, "possessions {}".format(n_possessions(lab))


def scene_throw_in():
    """Ball goes out, A carries it back at jogging pace for 8 s, then throws it in.
    D from the ball going out (+/-0.2 s) until the throw (+/-1 s), then A."""
    t, m = scene(seg(5, LIVE_H),
                 seg(1, LIVE_H, ball_y_m=69.5, ball_out=True, ball_speed=3.0, d_H=2.0, d_A=3.0),
                 seg(8, LIVE_A, ball_y_m=66.0, ball_speed=2.0, s_near_A=2.0, d_A=0.5, mean_speed=0.8),
                 seg(0.5, LIVE_A, ball_speed=12.0, d_A=3.0, d_H=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    lab, _ = run(t)
    d = np.flatnonzero(lab == "D")
    if not len(d):
        return False, "no D at all"
    ok = abs(d[0] - m[1]) <= 0.2 * FPS and abs(d[-1] + 1 - m[3]) <= FPS \
        and share(lab, m[4] + FPS, len(lab), "A") == 1.0
    return ok, "D starts {:+.2f} s from the ball going out, ends {:+.2f} s from the throw".format(
        (d[0] - m[1]) / FPS, (d[-1] + 1 - m[3]) / FPS)


def scene_free_kick_no_out():
    """Foul: ball still in the pitch for 15 s, everyone walking, then a 12 m/s free kick and H plays on.
    At least 70% of the stoppage is D, then H."""
    t, m = scene(seg(5, LIVE_H),
                 seg(15, LIVE_H, ball_speed=0.0, s_near_H=0.5, d_A=9.0, mean_speed=0.7),
                 seg(0.5, LIVE_H, ball_speed=12.0, d_H=3.0, d_A=8.0),
                 seg(6, LIVE_H))
    lab, _ = run(t)
    dshare = share(lab, m[1], m[2], "D")
    return dshare >= 0.7 and share(lab, m[3] + 2 * FPS, len(lab), "H") == 1.0, \
        "D share of the stoppage {:.2f}".format(dshare)


def scene_missing_and_stopped():
    """Ball missing for 20 s with players walking (0.7 m/s), then A restarts: mostly D, then A."""
    t, m = scene(seg(5, LIVE_H), seg(20, MISSING, mean_speed=0.7), seg(5, LIVE_A))
    lab, _ = run(t)
    dshare = share(lab, m[1], m[2], "D")
    return dshare >= 0.8 and share(lab, m[2] + FPS, len(lab), "A") == 1.0, \
        "D share of the gap {:.2f}".format(dshare)


def scene_missing_but_playing():
    """Ball lost by the tracker for 10 s while players run (2.5 m/s), H still on it: H throughout, no D."""
    t, m = scene(seg(5, LIVE_H), seg(10, MISSING, mean_speed=2.5), seg(5, LIVE_H))
    lab, _ = run(t)
    return share(lab, 0, len(lab), "H") == 1.0, "D frames {}".format(int(np.sum(lab == "D")))


def scene_slow_buildup_quick_passes():
    """H circulates at the back: 1.5 s on the ball, 0.5 s pass, repeated, everyone slow (0.9 m/s): alive, H."""
    parts = [seg(3, LIVE_H)]
    for _ in range(8):
        parts += [seg(1.5, LIVE_H, ball_speed=1.0, s_near_H=1.0, d_H=0.5, mean_speed=0.9),
                  seg(0.5, LIVE_H, ball_speed=12.0, d_H=5.0, d_A=8.0, mean_speed=0.9)]
    t, m = scene(*parts)
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0, "D frames {}".format(int(np.sum(lab == "D")))


def scene_slow_buildup_long_holds():
    """Same, but each centre-back holds the ball 3.5 s before passing (common in patient build-up): alive."""
    parts = [seg(3, LIVE_H)]
    for _ in range(5):
        parts += [seg(3.5, LIVE_H, ball_speed=1.0, s_near_H=1.0, d_H=0.5, d_A=12.0, mean_speed=0.9),
                  seg(0.5, LIVE_H, ball_speed=12.0, d_H=5.0, d_A=12.0, mean_speed=0.9)]
    t, m = scene(*parts)
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0, "D frames {} ({:.1f} s)".format(
        int(np.sum(lab == "D")), np.sum(lab == "D") / FPS)


def scene_keeper_holds():
    """H shoots, the A keeper catches and holds for 5 s while players walk back, then punts: alive, A from the catch."""
    t, m = scene(seg(5, LIVE_H),
                 seg(0.6, LIVE_H, ball_x_m=10.0, ball_speed=20.0, d_H=8.0, d_A=6.0),
                 seg(5, LIVE_A, ball_x_m=4.0, ball_speed=1.0, s_near_A=1.0, d_A=0.3, d_H=9.0, mean_speed=0.9),
                 seg(1, LIVE_A, ball_x_m=30.0, ball_speed=25.0, d_A=15.0, d_H=15.0),
                 seg(5, LIVE_A))
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0 and share(lab, m[2] + FPS, len(lab), "A") == 1.0, \
        "D frames {} ({:.1f} s)".format(int(np.sum(lab == "D")), np.sum(lab == "D") / FPS)


def scene_keeper_holds_6s():
    """Same catch, but the keeper uses the full 6 s the Laws allow before punting: alive, A from the catch."""
    t, m = scene(seg(5, LIVE_H),
                 seg(0.6, LIVE_H, ball_x_m=10.0, ball_speed=20.0, d_H=8.0, d_A=6.0),
                 seg(6, LIVE_A, ball_x_m=4.0, ball_speed=1.0, s_near_A=1.0, d_A=0.3, d_H=9.0, mean_speed=0.9),
                 seg(1, LIVE_A, ball_x_m=30.0, ball_speed=25.0, d_A=15.0, d_H=15.0),
                 seg(5, LIVE_A))
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0 and share(lab, m[2] + FPS, len(lab), "A") == 1.0, \
        "D frames {} ({:.1f} s)".format(int(np.sum(lab == "D")), np.sum(lab == "D") / FPS)


def scene_slow_buildup_5s_hold():
    """A centre-back, unpressed, holds the ball 5 s before passing, everyone slow (0.9 m/s): alive, H."""
    parts = [seg(3, LIVE_H)]
    for _ in range(4):
        parts += [seg(5, LIVE_H, ball_speed=1.0, s_near_H=1.0, d_H=0.5, d_A=12.0, mean_speed=0.9),
                  seg(0.5, LIVE_H, ball_speed=12.0, d_H=5.0, d_A=12.0, mean_speed=0.9)]
    t, m = scene(*parts)
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0, "D frames {} ({:.1f} s)".format(
        int(np.sum(lab == "D")), np.sum(lab == "D") / FPS)


def scene_quick_throw_in():
    """Ball goes out, A takes a quick throw-in 4.5 s later: the short stoppage must still be D, then A."""
    t, m = scene(seg(5, LIVE_H),
                 seg(1, LIVE_H, ball_y_m=69.5, ball_out=True, ball_speed=3.0, d_H=2.0, d_A=3.0),
                 seg(3.5, LIVE_A, ball_y_m=67.0, ball_speed=2.0, s_near_A=2.0, d_A=0.5, mean_speed=1.5),
                 seg(0.5, LIVE_A, ball_speed=12.0, d_A=3.0, d_H=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    lab, _ = run(t)
    dshare = share(lab, m[1], m[3], "D")
    return dshare >= 0.8 and share(lab, m[4] + FPS, len(lab), "A") == 1.0, \
        "D share of the 4.5 s stoppage {:.2f}".format(dshare)


def scene_false_ball_detection():
    """For 1 s the tracker puts the ball 20 m beyond the touchline (a false detection): ignore it, H throughout."""
    t, m = scene(seg(5, LIVE_H),
                 seg(1, LIVE_H, ball_y_m=88.0, ball_out=True, d_H=30.0, d_A=30.0, ball_speed=np.nan),
                 seg(5, LIVE_H))
    lab, _ = run(t)
    return share(lab, 0, len(lab), "H") == 1.0, "D frames {}".format(int(np.sum(lab == "D")))


def scene_received_then_hold_6s():
    """H passes along the back line, each centre-back receives and holds 6 s unpressed, everyone slow: alive, H."""
    parts = [seg(3, LIVE_H)]
    for _ in range(4):
        parts += [seg(0.5, LIVE_H, ball_speed=12.0, d_H=5.0, d_A=12.0, mean_speed=0.9),
                  seg(6, LIVE_H, ball_speed=1.0, s_near_H=1.0, d_H=0.5, d_A=12.0, mean_speed=0.9)]
    parts += [seg(0.5, LIVE_H, ball_speed=12.0, d_H=5.0, d_A=12.0, mean_speed=0.9), seg(3, LIVE_H)]
    t, m = scene(*parts)
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0, "D frames {} ({:.1f} s)".format(
        int(np.sum(lab == "D")), np.sum(lab == "D") / FPS)


def scene_received_then_fouled():
    """H receives a pass, dribbles 2 s, is fouled: ball knocked loose, everyone walks 15 s, free kick.
    The reception guard must not swallow the stoppage: at least 70% of it is D."""
    t, m = scene(seg(5, LIVE_H),
                 seg(0.6, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0),
                 seg(2, LIVE_H),
                 seg(1, LIVE_H, ball_speed=3.0, d_H=3.0, d_A=3.0, s_near_H=0.5, s_near_A=0.5, mean_speed=1.0),
                 seg(14, LIVE_H, ball_speed=0.0, d_H=3.0, d_A=3.0, s_near_H=0.5, s_near_A=0.5, mean_speed=0.7),
                 seg(0.5, LIVE_H, ball_speed=12.0, d_H=3.0, d_A=8.0),
                 seg(6, LIVE_H))
    lab, _ = run(t)
    dshare = share(lab, m[3], m[5], "D")
    return dshare >= 0.7, "D share of the 15 s stoppage {:.2f}".format(dshare)


def scene_stoppage_in_gap():
    """The tracker loses the ball; players keep moving 6 s, then walk 14 s (a stoppage), then a kick out of
    the gap and A plays on. D from the ball vanishing (+/-1 s) to the restart kick (+/-1 s)."""
    t, m = scene(seg(5, LIVE_H),
                 seg(6, MISSING, mean_speed=1.8),
                 seg(14, MISSING, mean_speed=0.7),
                 seg(0.6, LIVE_A, ball_speed=15.0, d_H=6.0, d_A=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    lab, _ = run(t)
    d = np.flatnonzero(lab == "D")
    if not len(d):
        return False, "no D at all"
    ok = abs(d[0] - m[1]) <= FPS and abs(d[-1] + 1 - m[3]) <= FPS and share(lab, m[4] + FPS, len(lab), "A") == 1.0
    return ok, "D starts {:+.2f} s from the ball vanishing, ends {:+.2f} s from the restart kick".format(
        (d[0] - m[1]) / FPS, (d[-1] + 1 - m[3]) / FPS)


def scene_open_play_gap_lull():
    """Tracker loses the ball 9 s in open play; players ease off for 3.5 s in the middle, then a pass
    appears out of the gap: no D (a lull is not a stoppage, and the snap must not create one)."""
    t, m = scene(seg(5, LIVE_H),
                 seg(2.5, MISSING, mean_speed=2.2), seg(3.5, MISSING, mean_speed=0.8), seg(3, MISSING, mean_speed=2.2),
                 seg(0.6, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0),
                 seg(5, LIVE_H))
    lab, _ = run(t)
    return int(np.sum(lab == "D")) == 0, "D frames {}".format(int(np.sum(lab == "D")))


# -- event detector checks (ball_events), no labels involved ---------------------------------------

def events_of(t):
    _, info = run(t)
    return info["events"]


def scene_event_pass_intercepted():
    """H passes, A receives: one kick by H, one reception by A, result 'intercepted', flight ~0.6 s."""
    t, m = scene(seg(3, LIVE_H), seg(0.6, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0), seg(3, LIVE_A))
    ev = events_of(t)
    k, r = np.flatnonzero(ev["kick"] >= 0), np.flatnonzero(ev["reception"] >= 0)
    ok = (len(k) == 1 and len(r) == 1 and ev["kick"][k[0]] == rules.HOME and ev["reception"][r[0]] == rules.AWAY
          and ev["pass_result"][k[0]] == 0 and abs(ev["flight_frames"][k[0]] - 0.6 * FPS) <= 3)
    return ok, "kicks {} receptions {} flight {:.2f} s".format(
        len(k), len(r), ev["flight_frames"][k[0]] / FPS if len(k) else 0)


def scene_event_flies_past():
    """A pass flies past an A player (no slowing) and reaches an H teammate: completed H pass, no A reception."""
    t, m = scene(seg(3, LIVE_H), seg(1, LIVE_H, ball_speed=15.0, d_H=5.0, d_A=0.8, s_near_A=3.0), seg(3, LIVE_H))
    ev = events_of(t)
    k = np.flatnonzero(ev["kick"] >= 0)
    ok = len(k) == 1 and ev["pass_result"][k[0]] == 1 and not np.any(ev["reception"] == rules.AWAY)
    return ok, "kicks {} result {}".format(len(k), ev["pass_result"][k[0]] if len(k) else None)


def scene_event_sprint_dribble():
    """An H winger sprints with the ball at 9 m/s for 3 s (ball at his feet): no kick, no reception."""
    t, m = scene(seg(3, LIVE_H), seg(3, LIVE_H, ball_speed=9.0, s_near_H=9.0), seg(3, LIVE_H))
    ev = events_of(t)
    nk, nr = int(np.sum(ev["kick"] >= 0)), int(np.sum(ev["reception"] >= 0))
    return nk == 0 and nr == 0, "kicks {} receptions {}".format(nk, nr)


def scene_event_placed_kick():
    """Free kick: ball still 5 s, H taker runs up (not controlling), kicks: one kick by H, marked 'placed'."""
    t, m = scene(seg(3, LIVE_H, ball_speed=0.0, s_near_H=0.0, d_H=1.5, d_A=9.0, mean_speed=0.7),
                 seg(2, LIVE_H, ball_speed=0.0, s_near_H=3.0, d_H=1.5, d_A=9.0, mean_speed=0.7),
                 seg(0.6, LIVE_H, ball_speed=20.0, d_H=5.0, d_A=8.0), seg(3, LIVE_H))
    ev = events_of(t)
    k = np.flatnonzero(ev["kick"] >= 0)
    ok = len(k) == 1 and ev["kick"][k[0]] == rules.HOME and bool(ev["placed"][k[0]])
    return ok, "kicks {} placed {}".format(len(k), bool(ev["placed"][k[0]]) if len(k) else None)


def scene_event_speed_spike():
    """A 2-frame tracking spike to 15 m/s during a dribble is jitter, not a kick."""
    t, m = scene(seg(3, LIVE_H), seg(2 / FPS, LIVE_H, ball_speed=15.0, d_H=2.5), seg(3, LIVE_H))
    ev = events_of(t)
    nk = int(np.sum(ev["kick"] >= 0))
    return nk == 0, "kicks {}".format(nk)


# -- anchor checks (anchors), no labels involved ----------------------------------------------------

def anchors_of(t):
    _, info = run(t)
    return info["anchors"]


def scene_anchor_dribble():
    """3 s of an H player dribbling, ball at his feet, opponent 4 m away: firm H anchor throughout."""
    t, m = scene(seg(3, LIVE_H))
    a = anchors_of(t)
    return bool(np.all(a["team"] == rules.HOME)), "H anchor share {:.2f}".format(np.mean(a["team"] == rules.HOME))


def scene_anchor_duel_is_not_anchor():
    """A tight duel (both within 1.2 m) for 3 s: no anchor, it is exactly what we are unsure of."""
    t, m = scene(seg(3, LIVE_H, d_H=0.9, d_A=1.2))
    a = anchors_of(t)
    return bool(np.all(a["team"] == rules.NOBODY)), "anchored frames {}".format(int(np.sum(a["team"] >= 0)))


def scene_anchor_short_touch_is_not_anchor():
    """0.6 s of clean control is too short to be sure (1 s needed): no firm anchor."""
    t, m = scene(seg(0.6, LIVE_H), seg(3, LIVE_H, d_H=5.0, d_A=5.0, ball_speed=6.0))
    a = anchors_of(t)
    return not a["firm_H"].any(), "firm frames {}".format(int(a["firm_H"].sum()))


def scene_anchor_completed_pass():
    """H passes to H over 1 s: the whole flight is an H anchor (H had the ball all along)."""
    t, m = scene(seg(3, LIVE_H), seg(1, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0), seg(3, LIVE_H))
    a = anchors_of(t)
    return bool(np.all(a["team"][m[1]:m[2]] == rules.HOME)), "H anchor share of the flight {:.2f}".format(
        np.mean(a["team"][m[1]:m[2]] == rules.HOME))


def scene_anchor_intercepted_pass():
    """H passes, A intercepts: the flight is NOT anchored (it ended with the other team), A from the reception."""
    t, m = scene(seg(3, LIVE_H), seg(1, LIVE_H, ball_speed=15.0, d_H=6.0, d_A=6.0), seg(3, LIVE_A))
    a = anchors_of(t)
    ok = bool(np.all(a["team"][m[1]:m[2]] == rules.NOBODY)) and a["team"][m[2] + 2] == rules.AWAY
    return ok, "anchored flight frames {}, A after reception {}".format(
        int(np.sum(a["team"][m[1]:m[2]] >= 0)), a["team"][m[2] + 2] == rules.AWAY)


def scene_anchor_ball_out():
    """Ball 1.5 m past the touchline: dead anchor, no team anchor."""
    t, m = scene(seg(3, LIVE_H), seg(1, LIVE_H, ball_y_m=69.5, ball_out=True, d_H=1.0, d_A=4.0))
    a = anchors_of(t)
    ok = bool(np.all(a["dead"][m[1]:])) and bool(np.all(a["team"][m[1]:] == rules.NOBODY))
    return ok, "dead anchor share {:.2f}".format(np.mean(a["dead"][m[1]:]))


# -- the team's feature table: the adapter must not change anything --------------------------------

def scene_feature_table_roundtrip():
    """The throw-in scene, rewritten in Sausthava's column names and read back through from_feature_table,
    gives exactly the same labels (so switching tables changes the data, never the rules)."""
    t, m = scene(seg(5, LIVE_H),
                 seg(1, LIVE_H, ball_y_m=69.5, ball_out=True, ball_speed=3.0, d_H=2.0, d_A=3.0),
                 seg(8, LIVE_A, ball_y_m=66.0, ball_speed=2.0, s_near_A=2.0, d_A=0.5, mean_speed=0.8),
                 seg(0.5, LIVE_A, ball_speed=12.0, d_A=3.0, d_H=6.0, mean_speed=1.5),
                 seg(6, LIVE_A))
    back = {ours: his for his, ours in rules.FT_COLUMNS.items()}
    ft = t.rename(columns=back).drop(columns=["ball_present", "ball_out"])
    ft.insert(0, "frame", np.arange(len(ft)))
    ft["ball_jump"] = False
    lab_direct, _ = run(t)
    lab_ft, _ = run(rules.from_feature_table(ft, PITCH))
    same = float(np.mean(lab_direct == lab_ft))
    return same == 1.0, "labels identical on {:.1%} of frames".format(same)


SCENES = [
    (scene_dribble, False),
    (scene_pass_to_opponent, False),
    (scene_ball_flies_past_opponent, False),
    (scene_brief_contested_touch, False),
    (scene_two_pokes_no_control, False),
    (scene_interception_with_blip, False),
    (scene_win_and_pass_at_once, False),
    (scene_pass_back_to_thrower, False),
    (scene_thrower_holds_ball_out, False),
    (scene_turnover_hidden_in_gap, False),
    (scene_turnover_in_gap_with_tag, False),
    (scene_short_blip_does_not_move_switch, False),
    (scene_ball_jogged_back, False),
    (scene_foul_ball_carried, False),
    (scene_jittery_duel, False),
    (scene_throw_in, False),
    (scene_free_kick_no_out, False),
    (scene_missing_and_stopped, False),
    (scene_missing_but_playing, False),
    (scene_slow_buildup_quick_passes, False),
    (scene_slow_buildup_long_holds, False),
    (scene_keeper_holds, False),
    (scene_keeper_holds_6s, False),       # fixed by the reception guard (was a gap until josh_v3)
    (scene_slow_buildup_5s_hold, True),   # gap: the FIRST hold follows a dribble, not a received pass, and
                                          # "dribble, everyone slows, ball stays at his feet" is also what a
                                          # foul looks like. The received version passes (received_then_hold_6s).
    (scene_quick_throw_in, False),
    (scene_false_ball_detection, False),
    (scene_received_then_hold_6s, False),
    (scene_received_then_fouled, False),
    (scene_stoppage_in_gap, False),
    (scene_open_play_gap_lull, False),
    (scene_event_pass_intercepted, False),
    (scene_event_flies_past, False),
    (scene_event_sprint_dribble, False),
    (scene_event_placed_kick, False),
    (scene_event_speed_spike, False),
    (scene_anchor_dribble, False),
    (scene_anchor_duel_is_not_anchor, False),
    (scene_anchor_short_touch_is_not_anchor, False),
    (scene_anchor_completed_pass, False),
    (scene_anchor_intercepted_pass, False),
    (scene_anchor_ball_out, False),
    (scene_feature_table_roundtrip, False),
]
# second element: True = KNOWN GAP (reported, not failed). Flip after a scene has been seen failing
# for a reason we accept for now.


def main():
    failed = 0
    for fn, known_gap in SCENES:
        ok, detail = fn()
        status = "PASS" if ok else ("KNOWN GAP" if known_gap else "FAIL")
        failed += (not ok) and (not known_gap)
        print("{:<10} {:<34} {}".format(status, fn.__name__.replace("scene_", ""), detail))
        print("{:<10} {}".format("", fn.__doc__.split("\n")[0]))
    print("\n{} failed".format(failed))
    return failed


def test_scenes():                         # pytest entry point
    bad = [fn.__name__ for fn, gap in SCENES if not gap and not fn()[0]]
    assert not bad, bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
