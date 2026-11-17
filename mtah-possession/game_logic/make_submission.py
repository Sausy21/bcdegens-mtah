"""One command from the provided data to the submission CSV, using the rules in game_logic/rules.py.

    python game_logic/make_submission.py                              # -> evaluation/results/josh_v7.csv
    python game_logic/make_submission.py --out TEAM_NAME.csv
    python game_logic/make_submission.py --compare evaluation/results/josh_v7.csv

Steps (deterministic; nothing is written except the output CSV):
1. Feature table, one row per video frame (metres, m/s), read from the tactical tracking:
   ball position and speed, nearest home/away player distance and speed, mean player speed, ball out.
   With --features, the team's features.csv is used instead (rules.from_feature_table); it needs the
   three player-speed columns listed in game_logic/RULEBOOK.md section 2.
2. Smart Tagging team per frame (rules.tag_team_from_events), team names from the tracking metadata.
3. rules.run_pipeline with the default parameters (rules.PARAMS) -> one label per frame.
4. Write frame,label. Frames outside the graded halves are X.

--compare reports whether the result is identical to an existing CSV, and if not, how many rows differ.
It prints counts only, never frame numbers.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(ROOT, "game_logic"), os.path.join(ROOT, "starter"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402
import rules  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
PITCH = (PITCH_L, PITCH_W)
OUT_MARGIN_M = 0.5        # ball "out" only when clearly past a line (tracking noise)
BALL_SPEED_MAX = 40.0     # m/s; faster than any real shot = tracking jump -> speed unknown
HOME, AWAY = rules.HOME, rules.AWAY
assert FPS == rules.FPS, "rules.py windows are in frames at {} fps".format(rules.FPS)


# -- feature table from the raw tracking --------------------------------------------------------

def _xy_s(group):
    """'x,y,s' -> floats; NaN when missing."""
    if group[0] == "N":
        return np.nan, np.nan, np.nan
    vals = group.split(",")
    return float(vals[0]), float(vals[1]), (float(vals[2]) if len(vals) > 2 else np.nan)


def read_raw_atd(raw_path, n_frames, n_slots):
    """Ball (x, y) and players (x, y, s) per video frame, in metres. frame_id counts from 1."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if parts[2] and parts[2][0] != "N":
                bx, by, _ = _xy_s(parts[2])
                ball[frame] = (bx * PITCH_L, by * PITCH_W)
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                if group and group[0] != "N":
                    x, y, s = _xy_s(group)
                    players[frame, slot] = (x * PITCH_L, y * PITCH_W, s)
    return ball, players


def ball_speed(ball):
    """m/s from a centred difference over +/-1 frame; NaN unless both neighbours exist."""
    speed = np.full(len(ball), np.nan, dtype=np.float32)
    step = ball[2:] - ball[:-2]
    speed[1:-1] = np.hypot(step[:, 0], step[:, 1]) * FPS / 2.0
    speed[speed > BALL_SPEED_MAX] = np.nan
    return speed


def nearest_per_team(ball, players, team_of_track, team):
    """(distance to the ball, speed) of the team's nearest player, per frame."""
    sel = players[:, team_of_track == team, :]
    dist = np.hypot(sel[:, :, 0] - ball[:, None, 0], sel[:, :, 1] - ball[:, None, 1])
    has = np.isfinite(dist).any(axis=1)
    idx = np.where(has, np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1), 0)
    rows = np.arange(len(ball))
    return (np.where(has, dist[rows, idx], np.nan).astype(np.float32),
            np.where(has, sel[rows, idx, 2], np.nan).astype(np.float32))


def build_table(meta_path, raw_path):
    """The feature table the rules use, from the raw tactical tracking."""
    team_of_track = team_of_each_track(meta_path)
    ball, players = read_raw_atd(raw_path, source_frames("tactical"), len(team_of_track))
    d_H, s_H = nearest_per_team(ball, players, team_of_track, HOME)
    d_A, s_A = nearest_per_team(ball, players, team_of_track, AWAY)
    px, ps = players[:, :, 0], players[:, :, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)       # frames with no players tracked
        t = pd.DataFrame({
            "ball_present": np.isfinite(ball[:, 0]),
            "ball_x_m": ball[:, 0], "ball_y_m": ball[:, 1], "ball_speed": ball_speed(ball),
            "d_H": d_H, "d_A": d_A, "s_near_H": s_H, "s_near_A": s_A,
            "n_H": np.isfinite(px[:, team_of_track == HOME]).sum(axis=1),
            "n_A": np.isfinite(px[:, team_of_track == AWAY]).sum(axis=1),
            "mean_speed": np.nanmean(ps, axis=1),
            "cx_H": np.nanmean(px[:, team_of_track == HOME], axis=1),
            "cx_A": np.nanmean(px[:, team_of_track == AWAY], axis=1),
        })
    bx, by = t["ball_x_m"], t["ball_y_m"]
    t["ball_out"] = t["ball_present"] & ((bx < -OUT_MARGIN_M) | (bx > PITCH_L + OUT_MARGIN_M)
                                         | (by < -OUT_MARGIN_M) | (by > PITCH_W + OUT_MARGIN_M))
    t.index.name = "video_frame"
    return t


def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


# -- main -----------------------------------------------------------------------------------------

def make_labels(features_path=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    if features_path:
        t = rules.from_feature_table(pd.read_csv(features_path), PITCH, OUT_MARGIN_M)
    else:
        t = build_table(meta_path, raw_path)
    home_name, away_name = team_names(meta_path)
    tag_team, _, _ = rules.tag_team_from_events(load_smart_tagging.load(), len(t), home_name, away_name)
    graded = graded_mask(len(t))
    labels, _ = rules.run_pipeline(t, tag_team, graded, PITCH, dict(rules.PARAMS))
    return labels, graded


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "josh_v7.csv"),
                    help="output CSV (default: evaluation/results/josh_v7.csv, gitignored)")
    ap.add_argument("--features", default=None, help="use the team's features.csv instead of the raw tracking")
    ap.add_argument("--compare", default=None, help="existing CSV to compare the result with")
    args = ap.parse_args()

    labels, graded = make_labels(args.features)
    if args.compare:
        old = pd.read_csv(args.compare)["label"].to_numpy().astype(str)
        if len(old) != len(labels):
            print("compare: row counts differ ({} vs {})".format(len(old), len(labels)))
        else:
            diff = old != labels
            print("compare: IDENTICAL" if not diff.any() else
                  "compare: {} rows differ ({:.3f}% of graded frames)".format(int(diff.sum()), 100 * diff[graded].mean()))
    write_submission(args.out, labels)
    counts = pd.Series(labels).value_counts()
    print("wrote {} ({} rows): {}".format(args.out, len(labels),
                                         "  ".join("{} {}".format(k, int(counts.get(k, 0))) for k in "HADX")))


if __name__ == "__main__":
    main()
