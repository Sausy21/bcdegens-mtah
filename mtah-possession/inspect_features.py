"""Report feature-table quality and coverage without possession ground truth."""

import argparse

import numpy as np

from games import GAME, GAMES
from src.features import load_feature_table


def _percent(value):
    return "{:6.2f}%".format(100.0 * value)


def inspect(path):
    expected = GAMES[GAME]["n_video_frames"]
    table = load_feature_table(path, expected)
    graded = table["graded"].to_numpy(dtype=bool)
    ball = (np.isfinite(table["ball_x_m"].to_numpy(dtype=float))
            & np.isfinite(table["ball_y_m"].to_numpy(dtype=float)))
    tracked = table["n_tracked"].to_numpy(dtype=float)
    home = np.isfinite(table["d_nearest_home_m"].to_numpy(dtype=float))
    away = np.isfinite(table["d_nearest_away_m"].to_numpy(dtype=float))
    jumps = table["ball_jump"].to_numpy(dtype=bool)

    print("feature file: {}".format(path))
    print("frames:       {} (0..{})".format(len(table), len(table) - 1))
    print("graded:       {}  {}".format(int(np.sum(graded)), _percent(np.mean(graded))))
    print("ball present: {}  {} overall".format(int(np.sum(ball)), _percent(np.mean(ball))))
    if graded.any():
        print("              {} within graded frames".format(_percent(np.mean(ball[graded]))))
    print("ball jumps:   {}  {}".format(int(np.sum(jumps)), _percent(np.mean(jumps))))
    print("team distance available: home {}  away {}  both {}".format(
        _percent(np.mean(home)), _percent(np.mean(away)), _percent(np.mean(home & away))))
    finite_tracked = tracked[np.isfinite(tracked)]
    if len(finite_tracked):
        print("tracked players: mean {:.2f}  p05 {:.1f}  median {:.1f}  p95 {:.1f}  max {:.0f}".format(
            float(np.mean(finite_tracked)), float(np.percentile(finite_tracked, 5)),
            float(np.median(finite_tracked)), float(np.percentile(finite_tracked, 95)),
            float(np.max(finite_tracked))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", help="local feature CSV")
    arguments = parser.parse_args()
    inspect(arguments.features)
