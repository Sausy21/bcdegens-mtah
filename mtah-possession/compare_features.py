"""Compare two local feature tables without using possession ground truth."""

import argparse

import numpy as np

from games import GAME, GAMES
from src.features import REQUIRED_COLUMNS, load_feature_table


def compare(left_path, right_path, expected=None):
    expected = expected or GAMES[GAME]["n_video_frames"]
    left = load_feature_table(left_path, expected)
    right = load_feature_table(right_path, expected)

    print("rows {}  frame alignment: exact".format(expected))
    print("column                         missing mismatch    MAE/disagreement       p95        max")
    print("-----------------------------  ----------------  ----------------  ---------  ---------")
    for name in REQUIRED_COLUMNS:
        if name == "frame":
            continue
        if left[name].dtype == bool and right[name].dtype == bool:
            disagreement = float(np.mean(left[name].to_numpy() != right[name].to_numpy()))
            print("{:<29}  {:>16}  {:>15.3%}".format(name, "-", disagreement))
            continue

        a = left[name].to_numpy(dtype=float)
        b = right[name].to_numpy(dtype=float)
        finite_a = np.isfinite(a)
        finite_b = np.isfinite(b)
        missing_mismatch = int(np.sum(finite_a != finite_b))
        both = finite_a & finite_b
        if both.any():
            difference = np.abs(a[both] - b[both])
            mae = float(np.mean(difference))
            p95 = float(np.percentile(difference, 95))
            maximum = float(np.max(difference))
            stats = "{:>16.6g}  {:>9.6g}  {:>9.6g}".format(mae, p95, maximum)
        else:
            stats = "{:>16}  {:>9}  {:>9}".format("n/a", "n/a", "n/a")
        print("{:<29}  {:>16}  {}".format(name, missing_mismatch, stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="first local feature CSV")
    parser.add_argument("right", help="second local feature CSV")
    parser.add_argument("--rows", type=int, default=GAMES[GAME]["n_video_frames"],
                        help="expected row count (default: full tactical match)")
    arguments = parser.parse_args()
    compare(arguments.left, arguments.right, arguments.rows)
