"""Ground-truth-free structural diagnostics for a possession submission."""

import csv
import json
import os

import numpy as np


def _runs(labels, wanted):
    runs = []
    start = None
    current = None
    for frame, label in enumerate(labels):
        if label in wanted:
            if label != current:
                if start is not None:
                    runs.append((start, frame, current))
                start = frame
                current = label
        else:
            if start is not None:
                runs.append((start, frame, current))
                start = None
                current = None
    if start is not None:
        runs.append((start, len(labels), current))
    return runs


def summarize_labels(labels, fps=25):
    labels = np.asarray(labels, dtype=object)
    counts = {name: int(np.sum(labels == name)) for name in ("H", "A", "D", "X")}
    possessions = _runs(labels, {"H", "A"})
    dead_spells = _runs(labels, {"D"})
    durations = np.array([end - start for start, end, _ in possessions], dtype=float)
    dead_durations = np.array([end - start for start, end, _ in dead_spells], dtype=float)
    alive = counts["H"] + counts["A"]
    graded = alive + counts["D"]

    return {
        "label_counts": counts,
        "possessions": len(possessions),
        "mean_possession_s": float(np.mean(durations) / fps) if len(durations) else 0.0,
        "median_possession_s": float(np.median(durations) / fps) if len(durations) else 0.0,
        "longest_possession_s": float(np.max(durations) / fps) if len(durations) else 0.0,
        "possessions_at_most_1s": int(np.sum(durations <= fps)),
        "dead_ball_spells": len(dead_spells),
        "dead_ball_s": float(np.sum(dead_durations) / fps),
        "dead_ball_share_graded": float(counts["D"] / graded) if graded else 0.0,
        "home_share_alive": float(counts["H"] / alive) if alive else 0.0,
    }


def print_summary(summary):
    print("structure: {} possessions, mean {:.1f}s, median {:.1f}s, longest {:.1f}s".format(
        summary["possessions"], summary["mean_possession_s"],
        summary["median_possession_s"], summary["longest_possession_s"]))
    print("           {} possessions <=1s; {} dead-ball spells, {:.1%} of graded frames".format(
        summary["possessions_at_most_1s"], summary["dead_ball_spells"],
        summary["dead_ball_share_graded"]))
    print("           home share of alive frames {:.1%}".format(summary["home_share_alive"]))


def _ensure_parent(path):
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)


def write_diagnostics(path, summary):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_transitions(path, transitions):
    _ensure_parent(path)
    columns = ("frame", "half", "from", "to", "reason", "evidence")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(transitions)
