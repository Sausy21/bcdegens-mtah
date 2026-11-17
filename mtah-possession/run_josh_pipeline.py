"""Build the submission with Josh's football rules (game_logic/rules.py).

Usage, from mtah-possession/:
    python tools/explore_data.py DATA/tactical/BAR-ATH_tactical_atd.txt DATA/tactical/BAR-ATH_tactical_atd.xml DATA/tactical/BAR-ATH_tactical_smart_tagging.xml --out outputs
    python run_josh_pipeline.py --data-dir DATA --features outputs/features.csv --output outputs/BarcaClubDegens.csv

Steps:
  1. Josh's rules turn the feature table into H/A/D per frame (control, dead ball from player
     movement and the ball leaving the pitch, restarts, Smart Tagging team in long tracking gaps).
  2. Each half is dead ball until its kick-off, using the kick-off anchors published in games.py
     (the same choice as run_pipeline.py).
  3. A graded frame the rules leave undecided (X) keeps the previous label in its half.
Frames outside the graded halves are X. Deterministic: same data, same file.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "game_logic"))
import rules  # noqa: E402
from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require  # noqa: E402


def smart_tagging_events(tagging_path, fps):
    events = []
    for inst in ET.parse(tagging_path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        events.append({
            "code": inst.findtext("code"),
            "tags": tags,
            "start_frame": int(round(float(inst.findtext("start")) * fps)),
            "end_frame": int(round(float(inst.findtext("end")) * fps)),
        })
    return events


def main():
    parser = argparse.ArgumentParser(description="Build the submission with Josh's rules")
    parser.add_argument("--data-dir", required=True, help="folder holding tactical/")
    parser.add_argument("--features", required=True, help="features.csv from tools/explore_data.py")
    parser.add_argument("--output", required=True, help="submission CSV to write")
    args = parser.parse_args()

    game = GAMES[GAME]
    fps, pitch, n_frames = game["fps"], tuple(game["pitch"]), game["n_video_frames"]
    meta_path, tagging_path = require(["atd_meta", "smart_tagging"], GAME, args.data_dir)

    features = pd.read_csv(args.features)
    table = rules.from_feature_table(features, pitch=pitch)
    team_names = [team.findtext("Name") for team in ET.parse(meta_path).getroot().iter("Team")]
    tag_team, _, _ = rules.tag_team_from_events(smart_tagging_events(tagging_path, fps),
                                                len(table), team_names[0], team_names[1])
    graded = features["graded"].to_numpy(bool)
    labels, _ = rules.run_pipeline(table, tag_team, graded, pitch)

    kickoffs = {1: anchor_video_frame(game["kickoff"]), 2: anchor_video_frame(game["kickoff_2nd"])}
    for half, first, last in half_video_blocks(GAME):
        labels[first:min(kickoffs[half], last + 1)] = "D"
        for frame in range(first + 1, last + 1):
            if labels[frame] == "X":
                labels[frame] = labels[frame - 1]

    if len(labels) != n_frames:
        sys.exit(f"expected {n_frames} labels, got {len(labels)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame({"frame": np.arange(n_frames), "label": labels}).to_csv(args.output, index=False)

    counts = {k: int((labels[graded] == k).sum()) for k in "HADX"}
    print(f"Wrote {args.output}. Graded frames: H {counts['H']}, A {counts['A']}, D {counts['D']}, X {counts['X']}")


if __name__ == "__main__":
    main()
