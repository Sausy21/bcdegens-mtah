"""Generate the local per-frame feature CSV from tactical ATD data."""

import argparse
import os

from games import GAME, GAMES, half_video_blocks, require, source_frames
from src.feature_builder import build_feature_table, load_feature_config
from src.features import validate_feature_table
from src.smart_tagging import (add_dead_ball_features, load_dead_ball_config,
                               load_events)


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "feature_generation.json")
DEFAULT_DEAD_BALL_CONFIG = os.path.join(ROOT, "config", "dead_ball_rules.json")


def run(data_dir, output_path, config_path=DEFAULT_CONFIG,
        dead_ball_config_path=DEFAULT_DEAD_BALL_CONFIG, use_smart_tagging=True):
    wanted = ["atd_meta", "atd_raw"]
    if use_smart_tagging:
        wanted.append("smart_tagging")
    paths = require(wanted, GAME, data_dir)
    meta_path, raw_path = paths[:2]
    config = load_feature_config(config_path)
    table = build_feature_table(
        meta_path=meta_path,
        raw_path=raw_path,
        n_frames=source_frames("tactical", GAME),
        pitch=GAMES[GAME]["pitch"],
        active_blocks=half_video_blocks(GAME),
        config=config,
        fps=GAMES[GAME]["fps"],
    )
    table = validate_feature_table(table, source_frames("tactical", GAME))
    if use_smart_tagging:
        table, report = add_dead_ball_features(
            table, load_events(paths[2], GAMES[GAME]["fps"]),
            load_dead_ball_config(dead_ball_config_path))
        detected = sum(row["restart_frame"] is not None for row in report)
        print("Smart Tagging restarts detected: {} of {} set pieces".format(
            detected, len(report)))
    folder = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    table.to_csv(output_path, index=False, lineterminator="\n", float_format="%.3f")
    print("wrote {} rows to {}".format(len(table), os.path.abspath(output_path)))
    print("ball present {:0.1f}%".format(100.0 * table["ball_x_m"].notna().mean()))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="root containing tactical/ and broadcast/")
    parser.add_argument("--output", required=True, help="local feature CSV to write")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="feature configuration JSON")
    parser.add_argument("--dead-ball-config", default=DEFAULT_DEAD_BALL_CONFIG,
                        help="restart/dead-ball configuration JSON")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="generate tracking features without restart evidence")
    arguments = parser.parse_args()
    run(arguments.data_dir, arguments.output, arguments.config,
        arguments.dead_ball_config, not arguments.no_smart_tagging)
