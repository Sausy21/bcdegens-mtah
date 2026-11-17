"""Generate a deterministic possession submission from the team feature table."""

import argparse
import os

import numpy as np

from games import (GAME, GAMES, anchor_video_frame, half_video_blocks, require,
                   source_frames)
from src.config import load_config
from src.decoder import decode_features
from src.diagnostics import (print_summary, summarize_labels, write_diagnostics,
                             write_transitions)
from src.feature_builder import build_feature_table, load_feature_config
from src.features import load_feature_table, validate_feature_table
from src.smart_tagging import (add_dead_ball_features, load_dead_ball_config,
                               load_events)
from src.submission import file_sha256, write_submission


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "possession_rules.json")
DEFAULT_FEATURE_CONFIG = os.path.join(ROOT, "config", "feature_generation.json")
DEFAULT_DEAD_BALL_CONFIG = os.path.join(ROOT, "config", "dead_ball_rules.json")


def run(output_path, features_path=None, data_dir=None, config_path=DEFAULT_CONFIG,
        feature_config_path=DEFAULT_FEATURE_CONFIG,
        dead_ball_config_path=DEFAULT_DEAD_BALL_CONFIG,
        tagging_path=None, use_smart_tagging=True, transitions_path=None,
        diagnostics_path=None):
    expected_frames = GAMES[GAME]["n_video_frames"]
    config = load_config(config_path)
    if (features_path is None) == (data_dir is None):
        raise ValueError("provide exactly one of features_path or data_dir")
    if features_path is not None:
        features = load_feature_table(features_path, expected_frames)
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
        features = build_feature_table(
            meta_path=meta_path,
            raw_path=raw_path,
            n_frames=source_frames("tactical", GAME),
            pitch=GAMES[GAME]["pitch"],
            active_blocks=half_video_blocks(GAME),
            config=load_feature_config(feature_config_path),
            fps=GAMES[GAME]["fps"],
        )
        features = validate_feature_table(features, expected_frames)
        if tagging_path is None and use_smart_tagging:
            tagging_path = require(["smart_tagging"], GAME, data_dir)[0]
    if tagging_path is not None and use_smart_tagging:
        features, report = add_dead_ball_features(
            features, load_events(tagging_path, GAMES[GAME]["fps"]),
            load_dead_ball_config(dead_ball_config_path))
        detected = sum(row["restart_frame"] is not None for row in report)
        print("Smart Tagging restarts detected: {} of {} set pieces".format(
            detected, len(report)))
    play_starts = {
        1: anchor_video_frame(GAMES[GAME]["kickoff"]),
        2: anchor_video_frame(GAMES[GAME]["kickoff_2nd"]),
    }
    transitions = []
    labels = decode_features(features, half_video_blocks(GAME), config,
                             play_start_frames=play_starts,
                             transition_log=transitions)

    output_folder = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    write_submission(output_path, labels, expected_frames)

    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("wrote {}".format(os.path.abspath(output_path)))
    print("rows {}  labels {}".format(expected_frames, counts))
    print("sha256 {}".format(file_sha256(output_path)))
    print("format validation: OK")
    summary = summarize_labels(labels, GAMES[GAME]["fps"])
    print_summary(summary)
    if transitions_path:
        write_transitions(transitions_path, transitions)
        print("transition log {}".format(os.path.abspath(transitions_path)))
    if diagnostics_path:
        write_diagnostics(diagnostics_path, summary)
        print("diagnostics {}".format(os.path.abspath(diagnostics_path)))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--features", help="local feature CSV (never committed)")
    inputs.add_argument("--data-dir", help="build features in memory from tactical ATD")
    parser.add_argument("--output", required=True, help="submission CSV to create")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="decoder configuration JSON")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG,
                        help="feature generation configuration JSON")
    parser.add_argument("--dead-ball-config", default=DEFAULT_DEAD_BALL_CONFIG,
                        help="restart/dead-ball configuration JSON")
    parser.add_argument("--tagging-xml",
                        help="optional Smart Tagging XML when using a saved feature CSV")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="disable Smart Tagging even for a raw data directory")
    parser.add_argument("--debug-transitions",
                        help="write a local frame-level transition log (never commit it)")
    parser.add_argument("--diagnostics",
                        help="write aggregate structural diagnostics as JSON")
    arguments = parser.parse_args()
    run(output_path=arguments.output, features_path=arguments.features,
        data_dir=arguments.data_dir, config_path=arguments.config,
        feature_config_path=arguments.feature_config,
        dead_ball_config_path=arguments.dead_ball_config,
        tagging_path=arguments.tagging_xml,
        use_smart_tagging=not arguments.no_smart_tagging,
        transitions_path=arguments.debug_transitions,
        diagnostics_path=arguments.diagnostics)
