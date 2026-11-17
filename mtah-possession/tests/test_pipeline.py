import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.config import DecoderConfig
from src.decoder import decode_features
from src.diagnostics import summarize_labels
from src.feature_builder import FeatureConfig, build_feature_table
from src.features import load_feature_table, validate_feature_table
from src.smart_tagging import DeadBallConfig, add_dead_ball_features, load_events
from src.submission import file_sha256, write_submission
from release_check import sensitive_tracked_paths


def feature_table(count):
    return pd.DataFrame({
        "frame": np.arange(count),
        "graded": np.ones(count, dtype=bool),
        "ball_x_m": np.ones(count),
        "ball_y_m": np.ones(count),
        "ball_speed_ms": np.zeros(count),
        "ball_speed_smooth_ms": np.zeros(count),
        "ball_jump": np.zeros(count, dtype=bool),
        "n_tracked": np.full(count, 22),
        "d_nearest_home_m": np.full(count, 1.0),
        "d_nearest_away_m": np.full(count, 4.0),
        "n_home_within_5m": np.ones(count),
        "n_away_within_5m": np.zeros(count),
    })


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = DecoderConfig(
            switch_hold_frames=3,
            restart_control_hold_frames=2,
            dead_ball_hold_frames=1,
        ).validate()

    def test_sustained_control_switches_but_single_touch_does_not(self):
        table = feature_table(10)
        table.loc[4:4, ["d_nearest_home_m", "d_nearest_away_m"]] = [4.0, 1.0]
        table.loc[7:9, ["d_nearest_home_m", "d_nearest_away_m"]] = [4.0, 1.0]
        labels = decode_features(table, [(1, 0, 9)], self.config)
        self.assertEqual(list(labels[:2]), ["D", "H"])
        self.assertEqual(labels[4], "H")
        self.assertEqual(list(labels[7:]), ["H", "H", "A"])

    def test_transition_log_explains_restart_and_team_switch(self):
        table = feature_table(7)
        table.loc[4:6, ["d_nearest_home_m", "d_nearest_away_m"]] = [4.0, 1.0]
        transitions = []
        decode_features(table, [(1, 0, 6)], self.config,
                        transition_log=transitions)
        self.assertEqual([row["reason"] for row in transitions],
                         ["restart", "sustained_control"])

    def test_missing_ball_and_tracking_jump_carry_state(self):
        table = feature_table(6)
        table.loc[2, ["ball_x_m", "ball_y_m"]] = [np.nan, np.nan]
        table.loc[3, "ball_jump"] = True
        labels = decode_features(table, [(1, 0, 5)], self.config)
        self.assertEqual(list(labels), ["D", "H", "H", "H", "H", "H"])

    def test_published_kickoff_keeps_prestart_frames_dead(self):
        table = feature_table(8)
        labels = decode_features(table, [(2, 0, 7)], self.config,
                                 play_start_frames={2: 4})
        self.assertEqual(list(labels[:5]), ["D", "D", "D", "D", "D"])
        self.assertEqual(labels[5], "H")

    def test_dead_ball_feature_and_outside_match_x(self):
        table = feature_table(8)
        table["dead_ball"] = "false"
        table.loc[4, "dead_ball"] = "true"
        labels = decode_features(table, [(1, 1, 6)], self.config)
        self.assertEqual(labels[0], "X")
        self.assertEqual(labels[4], "D")
        self.assertEqual(labels[7], "X")

    def test_feature_contract_rejects_frame_gap(self):
        table = feature_table(4)
        table.loc[2, "frame"] = 9
        with self.assertRaisesRegex(ValueError, "frame column"):
            validate_feature_table(table, 4)

    def test_feature_precision_is_canonical_across_memory_and_csv(self):
        table = feature_table(4)
        table.loc[1, "d_nearest_home_m"] = 1.9996
        normalized = validate_feature_table(table, 4)
        self.assertEqual(normalized.loc[1, "d_nearest_home_m"], 2.0)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "features.csv")
            normalized.to_csv(path, index=False, float_format="%.3f")
            loaded = load_feature_table(path, 4)
        pd.testing.assert_frame_equal(normalized, loaded, check_dtype=False)

    def test_submission_is_valid_and_deterministic(self):
        labels = np.array(["X", "D", "H", "H", "A", "X"])
        with tempfile.TemporaryDirectory() as folder:
            first = os.path.join(folder, "first.csv")
            second = os.path.join(folder, "second.csv")
            write_submission(first, labels, len(labels))
            write_submission(second, labels, len(labels))
            self.assertEqual(file_sha256(first), file_sha256(second))

    def test_raw_atd_feature_generation(self):
        xml = """<Root>
          <Team id="home"><Name>Home</Name></Team>
          <Team id="away"><Name>Away</Name></Team>
          <Player teamId="home"/><Player teamId="away"/>
        </Root>"""
        raw = "\n".join((
            "1:0.10,0.50,0;0.90,0.50,0:0.11,0.50,2",
            "2:0.11,0.50,0;0.89,0.50,0:NaN,NaN,NaN",
            "3:0.12,0.50,0;0.88,0.50,0:0.87,0.50,3",
            "4:NaN,NaN,NaN;0.87,0.50,0:0.86,0.50,2",
        )) + "\n"
        with tempfile.TemporaryDirectory() as folder:
            meta_path = os.path.join(folder, "atd.xml")
            raw_path = os.path.join(folder, "atd.txt")
            with open(meta_path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write(raw)
            table = build_feature_table(
                meta_path, raw_path, 4, (105.3, 68.0), [(1, 0, 3)], FeatureConfig())

        self.assertEqual(list(table["frame"]), [0, 1, 2, 3])
        self.assertEqual(list(table["n_tracked"]), [2, 2, 2, 1])
        self.assertAlmostEqual(table.loc[0, "ball_x_m"], 0.11 * 105.3)
        self.assertAlmostEqual(table.loc[0, "d_nearest_home_m"], 0.01 * 105.3)
        self.assertTrue(np.isnan(table.loc[1, "ball_x_m"]))
        self.assertTrue(np.isnan(table.loc[0, "ball_speed_ms"]))
        self.assertTrue(np.isnan(table.loc[2, "ball_speed_ms"]))
        self.assertAlmostEqual(table.loc[3, "ball_speed_ms"], 0.01 * 105.3 * 25)
        self.assertAlmostEqual(table.loc[2, "d_nearest_away_m"], 0.01 * 105.3)
        self.assertEqual(table.loc[3, "n_home_within_5m"], 0)

    def test_raw_atd_rejects_missing_frame(self):
        xml = """<Root><Team id="h"/><Team id="a"/>
          <Player teamId="h"/><Player teamId="a"/></Root>"""
        raw = "1:0,0,0;1,1,0:0,0,0\n3:0,0,0;1,1,0:0,0,0\n"
        with tempfile.TemporaryDirectory() as folder:
            meta_path = os.path.join(folder, "atd.xml")
            raw_path = os.path.join(folder, "atd.txt")
            with open(meta_path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.write(raw)
            with self.assertRaisesRegex(ValueError, "missing"):
                build_feature_table(meta_path, raw_path, 3, (100.0, 60.0),
                                    [(1, 0, 2)], FeatureConfig())

    def test_release_check_finds_restricted_artifacts(self):
        paths = [
            "mtah-possession/data/README.md",
            "mtah-possession/data/tactical/match.mp4",
            "mtah-possession/local_features.csv",
            "mtah-possession/src/decoder.py",
        ]
        self.assertEqual(sensitive_tracked_paths(paths), paths[1:3])

    def test_smart_tagging_marks_only_pre_restart_frames_dead(self):
        table = feature_table(12)
        table["ball_speed_ms"] = 0.0
        table.loc[5:7, "ball_speed_ms"] = 2.0
        xml = """<Root><instance><start>0.1</start><end>1.0</end>
          <code>SET PIECES</code>
          <label><group>Type</group><text>Free Kick</text></label>
          <label><group>Half</group><text>1st Half</text></label>
        </instance></Root>"""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tagging.xml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            events = load_events(path, fps=10)
        augmented, report = add_dead_ball_features(
            table, events, DeadBallConfig(restart_sustain_frames=3))
        self.assertEqual(report[0]["restart_frame"], 5)
        self.assertEqual(list(np.flatnonzero(augmented["dead_ball"])), [1, 2, 3, 4])
        self.assertTrue(augmented.loc[5, "smart_restart"])
        self.assertFalse(augmented.loc[5, "dead_ball"])

    def test_smart_tagging_without_sustained_motion_does_not_invent_dead_ball(self):
        table = feature_table(8)
        table["ball_speed_ms"] = 0.0
        events = [{
            "index": 0, "code": "SET PIECES", "start_frame": 1, "end_frame": 7,
            "tags": {"Type": "Throw-in"},
        }]
        augmented, report = add_dead_ball_features(table, events, DeadBallConfig())
        self.assertIsNone(report[0]["restart_frame"])
        self.assertFalse(augmented["dead_ball"].any())

    def test_long_tag_window_searches_fully_but_caps_dead_ball_duration(self):
        table = feature_table(30)
        table["ball_speed_ms"] = 0.0
        table.loc[25:27, "ball_speed_ms"] = 2.0
        events = [{
            "index": 0, "code": "SET PIECES", "start_frame": 0, "end_frame": 30,
            "tags": {"Type": "Throw-in"},
        }]
        augmented, report = add_dead_ball_features(
            table, events, DeadBallConfig(restart_sustain_frames=3,
                                          maximum_event_window_frames=10))
        self.assertEqual(report[0]["restart_frame"], 25)
        self.assertEqual(report[0]["dead_start_frame"], 15)
        self.assertEqual(list(np.flatnonzero(augmented["dead_ball"])), list(range(15, 25)))

    def test_structural_diagnostics(self):
        summary = summarize_labels(np.array(["X", "D", "D", "H", "H", "A", "D", "X"]),
                                   fps=2)
        self.assertEqual(summary["possessions"], 2)
        self.assertEqual(summary["dead_ball_spells"], 2)
        self.assertEqual(summary["possessions_at_most_1s"], 2)
        self.assertAlmostEqual(summary["dead_ball_share_graded"], 3 / 6)


if __name__ == "__main__":
    unittest.main()
