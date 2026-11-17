"""Parse Smart Tagging and derive conservative set-piece dead-ball evidence."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .features import coerce_bool


@dataclass(frozen=True)
class DeadBallConfig:
    set_piece_code: str = "SET PIECES"
    restart_motion_speed_m_s: float = 1.0
    restart_control_radius_m: float = 3.0
    restart_sustain_frames: int = 3
    maximum_event_window_frames: int = 375

    def validate(self):
        if not self.set_piece_code.strip():
            raise ValueError("set_piece_code cannot be empty")
        if self.restart_motion_speed_m_s < 0:
            raise ValueError("restart_motion_speed_m_s cannot be negative")
        if self.restart_control_radius_m <= 0:
            raise ValueError("restart_control_radius_m must be positive")
        if self.restart_sustain_frames < 1:
            raise ValueError("restart_sustain_frames must be at least 1")
        if self.maximum_event_window_frames < self.restart_sustain_frames:
            raise ValueError("maximum_event_window_frames is too short")
        return self


def load_dead_ball_config(path):
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    allowed = set(DeadBallConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown dead-ball configuration keys: {}".format(unknown))
    return DeadBallConfig(**values).validate()


def load_events(path, fps=25):
    """Read SportsCode instances into frame-aligned dictionaries."""
    events = []
    for index, instance in enumerate(ET.parse(path).iter("instance")):
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError) as error:
            raise ValueError("Smart Tagging instance {} has invalid timestamps".format(
                index)) from error
        if end_s < start_s:
            raise ValueError("Smart Tagging instance {} ends before it starts".format(index))
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        events.append({
            "index": index,
            "code": (instance.findtext("code") or "").strip(),
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": tags,
        })
    return events


def _first_sustained(mask, start, end, length):
    run = 0
    for frame in range(start, end):
        if mask[frame]:
            run += 1
            if run >= length:
                return frame - length + 1
        else:
            run = 0
    return None


def add_dead_ball_features(features, events, config=None):
    """Add conservative dead-ball evidence and return (table, event report).

    A set-piece window is not itself a dead-ball interval. The inferred restart is the first
    sustained, non-jump ball movement with a player close enough to control the ball. Only the
    portion from the tag-window start up to that detected restart is marked dead. Events without
    a defensible restart are reported but do not create labels.
    """
    config = (config or DeadBallConfig()).validate()
    table = features.copy()
    count = len(table)
    set_piece = np.zeros(count, dtype=bool)
    restart = np.zeros(count, dtype=bool)
    dead = np.zeros(count, dtype=bool)

    ball_known = (np.isfinite(table["ball_x_m"].to_numpy(dtype=float))
                  & np.isfinite(table["ball_y_m"].to_numpy(dtype=float)))
    speed = table["ball_speed_ms"].to_numpy(dtype=float)
    jump = table["ball_jump"].to_numpy(dtype=bool)
    home = table["d_nearest_home_m"].to_numpy(dtype=float)
    away = table["d_nearest_away_m"].to_numpy(dtype=float)
    nearest = np.fmin(home, away)
    moving_controlled = (ball_known & np.isfinite(speed) & ~jump
                         & (speed >= config.restart_motion_speed_m_s)
                         & np.isfinite(nearest)
                         & (nearest <= config.restart_control_radius_m))

    report = []
    wanted_code = config.set_piece_code.strip().casefold()
    for event in events:
        if event["code"].casefold() != wanted_code:
            continue
        original_start = event["start_frame"]
        original_end = event["end_frame"]
        start = max(0, min(count, original_start))
        end = max(start, min(count, original_end))
        if start == end:
            detected = None
        else:
            set_piece[start:end] = True
            detected = _first_sustained(
                moving_controlled, start, end, config.restart_sustain_frames)
        if detected is not None:
            dead_start = max(start, detected - config.maximum_event_window_frames)
            dead[dead_start:detected] = True
            restart[detected] = True
        else:
            dead_start = None
        report.append({
            "event_index": event["index"],
            "type": event["tags"].get("Type"),
            "half": event["tags"].get("Half"),
            "start_frame": original_start,
            "end_frame": original_end,
            "restart_frame": detected,
            "dead_start_frame": dead_start,
            "dead_frames": 0 if detected is None else max(0, detected - dead_start),
        })

    if "dead_ball" in table.columns:
        existing = coerce_bool(table["dead_ball"], "dead_ball").to_numpy()
        dead |= existing
    table["smart_set_piece"] = set_piece
    table["smart_restart"] = restart
    table["dead_ball"] = dead
    return table, report
