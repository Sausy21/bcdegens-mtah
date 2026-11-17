"""Generate the decoder feature table directly from raw tactical ATD files."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd


HOME = 0
AWAY = 1


@dataclass(frozen=True)
class FeatureConfig:
    speed_smoothing_window_frames: int = 5
    jump_speed_m_s: float = 40.0

    def validate(self):
        window = self.speed_smoothing_window_frames
        if window < 1 or window % 2 == 0:
            raise ValueError("speed_smoothing_window_frames must be a positive odd integer")
        if self.jump_speed_m_s <= 0:
            raise ValueError("jump_speed_m_s must be positive")
        return self


def load_feature_config(path):
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    allowed = set(FeatureConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown feature configuration keys: {}".format(unknown))
    return FeatureConfig(**values).validate()


def team_of_each_track(meta_path):
    """Return a HOME/AWAY code for every track slot in FIFA/EPTS order."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    unknown = sorted(set(slots) - set(team_ids))
    if unknown:
        raise ValueError("{}: player slots reference unknown teams {}".format(meta_path, unknown))
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots],
                    dtype=np.int8)


def _parse_group(group, pitch):
    """Parse x,y,speed; return metre coordinates and m/s speed, or None."""
    group = group.strip()
    if not group or group[0].upper() == "N":
        return None
    values = group.split(",")
    if len(values) < 3:
        raise ValueError("invalid ATD object group: {!r}".format(group))
    try:
        x = float(values[0]) * pitch[0]
        y = float(values[1]) * pitch[1]
        speed = float(values[2])
    except ValueError as error:
        raise ValueError("invalid ATD object group: {!r}".format(group)) from error
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y, speed


def _graded_mask(n_frames, active_blocks):
    graded = np.zeros(n_frames, dtype=bool)
    for _half, first, last in active_blocks:
        if first < 0 or last < first or last >= n_frames:
            raise ValueError("invalid active block {}..{}".format(first, last))
        graded[first:last + 1] = True
    return graded


def build_feature_table(meta_path, raw_path, n_frames, pitch, active_blocks,
                        config=None, fps=25):
    """Build one feature row per video frame with no kloppy dependency.

    The raw ATD frame counter is one-based and positions are normalized pitch coordinates.
    Ball speed is derived from consecutive positions because the raw ball speed channel does
    not behave like metres per second on the provided export.
    """
    config = (config or FeatureConfig()).validate()
    teams = team_of_each_track(meta_path)

    ball_x = np.full(n_frames, np.nan, dtype=float)
    ball_y = np.full(n_frames, np.nan, dtype=float)
    n_tracked = np.zeros(n_frames, dtype=np.int16)
    d_home = np.full(n_frames, np.nan, dtype=float)
    d_away = np.full(n_frames, np.nan, dtype=float)
    n_home_near = np.full(n_frames, np.nan, dtype=float)
    n_away_near = np.full(n_frames, np.nan, dtype=float)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}:{} expected three colon-separated fields".format(
                    raw_path, line_number))
            try:
                frame = int(parts[0]) - 1
            except ValueError as error:
                raise ValueError("{}:{} invalid frame id {!r}".format(
                    raw_path, line_number, parts[0])) from error
            if frame < 0 or frame >= n_frames:
                raise ValueError("{}:{} frame {} is outside 1..{}".format(
                    raw_path, line_number, frame + 1, n_frames))
            if seen[frame]:
                raise ValueError("{}:{} duplicate frame {}".format(
                    raw_path, line_number, frame + 1))
            seen[frame] = True

            groups = parts[1].split(";")
            if len(groups) != len(teams):
                raise ValueError("{}:{} player group count {} != metadata slots {}".format(
                    raw_path, line_number, len(groups), len(teams)))
            players = []
            player_teams = []
            for slot, group in enumerate(groups):
                parsed = _parse_group(group, pitch)
                if parsed is not None:
                    players.append(parsed[:2])
                    player_teams.append(teams[slot])
            n_tracked[frame] = len(players)

            ball = _parse_group(parts[2], pitch)
            if ball is None:
                continue
            ball_x[frame], ball_y[frame] = ball[:2]
            n_home_near[frame] = 0
            n_away_near[frame] = 0
            if not players:
                continue

            positions = np.asarray(players, dtype=float)
            player_teams = np.asarray(player_teams, dtype=np.int8)
            distances = np.sqrt(np.sum(
                (positions - np.asarray(ball[:2], dtype=float)) ** 2, axis=1))
            for team, destination, near_destination in (
                    (HOME, d_home, n_home_near), (AWAY, d_away, n_away_near)):
                selected = distances[player_teams == team]
                if len(selected):
                    destination[frame] = float(np.min(selected))
                    near_destination[frame] = int(np.sum(selected <= 5.0))

    missing = np.flatnonzero(~seen)
    if len(missing):
        preview = ", ".join(str(int(frame + 1)) for frame in missing[:5])
        raise ValueError("ATD is missing {} frame records; first: {}".format(
            len(missing), preview))

    ball_speed = np.full(n_frames, np.nan, dtype=float)
    consecutive = (np.isfinite(ball_x[1:]) & np.isfinite(ball_y[1:])
                   & np.isfinite(ball_x[:-1]) & np.isfinite(ball_y[:-1]))
    delta = np.hypot(ball_x[1:] - ball_x[:-1], ball_y[1:] - ball_y[:-1])
    consecutive_frames = np.flatnonzero(consecutive) + 1
    ball_speed[consecutive_frames] = delta[consecutive] * float(fps)

    speed_series = pd.Series(ball_speed)
    smooth_speed = speed_series.rolling(
        config.speed_smoothing_window_frames, center=True,
        min_periods=(config.speed_smoothing_window_frames + 1) // 2
    ).median().to_numpy(copy=True)
    ball_jump = np.isfinite(ball_speed) & (ball_speed > config.jump_speed_m_s)

    return pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int64),
        "graded": _graded_mask(n_frames, active_blocks),
        "ball_x_m": ball_x,
        "ball_y_m": ball_y,
        "ball_speed_ms": ball_speed,
        "ball_speed_smooth_ms": smooth_speed,
        "ball_jump": ball_jump,
        "n_tracked": n_tracked,
        "d_nearest_home_m": d_home,
        "d_nearest_away_m": d_away,
        "n_home_within_5m": n_home_near,
        "n_away_within_5m": n_away_near,
    })
