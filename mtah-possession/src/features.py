"""Read and validate the team feature-table contract."""

import numpy as np
import pandas as pd


FEATURE_DECIMALS = 3

REQUIRED_COLUMNS = (
    "frame",
    "graded",
    "ball_x_m",
    "ball_y_m",
    "ball_speed_ms",
    "ball_speed_smooth_ms",
    "ball_jump",
    "n_tracked",
    "d_nearest_home_m",
    "d_nearest_away_m",
    "n_home_within_5m",
    "n_away_within_5m",
)

NUMERIC_COLUMNS = tuple(name for name in REQUIRED_COLUMNS
                        if name not in ("graded", "ball_jump"))
FLOAT_FEATURE_COLUMNS = (
    "ball_x_m",
    "ball_y_m",
    "ball_speed_ms",
    "ball_speed_smooth_ms",
    "d_nearest_home_m",
    "d_nearest_away_m",
)


def coerce_bool(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "nan": False, "": False,
    }
    invalid = sorted(set(lowered) - set(mapping))
    if invalid:
        raise ValueError("{} contains non-boolean values: {}".format(name, invalid[:5]))
    return lowered.map(mapping).astype(bool)


def validate_feature_table(table, expected_frames):
    """Return a normalized copy or raise with a precise contract violation."""
    table = table.copy()
    table.columns = [str(column).strip() for column in table.columns]

    missing = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError("feature table is missing columns: {}".format(missing))
    if len(table) != expected_frames:
        raise ValueError("feature row count {} != {}".format(len(table), expected_frames))

    for name in NUMERIC_COLUMNS:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    frames = table["frame"].to_numpy()
    if np.isnan(frames).any() or not np.array_equal(frames, np.arange(expected_frames)):
        raise ValueError("feature frame column must be the integers 0..{} in order".format(
            expected_frames - 1))
    table["frame"] = table["frame"].astype(np.int64)
    table["graded"] = coerce_bool(table["graded"], "graded")
    table["ball_jump"] = coerce_bool(table["ball_jump"], "ball_jump")
    for name in FLOAT_FEATURE_COLUMNS:
        table[name] = table[name].round(FEATURE_DECIMALS)

    tracked = table["n_tracked"].to_numpy(dtype=float)
    if not np.isfinite(tracked).all() or (tracked < 0).any() or not np.equal(
            tracked, np.floor(tracked)).all():
        raise ValueError("n_tracked must contain non-negative integer counts")
    table["n_tracked"] = table["n_tracked"].astype(np.int64)
    return table


def load_feature_table(path, expected_frames):
    return validate_feature_table(pd.read_csv(path), expected_frames)
