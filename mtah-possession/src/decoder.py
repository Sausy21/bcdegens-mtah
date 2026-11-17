"""Deterministic frame evidence and possession-state decoding."""

import numpy as np

from .config import DecoderConfig
from .features import coerce_bool


TEAM_LABELS = ("H", "A")
VALID_EVIDENCE = ("H", "A", "D", None)


def infer_raw_evidence(features, config):
    """Infer H/A/D evidence per frame without applying temporal state rules.

    Missing ball locations, tracking jumps, and inconclusive distances intentionally
    produce None. The temporal decoder decides how uncertainty affects the current state.
    """
    if not isinstance(config, DecoderConfig):
        raise TypeError("config must be DecoderConfig")

    count = len(features)
    evidence = np.full(count, None, dtype=object)

    if config.dead_ball_feature in features.columns:
        dead = coerce_bool(features[config.dead_ball_feature],
                           config.dead_ball_feature).to_numpy()
        evidence[dead] = "D"
    else:
        dead = np.zeros(count, dtype=bool)

    ball_known = (np.isfinite(features["ball_x_m"].to_numpy(dtype=float))
                  & np.isfinite(features["ball_y_m"].to_numpy(dtype=float)))
    usable = ball_known & ~features["ball_jump"].to_numpy(dtype=bool) & ~dead

    home = features["d_nearest_home_m"].to_numpy(dtype=float)
    away = features["d_nearest_away_m"].to_numpy(dtype=float)
    home_known = np.isfinite(home)
    away_known = np.isfinite(away)
    if config.require_both_team_distances:
        usable &= home_known & away_known

    home_control = (usable & home_known & (home <= config.control_radius_m)
                    & (~away_known | (home + config.opponent_margin_m <= away)))
    away_control = (usable & away_known & (away <= config.control_radius_m)
                    & (~home_known | (away + config.opponent_margin_m <= home)))

    evidence[home_control] = "H"
    evidence[away_control] = "A"
    return evidence


def _hold_required(current, candidate, config):
    if candidate == "D":
        return config.dead_ball_hold_frames
    if current == "D" and candidate in TEAM_LABELS:
        return config.restart_control_hold_frames
    return config.switch_hold_frames


def decode_evidence(evidence, active_blocks, config, play_start_frames=None,
                    transition_log=None):
    """Decode raw evidence into H/A/D/X labels.

    `active_blocks` contains inclusive (half, first_frame, last_frame) tuples. State
    is reset at each half; frames outside those blocks are X. `play_start_frames`
    can keep the early part of a graded half at D until its published kickoff frame.
    """
    evidence = np.asarray(evidence, dtype=object)
    invalid = sorted({value for value in evidence if value not in VALID_EVIDENCE}, key=str)
    if invalid:
        raise ValueError("invalid raw evidence values: {}".format(invalid))

    labels = np.full(len(evidence), "X", dtype="<U1")
    occupied = np.zeros(len(evidence), dtype=bool)

    play_start_frames = play_start_frames or {}
    for half, first, last in active_blocks:
        if first < 0 or last < first or last >= len(evidence):
            raise ValueError("invalid active block {}..{} for {} frames".format(
                first, last, len(evidence)))
        if occupied[first:last + 1].any():
            raise ValueError("active blocks overlap")
        occupied[first:last + 1] = True

        current = config.initial_half_label
        candidate = None
        candidate_frames = 0
        loose_frames = 0

        for frame in range(first, last + 1):
            if frame < play_start_frames.get(half, first):
                labels[frame] = "D"
                continue
            raw = evidence[frame]
            if raw is None:
                candidate = None
                candidate_frames = 0
                loose_frames += 1
                if (config.max_loose_frames is not None
                        and current in TEAM_LABELS
                        and loose_frames > config.max_loose_frames):
                    previous = current
                    current = "D"
                    if transition_log is not None:
                        transition_log.append({
                            "frame": frame, "half": half, "from": previous, "to": current,
                            "reason": "loose_timeout", "evidence": "",
                        })
            else:
                loose_frames = 0
                if raw == current:
                    candidate = None
                    candidate_frames = 0
                else:
                    if raw == candidate:
                        candidate_frames += 1
                    else:
                        candidate = raw
                        candidate_frames = 1
                    if candidate_frames >= _hold_required(current, candidate, config):
                        previous = current
                        if candidate == "D":
                            reason = "dead_ball"
                        elif current == "D":
                            reason = "restart"
                        else:
                            reason = "sustained_control"
                        current = candidate
                        if transition_log is not None:
                            transition_log.append({
                                "frame": frame, "half": half, "from": previous,
                                "to": current, "reason": reason, "evidence": raw,
                            })
                        candidate = None
                        candidate_frames = 0
            labels[frame] = current

    return labels


def decode_features(features, active_blocks, config, play_start_frames=None,
                    transition_log=None):
    return decode_evidence(infer_raw_evidence(features, config), active_blocks, config,
                           play_start_frames=play_start_frames,
                           transition_log=transition_log)
