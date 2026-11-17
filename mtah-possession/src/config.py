"""Configuration loading and validation for the possession decoder."""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DecoderConfig:
    control_radius_m: float = 2.0
    opponent_margin_m: float = 0.5
    switch_hold_frames: int = 10
    restart_control_hold_frames: int = 10
    dead_ball_hold_frames: int = 1
    max_loose_frames: Optional[int] = None
    require_both_team_distances: bool = True
    initial_half_label: str = "D"
    dead_ball_feature: str = "dead_ball"

    def validate(self):
        if self.control_radius_m <= 0:
            raise ValueError("control_radius_m must be positive")
        if self.opponent_margin_m < 0:
            raise ValueError("opponent_margin_m cannot be negative")
        for name in ("switch_hold_frames", "restart_control_hold_frames",
                     "dead_ball_hold_frames"):
            if getattr(self, name) < 1:
                raise ValueError("{} must be at least 1".format(name))
        if self.max_loose_frames is not None and self.max_loose_frames < 1:
            raise ValueError("max_loose_frames must be null or at least 1")
        if self.initial_half_label not in ("H", "A", "D"):
            raise ValueError("initial_half_label must be H, A, or D")
        if not self.dead_ball_feature:
            raise ValueError("dead_ball_feature cannot be empty")
        return self


def load_config(path):
    """Load a JSON decoder configuration and reject unknown keys."""
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    allowed = set(DecoderConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown decoder configuration keys: {}".format(unknown))
    return DecoderConfig(**values).validate()
