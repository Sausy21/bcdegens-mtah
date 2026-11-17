"""Explore the tactical tracking data and build a per-frame feature table.

Usage:
    python tools/explore_data.py ATD_TXT ATD_XML TAGGING_XML --out outputs

Prints a few summary stats (ball gaps, players tracked, ball speed, Smart Tagging)
and writes outputs/features.csv with one row per video frame.

features.csv is derived from the match data, so it stays out of git.
"""
import argparse
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.3  # metres
PITCH_WIDTH = 68.0
FPS = 25
N_FRAMES = 199319

# Graded frames per half (inclusive), from the challenge docs.
HALVES = [(11185, 90651), (110808, 190197)]

# Faster than any real ground ball. Usually a ball in the air or a tracking jump.
MAX_BALL_SPEED = 40.0  # m/s

# Faster than any sprint (about 10 m/s). Usually a tracking jump, not a player.
MAX_PLAYER_SPEED = 12.0  # m/s


def read_teams(xml_path):
    """Return the team of each track slot (0 = home, 1 = away) and a club name -> H/A map.

    The first team in the metadata is the home side.
    """
    root = ET.parse(xml_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"Expected 2 teams in {xml_path}, found {len(teams)}")

    home_id = teams[0].get("id")
    slot_team = np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")])

    club_side = {}
    for team, side in zip(teams, ["H", "A"]):
        name = (team.findtext("Name") or "").strip().lower()
        club_side[name] = side
    return slot_team, club_side


def read_keepers(xml_path):
    """True for each track slot the metadata marks as a goalkeeper (position_type = Goalkeeper)."""
    root = ET.parse(xml_path).getroot()
    keepers = []
    for player in root.iter("Player"):
        values = [param.findtext("Value") for param in player.iter("ProviderParameter")]
        keepers.append("Goalkeeper" in values)
    return np.array(keepers)


def read_atd_arrays(txt_path, n_slots):
    """Read the raw tracking file into numpy arrays.

    Returns x, y (frames x slots) and ball_x, ball_y (frames), normalised 0-1, NaN where
    nothing is tracked. Row i is video frame i (the file's frame ids start at 1).

    We parse the text directly instead of using starter/load_atd.py, which goes through
    kloppy and is slow and memory-heavy on the full match. The official baseline does the same.
    """
    x = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    y = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    ball_x = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_y = np.full(N_FRAMES, np.nan, dtype=np.float32)

    with open(txt_path) as f:
        for line in f:
            fields = line.rstrip("\n").split(":")
            if len(fields) != 3:
                continue
            frame_id, players, ball = fields
            frame = int(frame_id) - 1

            for slot, player in enumerate(players.split(";")):
                if not player.startswith("N"):  # "NaN,NaN" means not tracked
                    px, py = player.split(",")[:2]
                    x[frame, slot] = float(px)
                    y[frame, slot] = float(py)

            if not ball.startswith("N"):
                bx, by = ball.split(",")[:2]
                ball_x[frame] = float(bx)
                ball_y[frame] = float(by)

    return x, y, ball_x, ball_y


def graded_frames():
    mask = np.zeros(N_FRAMES, dtype=bool)
    for start, end in HALVES:
        mask[start:end + 1] = True
    return mask


def run_lengths(flags):
    """Lengths of each run of consecutive True values."""
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def summarise_ball_gaps(has_ball, graded):
    print(f"Ball tracked on {100 * has_ball.mean():.1f}% of all frames, "
          f"{100 * has_ball[graded].mean():.1f}% of graded frames")

    # Gaps are measured inside each half, so a gap at a half's edge is cut off there.
    gaps = np.concatenate([run_lengths(~has_ball[start:end + 1]) for start, end in HALVES])
    print(f"{int((~has_ball[graded]).sum())} missing frames in graded play, in {len(gaps)} gaps")
    print(f"Gap length (s): median {np.median(gaps) / FPS:.2f}, "
          f"90th pct {np.percentile(gaps, 90) / FPS:.2f}, max {gaps.max() / FPS:.1f}")

    buckets = [(1, 5), (6, 25), (26, 75), (76, None)]
    for low, high in buckets:
        in_bucket = (gaps >= low) if high is None else (gaps >= low) & (gaps <= high)
        label = f"{low}+" if high is None else f"{low}-{high}"
        share = 100 * gaps[in_bucket].sum() / gaps.sum()
        print(f"  gaps of {label:>6} frames: {int(in_bucket.sum()):>4} gaps, {share:5.1f}% of missing frames")


def nearest_distance(dist, slot_team, team):
    """Distance from the ball to the closest player of one team, per frame (NaN if none)."""
    team_dist = np.where(slot_team == team, dist, np.inf)
    team_dist = np.where(np.isnan(team_dist), np.inf, team_dist)
    nearest = team_dist.min(axis=1)
    nearest[np.isinf(nearest)] = np.nan
    return nearest


def players_within(dist, slot_team, team, has_ball, radius=5.0):
    """How many players of one team are within `radius` metres of the ball.

    Left blank when the ball is missing, so "unknown" can't be mistaken for "nobody close".
    """
    close = (slot_team == team) & (dist < radius)
    return np.where(has_ball, close.sum(axis=1), np.nan)


def player_speeds(x, y):
    """Speed of every player on every frame (m/s), from positions.

    Centred over +/-2 frames (0.16 s) so the jitter of point tracking doesn't look like running.
    Anything above MAX_PLAYER_SPEED is treated as a tracking jump and left blank.
    """
    px, py = x * PITCH_LENGTH, y * PITCH_WIDTH
    speed = np.full(px.shape, np.nan, dtype=np.float32)
    speed[2:-2] = np.hypot(px[4:] - px[:-4], py[4:] - py[:-4]) * FPS / 4.0
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    return speed


def nearest_player_columns(dist, speed, slot_team, keepers, team):
    """For the player of one team nearest the ball: their speed, and whether they're the keeper.

    Both are blank when the ball is missing or nobody from that team is tracked.
    """
    team_dist = np.where((slot_team == team) & ~np.isnan(dist), dist, np.inf)
    has_player = np.isfinite(team_dist).any(axis=1)
    nearest = team_dist.argmin(axis=1)
    rows = np.arange(len(dist))
    nearest_speed = np.where(has_player, speed[rows, nearest], np.nan)
    nearest_is_keeper = np.where(has_player, keepers[nearest], np.nan)
    return nearest_speed, nearest_is_keeper


def team_mean_x(x, slot_team, team):
    """Mean x position (m) of one team's tracked players: shows which goal they defend."""
    team_x = np.where(slot_team == team, x * PITCH_LENGTH, np.nan)
    with warnings.catch_warnings():  # frames with nobody tracked -> blank, quietly
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(team_x, axis=1)


def build_features(x, y, ball_x, ball_y, slot_team, keepers, graded):
    has_ball = ~np.isnan(ball_x)
    n_tracked = (~np.isnan(x)).sum(axis=1)

    # Distance from every player to the ball, in metres.
    dx = (x - ball_x[:, None]) * PITCH_LENGTH
    dy = (y - ball_y[:, None]) * PITCH_WIDTH
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Ball speed from frame-to-frame movement. NaN on the first frame after a gap.
    step_x = np.diff(ball_x * PITCH_LENGTH, prepend=np.nan)
    step_y = np.diff(ball_y * PITCH_WIDTH, prepend=np.nan)
    speed = np.sqrt(step_x ** 2 + step_y ** 2) * FPS
    speed_smooth = pd.Series(speed).rolling(5, center=True, min_periods=3).median().values
    jump = speed > MAX_BALL_SPEED

    print(f"Ball speed (m/s): median {np.nanmedian(speed):.2f}, 90th pct {np.nanpercentile(speed, 90):.2f}; "
          f"{100 * jump.sum() / (~np.isnan(speed)).sum():.2f}% of frames above {MAX_BALL_SPEED:.0f} m/s flagged")

    # Extra columns for Josh's rules (game_logic/rules.py): player speeds, keepers, team positions.
    speed_players = player_speeds(x, y)
    s_home, home_is_keeper = nearest_player_columns(dist, speed_players, slot_team, keepers, 0)
    s_away, away_is_keeper = nearest_player_columns(dist, speed_players, slot_team, keepers, 1)
    with warnings.catch_warnings():  # frames with nobody tracked -> blank, quietly
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_player_speed = np.nanmean(speed_players, axis=1)

    return pd.DataFrame({
        "frame": np.arange(N_FRAMES),
        "graded": graded,
        "ball_x_m": ball_x * PITCH_LENGTH,
        "ball_y_m": ball_y * PITCH_WIDTH,
        "ball_speed_ms": speed,
        "ball_speed_smooth_ms": speed_smooth,
        "ball_jump": jump,
        "n_tracked": n_tracked,
        "d_nearest_home_m": nearest_distance(dist, slot_team, 0),
        "d_nearest_away_m": nearest_distance(dist, slot_team, 1),
        "n_home_within_5m": players_within(dist, slot_team, 0, has_ball),
        "n_away_within_5m": players_within(dist, slot_team, 1, has_ball),
        "s_nearest_home_ms": s_home,
        "s_nearest_away_ms": s_away,
        "mean_player_speed_ms": mean_player_speed,
        "cx_home_m": team_mean_x(x, slot_team, 0),
        "cx_away_m": team_mean_x(x, slot_team, 1),
        "nearest_home_is_gk": home_is_keeper,
        "nearest_away_is_gk": away_is_keeper,
    })


def summarise_tagging(tagging_xml, club_side):
    rows = []
    for inst in ET.parse(tagging_xml).iter("instance"):
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        club = (labels.get("Team") or "").strip().lower()
        rows.append({
            "code": inst.findtext("code"),
            "type": labels.get("Type"),
            "side": club_side.get(club, "?"),  # "?" = no team (N/A)
            "length_s": float(inst.findtext("end")) - float(inst.findtext("start")),
        })
    tags = pd.DataFrame(rows)

    print(f"Smart Tagging: {len(tags)} instances")
    print(tags["side"].value_counts().to_string())

    set_pieces = tags[tags["code"].str.upper() == "SET PIECES"]
    print("\nSet pieces by type and team:")
    print(pd.crosstab(set_pieces["type"], set_pieces["side"]).to_string())
    # Each type has a fixed window length, so these mark roughly when, not how long.
    print("\nSet-piece window length (s):")
    print(set_pieces.groupby("type")["length_s"].agg(["min", "max"]).round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description="Explore the tracking data and build features.csv")
    parser.add_argument("atd_txt", help="tracking data (BAR-ATH_tactical_atd.txt)")
    parser.add_argument("atd_xml", help="tracking metadata (BAR-ATH_tactical_atd.xml)")
    parser.add_argument("tagging_xml", help="Smart Tagging (BAR-ATH_tactical_smart_tagging.xml)")
    parser.add_argument("--out", default="outputs", help="folder for features.csv (default: outputs)")
    args = parser.parse_args()

    slot_team, club_side = read_teams(args.atd_xml)
    keepers = read_keepers(args.atd_xml)
    x, y, ball_x, ball_y = read_atd_arrays(args.atd_txt, len(slot_team))
    graded = graded_frames()
    print(f"{len(slot_team)} track slots: {(slot_team == 0).sum()} home, {(slot_team == 1).sum()} away; "
          f"goalkeeper slots {np.flatnonzero(keepers).tolist()}\n")

    summarise_ball_gaps(~np.isnan(ball_x), graded)

    n_tracked = (~np.isnan(x)).sum(axis=1)[graded]
    print(f"\nPlayers tracked per graded frame: mean {n_tracked.mean():.1f}, "
          f"5th pct {np.percentile(n_tracked, 5):.0f}, 95th pct {np.percentile(n_tracked, 95):.0f}\n")

    features = build_features(x, y, ball_x, ball_y, slot_team, keepers, graded)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "features.csv")
    features.to_csv(path, index=False, float_format="%.3f")
    print(f"Wrote {path} ({len(features)} rows)\n")

    summarise_tagging(args.tagging_xml, club_side)


if __name__ == "__main__":
    main()
