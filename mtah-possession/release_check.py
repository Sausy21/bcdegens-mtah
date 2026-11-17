"""Run repository-safety and submission checks before uploading."""

import argparse
import os
import subprocess
import sys

from evaluation import possession_metrics as pm
from games import GAME, GAMES


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
REQUIRED_FILES = (
    "README.md",
    "SOLUTION.md",
    "TECHNICAL_DOCUMENT.md",
    "EXPERIMENTS.md",
    "requirements.txt",
    "run_pipeline.py",
    "build_features.py",
    "config/dead_ball_rules.json",
    "config/feature_generation.json",
    "config/possession_rules.json",
)
SENSITIVE_EXTENSIONS = (".mp4", ".mov", ".mkv", ".parquet", ".feather",
                        ".pkl", ".joblib", ".pt", ".pth")


def sensitive_tracked_paths(paths):
    unsafe = []
    for original in paths:
        path = original.replace("\\", "/")
        lowered = path.lower()
        basename = lowered.rsplit("/", 1)[-1]
        if "/data/" in "/{}/".format(lowered.strip("/")) and basename != "readme.md":
            unsafe.append(original)
        elif lowered.endswith(SENSITIVE_EXTENSIONS):
            unsafe.append(original)
        elif "feature" in basename and basename.endswith((".csv", ".parquet", ".feather")):
            unsafe.append(original)
    return unsafe


def tracked_files():
    result = subprocess.run(
        ["git", "-C", REPO_ROOT, "ls-files"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return [line for line in result.stdout.splitlines() if line]


def check_submission(path):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
    if reason:
        raise ValueError(reason)


def run(submission):
    failures = []
    missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(ROOT, name))]
    if missing:
        failures.append("missing required project files: {}".format(missing))

    unsafe = sensitive_tracked_paths(tracked_files())
    if unsafe:
        failures.append("restricted or derived artifacts are tracked: {}".format(unsafe))

    try:
        check_submission(submission)
    except Exception as error:
        failures.append("submission failed validation: {}".format(error))

    if failures:
        print("RELEASE CHECK FAILED")
        for failure in failures:
            print("- {}".format(failure))
        return 1
    print("RELEASE CHECK PASSED")
    print("- required code and documentation present")
    print("- no obvious restricted/derived artifacts tracked")
    print("- submission has {} valid rows".format(GAMES[GAME]["n_video_frames"]))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="exact CSV intended for upload")
    arguments = parser.parse_args()
    sys.exit(run(arguments.submission))
