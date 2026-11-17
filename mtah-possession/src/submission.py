"""Write, validate, and fingerprint competition submissions."""

import hashlib

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm


def submission_table(labels):
    labels = np.asarray(labels, dtype=object)
    return pd.DataFrame({"frame": np.arange(len(labels), dtype=np.int64), "label": labels})


def validate_table(table, expected_frames):
    reason = pm.validate_submission(table, expected_frames)
    if reason:
        raise ValueError("invalid submission: {}".format(reason))


def write_submission(path, labels, expected_frames):
    table = submission_table(labels)
    validate_table(table, expected_frames)
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
