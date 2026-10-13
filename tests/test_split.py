import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from diabetes_risk.split import assign_split, build_split


def test_assign_split_matches_fractions_within_a_point():
    target = pd.Series(["diabetes"] * 1000 + ["no_diabetes"] * 1000)
    labels = assign_split(target, seed=2024, train_frac=0.6, val_frac=0.2)
    counts = labels.value_counts()
    total = len(target)
    assert abs(counts["train"] / total - 0.6) < 0.01
    assert abs(counts["validation"] / total - 0.2) < 0.01
    assert abs(counts["test"] / total - 0.2) < 0.01


def test_assign_split_keeps_each_class_mix_within_a_point():
    target = pd.Series(["diabetes"] * 100 + ["no_diabetes"] * 900)
    labels = assign_split(target, seed=2024, train_frac=0.6, val_frac=0.2)
    train_target = target[labels == "train"]
    share = (train_target == "diabetes").mean()
    assert abs(share - 0.1) < 0.01


def test_assign_split_honors_the_test_fraction():
    # Size 7 is where rounding each share disagrees with leaving test as the leftover.
    target = pd.Series(["diabetes"] * 7)
    labels = assign_split(target, seed=1, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    assert labels.value_counts().to_dict() == {"train": 4, "validation": 2, "test": 1}


def test_assign_split_leaves_unlabeled_rows_out():
    target = pd.Series(["diabetes", "no_diabetes", np.nan, np.nan])
    labels = assign_split(target, seed=1, train_frac=0.6, val_frac=0.2)
    assert labels.isna().sum() == 2


def test_assign_split_is_deterministic():
    target = pd.Series(["diabetes"] * 30 + ["no_diabetes"] * 70)
    first = assign_split(target, seed=2024, train_frac=0.6, val_frac=0.2)
    second = assign_split(target, seed=2024, train_frac=0.6, val_frac=0.2)
    pd.testing.assert_series_equal(first, second)


def test_build_split_is_deterministic_and_covers_labeled_rows():
    df = pd.DataFrame(
        {
            "row_id": range(1, 101),
            "DIABETE4": [1] * 30 + [3] * 60 + [4] * 10,
            "_AGE80": 40,
            "_BMI5": 2500,
            "EXERANY2": 1,
            "_SMOKER3": 4,
            "USENOW3": 3,
            "ECIGNOW3": 1,
            "_RFDRHV9": 1,
            "SEXVAR": 1,
            "SSBSUGR2": 888,
        }
    )
    first = build_split(df)
    second = build_split(df)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"train", "validation", "test"}
    assert len(first) == 100


def test_build_split_excludes_unlabeled_rows():
    df = pd.DataFrame(
        {
            "row_id": range(1, 11),
            "DIABETE4": [1] * 5 + [2] * 5,  # pregnancy-only rows get no target
            "_AGE80": 40,
            "_BMI5": 2500,
            "EXERANY2": 1,
            "_SMOKER3": 4,
            "USENOW3": 3,
            "ECIGNOW3": 1,
            "_RFDRHV9": 1,
            "SEXVAR": 1,
            "SSBSUGR2": 888,
        }
    )
    split = build_split(df)
    assert len(split) == 5
    assert set(split["row_id"]) == {1, 2, 3, 4, 5}
