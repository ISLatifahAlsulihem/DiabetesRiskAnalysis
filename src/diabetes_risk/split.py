"""Task 6: freeze the train, validation, and test ids.

Splits labeled rows 60/20/20, stratified on `target`, with a fixed
seed, and saves the assignment so every later task trains and scores
on the same rows. Unlabeled rows (see src/diabetes_risk/clean.py) are
left out of the split.

Run from the project root:
  python src/diabetes_risk/split.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diabetes_risk.clean import prepare_table
from diabetes_risk.config import SPLIT_SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC

BASE_DIR = Path(__file__).resolve().parent.parent.parent
THIN_TABLE_PATH = BASE_DIR / "data" / "processed" / "brfss2024_thin.parquet"
SPLIT_PATH = BASE_DIR / "data" / "processed" / "split.csv"


def _split_sizes(n, train_frac, val_frac, test_frac):
    """Largest-remainder counts so the three fractions partition n rows."""
    raw = [n * frac for frac in (train_frac, val_frac, test_frac)]
    sizes = [int(value) for value in raw]
    shortfall = n - sum(sizes)
    if shortfall < 0:
        raise ValueError("split fractions must sum to 1")
    order = sorted(range(3), key=lambda i: (-(raw[i] - sizes[i]), i))
    for i in order[:shortfall]:
        sizes[i] += 1
    if sum(sizes) != n:
        raise ValueError("split fractions must sum to 1")
    return sizes


def assign_split(
    target, seed=SPLIT_SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_frac=TEST_FRAC
):
    """Stratified train/validation/test labels for a `target` Series.

    Rows where `target` is missing get no split (NaN). The result
    shares `target`'s index, so it can be assigned back to a column.
    """
    labels = pd.Series(np.nan, index=target.index, dtype="object")
    labeled = target.dropna()
    rng = np.random.default_rng(seed)

    for _, group in labeled.groupby(labeled):
        idx = rng.permutation(group.index.to_numpy())
        n = len(idx)
        n_train, n_val, n_test = _split_sizes(n, train_frac, val_frac, test_frac)
        labels.loc[idx[:n_train]] = "train"
        labels.loc[idx[n_train:n_train + n_val]] = "validation"
        labels.loc[idx[n_train + n_val:n_train + n_val + n_test]] = "test"

    return labels


def build_split(df):
    """Return the `row_id`/`split` table for the thin table `df`."""
    prepared = prepare_table(df)
    row_id = prepared["row_id"] if "row_id" in prepared.columns else pd.Series(
        prepared.index + 1, index=prepared.index
    )
    split = assign_split(prepared["target"])
    result = pd.DataFrame({"row_id": row_id, "split": split})
    return result.dropna(subset=["split"]).reset_index(drop=True)


def main():
    if not THIN_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Thin table not found at {THIN_TABLE_PATH}. Run src/diabetes_risk/load.py first."
        )
    df = pd.read_parquet(THIN_TABLE_PATH)
    split = build_split(df)

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)

    print(split["split"].value_counts())
    print(f"Saved {SPLIT_PATH}")


if __name__ == "__main__":
    main()
