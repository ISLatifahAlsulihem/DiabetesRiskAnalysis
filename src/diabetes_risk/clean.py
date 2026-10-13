"""Task 5: cleaning functions shared by every later task.

Turns the raw thin-table codes (see reports/data_dictionary.md and
reports/contract.md) into the three-class target and the main model
columns. Pregnancy-only, don't know, refused, and blank DIABETE4 rows
get no target. Non-answer codes for every other column become missing
(NaN) rather than a real value.

Run from the project root to also write reports/class_counts.csv:
  python src/diabetes_risk/clean.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
THIN_TABLE_PATH = BASE_DIR / "data" / "processed" / "brfss2024_thin.parquet"
CLASS_COUNTS_PATH = BASE_DIR / "reports" / "class_counts.csv"

# DIABETE4 codes -> target class. Codes 2 (pregnancy-only), 7 (don't know),
# 9 (refused), and blank are left out of the model. See reports/contract.md.
TARGET_MAP = {1: "diabetes", 3: "no_diabetes", 4: "prediabetes"}

EXERANY2_MAP = {1: True, 2: False}

SMOKER3_MAP = {
    1: "current_every_day",
    2: "current_some_days",
    3: "former",
    4: "never",
}

USENOW3_MAP = {1: "every_day", 2: "some_days", 3: "not_at_all"}

ECIGNOW3_MAP = {1: "never", 2: "every_day", 3: "some_days", 4: "former"}

RFDRHV9_MAP = {1: False, 2: True}

SEXVAR_MAP = {1: "male", 2: "female"}

MAIN_COLUMNS = [
    "target",
    "age",
    "bmi",
    "any_exercise",
    "smoker_status",
    "smokeless_tobacco",
    "ecigarette",
    "heavy_drinker",
    "sex",
    "sugar_drinks",
]


def _mapped(series, mapping):
    """Map known codes to their clean value; every other code becomes missing."""
    codes = pd.to_numeric(series, errors="coerce").round()
    return codes.map(mapping)


def _clean_bmi(series):
    """_BMI5 has two implied decimal places: 2845 -> 28.45. See reports/data_dictionary.md."""
    return (pd.to_numeric(series, errors="coerce") / 100).round(2)


def _clean_sugar_drinks(series):
    """SSBSUGR2: first digit is the time unit, last two digits are the count.

    101-199 times/day, 201-299 times/week, 301-399 times/month, 888 never.
    777/999/blank are missing. See reports/contract.md.
    """
    codes = pd.to_numeric(series, errors="coerce").round()
    unit = codes // 100
    count = codes % 100
    # 100, 200, and 300 are not in the contract ranges (101-199, 201-299, 301-399).
    counted = count.where(count.between(1, 99))

    per_day = counted.where(unit == 1)
    per_week = (counted / 7).where(unit == 2)
    per_month = (counted / 30).where(unit == 3)
    never = pd.Series(0.0, index=codes.index).where(codes == 888)

    return per_day.combine_first(per_week).combine_first(per_month).combine_first(never)


def prepare_table(df):
    """Return a dataframe with `target` and the main model columns.

    `df` is the thin table from Task 3 (raw BRFSS codes), indexed the
    same way. See reports/contract.md for the code lists this follows.
    """
    diabete4 = pd.to_numeric(df["DIABETE4"], errors="coerce").round()

    out = pd.DataFrame(index=df.index)
    out["target"] = diabete4.map(TARGET_MAP)
    out["age"] = pd.to_numeric(df["_AGE80"], errors="coerce")
    out["bmi"] = _clean_bmi(df["_BMI5"])
    out["any_exercise"] = _mapped(df["EXERANY2"], EXERANY2_MAP)
    out["smoker_status"] = _mapped(df["_SMOKER3"], SMOKER3_MAP)
    out["smokeless_tobacco"] = _mapped(df["USENOW3"], USENOW3_MAP)
    out["ecigarette"] = _mapped(df["ECIGNOW3"], ECIGNOW3_MAP)
    out["heavy_drinker"] = _mapped(df["_RFDRHV9"], RFDRHV9_MAP)
    out["sex"] = _mapped(df["SEXVAR"], SEXVAR_MAP)
    out["sugar_drinks"] = _clean_sugar_drinks(df["SSBSUGR2"])

    if "row_id" in df.columns:
        out.insert(0, "row_id", df["row_id"])
    return out


def class_counts(prepared):
    """Row counts for each target class, plus unlabeled rows."""
    counts = prepared["target"].value_counts(dropna=False)
    counts = counts.rename(index=lambda cls: "unlabeled" if pd.isna(cls) else cls)
    return counts.rename_axis("target").reset_index(name="count")


def main():
    if not THIN_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Thin table not found at {THIN_TABLE_PATH}. Run src/diabetes_risk/load.py first."
        )
    df = pd.read_parquet(THIN_TABLE_PATH)
    prepared = prepare_table(df)
    counts = class_counts(prepared)

    CLASS_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(CLASS_COUNTS_PATH, index=False)

    print(counts.to_string(index=False))
    print(f"Saved {CLASS_COUNTS_PATH}")


if __name__ == "__main__":
    main()
