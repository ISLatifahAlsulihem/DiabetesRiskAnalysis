"""Tasks 10/11: fit on frozen train IDs and select by validation macro F1.

Only row-ID metadata is read for the complete Parquet table. Test features
and targets are excluded by a Parquet filter before cleaning or modeling.
Run from the repository root: PYTHONPATH=src python -m diabetes_risk.train
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from diabetes_risk.clean import THIN_TABLE_PATH, prepare_table
from diabetes_risk.config import SPLIT_SEED
from diabetes_risk.evaluate import LABELS, score
from diabetes_risk.split import SPLIT_PATH

ROOT = Path(__file__).resolve().parents[2]
THIN = THIN_TABLE_PATH
SPLIT = SPLIT_PATH
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
PRED_DIR = ROOT / "data/processed"
ROW_ID = "row_id"
NUMERIC = ["age", "bmi", "sugar_drinks"]
BOOLEAN = ["any_exercise", "heavy_drinker"]
CATEGORICAL = ["smoker_status", "smokeless_tobacco", "ecigarette", "sex"]
FEATURES = NUMERIC + BOOLEAN + CATEGORICAL


def _check_ids(frame, name):
    if ROW_ID not in frame or frame[ROW_ID].isna().any():
        raise ValueError(f"{name}: row_id must exist and contain no missing values")
    if not frame[ROW_ID].is_unique:
        raise ValueError(f"{name}: row_id must be unique")


def load_rows():
    """Return train/validation rows and the non-test missing-target count.

    Unassigned non-test rows are inspected to account for missing targets.
    Labeled rows without a frozen assignment are an error, not a silent drop.
    Test IDs are checked for uniqueness/coverage, but their payload is unused.
    """
    split = pd.read_csv(SPLIT)
    _check_ids(split, "split")
    if "split" not in split or not split["split"].isin(
        ["train", "validation", "test"]
    ).all():
        raise ValueError("split must contain only train, validation, or test")
    ids = pd.read_parquet(THIN, columns=[ROW_ID], engine="pyarrow")
    _check_ids(ids, "thin table")
    if not split[ROW_ID].isin(ids[ROW_ID]).all():
        raise ValueError("Frozen split contains IDs absent from the thin table")
    test_ids = split.loc[split["split"] == "test", ROW_ID]
    allowed_ids = ids.loc[~ids[ROW_ID].isin(test_ids), ROW_ID]
    if allowed_ids.empty:
        raise ValueError("No non-test rows available")
    thin = pd.read_parquet(
        THIN, engine="pyarrow", filters=[(ROW_ID, "in", allowed_ids.tolist())]
    )
    _check_ids(thin, "filtered thin table")
    if set(thin[ROW_ID]) != set(allowed_ids):
        raise ValueError("Parquet filter did not return exactly the non-test IDs")
    table = prepare_table(thin)
    _check_ids(table, "cleaned table")
    assignments = split.loc[split["split"] != "test"]
    table = table.merge(assignments, on=ROW_ID, how="left", validate="one_to_one")
    missing_target_rows = int(table["target"].isna().sum())
    table = table.loc[table["target"].notna()].copy()
    if table["split"].isna().any():
        raise ValueError("Labeled non-test rows lack a frozen split assignment")
    if not table["target"].isin(LABELS).all():
        raise ValueError("Unexpected target class")
    for name in ["train", "validation"]:
        if not table["split"].eq(name).any():
            raise ValueError(f"No labeled {name} rows")
    return table, missing_target_rows


def _boolean_values(frame):
    """Convert verified cleaned booleans to 0/1, retaining NaN for imputation."""
    result = frame.copy()
    for col in result:
        present = result[col].dropna()
        if not present.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{col}: expected cleaned Boolean values or missingness")
        result[col] = result[col].map({False: 0.0, True: 1.0}).astype(float)
    return result


def _preprocessor(scale):
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric), NUMERIC),
        ("bool", Pipeline([
            ("normalize", FunctionTransformer(_boolean_values)),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), BOOLEAN),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def build_models():
    # Each estimator owns its preprocessing; fitting one cannot mutate another.
    return {
        "majority_class": DummyClassifier(strategy="most_frequent", random_state=SPLIT_SEED),
        "logistic_regression": Pipeline([
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SPLIT_SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=20,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SPLIT_SEED)),
        ]),
    }


def run():
    table, missing_target_rows = load_rows()
    train = table.loc[table["split"] == "train"]
    val = table.loc[table["split"] == "validation"]
    rows, fitted, predictions = [], {}, {}
    for name, model in build_models().items():
        # A failure aborts the run: never remove difficult rows for one candidate.
        model.fit(train[FEATURES], train["target"])
        pred = model.predict(val[FEATURES])
        if len(pred) != len(val) or not pd.Series(pred).isin(LABELS).all():
            raise ValueError(f"{name}: invalid or incomplete validation predictions")
        fitted[name] = model
        predictions[name] = pd.DataFrame({
            ROW_ID: val[ROW_ID], "target": val["target"], "pred": pred,
        })
        rows.append({
            "model": name,
            **score(val["target"], pred, LABELS),
            "train_rows": len(train),
            "validation_rows": len(val),
            "excluded_rows": missing_target_rows,
            "missing_target_rows": missing_target_rows,
            "preprocessing_failure_rows": 0,
            "prediction_failure_rows": 0,
        })
    metrics = pd.DataFrame(rows)
    # Write results only after every candidate has been evaluated successfully.
    for directory in [MODELS_DIR, REPORTS, PRED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        predictions[name].to_csv(PRED_DIR / f"validation_predictions_{name}.csv", index=False)
    train[[ROW_ID]].to_csv(PRED_DIR / "training_row_ids.csv", index=False)
    table.groupby("split")[FEATURES].agg(lambda s: int(s.isna().sum())).to_csv(
        REPORTS / "model_input_missingness.csv"
    )
    metrics.to_csv(REPORTS / "metrics_validation.csv", index=False)
    write_comparison(metrics)
    return metrics


def write_comparison(metrics):
    """Use the same metrics as the CSV, with stable candidate-order tie breaking."""
    ranked = metrics.sort_values("macro_f1", ascending=False, kind="mergesort")
    best = ranked.iloc[0]
    ref = metrics.loc[metrics["model"] == "majority_class", "macro_f1"].iloc[0]
    columns = ["model", "macro_f1", *(f"recall_{label}" for label in LABELS), "accuracy"]
    lines = [
        "# Model comparison", "",
        "All candidates use the same frozen training and validation IDs. "
        f"Inputs: {', '.join(FEATURES)}.", "",
        f"Training rows: {int(best.train_rows)}. Validation rows: {int(best.validation_rows)}.",
        "Test features and targets are not loaded into the modeling workflow or scored; "
        "only full-table row-ID metadata and frozen assignments are inspected.", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in ranked.iterrows():
        lines.append("| " + " | ".join(
            str(row[c]) if c == "model" else f"{row[c]:.6f}" for c in columns
        ) + " |")
    lines += ["", f"**Chosen model: `{best['model']}`** "
              f"(validation macro F1 = {best.macro_f1:.6f})."]
    if best.macro_f1 > ref:
        lines.append(f"It beats the majority-class reference ({ref:.6f}) "
                     f"by {best.macro_f1 - ref:.6f}.")
    else:
        lines.append("No candidate beats the majority-class reference.")
    lines += [
        "Ties retain candidate insertion order (majority-class reference first).", "",
        f"Excluded rows: {int(best.excluded_rows)}. `excluded_rows` is the number of "
        "non-test rows excluded for a missing target, including unassigned rows. "
        "It does not count the held-out test set. No rows are dropped for missing features. "
        "Preprocessing/prediction failure exclusions are zero: any such failure aborts the run.",
        "Numeric values use training medians; categorical and Boolean values use training "
        "modes and one-hot encoding with unknown categories ignored. Cleaned booleans "
        "are normalized to 0/1 before imputation. Only logistic regression scales numeric inputs.",
        "Missing-input counts by split are in `model_input_missingness.csv`; exact fitted "
        "IDs and validation predictions are saved under `data/processed/`.", "",
        "Metrics describe unweighted survey interview responses in this analysis sample. "
        "They are not population prevalence estimates. This model is not a medical "
        "diagnosis or a clinical tool.",
    ]
    (REPORTS / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Keep the transformer importable when a saved pipeline is loaded elsewhere.
    from diabetes_risk.train import run
    print(run().to_string(index=False))
