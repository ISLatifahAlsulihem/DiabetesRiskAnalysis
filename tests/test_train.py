"""Focused split-boundary, imputation, and reporting regression tests."""
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diabetes_risk import train
from diabetes_risk.clean import prepare_table


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    raw = pd.DataFrame({
        "row_id": range(1, 14), "DIABETE4": [1, 3, 4] * 4 + [2],
        "_AGE80": [30, 50, 70] * 4 + [20], "_BMI5": 2500,
        "EXERANY2": [1, 2, 7] * 4 + [1], "_SMOKER3": 4,
        "USENOW3": 3, "ECIGNOW3": 1, "_RFDRHV9": [1, 2, 9] * 4 + [1],
        "SEXVAR": 1, "SSBSUGR2": 888,
    })
    raw.loc[0, "_BMI5"] = np.nan
    raw.loc[7, "_BMI5"] = 99900  # Validation must not affect the training median.
    raw.loc[7, "_SMOKER3"] = 1  # Previously unseen category.
    split = pd.DataFrame({"row_id": range(1, 13),
                          "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3})
    monkeypatch.setattr(train, "THIN", tmp_path / "thin.parquet")
    monkeypatch.setattr(train, "SPLIT", tmp_path / "split.csv")
    monkeypatch.setattr(train, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "PRED_DIR", tmp_path / "predictions")
    raw.to_parquet(train.THIN, index=False)
    split.to_csv(train.SPLIT, index=False)
    return raw, split


def test_load_excludes_test_payload_before_cleaning(inputs, monkeypatch):
    original = train.prepare_table
    seen = []
    def guarded(frame):
        seen.extend(frame.row_id)
        assert not frame.row_id.isin([10, 11, 12]).any()
        return original(frame)
    monkeypatch.setattr(train, "prepare_table", guarded)
    table, excluded = train.load_rows()
    assert set(seen) == set(range(1, 10)) | {13}
    assert table.row_id.tolist() == list(range(1, 10))
    assert excluded == 1
    assert len(table) == 9  # Missing features do not remove rows.


@pytest.mark.parametrize("issue", ["duplicate_split", "duplicate_raw", "null_id",
                                    "orphan", "bad_split", "unassigned"])
def test_invalid_ids_and_assignments_fail(inputs, issue):
    raw, split = inputs
    if issue == "duplicate_split":
        split = pd.concat([split, split.iloc[[0]]])
    elif issue == "duplicate_raw":
        raw = pd.concat([raw, raw.iloc[[0]]])
    elif issue == "null_id":
        split.loc[0, "row_id"] = np.nan
    elif issue == "orphan":
        split.loc[0, "row_id"] = 999
    elif issue == "bad_split":
        split.loc[0, "split"] = "valid"
    else:
        split = split.iloc[1:]
    raw.to_parquet(train.THIN, index=False)
    split.to_csv(train.SPLIT, index=False)
    with pytest.raises(ValueError):
        train.load_rows()


def test_imputation_unknowns_and_independent_preprocessors(inputs):
    table, _ = train.load_rows()
    fit = table[table.split == "train"]
    val = table[table.split == "validation"]
    models = train.build_models()
    assert models['logistic_regression']['prep'] is not models['random_forest']['prep']
    for name, model in models.items():
        model.fit(fit[train.FEATURES], fit.target)
        assert len(model.predict(val[train.FEATURES])) == len(val)
        if name != 'majority_class':
            imputer = model['prep'].named_transformers_['num']['impute']
            assert imputer.statistics_[1] == 25
    path = train.MODELS_DIR.parent / 'pipeline.joblib'
    joblib.dump(models['logistic_regression'], path)
    np.testing.assert_array_equal(
        joblib.load(path).predict(val[train.FEATURES]),
        models['logistic_regression'].predict(val[train.FEATURES]))


def test_boolean_conversion_rejects_arbitrary_strings():
    with pytest.raises(ValueError, match="Boolean"):
        train._boolean_values(pd.DataFrame({"any_exercise": [True, "unexpected"]}))


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    observed = []
    def fit(self, X, y):
        self.observed.append(('fit', X.index.tolist(), y.tolist(), X.columns.tolist()))
        self.reference_ = DummyClassifier(strategy='most_frequent').fit(X, y)
        return self
    def predict(self, X):
        self.observed.append(('predict', X.index.tolist(), X.columns.tolist()))
        return self.reference_.predict(X)


def test_same_validation_reproducibility_reports_and_unchanged_split(inputs, monkeypatch):
    RecordingClassifier.observed = []
    monkeypatch.setattr(train, 'build_models', lambda: {
        'majority_class': RecordingClassifier(), 'other': RecordingClassifier()})
    before = hashlib.sha256(train.SPLIT.read_bytes()).hexdigest()
    first = train.run()
    events = RecordingClassifier.observed
    assert events[0] == events[2]
    assert events[1] == events[3]
    assert events[0][1] == list(range(6))
    assert events[1][1] == [6, 7, 8]
    assert events[0][3] == train.FEATURES
    pd.testing.assert_frame_equal(first, train.run())
    assert hashlib.sha256(train.SPLIT.read_bytes()).hexdigest() == before
    pd.testing.assert_frame_equal(first, pd.read_csv(train.REPORTS / 'metrics_validation.csv'))
    report = (train.REPORTS / 'model_comparison.md').read_text()
    assert 'Chosen model: `majority_class`' in report
    assert 'No candidate beats' in report
    assert 'Training rows: 6. Validation rows: 3.' in report
    assert 'Excluded rows: 1.' in report
    for row in first.itertuples():
        assert f'{row.macro_f1:.6f}' in report
        assert f'{row.recall_prediabetes:.6f}' in report
    a = pd.read_csv(train.PRED_DIR / 'validation_predictions_majority_class.csv')
    b = pd.read_csv(train.PRED_DIR / 'validation_predictions_other.csv')
    pd.testing.assert_frame_equal(a, b)
    assert a.row_id.tolist() == [7, 8, 9]


def test_prediction_failure_aborts_without_reports(inputs, monkeypatch):
    class Broken(DummyClassifier):
        def predict(self, X):
            return np.array(['diabetes'])
    monkeypatch.setattr(train, 'build_models', lambda: {'broken': Broken()})
    with pytest.raises(ValueError, match='incomplete'):
        train.run()
    assert not train.REPORTS.exists()
