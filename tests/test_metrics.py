import pytest

from diabetes_risk.evaluate import score

LABELS = ["A", "B", "C"]


def test_hand_built_example():
    # recall: A 1/2, B 2/2, C 1/2
    # precision: A 1/2, B 2/3, C 1/1 -> F1: 0.5, 0.8, 2/3
    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "B", "B", "B", "C", "A"]
    out = score(y_true, y_pred, labels=LABELS)
    assert out["recall_A"] == pytest.approx(0.5)
    assert out["recall_B"] == pytest.approx(1.0)
    assert out["recall_C"] == pytest.approx(0.5)
    assert out["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert out["accuracy"] == pytest.approx(4 / 6)


def test_majority_class_reference():
    # Predicting only A: precision 2/4, recall 1 -> F1 2/3; others 0 -> macro 2/9
    y_true = ["A", "A", "B", "C"]
    y_pred = ["A", "A", "A", "A"]
    out = score(y_true, y_pred, labels=LABELS)
    assert out["recall_A"] == 1.0
    assert out["recall_B"] == 0.0
    assert out["macro_f1"] == pytest.approx(2 / 9)
