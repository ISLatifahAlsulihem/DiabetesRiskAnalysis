"""Metric helpers shared by training and comparison."""
from sklearn.metrics import accuracy_score, f1_score, recall_score

LABELS = ["no_diabetes", "prediabetes", "diabetes"]


def score(y_true, y_pred, labels=None):
    """Return macro F1, accuracy (recorded only, never used to pick a model), and per-class recall."""
    labels = labels or LABELS
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {"macro_f1": float(macro), "accuracy": float(accuracy_score(y_true, y_pred))}
    out.update({f"recall_{lab}": float(r) for lab, r in zip(labels, recalls)})
    return out
