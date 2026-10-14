# Task 10 / Task 11 implementation audit

Repository: https://github.com/owAziz/DiabetesRiskAnalysis

Reviewed base commit: `30ec1bd8b0d8a59d5b13a0ec9a5d35e2545be669`.

## Scope and compatibility

Inspected the actual contract, plan, configuration, cleaning, split, data dictionary,
loader, existing tests, processed Parquet, and frozen CSV. The checked-out repository
has no train.py, evaluate.py, or test_metrics.py. The three attached files supplied
those proposed implementations. This change adds them at the intended repository
paths, preserving the metric helper and its tests, and correcting the training
workflow. No existing tracked file, cleaning rule, split assignment, or contract
was changed. Changes are local; no GitHub push or pull request was made.

## Assumptions checked

| Assumption | Finding | Evidence |
| --- | --- | --- |
| prepare_table retains row_id and expected inputs/target | Confirmed | Actual function and output contain row_id, target, and all nine contract features |
| sugar_drinks is numeric | Confirmed | float64, times/day from the existing cleaner; actual range 0 to 99, no missing values in this sample |
| Exercise/heavy-drinker are nullable Boolean-like values | Confirmed | Both object dtype, actual distinct values False, True, NaN; not arbitrary strings |
| Configuration owns seeds | Confirmed | SAMPLE_SEED and SPLIT_SEED are 2024; training imports SPLIT_SEED |
| Frozen split schema and values | Confirmed | row_id/split; train, validation, test; unique nonmissing IDs; all match the thin table |
| 40,000 sampled rows versus labeled rows | Both are 40,000 in this file | All sampled targets are labeled, consistent with the updated sample eligibility rule; these counts need not agree in other data |
| Stratified, deterministic, approximately 60/20/20 | Confirmed | Existing split reproduced in memory for audit and exactly matched; CSV never regenerated or overwritten |

The one-time data/split audit inspected full-table class counts to verify the
requested stratification. This was not model evaluation. Normal model selection
reads full-table row IDs only and uses a Parquet predicate to return non-test
payloads before calling prepare_table. It never fits, transforms, or scores test
rows. Storage engines may physically read shared Parquet pages; this does not
expose test payloads to the modeling workflow.

| Target | Train | Validation | Test (assignment audit only) |
| --- | ---: | ---: | ---: |
| no_diabetes | 20,047 | 6,682 | 6,682 |
| prediabetes | 571 | 190 | 190 |
| diabetes | 3,383 | 1,128 | 1,127 |
| Total | 24,001 | 8,000 | 7,999 |

Class-wise largest-remainder rounding explains the one-row difference from
24,000/8,000/8,000. Missing targets: zero. All nine inputs are used; IDs, split,
target, raw diabetes codes, survey weights, and descriptive-only fields are not
predictors.

## Issues fixed in the attached proposal

- Complete-case deletion discarded usable rows; train-only median/mode imputation now retains them.
- Blanket categorical string conversion has been replaced by checked Boolean-to-0/1 normalization and separate string categorical processing.
- Uniqueness, nonmissing IDs, legal assignments, split coverage, exact filtering, one-to-one joins, and missing labeled assignments are validated.
- Shared mutable preprocessing between candidate pipelines has been removed.
- Numeric sugar_drinks remains numeric. Logistic regression scales numeric inputs; random forest does not need scaling.
- Logistic regression and random forest reuse the configured seed.
- The report now contains per-class recall, accuracy, evaluation counts, exclusion definitions, honest reference comparison, and stable tie handling.
- Claims that test data were never read have been narrowed to the actual modeling boundary, acknowledging row-ID metadata and the separate audit.
- All candidates are scored before outputs are written; any fit/prediction error aborts instead of silently deleting rows.

## Files added

| File | Purpose |
| --- | --- |
| src/diabetes_risk/train.py | Corrected Task 10/11 workflow based on the attachment |
| src/diabetes_risk/evaluate.py | Attached fixed-label metric helper, unchanged |
| tests/test_metrics.py | Attached metric tests, unchanged |
| tests/test_train.py | Split-boundary, imputation, independent preprocessing, reproducibility, saved model, failure, and reporting regression tests |
| reports/metrics_validation.csv | Full-precision actual metrics and exclusion counts |
| reports/model_comparison.md | Generated selection report |
| reports/model_input_missingness.csv | Missing input counts in train/validation before imputation |
| reports/task10_11_audit.md | This audit and verification record |
| data/processed/training_row_ids.csv | Exact row IDs used to fit every candidate |
| data/processed/validation_predictions_majority_class.csv | Actual reference predictions and target/row IDs |
| data/processed/validation_predictions_logistic_regression.csv | Actual logistic regression predictions and target/row IDs |
| data/processed/validation_predictions_random_forest.csv | Actual forest predictions and target/row IDs |

The training command also writes models/majority_class.joblib,
models/logistic_regression.joblib, and models/random_forest.joblib, following the
existing ignored models directory convention. These are local generated outputs.

## Actual results

| Model | Validation macro F1 | Recall no_diabetes | Recall prediabetes | Recall diabetes | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority_class | 0.303410 | 1.000000 | 0.000000 | 0.000000 | 0.835250 |
| logistic_regression | 0.398026 | 0.613738 | 0.257895 | 0.605496 | 0.604125 |
| random_forest | 0.401248 | 0.637534 | 0.121053 | 0.675532 | 0.630625 |

Selected: random_forest, macro F1 0.4012481338323585, improving on the reference
by 0.09783806254325167. Its prediabetes recall is only 0.121053; selection by macro
F1 does not establish strong performance for every class.

Every candidate fits 24,001 rows and scores the same 8,000 rows. Excluded missing
targets, preprocessing-failure rows, and prediction-failure rows are all zero.
excluded_rows counts non-test rows missing the target (including unassigned
unlabeled rows), not the held-out test set. Failures abort rather than create
candidate-specific evaluation samples.

## Commands and verification

Executed from the repository root:

```bash
python -m pip install pytest
python -m pip install pyarrow
PYTHONPATH=src python -m pytest tests/test_clean.py tests/test_split.py tests/test_metrics.py -q
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m diabetes_risk.train
```

The requested cleaning/split/metric subset passed: 18 tests. The complete suite,
including the added training tests, passed: 38 tests. Training completed without
warnings. A separate repeat-run verification compared generated CSV/Markdown and
prediction files byte-for-byte, recomputed metrics from predictions, checked
identical validation IDs and labels across models, loaded every saved model and
reproduced its predictions, checked no test IDs appeared in predictions, and
compared both original input hashes. All passed.

An initial exact-float check compared a CSV parsed with pandas defaults to
in-memory values and found last-bit decimal parsing differences. The corrected
check compared actual output bytes across runs; they matched exactly. This was
verification-code precision, not changed model predictions or nondeterminism.

Original and final SHA-256:

- split.csv: `6f85aba6e73a65b678766f9e548dd1582dc9bd1b830e378384ec177e148f41a0`
- brfss2024_thin.parquet: `46e09270e7d86eb55fdba0ba24aa092576a13e1efd416b559e85460a8337a986`

Runtime: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0,
PyArrow 26.0.0, pytest 9.1.1. PyArrow and pytest were initially absent and installed
successfully. The repository has no dependency manifest or version pins, so
compatibility with an unspecified teammate environment cannot be certified.
The implementation avoids newer optional imputer/encoder keyword arguments.
PyArrow is explicitly required for filtered Parquet loading.

## Remaining boundaries

Implementation requirements were verified in the tested runtime, with the documented exception that test data were read during the initial split audit, but were not used for model training, evaluation, or selection.
The full raw SAS sample draw was not rerun; verification used the committed
processed table. No test-set metrics were generated. Task 13's metrics.json and
single chosen-model command, and the final test evaluation, remain later tasks.
Results describe unweighted interview responses, not population prevalence or
clinical diagnosis. No claim of diagnostic suitability is made.
