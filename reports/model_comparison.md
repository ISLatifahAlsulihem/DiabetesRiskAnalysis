# Model comparison

All candidates use the same frozen training and validation IDs. Inputs: age, bmi, sugar_drinks, any_exercise, heavy_drinker, smoker_status, smokeless_tobacco, ecigarette, sex.

Training rows: 24001. Validation rows: 8000.
Test features and targets are not loaded into the modeling workflow or scored; only full-table row-ID metadata and frozen assignments are inspected.

| model | macro_f1 | recall_no_diabetes | recall_prediabetes | recall_diabetes | accuracy |
| --- | --- | --- | --- | --- | --- |
| random_forest | 0.401248 | 0.637534 | 0.121053 | 0.675532 | 0.630625 |
| logistic_regression | 0.398026 | 0.613738 | 0.257895 | 0.605496 | 0.604125 |
| majority_class | 0.303410 | 1.000000 | 0.000000 | 0.000000 | 0.835250 |

**Chosen model: `random_forest`** (validation macro F1 = 0.401248).
It beats the majority-class reference (0.303410) by 0.097838.
Ties retain candidate insertion order (majority-class reference first).

Excluded rows: 0. `excluded_rows` is the number of non-test rows excluded for a missing target, including unassigned rows. It does not count the held-out test set. No rows are dropped for missing features. Preprocessing/prediction failure exclusions are zero: any such failure aborts the run.
Numeric values use training medians; categorical and Boolean values use training modes and one-hot encoding with unknown categories ignored. Cleaned booleans are normalized to 0/1 before imputation. Only logistic regression scales numeric inputs.
Missing-input counts by split are in `model_input_missingness.csv`; exact fitted IDs and validation predictions are saved under `data/processed/`.

Metrics describe unweighted survey interview responses in this analysis sample. They are not population prevalence estimates. This model is not a medical diagnosis or a clinical tool.
