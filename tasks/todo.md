# Tasks: Diabetes risk model

Any teammate can take the next task whose dependencies are done. One owner per task.

## Task 1: Create the project layout

**Owner:** Abdulaziz

**Description:** Add the folders and starter files the team will share. Raw survey data stays out of git. Notebooks, source code, reports, and models each have one place.

**Acceptance criteria:**

- [x] These folders exist: `data/raw/`, `data/processed/`, `notebooks/`, `src/diabetes_risk/`, `tests/`, `reports/figures/`, `models/`
- [x] `src/diabetes_risk/__init__.py` exists so the training code can be imported
- [x] `.gitignore` ignores `data/raw/`, `data/processed/`, and `models/`
- [x] README gains a "Layout" section that points to `tasks/plan.md` and states that notebooks are for exploration and `src/diabetes_risk/` is for shared code

**Verification:**

- [x] Manual check: a dummy file placed in `data/raw/` does not show up as a file git would track (`git check-ignore -v data/raw/dummy` once git is in use)
- [x] Manual check: another person can tell from the README where to put a new notebook versus a shared function

**Dependencies:** None

**Files likely touched:**

- `.gitignore`
- `README.md`
- `src/diabetes_risk/__init__.py`

**Estimated scope:** Small: 1-2 files, plus empty folders

## Task 2: Write the analysis contract

**Owner:** Abdulaziz

**Description:** Write a contract the whole team follows. It defines the three-class target, who is excluded, the main columns, the sample, the metric, and the split sizes. It does not fix the model type. Later tasks follow this page.

**Acceptance criteria:**

- [x] `reports/contract.md` maps `DIABETE4` codes to `no_diabetes`, `prediabetes`, and `diabetes`, and lists the codes left out (pregnancy-only, don't know, refused, blank), using the 2024 codebook
- [x] Main model inputs are age (`_AGE80`), BMI (`_BMI5`), any exercise (`EXERANY2`), smoking status (`_SMOKER3`), smokeless tobacco (`USENOW3`), e-cigarettes (`ECIGNOW3`), heavy drinking (`_RFDRHV9`), sex (`SEXVAR`), and sugary soda (`SSBSUGR2`). The list is not closed.
- [x] Height (`HEIGHT3`) and weight (`WEIGHT2`) are marked descriptive-only
- [x] The decision metric is validation macro F1, with per-class recall reported beside it
- [x] Split is 60% train, 20% validation, 20% test, stratified on the target
- [x] The modeling table is 40,000 interviews with `DIABETE4` in {1, 3, 4} and a real `SSBSUGR2` answer, stratified on the target
- [x] Model type is not fixed. The reference predicts the majority class, and the chosen model is the highest validation macro F1
- [x] The page states the model predicts interview responses and is not a diagnosis

**Verification:**

- [x] Manual check: every teammate can point to the target definition and the metric without opening a notebook
- [x] Manual check: code lists match the 2024 BRFSS codebook, not memory

**Dependencies:** None. Can be written beside Task 1.

**Files likely touched:**

- `reports/contract.md`

**Estimated scope:** Small: 1-2 files

## Checkpoint: Foundation

- [x] The team has read `reports/contract.md` and agrees on the target, the exclusions, the main inputs, the 40,000-row sample, macro F1, and that model type is not fixed
- [x] Folders from Task 1 exist and raw data is gitignored
- [x] Review with the team before anyone downloads or models

## Task 3: Load BRFSS into a thin Parquet table
**Owner:** Latifah

**Description:** Download the 2024 combined landline and cell-phone public-use file from the CDC page in the README. Read it once, keep the main contract columns plus the survey weight and a row id, draw the 40,000-row sample defined in `reports/contract.md`, and save a Parquet file later tasks will use.

**Acceptance criteria:**

- [x] `data/raw/` holds the original SAS transport file, unmodified
- [x] A script writes `data/processed/brfss2024_thin.parquet`
- [x] The Parquet file contains: a row id, `DIABETE4`, `DIABTYPE`, `_AGE80`, `HEIGHT3`, `WEIGHT2`, `_BMI5`, `EXERANY2`, `_SMOKER3`, `USENOW3`, `ECIGNOW3`, `_RFDRHV9`, `SEXVAR`, `SSBSUGR2`, and the weight column named in the codebook
- [x] The script prints row count and column names, and the row count is 40,000
- [x] README says the download URL and the command that rebuilds the Parquet file

**Verification:**

- [x] Manual check: rerunning the script replaces the Parquet file and prints the same row count
- [x] Manual check: the Parquet schema does not include the other BRFSS columns

**Dependencies:** Task 1, Task 2

**Files likely touched:**

- `src/diabetes_risk/load.py`
- `README.md`

**Estimated scope:** Medium: 3-5 files

## Task 4: Write the column dictionary
**Owner:** Latifah

**Description:** Document every column in the thin table so special codes are not treated as real ages, BMIs, or answers. Include the `_BMI5` scale and the weight variable after checking the codebook.

**Acceptance criteria:**

- [x] `reports/data_dictionary.md` has one row per thin-table column: meaning, valid values, codes that mean missing or refused, and whether the model may use it
- [x] `_BMI5` scale is confirmed (raw value versus implied decimals) with one worked example
- [x] `DIABTYPE` notes that the question was asked only in states that fielded the diabetes module
- [x] Weight column name matches the codebook

**Verification:**

- [x] Manual check: each special code listed in the dictionary is visible in a value-count of the Parquet column
- [x] Manual check: a teammate who did not load the file can clean a column using only this dictionary

**Dependencies:** Task 2, Task 3

**Files likely touched:**

- `reports/data_dictionary.md`

**Estimated scope:** Small: 1-2 files

## Task 5: Implement cleaning as importable functions

**Owner:** Nasser Abanmy

**Description:** Turn the dictionary into functions other tasks import. Map non-answers to missing, apply the BMI scale, and build the three-class target. Height and weight are parsed only far enough to audit BMI, because their raw codes mix units.

**Acceptance criteria:**

- [x] `prepare_table` returns a dataframe with `target` in {`no_diabetes`, `prediabetes`, `diabetes`} and main model columns `age`, `bmi`, `any_exercise`, `smoker_status`, `smokeless_tobacco`, `ecigarette`, `heavy_drinker`, `sex`, `sugar_drinks`
- [x] Pregnancy-only, don't know, refused, and blank `DIABETE4` values are not assigned a target class
- [x] Non-answer codes for every model input become missing, using the code lists in `reports/contract.md`
- [x] Tests cover at least: a yes/no/prediabetes mapping, a pregnancy-only row left unlabeled, a refused BMI becoming missing, and the BMI scale example from the dictionary
- [x] No notebook contains a second copy of these rules

**Verification:**

- [x] Tests pass: `python -m pytest tests/test_clean.py`
- [x] Manual check: row counts of each target class are written to `reports/class_counts.csv`

**Dependencies:** Task 2, Task 4

**Files likely touched:**

- `src/diabetes_risk/clean.py`
- `tests/test_clean.py`
- `reports/class_counts.csv`

**Estimated scope:** Medium: 3-5 files

## Task 6: Freeze the train, validation, and test ids

**Owner:** Nasser Abanmy

**Description:** Split labeled rows 60/20/20, stratified on `target`, with a fixed seed. Save the assignment so every model uses the same people. Unlabeled rows stay out of the split.

**Acceptance criteria:**

- [x] `data/processed/split.csv` has `row_id` and `split` (`train`, `validation`, `test`)
- [x] Shares are 60/20/20 within a rounding tolerance of one percentage point, and each split's target mix matches the overall mix within one percentage point
- [x] The seed, the split fractions, and the 40,000-row sample size live in one config module named by the contract
- [x] Calling the split function twice on the same input returns the same ids

**Verification:**

- [x] Tests pass: `python -m pytest tests/test_split.py`
- [x] Manual check: deleting `split.csv` and rerunning the script recreates the same ids

**Dependencies:** Task 2, Task 5

**Files likely touched:**

- `src/diabetes_risk/split.py`
- `src/diabetes_risk/config.py`
- `tests/test_split.py`

**Estimated scope:** Medium: 3-5 files

## Checkpoint: Shared table

- [ ] One command rebuilds the modeling table from `data/raw`
- [ ] `data/processed/split.csv` is the only split
- [ ] `python -m pytest tests/test_clean.py tests/test_split.py` passes
- [ ] Review with the team before modeling

## Task 7: Data-quality notebook and saved summary

**Owner:** Abdulmalk Alnajem

**Description:** Profile the thin table in a notebook, then save the conclusions as files. The notebook is the author's scratch work. The summary is what the team reads.

**Acceptance criteria:**

- [ ] `notebooks/01_data_quality.ipynb` shows missingness, target counts, and impossible values (for example BMI far outside a human range) using the cleaning functions
- [ ] `reports/data_quality.md` states how many rows are unlabeled, how many labeled rows remain, and any column that is too missing to use
- [ ] Figures cited in that note are saved under `reports/figures/`

**Verification:**

- [ ] Manual check: the notebook runs from top to bottom on the Parquet file
- [ ] Manual check: `reports/data_quality.md` numbers match `reports/class_counts.csv`

**Dependencies:** Task 3, Task 5

**Files likely touched:**

- `notebooks/01_data_quality.ipynb`
- `reports/data_quality.md`
- `reports/figures/`

**Estimated scope:** Medium: 3-5 files

## Task 8: Relationship charts for age, body size, and activity

**Owner:** Abdulmalk Alnajem

**Description:** Answer the first half of the project question with weighted descriptive charts: how age, BMI, height, weight, and physical activity differ across the three diabetes statuses. This task does not fit a model.

**Acceptance criteria:**

- [ ] Rates and means that describe the survey use the weight column from the dictionary
- [ ] Saved figures compare the three target classes on age, BMI, and exercise
- [ ] A short height-and-weight note shows they track BMI, which is why the model does not also include them
- [ ] `reports/relationships.md` states each comparison in a few sentences and links the figure files
- [ ] The notebook imports cleaning from `src/diabetes_risk/` rather than recoding `DIABETE4` inline

**Verification:**

- [ ] Manual check: `notebooks/02_relationships.ipynb` runs from top to bottom
- [ ] Manual check: every number in `reports/relationships.md` is reproducible from the saved table or figure

**Dependencies:** Task 5. Can run beside Tasks 9 and 10.

**Files likely touched:**

- `notebooks/02_relationships.ipynb`
- `reports/relationships.md`
- `reports/figures/`

**Estimated scope:** Medium: 3-5 files

## Task 9: Diabetes-type table for respondents with diabetes

**Owner:** Bandar Alshahrani

**Description:** Answer the second half of the project question. Among people who reported diabetes, summarize `DIABTYPE`, only on interviews where the diabetes module was used. Keep this separate from the classifier.

**Acceptance criteria:**

- [x] The table is limited to `DIABETE4` diabetes and to rows where `DIABTYPE` was actually asked
- [x] Counts and weighted percents are saved to `reports/diabetes_type.csv` for each type code, including don't know and refused
- [x] `reports/diabetes_type.md` states how many interviews were excluded because the module was not fielded
- [x] The type labels match the codebook

**Verification:**

- [x] Manual check: unweighted counts in the CSV equal a direct filter of the thin table
- [x] Manual check: the note does not describe type shares as a share of all 457,670 interviews

**Dependencies:** Task 3, Task 4. Can run beside Tasks 7 and 8.

**Files likely touched:**

- `src/diabetes_risk/diabetes_type.py`
- `reports/diabetes_type.csv`
- `reports/diabetes_type.md`

**Estimated scope:** Small: 1-2 files

## Checkpoint: Descriptive answer

- [ ] `reports/relationships.md` and `reports/diabetes_type.md` stand alone without a notebook
- [ ] Weighted figures name the weight column
- [ ] Review with the team. The written question is answerable even before the model is strong

## Task 10: Fit models against a majority-class reference

**Description:** Fit a majority-class reference and any other classifiers on the train split, using the main inputs named in `reports/contract.md`. Model type is not fixed. Score every model on the validation split. Record how many rows were fit and how many were left out.

**Acceptance criteria:**

- [ ] Training rows come from `split == train` joined to the cleaned table
- [ ] `reports/metrics_validation.csv` has one row per model, including the majority-class reference, with macro F1 and per-class recall
- [ ] Each fitted model is saved under `models/`
- [ ] The metrics record how many rows were fit and how many were left out
- [ ] The test split is not read

**Verification:**

- [ ] Tests pass: `python -m pytest tests/test_metrics.py` for the metric helper on a tiny hand-built example
- [ ] Manual check: rerunning training does not change `split.csv`

**Dependencies:** Task 5, Task 6

**Files likely touched:**

- `src/diabetes_risk/train.py`
- `src/diabetes_risk/evaluate.py`
- `tests/test_metrics.py`
- `reports/metrics_validation.csv`

**Estimated scope:** Medium: 3-5 files

## Task 11: Compare models and name the chosen one

**Description:** Compare the models from Task 10 on the same rows, main inputs, and validation metric. Further models may be added. Model type is not fixed. The chosen model is the one with the highest validation macro F1.

**Acceptance criteria:**

- [ ] Every compared model uses the same train rows and the same main inputs
- [ ] `reports/model_comparison.md` lists each model, including the majority-class reference, with validation macro F1, and names the chosen model
- [ ] The test split is still unread

**Verification:**

- [ ] Manual check: the comparison table matches `reports/metrics_validation.csv`
- [ ] Manual check: the chosen model is the highest validation macro F1; if that is the majority-class reference, the write-up says so

**Dependencies:** Task 10

**Files likely touched:**

- `src/diabetes_risk/train.py`
- `reports/metrics_validation.csv`
- `reports/model_comparison.md`

**Estimated scope:** Medium: 3-5 files

## Task 12: Error-analysis notebook on saved predictions

**Description:** Look at where the chosen model is wrong. Slice validation errors by age band, BMI band, and exercise. The notebook reviews the saved predictions for the chosen model.

**Acceptance criteria:**

- [ ] `notebooks/03_error_analysis.ipynb` reads predictions written by the training code
- [ ] `reports/error_analysis.md` names the slices with the weakest recall
- [ ] Figures for those slices are saved under `reports/figures/`
- [ ] The error review uses the saved predictions for the chosen model

**Verification:**

- [ ] Manual check: the notebook runs from top to bottom after Task 11
- [ ] Manual check: slice counts add up to the validation rows that were scored

**Dependencies:** Task 11

**Files likely touched:**

- `notebooks/03_error_analysis.ipynb`
- `reports/error_analysis.md`
- `reports/figures/`

**Estimated scope:** Medium: 3-5 files

## Task 13: Single training command for the chosen model

**Description:** Make one command rebuild the chosen model from the processed table and the frozen split, then write the model and a metrics file. This is the command a new teammate runs.

**Acceptance criteria:**

- [ ] `python -m diabetes_risk.train` trains the model named in `reports/model_comparison.md`
- [ ] It writes the model artifact under `models/` and writes `reports/metrics.json` with validation macro F1 and per-class recall
- [ ] It scores the test split only when passed an explicit flag, and writes those metrics to a separate key in `reports/metrics.json`
- [ ] README documents the command and the flag

**Verification:**

- [ ] Manual check: running the command twice on unchanged data reproduces the same validation macro F1
- [ ] Manual check: a run without the flag leaves test metrics absent from `reports/metrics.json`

**Dependencies:** Task 11

**Files likely touched:**

- `src/diabetes_risk/__main__.py`
- `src/diabetes_risk/train.py`
- `README.md`
- `reports/metrics.json`

**Estimated scope:** Medium: 3-5 files

## Checkpoint: Model

- [ ] The chosen model is the highest validation macro F1, and the comparison says whether it beats the majority-class reference
- [ ] `python -m diabetes_risk.train` rewrites the model and `reports/metrics.json`
- [ ] Test metrics are produced once, with the flag, and set aside for Task 14
- [ ] Review with the team before writing the results page

## Task 14: Results page with limitations

**Description:** Write the page a reader can finish in a few minutes. It answers both parts of the project question, names the model and the test metric, and states the limits of self-report BRFSS data.

**Acceptance criteria:**

- [ ] `reports/results.md` covers: the question, the 2024 BRFSS source, the target and exclusions, the 40,000-row sample, the main inputs, the chosen model, validation and one-time test macro F1, and the diabetes-type result
- [ ] Limitations include: self-report rather than a lab test, not a diagnosis, pregnancy-only diabetes excluded, `DIABTYPE` only where the module was asked, descriptive rates are weighted and model scores are not population prevalence
- [ ] README links to `reports/results.md` and to the training command
- [ ] Numbers match `reports/metrics.json`, `reports/relationships.md`, and `reports/diabetes_type.md`

**Verification:**

- [ ] Manual check: a teammate who did not train the model can retell the target, the metric, and the limitations from this page alone
- [ ] Manual check: test metrics appear once and match the file produced by the flagged training command

**Dependencies:** Task 8, Task 9, Task 12, Task 13

**Files likely touched:**

- `reports/results.md`
- `README.md`

**Estimated scope:** Small: 1-2 files

## Checkpoint: Complete

- [ ] A new teammate can follow the README, run `python -m diabetes_risk.train`, and match `reports/metrics.json`
- [ ] `reports/results.md` states self-report, exclusions, the one-time test rule, and that this is not a diagnosis
- [ ] Review with the team before sharing results outside the project
