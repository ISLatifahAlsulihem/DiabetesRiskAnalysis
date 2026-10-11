# Diabetes risk analysis

## 1. Question

How do age, height, weight, BMI, and physical activity relate to self-reported diabetes status (no diabetes, prediabetes, or diabetes), and what type of diabetes do people with diabetes report?

## 2. Data

| Item | Detail |
| --- | --- |
| Source | U.S. Centers for Disease Control and Prevention, Behavioral Risk Factor Surveillance System (BRFSS), combined landline and cell phone public-use file |
| Link | https://www.cdc.gov/brfss/annual_data/annual_2024.html |
| Time period | Interviews collected in 2024. Some states finished interviews in early 2025. CDC released the public file in August 2025. |
| Coverage | 49 states, the District of Columbia, Guam, Puerto Rico, and the U.S. Virgin Islands. |
| Rows | 457,670 interviews |
| Columns | 345 variables in the SAS transport file |

The diabetes status question is `DIABETE4` (yes, only during pregnancy, no, or prediabetes/borderline). Diabetes type is `DIABTYPE` and was asked only in states that used the diabetes module. Age, height, weight, BMI, and activity are in the same file (`_AGE80`, `HEIGHT3`, `WEIGHT2`, `_BMI5`, `EXERANY2`). The file does not ask whether a relative has diabetes.

The main model inputs are listed in [reports/contract.md](reports/contract.md). Besides age, BMI, and exercise, they include cigarette smoking (`_SMOKER3`), smokeless tobacco (`USENOW3`), e-cigarettes (`ECIGNOW3`), heavy drinking (`_RFDRHV9`), sex (`SEXVAR`), and sugary soda (`SSBSUGR2`). That list is the starting set, not a closed one. The modeling table is a 40,000-interview sample from people who were asked the soda question and who have one of the three diabetes classes. The public file itself still has 457,670 interviews.

The data are a U.S. government work in the public domain. Cite CDC when the results are shared.

## 3. Team members

- Abdulaziz -/owAziz
- Latifah Alsulihem -/ISLatifahAlsulihem
- Bandar Alshahrani -/bandar345
- Abdulmalk Alnajem -/AbdulmalkNaj
- Latifa Aljbali -/laajbali
## 4. Status

Dataset selected. The project layout and the analysis contract are in place. Modeling has not started.

## 5. Layout

The implementation plan is in [tasks/plan.md](tasks/plan.md). The analysis contract is in [reports/contract.md](reports/contract.md). It defines the target, the main inputs, the 40,000-row sample, the metric, and the split. It does not fix the model type.

Notebooks are for exploration. Shared code belongs in `src/diabetes_risk/`.

| Place | What goes here |
| --- | --- |
| `notebooks/` | A new notebook: plots, questions, and error review |
| `src/diabetes_risk/` | A new shared function: loading, cleaning, the split, training, and evaluation |
| `data/raw/` | The CDC file, unchanged. Git ignores this folder. |
| `data/processed/` | Tables produced by scripts. Git ignores this folder. |
| `reports/` | The contract, tables, and the short write-up |
| `reports/figures/` | Saved figures |
| `models/` | Saved models. Git ignores this folder. |
| `tests/` | Tests for shared code |

## Data Loading
- Raw Data Source: [CDC 2024 BRFSS Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
- To build the thin Parquet table, place `LLCP2024.XPT` in `data/raw/` and run:
```bash
python src/diabetes_risk/load.py
```
