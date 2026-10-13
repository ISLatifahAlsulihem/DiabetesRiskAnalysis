import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from diabetes_risk.clean import class_counts, prepare_table


def make_rows():
    # DIABETE4, _AGE80, _BMI5, EXERANY2, _SMOKER3, USENOW3, ECIGNOW3, _RFDRHV9, SEXVAR, SSBSUGR2
    return pd.DataFrame(
        [
            (1, 45, 2845, 1, 1, 1, 1, 1, 1, 102),   # diabetes
            (3, 30, 3500, 2, 4, 3, 1, 2, 2, 888),   # no diabetes, never drinks soda
            (4, 60, None, 1, 9, 7, 7, 9, 1, 777),   # prediabetes, everything refused/unknown
            (2, 28, 2200, 1, 3, 1, 1, 1, 2, 201),   # pregnancy-only: no target
            (7, 50, 2500, 2, 2, 2, 2, 2, 1, 330),   # don't know DIABETE4: no target
        ],
        columns=[
            "DIABETE4", "_AGE80", "_BMI5", "EXERANY2", "_SMOKER3",
            "USENOW3", "ECIGNOW3", "_RFDRHV9", "SEXVAR", "SSBSUGR2",
        ],
    )


def test_target_maps_yes_no_prediabetes():
    prepared = prepare_table(make_rows())
    assert prepared["target"].tolist()[:3] == ["diabetes", "no_diabetes", "prediabetes"]


def test_pregnancy_only_and_dont_know_are_left_unlabeled():
    prepared = prepare_table(make_rows())
    assert prepared["target"].iloc[3:].isna().all()


def test_bmi_scale_matches_the_dictionary_example():
    prepared = prepare_table(make_rows())
    assert prepared["bmi"].iloc[0] == 28.45
    assert prepared["bmi"].iloc[1] == 35.00


def test_refused_bmi_becomes_missing():
    prepared = prepare_table(make_rows())
    assert pd.isna(prepared["bmi"].iloc[2])


def test_non_answer_codes_become_missing():
    prepared = prepare_table(make_rows())
    row = prepared.iloc[2]  # _SMOKER3=9, USENOW3=7, ECIGNOW3=7, _RFDRHV9=9
    assert pd.isna(row["smoker_status"])
    assert pd.isna(row["smokeless_tobacco"])
    assert pd.isna(row["ecigarette"])
    assert pd.isna(row["heavy_drinker"])


def test_known_codes_map_to_readable_values():
    prepared = prepare_table(make_rows())
    row = prepared.iloc[0]
    assert row["any_exercise"] == True
    assert row["smoker_status"] == "current_every_day"
    assert row["smokeless_tobacco"] == "every_day"
    assert row["ecigarette"] == "never"
    assert row["heavy_drinker"] == False
    assert row["sex"] == "male"


def test_sugar_drinks_unit_conversion():
    prepared = prepare_table(make_rows())
    assert prepared["sugar_drinks"].iloc[0] == 2.0           # 102 -> 2 times/day
    assert prepared["sugar_drinks"].iloc[1] == 0.0            # 888 -> never
    assert pd.isna(prepared["sugar_drinks"].iloc[2])          # 777 -> missing
    assert prepared["sugar_drinks"].iloc[3] == pytest.approx(1 / 7)   # 201 -> 1/7 per day
    assert prepared["sugar_drinks"].iloc[4] == pytest.approx(1.0)      # 330 -> 1/day


def test_refused_blank_and_out_of_range_codes_are_missing():
    frame = pd.DataFrame(
        [
            (9, 40, 2500, 7, 9, 9, 9, 9, 1, 999),
            (None, 40, 2500, 9, 4, 3, 1, 1, 2, 200),
            (1, 40, 2500, 1, 4, 3, 1, 1, 1, 100),
            (3, 40, 2500, 2, 4, 3, 1, 1, 2, 300),
        ],
        columns=[
            "DIABETE4", "_AGE80", "_BMI5", "EXERANY2", "_SMOKER3",
            "USENOW3", "ECIGNOW3", "_RFDRHV9", "SEXVAR", "SSBSUGR2",
        ],
    )
    prepared = prepare_table(frame)
    assert prepared["target"].iloc[:2].isna().all()
    assert prepared["any_exercise"].iloc[:2].isna().all()
    assert prepared["sugar_drinks"].isna().all()


def test_class_counts_reports_unlabeled_rows():
    prepared = prepare_table(make_rows())
    counts = class_counts(prepared).set_index("target")["count"].to_dict()
    assert counts["diabetes"] == 1
    assert counts["no_diabetes"] == 1
    assert counts["prediabetes"] == 1
    assert counts["unlabeled"] == 2
