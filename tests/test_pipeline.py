# This file checks that the important pieces still behave correctly: scores, life-left labels, features, costs, windows, and the fake-data training path.

# These lines bring in paths, numbers, tables, and the pieces of the project that are being checked.
from pathlib import Path

import numpy as np
import pandas as pd
from src import data as D
from src.business_case import BusinessCaseAssumptions, compare_business_case
from src.deep import last_cycle_indices, make_windows
from src.eda import analyze_training_data
from src.decision import CostAssumptions, compare_policies, band
from src.explain import facts_text
from src.metrics import rmse, nasa_score
from src import train as T


# This checks that guessing too much life left scores worse than guessing too little, and that a perfect guess scores zero.
def test_metrics_penalise_late_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert rmse([1, 2], [1, 2]) == 0


# This checks that life left never goes above the cap and never goes below zero.
def test_rul_capped_and_nonnegative():
    # A single engine with 200 cycles.
    df = pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)})
    out = D.add_rul(df)
    assert out["rul"].max() == D.RUL_CAP and out["rul"].min() == 0


# This checks that one engine's recent average is not polluted by another engine's readings.
def test_features_do_not_mix_engines():
    # Two engines, two cycles each. Engine 2's sensor is much larger.
    df = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2], "s1": [1.0, 2.0, 100.0, 101.0]})
    f = D.build_features(df, ["s1"], window=2)
    # Engine 2's first cycle should still average to its own reading.
    assert f.loc[(f.unit == 2) & (f.cycle == 1), "s1_mean"].iloc[0] == 100.0


# This checks that a perfect life-left guess costs less than servicing every engine at a fixed age, and causes no surprise failures.
def test_perfect_prediction_beats_fixed_schedule():
    true = np.array([10, 40, 80, 120])
    res = compare_policies(true, true, np.full(4, 100), CostAssumptions(fixed_age=100))
    assert res["predictive"] < res["fixed_schedule"] and res["failures_predictive"] == 0


# This checks the three urgency labels: very little life, some life, and plenty of life.
def test_bands():
    assert band(10) == "Critical" and band(50) == "Plan soon" and band(100) == "Healthy"


# This checks that a flat sensor is reported as constant, and that a sensor which rises with age is reported as trending up.
def test_eda_reports_within_engine_trends_and_constant_sensors():
    # Two engines. Sensor 1 rises. Sensor 2 never moves.
    train = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 1, 2, 3],
        "s1": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        "s2": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })
    analysis = analyze_training_data(train)
    assert analysis["constant_sensors"] == ["s2"]
    assert analysis["lifetimes"]["engine_count"] == 2
    assert analysis["lifetimes"]["median_cycles"] == 3
    assert analysis["sensor_trends"].iloc[0]["median_spearman_with_cycle"] == 1


# This checks that each window only uses the current reading and earlier ones, with the first reading repeated to fill the start.
def test_deep_windows_are_causal_and_left_padded():
    values = np.array([[1.0], [2.0], [3.0]])
    windows, targets = make_windows(values, [10, 20, 30], window=3)
    np.testing.assert_array_equal(windows[:, :, 0], [[1, 1, 1], [1, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(targets, [10, 20, 30])


# This checks that the last-cycle finder points at the newest row of each engine.
def test_deep_test_metrics_use_last_cycle_per_engine():
    frame = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2]})
    np.testing.assert_array_equal(last_cycle_indices(frame), [1, 3])


# This checks the downtime calculator: a breakdown price, a planned visit plus a false-alarm fee, and a counted false alarm.
def test_business_case_applies_downtime_and_false_alarm_assumptions():
    assumptions = BusinessCaseAssumptions(
        downtime_cost_per_hour=100,
        downtime_hours_per_failure=10,
        planned_maintenance_cost=200,
        false_alarm_cost=50,
        false_alarm_horizon_cycles=60,
        safety_margin=10,
        fixed_age=100,
    )
    result = compare_business_case([200], [300], [0], assumptions)
    assert result["run_to_failure"] == 1000
    assert result["predictive"] == 250
    assert result["false_alarms_predictive"] == 1


# This checks that the maintenance note is readable text with a blank line before the engine and before the cost section.
def test_memo_facts_are_readable_markdown():
    predictions = pd.DataFrame({"unit": [3], "pred_rul": [12], "band": ["Critical"]})
    costs = {
        "n_engines": 1,
        "run_to_failure": 100,
        "fixed_schedule": 80,
        "predictive": 90,
        "saving_vs_fixed_pct": -12.5,
        "failures_fixed": 0,
        "failures_predictive": 1,
    }
    importance = pd.DataFrame({"feature": ["s4_mean"]})
    facts = facts_text(predictions, costs, importance)
    assert "\n\n- Engine 3" in facts
    assert "\n\n**Simulated policy costs:**" in facts


# This checks that a fake-data training run writes into temporary folders and never touches the real data or report folders.
def test_synthetic_training_uses_temporary_input_and_output_paths(monkeypatch, tmp_path):
    # A stand-in that only creates the folder, instead of writing fake engines.
    def fake_make_synthetic(out_dir):
        Path(out_dir).mkdir(parents=True)

    # A stand-in that checks the folders are temporary, then pretends training finished.
    def fake_train(subset, synthetic, raw_dir, output_root):
        assert subset == "FD001" and synthetic is True
        assert Path(raw_dir) != D.RAW and Path(output_root) != T.ROOT
        assert Path(raw_dir).is_dir()
        return "smoke complete"

    # Swap the real functions for the stand-ins, then run the fake-data path.
    monkeypatch.setattr(D, "make_synthetic", fake_make_synthetic)
    monkeypatch.setattr(T, "_train", fake_train)
    assert T.main(synthetic=True) == "smoke complete"
