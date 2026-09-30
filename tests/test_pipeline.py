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


def test_metrics_penalise_late_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40])  # late worse than early
    assert rmse([1, 2], [1, 2]) == 0


def test_rul_capped_and_nonnegative():
    df = pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)})
    out = D.add_rul(df)
    assert out["rul"].max() == D.RUL_CAP and out["rul"].min() == 0


def test_features_do_not_mix_engines():
    df = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2], "s1": [1.0, 2.0, 100.0, 101.0]})
    f = D.build_features(df, ["s1"], window=2)
    assert f.loc[(f.unit == 2) & (f.cycle == 1), "s1_mean"].iloc[0] == 100.0


def test_perfect_prediction_beats_fixed_schedule():
    true = np.array([10, 40, 80, 120])
    res = compare_policies(true, true, np.full(4, 100), CostAssumptions(fixed_age=100))
    assert res["predictive"] < res["fixed_schedule"] and res["failures_predictive"] == 0


def test_bands():
    assert band(10) == "Critical" and band(50) == "Plan soon" and band(100) == "Healthy"


def test_eda_reports_within_engine_trends_and_constant_sensors():
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


def test_deep_windows_are_causal_and_left_padded():
    values = np.array([[1.0], [2.0], [3.0]])
    windows, targets = make_windows(values, [10, 20, 30], window=3)
    np.testing.assert_array_equal(windows[:, :, 0], [[1, 1, 1], [1, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(targets, [10, 20, 30])


def test_deep_test_metrics_use_last_cycle_per_engine():
    frame = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2]})
    np.testing.assert_array_equal(last_cycle_indices(frame), [1, 3])


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


def test_synthetic_training_uses_temporary_input_and_output_paths(monkeypatch, tmp_path):
    def fake_make_synthetic(out_dir):
        Path(out_dir).mkdir(parents=True)

    def fake_train(subset, synthetic, raw_dir, output_root):
        assert subset == "FD001" and synthetic is True
        assert Path(raw_dir) != D.RAW and Path(output_root) != T.ROOT
        assert Path(raw_dir).is_dir()
        return "smoke complete"

    monkeypatch.setattr(D, "make_synthetic", fake_make_synthetic)
    monkeypatch.setattr(T, "_train", fake_train)
    assert T.main(synthetic=True) == "smoke complete"
