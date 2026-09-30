"""Optional downtime and false-alarm cost scenario, separate from decision.py."""
from dataclasses import dataclass

import numpy as np


@dataclass
class BusinessCaseAssumptions:
    downtime_cost_per_hour: float
    downtime_hours_per_failure: float
    planned_maintenance_cost: float
    false_alarm_cost: float
    false_alarm_horizon_cycles: int = 60
    safety_margin: int = 15
    fixed_age: int = 139


def compare_business_case(pred_rul, true_rul, current_cycle, assumptions):
    """Compare policies using downtime and false-alarm costs supplied by the user.

    A preventive action is counted as a false alarm when more than
    ``false_alarm_horizon_cycles`` of actual life would remain at service time.
    """
    pred = np.asarray(pred_rul, dtype=float)
    true = np.asarray(true_rul, dtype=float)
    cycles = np.asarray(current_cycle, dtype=float)
    if pred.shape != true.shape or true.shape != cycles.shape or true.ndim != 1 or not true.size:
        raise ValueError("pred_rul, true_rul, and current_cycle must be non-empty 1D arrays of equal length")
    if min(assumptions.downtime_cost_per_hour, assumptions.downtime_hours_per_failure,
           assumptions.planned_maintenance_cost, assumptions.false_alarm_cost) < 0:
        raise ValueError("cost and downtime assumptions must be non-negative")

    failure_cost = assumptions.downtime_cost_per_hour * assumptions.downtime_hours_per_failure

    def policy_costs(service_cycles):
        failure = service_cycles >= true
        remaining_life = true - service_cycles
        preventive = ~failure
        false_alarm = preventive & (remaining_life > assumptions.false_alarm_horizon_cycles)
        costs = np.where(failure, failure_cost, assumptions.planned_maintenance_cost)
        costs += false_alarm * assumptions.false_alarm_cost
        return float(costs.sum()), int(failure.sum()), int(false_alarm.sum())

    fixed_service = np.maximum(0.0, assumptions.fixed_age - cycles)
    predictive_service = np.maximum(0.0, pred - assumptions.safety_margin)
    fixed_cost, fixed_failures, fixed_false_alarms = policy_costs(fixed_service)
    predictive_cost, predictive_failures, predictive_false_alarms = policy_costs(predictive_service)
    run_to_failure = failure_cost * len(true)
    return {
        "run_to_failure": float(run_to_failure),
        "fixed_schedule": fixed_cost,
        "predictive": predictive_cost,
        "failure_cost_per_engine": float(failure_cost),
        "failures_fixed": fixed_failures,
        "failures_predictive": predictive_failures,
        "false_alarms_fixed": fixed_false_alarms,
        "false_alarms_predictive": predictive_false_alarms,
        "n_engines": int(len(true)),
        "saving_vs_fixed_pct": 100 * (1 - predictive_cost / fixed_cost) if fixed_cost else 0.0,
    }