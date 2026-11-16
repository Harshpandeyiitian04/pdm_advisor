# This file is a second cost check, separate from the main decision file.
# It prices plans using hours of downtime and the cost of servicing an engine that was not actually close to failing.

# These lines bring in a simple way to store settings, and number tools.
from dataclasses import dataclass

import numpy as np


# These are the plant numbers someone types in for this extra cost check.
@dataclass
class BusinessCaseAssumptions:
    # How much one hour of stopped production costs.
    downtime_cost_per_hour: float
    # How many hours a breakdown typically stops the plant.
    downtime_hours_per_failure: float
    # What one planned service visit costs.
    planned_maintenance_cost: float
    # Extra cost of servicing an engine that still had plenty of life left.
    false_alarm_cost: float
    # A planned service counts as a false alarm if more than this many cycles of life would still remain.
    false_alarm_horizon_cycles: int = 60
    # The predictive plan services this many cycles before the guessed end.
    safety_margin: int = 15
    # The fixed plan services every engine at this age.
    fixed_age: int = 139


# This compares do-nothing, service-by-age, and service-from-the-guess, using downtime and false-alarm costs.
def compare_business_case(pred_rul, true_rul, current_cycle, assumptions):
    # Turn the guesses, the real life left, and the current ages into number lists.
    pred = np.asarray(pred_rul, dtype=float)
    true = np.asarray(true_rul, dtype=float)
    cycles = np.asarray(current_cycle, dtype=float)
    # Stop if the three lists are not the same shape, are empty, or are not a single list.
    if pred.shape != true.shape or true.shape != cycles.shape or true.ndim != 1 or not true.size:
        raise ValueError("pred_rul, true_rul, and current_cycle must be non-empty 1D arrays of equal length")
    # Stop if any cost or downtime number is negative.
    if min(assumptions.downtime_cost_per_hour, assumptions.downtime_hours_per_failure,
           assumptions.planned_maintenance_cost, assumptions.false_alarm_cost) < 0:
        raise ValueError("cost and downtime assumptions must be non-negative")

    # A breakdown costs the hourly rate times the hours the plant is down.
    failure_cost = assumptions.downtime_cost_per_hour * assumptions.downtime_hours_per_failure

    # This prices one plan: pay the breakdown cost if we are too late, otherwise the planned visit, plus a false-alarm fee when we were far too early.
    def policy_costs(service_cycles):
        # True when the service would happen at or after the real failure.
        failure = service_cycles >= true
        # How much life would still be left at the service moment.
        remaining_life = true - service_cycles
        # A preventive visit is any service that happens before failure.
        preventive = ~failure
        # A false alarm is a preventive visit that still leaves more life than the chosen horizon.
        false_alarm = preventive & (remaining_life > assumptions.false_alarm_horizon_cycles)
        # Start with the breakdown cost or the planned-visit cost.
        costs = np.where(failure, failure_cost, assumptions.planned_maintenance_cost)
        # Add the false-alarm fee where it applies.
        costs += false_alarm * assumptions.false_alarm_cost
        # Return the total cost, the number of breakdowns, and the number of false alarms.
        return float(costs.sum()), int(failure.sum()), int(false_alarm.sum())

    # The age-based plan waits until the engine reaches a set age.
    fixed_service = np.maximum(0.0, assumptions.fixed_age - cycles)
    # The predictive plan services a safety buffer before the guessed end.
    predictive_service = np.maximum(0.0, pred - assumptions.safety_margin)
    # Price both plans.
    fixed_cost, fixed_failures, fixed_false_alarms = policy_costs(fixed_service)
    predictive_cost, predictive_failures, predictive_false_alarms = policy_costs(predictive_service)
    # Doing nothing means every engine eventually breaks down.
    run_to_failure = failure_cost * len(true)
    # Hand back the costs, the failure counts, the false-alarm counts, and the percent saved.
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
