# This file turns a "life left" guess into a maintenance choice, then compares three ways of deciding when to service an engine.
# Every dollar figure here is an assumption someone typed in. These are not measured costs from a real plant.

# These lines bring in a simple way to store settings, number tools, and data tables.
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


# These are the cost assumptions used to compare the plans.
@dataclass
class CostAssumptions:
    # What an unexpected breakdown costs: lost time plus the repair, per engine.
    failure_cost: float = 50000.0
    # What one planned service visit costs.
    planned_cost: float = 8000.0
    # Value thrown away for each cycle of unused life when we service too early.
    waste_per_cycle: float = 60.0
    # The predictive plan services this many cycles before the guessed end, as a safety buffer.
    safety_margin: int = 15
    # The fixed plan services every engine at this age, without looking at its condition.
    fixed_age: int = 150


# This labels an engine as urgent, coming due, or fine, based on how many cycles are left.
def band(rul, critical=30, soon=60):
    return "Critical" if rul <= critical else ("Plan soon" if rul <= soon else "Healthy")


# This works out the cost of servicing at a chosen moment when the engine really fails later.
def _cost(service_at, true_rul, a):
    # If we wait until after the real failure, we pay the breakdown cost.
    if service_at >= true_rul:
        return a.failure_cost
    # If we service in time, we pay the planned visit plus the unused life we threw away.
    return a.planned_cost + a.waste_per_cycle * (true_rul - service_at)


# This compares three plans against the real life left: do nothing, service by age, or service from the model's guess.
def compare_policies(pred_rul, true_rul, current_cycle, a=None):
    # Use the default assumptions if none were passed in.
    a = a or CostAssumptions()
    # Turn the guesses, the real answers, and the current ages into number lists.
    pred, true, cyc = np.asarray(pred_rul, float), np.asarray(true_rul, float), np.asarray(current_cycle, float)
    # The fixed plan waits until the engine reaches a set age, then services it.
    fixed_at = np.maximum(0.0, a.fixed_age - cyc)
    # Cost of that age-based plan for every engine.
    fixed = np.array([_cost(f, t, a) for f, t in zip(fixed_at, true)])
    # The predictive plan services a safety buffer before the guessed end.
    predictive = np.array([_cost(max(0.0, p - a.safety_margin), t, a) for p, t in zip(pred, true)])
    # Add up the three plans and count how many engines still fail under each plan.
    res = {
        "run_to_failure": float(a.failure_cost * len(true)),
        "fixed_schedule": float(fixed.sum()),
        "predictive": float(predictive.sum()),
        "failures_fixed": int((fixed_at >= true).sum()),
        "failures_predictive": int((np.maximum(0, pred - a.safety_margin) >= true).sum()),
        "n_engines": int(len(true)),
    }
    # Turn the cost gaps into a percent saved.
    res["saving_vs_run_to_failure_pct"] = 100 * (1 - res["predictive"] / res["run_to_failure"])
    res["saving_vs_fixed_pct"] = 100 * (1 - res["predictive"] / res["fixed_schedule"])
    return res


# This checks how the savings change if the breakdown-cost assumption is half, the same, or double.
def sensitivity(pred_rul, true_rul, current_cycle, base=None, multipliers=(0.5, 1, 2)):
    # Start from the given assumptions, or the defaults.
    base = base or CostAssumptions()
    rows = []
    for m in multipliers:
        # Rebuild the assumptions with a different breakdown cost, and compare the plans again.
        a = CostAssumptions(**{**asdict(base), "failure_cost": base.failure_cost * m})
        r = compare_policies(pred_rul, true_rul, current_cycle, a)
        # Keep the costs and the percent saved for this version of the assumption.
        rows.append({"failure_cost_x": m, "run_to_failure": r["run_to_failure"], "fixed_schedule": r["fixed_schedule"],
                     "predictive": r["predictive"], "saving_vs_fixed_pct": r["saving_vs_fixed_pct"]})
    # Hand back a table of those what-if results.
    return pd.DataFrame(rows)
