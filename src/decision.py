"""Turns predicted RUL into maintenance decisions and compares policies on cost.
All cost figures are ASSUMPTIONS supplied by the user, not measured data."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class CostAssumptions:
    failure_cost: float = 50000.0   # unplanned failure: downtime plus repair, per engine
    planned_cost: float = 8000.0    # one planned maintenance visit
    waste_per_cycle: float = 60.0   # value lost per cycle of unused life at service time
    safety_margin: int = 15         # predictive policy services this many cycles before predicted end
    fixed_age: int = 150            # fixed policy: service every engine at this age (cycles), blind to condition


def band(rul, critical=30, soon=60):
    return "Critical" if rul <= critical else ("Plan soon" if rul <= soon else "Healthy")


def _cost(service_at, true_rul, a):
    """Cost of servicing `service_at` cycles from now when the engine truly fails at `true_rul`."""
    if service_at >= true_rul:
        return a.failure_cost
    return a.planned_cost + a.waste_per_cycle * (true_rul - service_at)


def compare_policies(pred_rul, true_rul, current_cycle, a=None):
    """Fixed policy uses only engine age; predictive uses the model. Both judged against true RUL."""
    a = a or CostAssumptions()
    pred, true, cyc = np.asarray(pred_rul, float), np.asarray(true_rul, float), np.asarray(current_cycle, float)
    fixed_at = np.maximum(0.0, a.fixed_age - cyc)
    fixed = np.array([_cost(f, t, a) for f, t in zip(fixed_at, true)])
    predictive = np.array([_cost(max(0.0, p - a.safety_margin), t, a) for p, t in zip(pred, true)])
    res = {
        "run_to_failure": float(a.failure_cost * len(true)),
        "fixed_schedule": float(fixed.sum()),
        "predictive": float(predictive.sum()),
        "failures_fixed": int((fixed_at >= true).sum()),
        "failures_predictive": int((np.maximum(0, pred - a.safety_margin) >= true).sum()),
        "n_engines": int(len(true)),
    }
    res["saving_vs_run_to_failure_pct"] = 100 * (1 - res["predictive"] / res["run_to_failure"])
    res["saving_vs_fixed_pct"] = 100 * (1 - res["predictive"] / res["fixed_schedule"])
    return res


def sensitivity(pred_rul, true_rul, current_cycle, base=None, multipliers=(0.5, 1, 2)):
    """How much do savings change if the failure cost assumption is wrong?"""
    base = base or CostAssumptions()
    rows = []
    for m in multipliers:
        a = CostAssumptions(**{**asdict(base), "failure_cost": base.failure_cost * m})
        r = compare_policies(pred_rul, true_rul, current_cycle, a)
        rows.append({"failure_cost_x": m, "run_to_failure": r["run_to_failure"], "fixed_schedule": r["fixed_schedule"],
                     "predictive": r["predictive"], "saving_vs_fixed_pct": r["saving_vs_fixed_pct"]})
    return pd.DataFrame(rows)
