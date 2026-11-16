# This file is the on-screen dashboard. It shows the fleet, one engine, the model scores, a cost what-if, and the maintenance note.
# Run it with: python -m streamlit run app/dashboard.py

# These lines bring in file reading, the project path, model loading, tables, and the dashboard tools.
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Make sure this project folder is on the search path, so the src package can be imported when the app is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data as D
from src.business_case import BusinessCaseAssumptions, compare_business_case
from src.decision import CostAssumptions, compare_policies, sensitivity

# Where the saved reports and the trained model live.
R = PROJECT_ROOT / "reports"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
# Set the browser tab title and use the wide page layout, then show the page heading.
st.set_page_config(page_title="Predictive Maintenance Advisor", layout="wide")
st.title("Predictive Maintenance Advisor")

# Stop the page with a clear message if training has not been run yet.
if not (R / "results.json").exists():
    st.error("No results found. Run `python -m src.train` first.")
    st.stop()

# Load the saved summary and the per-engine predictions.
res = json.loads((R / "results.json").read_text())
preds = pd.read_csv(R / "test_predictions.csv")


# Remember this result so the page does not re-read the engine files on every refresh.
@st.cache_data
def load_test_subset(subset):
    # Load the test engines and the real life-left numbers for the chosen data subset.
    _, test_data, test_rul = D.load_cmapss(subset)
    return test_data, test_rul


# Remember the loaded model so the page does not reload the model file on every refresh.
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Warn the reader if these numbers came from fake data, then remind them the costs are assumptions.
if res.get("synthetic_data"):
    st.warning("These results come from SYNTHETIC data and are not real.")
st.caption("Benchmark: NASA C-MAPSS turbofan data. Costs are simulated from adjustable assumptions, not real plant data.")

# The five pages of the dashboard.
t1, t2, t3, t4, t5 = st.tabs(["Fleet overview", "Engine detail", "Model comparison", "Cost simulator", "Maintenance memo"])

# First page: count engines in each urgency group, then show the full prediction table.
with t1:
    c = st.columns(3)
    for col, b in zip(c, ["Critical", "Plan soon", "Healthy"]):
        col.metric(b, int((preds["band"] == b).sum()))
    st.dataframe(preds, width="stretch")

# Second page: one engine's guess, a life-left chart over time, and one sensor's history.
with t2:
    # Load the raw test history for the same subset the results came from.
    test_data, test_rul = load_test_subset(res.get("subset", "FD001"))
    # Let the user pick an engine, then show its guess, urgency, and the real life left.
    unit = int(st.selectbox("Engine", sorted(preds["unit"].astype(int).tolist())))
    row = preds[preds["unit"] == unit].iloc[0]
    st.write(f"Engine {unit}: predicted **{row.pred_rul:.0f}** cycles left, band **{row.band}**, actual **{row.true_rul:.0f}**.")
    # Keep this engine's rows in time order.
    engine = test_data[test_data["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    bundle = load_model(str(MODEL_PATH))
    # Rebuild the clues the model expects, then guess life left at every recorded cycle.
    engine_features = D.build_features(engine, bundle["sensors"], bundle["window"])
    predicted_curve = pd.Series(
        bundle["model"].predict(engine_features[bundle["features"]]), index=engine["cycle"]
    ).clip(0, bundle["cap"])
    # Rebuild the real life left at every cycle from the final answer, and cap it the same way.
    actual_curve = (float(test_rul.loc[unit]) + engine["cycle"].max() - engine["cycle"]).clip(upper=bundle["cap"])
    rul_curve = pd.DataFrame({
        "Cycle": engine["cycle"].to_numpy(),
        "Actual RUL": actual_curve.to_numpy(),
        "Predicted RUL": predicted_curve.to_numpy(),
    })
    # Draw the real and guessed life left on one chart.
    st.subheader("RUL over observed cycles")
    st.line_chart(rul_curve, x="Cycle", y=["Actual RUL", "Predicted RUL"])
    # Let the user pick a sensor and draw that sensor's readings over time.
    sensor = st.selectbox("Sensor history", bundle["sensors"])
    sensor_curve = pd.DataFrame({"Cycle": engine["cycle"], "Sensor reading": engine[sensor]})
    st.line_chart(sensor_curve, x="Cycle", y="Sensor reading")
    st.caption("Actual RUL is available here because this view uses labeled benchmark test data; it is not available for a live engine.")

# Third page: compare the models, show which clues mattered, and show which sensors drift with age.
with t3:
    model_rows = pd.DataFrame(res["models"])
    # If the optional sequence-model result exists, add it to the comparison table.
    deep_path = R / f"deep_results_{res.get('subset', 'FD001')}.json"
    if deep_path.exists():
        deep = json.loads(deep_path.read_text())
        model_rows = pd.concat([model_rows, pd.DataFrame([{
            "model": deep["model"],
            "val_rmse": deep["val_rmse"],
            "test_rmse": deep["test_rmse"],
            "test_nasa_score": deep["test_nasa_score"],
        }])], ignore_index=True)
    # Show the scores, which model was chosen, and a bar chart of the strongest clues.
    st.dataframe(model_rows.round(2), width="stretch", hide_index=True)
    st.write(f"Selected on validation error: **{res['best_model']}**. Final test RMSE {res['final_test_rmse']:.2f}, NASA score {res['final_test_nasa_score']:.0f}.")
    st.subheader("Top signals")
    st.bar_chart(pd.read_csv(R / "feature_importance.csv").set_index("feature")["importance"])
    # If the sensor-trend file exists, show the ten strongest age links.
    trend_path = R / "sensor_trends.csv"
    if trend_path.exists():
        st.subheader("Training-set sensor trends")
        st.caption("Median within-engine Spearman correlation with cycle; exploratory, not causal.")
        trends = pd.read_csv(trend_path).head(10).set_index("sensor")["median_spearman_with_cycle"]
        st.bar_chart(trends)

# Fourth page: sliders for the cost assumptions, a chart of the three plans, and a separate downtime what-if.
with t4:
    # Start the sliders from the assumptions that were saved with the results.
    base = res["cost_assumptions"]
    a = CostAssumptions(
        failure_cost=st.number_input("Unplanned failure cost", value=float(base["failure_cost"]), step=1000.0),
        planned_cost=st.number_input("Planned maintenance cost", value=float(base["planned_cost"]), step=500.0),
        waste_per_cycle=st.number_input("Cost per wasted cycle", value=float(base["waste_per_cycle"]), step=10.0),
        safety_margin=int(st.slider("Safety margin (cycles)", 0, 60, int(base["safety_margin"]))),
        fixed_age=int(st.slider("Fixed-schedule service age", 50, 250, int(base["fixed_age"]))))
    # Price the three plans with those slider values and show the totals.
    r = compare_policies(preds["pred_rul"], preds["true_rul"], preds["cycle"], a)
    st.bar_chart(pd.Series({"Run to failure": r["run_to_failure"], "Fixed schedule": r["fixed_schedule"], "Predictive": r["predictive"]}))
    st.write(f"Unplanned failures: fixed {r['failures_fixed']}, predictive {r['failures_predictive']} of {r['n_engines']}. "
             f"Predictive vs fixed: {r['saving_vs_fixed_pct']:.1f}%.")
    # Show how the answer moves if the breakdown cost is wrong.
    st.subheader("Sensitivity to failure cost")
    st.dataframe(sensitivity(preds["pred_rul"], preds["true_rul"], preds["cycle"], a).round(1))

    # A second calculator that uses hours of downtime and a false-alarm fee.
    st.subheader("Plant-input downtime and false-alarm scenario")
    st.caption("Separate from the existing simulator. Enter plant-validated costs. A false alarm means preventive service would occur more than the chosen cycle horizon before failure.")
    inputs = st.columns(4)
    downtime_cost = inputs[0].number_input("Downtime cost per hour ($)", min_value=0.0, value=0.0, step=100.0)
    downtime_hours = inputs[1].number_input("Downtime hours per failure", min_value=0.0, value=0.0, step=1.0)
    planned_cost = inputs[2].number_input("Planned maintenance cost ($)", min_value=0.0, value=0.0, step=500.0)
    false_alarm_cost = inputs[3].number_input("False-alarm cost ($)", min_value=0.0, value=0.0, step=100.0)
    horizon = st.slider("False-alarm horizon (cycles)", min_value=0, max_value=250, value=60)
    # Only calculate once the user has entered a real downtime cost and a real downtime length.
    if downtime_cost > 0 and downtime_hours > 0:
        scenario = BusinessCaseAssumptions(
            downtime_cost_per_hour=downtime_cost,
            downtime_hours_per_failure=downtime_hours,
            planned_maintenance_cost=planned_cost,
            false_alarm_cost=false_alarm_cost,
            false_alarm_horizon_cycles=horizon,
            safety_margin=a.safety_margin,
            fixed_age=a.fixed_age,
        )
        case = compare_business_case(preds["pred_rul"], preds["true_rul"], preds["cycle"], scenario)
        # Show the three plan costs, then the failure and false-alarm counts.
        st.dataframe(pd.DataFrame([
            {"Policy": "Run to failure", "Estimated cost ($)": case["run_to_failure"]},
            {"Policy": "Fixed schedule", "Estimated cost ($)": case["fixed_schedule"]},
            {"Policy": "Predictive", "Estimated cost ($)": case["predictive"]},
        ]), width="stretch", hide_index=True)
        st.write(f"Failures, fixed vs predictive: {case['failures_fixed']} vs {case['failures_predictive']}; false alarms: {case['false_alarms_fixed']} vs {case['false_alarms_predictive']}; predictive vs fixed: {case['saving_vs_fixed_pct']:.1f}%.")
    else:
        st.info("Enter positive downtime cost per hour and downtime hours per failure to calculate this scenario.")

# Fifth page: show the saved maintenance note.
with t5:
    st.markdown((R / "maintenance_memo.md").read_text())
