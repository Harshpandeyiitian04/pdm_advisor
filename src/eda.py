# This file looks at the training engines before any model is built.
# It counts how long engines last and which sensors drift as an engine gets older.
# The first line below is kept as a short description because the run block reads it for the help text.
"""This file looks at the training engines and writes a simple report about how long they last and which sensors drift with age."""
import argparse
from pathlib import Path

import pandas as pd

from . import data as D

# This is the project folder, used later to find where reports should be saved.
ROOT = Path(__file__).resolve().parents[1]


# This studies the training table: which sensors never move, how long engines last, and which sensors rise or fall with age.
def analyze_training_data(train):
    # The columns that might be useful: running settings and sensors that are actually in the table.
    candidates = [c for c in D.COLS if c.startswith(("op", "s")) and c in train.columns]
    # Sensors that change.
    variable = [sensor for sensor in candidates if train[sensor].std() > 1e-6]
    # Sensors that stay flat.
    constant = [sensor for sensor in candidates if sensor not in variable]

    trend_rows = []
    for sensor in variable:
        correlations = []
        for _, engine in train.groupby("unit"):
            # Skip an engine if it has too little history, or this sensor never moves on that engine.
            if len(engine) < 2 or engine[sensor].nunique() < 2:
                continue
            # See whether this sensor tends to rise or fall as the cycle number rises, for this one engine.
            correlation = engine["cycle"].rank().corr(engine[sensor].rank())
            if pd.notna(correlation):
                correlations.append(float(correlation))
        # Summarize that sensor across engines: typical link with age, and how often it trends up.
        trend_rows.append({
            "sensor": sensor,
            "median_spearman_with_cycle": float(pd.Series(correlations).median()) if correlations else 0.0,
            "engines_with_valid_correlation": len(correlations),
            "engines_trending_up_pct": 100 * sum(value > 0 for value in correlations) / len(correlations) if correlations else 0.0,
        })

    trends = pd.DataFrame(trend_rows)
    # Put the strongest age links first, whether the sensor goes up or down.
    if not trends.empty:
        trends = trends.sort_values("median_spearman_with_cycle", key=lambda values: values.abs(), ascending=False)
    # How many cycles each engine lasted.
    lifetimes = train.groupby("unit")["cycle"].max()
    # A short summary of those lifetimes: how many engines, and the shortest, typical, average, and longest lives.
    lifetime_summary = {
        "engine_count": int(lifetimes.size),
        "min_cycles": int(lifetimes.min()),
        "median_cycles": float(lifetimes.median()),
        "mean_cycles": float(lifetimes.mean()),
        "max_cycles": int(lifetimes.max()),
        "std_cycles": float(lifetimes.std(ddof=0)),
    }
    # Hand back the flat sensors, the moving sensors, the lifetime summary, and the trend table.
    return {
        "constant_sensors": constant,
        "variable_sensors": variable,
        "lifetimes": lifetime_summary,
        "sensor_trends": trends.reset_index(drop=True),
    }


# This writes the study to a markdown report and a spreadsheet of sensor trends.
def write_report(analysis, output_dir):
    # Create the output folder if it is missing.
    output_dir.mkdir(parents=True, exist_ok=True)
    trends = analysis["sensor_trends"]
    # Save the full trend table.
    trends.to_csv(output_dir / "sensor_trends.csv", index=False)
    life = analysis["lifetimes"]
    # Build the report as a list of lines: a title, the lifetime numbers, the flat sensors, and the strongest trends.
    rows = [
        "# C-MAPSS training data exploratory analysis",
        "",
        "This report uses the training split only. Sensor trends are per-engine Spearman correlations with cycle, then summarized across engines; they are exploratory, not causal.",
        "",
        "## Engine lifetime",
        "",
        f"- Engines: {life['engine_count']}",
        f"- Cycles: minimum {life['min_cycles']}, median {life['median_cycles']:.1f}, mean {life['mean_cycles']:.1f}, maximum {life['max_cycles']}",
        f"- Population standard deviation: {life['std_cycles']:.1f} cycles",
        "",
        "## Constant sensors/settings removed by preprocessing",
        "",
        ", ".join(analysis["constant_sensors"]) if analysis["constant_sensors"] else "None",
        "",
        "## Strongest sensor trends with cycle",
        "",
        "| Sensor | Median within-engine Spearman correlation | Engines with valid correlation | Engines trending up |",
        "| --- | ---: | ---: | ---: |",
    ]
    # Add the ten strongest sensors as table rows.
    for row in trends.head(10).itertuples(index=False):
        rows.append(f"| {row.sensor} | {row.median_spearman_with_cycle:.3f} | {row.engines_with_valid_correlation} | {row.engines_trending_up_pct:.1f}% |")
    # Save the report.
    (output_dir / "eda_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


# This loads one data subset, studies the training engines, and writes the report into the reports folder.
def main(subset="FD001"):
    # Load the training engines. The other two return values are not needed here.
    train, _, _ = D.load_cmapss(subset)
    analysis = analyze_training_data(train)
    output_dir = ROOT / "reports"
    write_report(analysis, output_dir)
    print(f"Wrote {output_dir / 'eda_report.md'} and {output_dir / 'sensor_trends.csv'}")


# When this file is run directly, read an optional subset name and start the study.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    args = parser.parse_args()
    main(args.subset)
