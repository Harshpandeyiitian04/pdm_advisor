"""Exploratory analysis for the training portion of a C-MAPSS subset."""
import argparse
from pathlib import Path

import pandas as pd

from . import data as D

ROOT = Path(__file__).resolve().parents[1]


def analyze_training_data(train):
    candidates = [c for c in D.COLS if c.startswith(("op", "s")) and c in train.columns]
    variable = [sensor for sensor in candidates if train[sensor].std() > 1e-6]
    constant = [sensor for sensor in candidates if sensor not in variable]

    trend_rows = []
    for sensor in variable:
        correlations = []
        for _, engine in train.groupby("unit"):
            if len(engine) < 2 or engine[sensor].nunique() < 2:
                continue
            correlation = engine["cycle"].rank().corr(engine[sensor].rank())
            if pd.notna(correlation):
                correlations.append(float(correlation))
        trend_rows.append({
            "sensor": sensor,
            "median_spearman_with_cycle": float(pd.Series(correlations).median()) if correlations else 0.0,
            "engines_with_valid_correlation": len(correlations),
            "engines_trending_up_pct": 100 * sum(value > 0 for value in correlations) / len(correlations) if correlations else 0.0,
        })

    trends = pd.DataFrame(trend_rows)
    if not trends.empty:
        trends = trends.sort_values("median_spearman_with_cycle", key=lambda values: values.abs(), ascending=False)
    lifetimes = train.groupby("unit")["cycle"].max()
    lifetime_summary = {
        "engine_count": int(lifetimes.size),
        "min_cycles": int(lifetimes.min()),
        "median_cycles": float(lifetimes.median()),
        "mean_cycles": float(lifetimes.mean()),
        "max_cycles": int(lifetimes.max()),
        "std_cycles": float(lifetimes.std(ddof=0)),
    }
    return {
        "constant_sensors": constant,
        "variable_sensors": variable,
        "lifetimes": lifetime_summary,
        "sensor_trends": trends.reset_index(drop=True),
    }


def write_report(analysis, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    trends = analysis["sensor_trends"]
    trends.to_csv(output_dir / "sensor_trends.csv", index=False)
    life = analysis["lifetimes"]
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
    for row in trends.head(10).itertuples(index=False):
        rows.append(f"| {row.sensor} | {row.median_spearman_with_cycle:.3f} | {row.engines_with_valid_correlation} | {row.engines_trending_up_pct:.1f}% |")
    (output_dir / "eda_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(subset="FD001"):
    train, _, _ = D.load_cmapss(subset)
    analysis = analyze_training_data(train)
    output_dir = ROOT / "reports"
    write_report(analysis, output_dir)
    print(f"Wrote {output_dir / 'eda_report.md'} and {output_dir / 'sensor_trends.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    args = parser.parse_args()
    main(args.subset)