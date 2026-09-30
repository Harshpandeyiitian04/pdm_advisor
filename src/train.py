"""Train and compare RUL models. Run:  python -m src.train [--subset FD001] [--synthetic]"""
import argparse, json
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from .decision import CostAssumptions, compare_policies, sensitivity, band
from .explain import top_importances, write_memo
from .metrics import rmse, nasa_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def candidate_models():
    models = {
        "Ridge (baseline)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                                         colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    except ImportError:
        pass
    return models


def main(subset="FD001", synthetic=False):
    if not synthetic:
        return _train(subset, False, D.RAW, ROOT)
    with TemporaryDirectory(prefix="pdm-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        raw_dir = temp_root / "raw"
        output_root = temp_root / "outputs"
        D.make_synthetic(out_dir=raw_dir)
        return _train(subset, True, raw_dir, output_root)


def _train(subset, synthetic, raw_dir, output_root):
    train_raw, test_raw, test_rul = D.load_cmapss(subset, raw_dir=raw_dir)
    train_raw = D.add_rul(train_raw)
    sensors = D.select_sensors(train_raw)

    tr_feats = D.build_features(train_raw, sensors)
    feats = D.feature_columns(tr_feats)  # computed BEFORE adding the label, so the target never becomes a feature
    tr = tr_feats.merge(train_raw[["unit", "cycle", "rul"]], on=["unit", "cycle"])

    # Split BY ENGINE so no engine appears in both train and validation (avoids leakage).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(tr, groups=tr["unit"]))
    tr_part, va_part = tr.iloc[tr_idx], tr.iloc[va_idx]

    te = D.last_cycle(D.build_features(test_raw, sensors))
    y_te = test_rul.loc[te["unit"]].clip(upper=D.RUL_CAP).values  # same cap as training, standard practice

    results, fitted = [], {}
    for name, model in candidate_models().items():
        model.fit(tr_part[feats], tr_part["rul"])
        val_pred = np.clip(model.predict(va_part[feats]), 0, D.RUL_CAP)
        te_pred = np.clip(model.predict(te[feats]), 0, D.RUL_CAP)
        results.append({"model": name,
                        "val_rmse": rmse(va_part["rul"], val_pred),
                        "test_rmse": rmse(y_te, te_pred),
                        "test_nasa_score": nasa_score(y_te, te_pred)})
        fitted[name] = te_pred
        print(f"{name:24s} val RMSE {results[-1]['val_rmse']:.2f} | test RMSE {results[-1]['test_rmse']:.2f} | NASA score {results[-1]['test_nasa_score']:.0f}")

    # Choose the model on VALIDATION error (not on the test set), then refit on all training engines.
    best_name = min(results, key=lambda r: r["val_rmse"])["model"]
    best = candidate_models()[best_name]
    best.fit(tr[feats], tr["rul"])
    te_pred = np.clip(best.predict(te[feats]), 0, D.RUL_CAP)

    model_dir = output_root / "models"
    report_dir = output_root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "sensors": sensors, "features": feats, "cap": D.RUL_CAP, "window": D.WINDOW, "name": best_name},
                model_dir / "best_model.joblib")

    preds = pd.DataFrame({"unit": te["unit"], "cycle": te["cycle"], "pred_rul": te_pred.round(1),
                          "true_rul": y_te, "band": [band(p) for p in te_pred]}).sort_values("pred_rul")
    preds.to_csv(report_dir / "test_predictions.csv", index=False)

    lifetimes = train_raw.groupby("unit")["cycle"].max()
    a = CostAssumptions(fixed_age=int(0.9 * np.percentile(lifetimes, 10)))  # cautious age-based schedule
    cost = compare_policies(te_pred, y_te, te["cycle"].values, a)
    sens = sensitivity(te_pred, y_te, te["cycle"].values, a)
    imp = top_importances(best, tr_part.sample(min(2000, len(tr_part)), random_state=SEED), feats)
    imp.to_csv(report_dir / "feature_importance.csv", index=False)
    sens.to_csv(report_dir / "sensitivity.csv", index=False)
    summary = {"subset": subset, "synthetic_data": synthetic, "best_model": best_name, "models": results,
               "final_test_rmse": rmse(y_te, te_pred), "final_test_nasa_score": nasa_score(y_te, te_pred),
               "cost_assumptions": a.__dict__, "cost_comparison": cost}
    (report_dir / "results.json").write_text(json.dumps(summary, indent=2))
    write_memo(preds, cost, imp, report_dir / "maintenance_memo.md", synthetic)
    print(f"\nBest model: {best_name} | final test RMSE {summary['final_test_rmse']:.2f}")
    print(f"Predictive saves {cost['saving_vs_fixed_pct']:.1f}% vs fixed schedule (simulation, assumptions in results.json)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--synthetic", action="store_true", help="use generated fake data (smoke test only)")
    args = ap.parse_args()
    main(args.subset, args.synthetic)
