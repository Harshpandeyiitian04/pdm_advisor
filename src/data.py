"""Data loading, RUL labelling and feature engineering for NASA C-MAPSS."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COLS = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125
WINDOW = 10


def load_cmapss(subset="FD001", raw_dir=RAW):
    """Return (train_df, test_df, test_rul_series). Files must be in data/raw."""
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(raw_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul"])["rul"]
    rul.index = np.sort(test["unit"].unique())  # RUL file rows follow engine order
    return train, test, rul


def add_rul(train, cap=RUL_CAP):
    """RUL per row = engines last cycle - current cycle, capped (early life is treated as healthy)."""
    out = train.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


def select_sensors(train, tol=1e-6):
    """Keep operating settings and sensors that actually vary in the training data."""
    cand = [c for c in COLS if c.startswith(("op", "s"))]
    return [c for c in cand if train[c].std() > tol]


def build_features(df, sensors, window=WINDOW):
    """Raw values plus rolling mean, rolling std and change-over-window, per engine.
    Uses only current and past cycles, so no future information leaks in."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")[sensors]
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).add_suffix("_mean")
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0).add_suffix("_std")
    delta = g.transform(lambda s: s - s.shift(window - 1).fillna(s.iloc[0])).add_suffix("_delta")
    return pd.concat([df[["unit", "cycle"]], df[sensors], mean, std, delta], axis=1)


def feature_columns(feats):
    return [c for c in feats.columns if c not in ("unit", "cycle")]


def last_cycle(feats):
    """One row per engine: its most recent cycle (used for test-time prediction)."""
    return feats.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)


def make_synthetic(out_dir=RAW, n_train=40, n_test=20, seed=0):
    """Write a small SYNTHETIC dataset in C-MAPSS format, for smoke tests and CI only.
    Results on this data are meaningless and must never be reported."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    constant = {0, 4, 9, 15, 17, 18}

    def engine(uid, life, cut=None):
        n = life if cut is None else cut
        rows = []
        for i in range(1, n + 1):
            wear = (i / life) ** 2
            sens = [500.0 + k if k in constant else 500.0 + k + (8 if k % 2 else -8) * wear + rng.normal(0, 0.5)
                    for k in range(21)]
            rows.append([uid, i, 0.0, 0.0, 100.0, *sens])
        return rows

    train_rows = [r for u in range(1, n_train + 1) for r in engine(u, int(rng.integers(130, 260)))]
    test_rows, ruls = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(130, 260))
        cut = int(rng.integers(50, life - 5))
        test_rows += engine(u, life, cut)
        ruls.append(life - cut)
    pd.DataFrame(train_rows).to_csv(out_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(ruls).to_csv(out_dir / "RUL_FD001.txt", header=False, index=False)
