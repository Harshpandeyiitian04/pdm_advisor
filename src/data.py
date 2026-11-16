# This file reads the engine history files, works out how much life each engine had left, and turns the raw sensor numbers into extra clues a model can learn from.

# These lines bring in tools for file paths, number math, and tables of data.
from pathlib import Path
import numpy as np
import pandas as pd

# This is the folder where the raw engine files live.
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
# These are the column names: which engine, which cycle, three running settings, and 21 sensors.
COLS = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
# We never say an engine has more than 125 cycles of life left. Early on, we just call it healthy.
RUL_CAP = 125
# When we look at recent history, we look back over the last 10 cycles.
WINDOW = 10


# This opens the training file, the test file, and the file that says how much life the test engines really had left.
def load_cmapss(subset="FD001", raw_dir=RAW):
    # Turn the folder path into a proper path.
    raw_dir = Path(raw_dir)
    # Read the training engines. Each row is one moment in time for one engine.
    train = pd.read_csv(raw_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    # Read the test engines the same way.
    test = pd.read_csv(raw_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    # Read how many cycles each test engine still had when its record stopped.
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul"])["rul"]
    # Match those life-left numbers to the engine numbers, in engine order.
    rul.index = np.sort(test["unit"].unique())
    # Hand back all three pieces.
    return train, test, rul


# This adds a "life left" number to every training row.
def add_rul(train, cap=RUL_CAP):
    # Work on a copy so the original table stays untouched.
    out = train.copy()
    # Find the last cycle each engine ever reached.
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    # Life left is the last cycle minus this cycle. Cap it so early life counts as healthy.
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


# This keeps only the settings and sensors that actually change. A sensor that never moves tells us nothing.
def select_sensors(train, tol=1e-6):
    # Look at the running settings and the sensor columns.
    cand = [c for c in COLS if c.startswith(("op", "s"))]
    # Drop any column that barely moves.
    return [c for c in cand if train[c].std() > tol]


# This adds extra clues from recent history: the recent average, how jumpy the readings are, and how much they changed.
# It only looks at the current moment and earlier moments, never the future.
def build_features(df, sensors, window=WINDOW):
    # Put each engine's rows in time order.
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    # Group the sensor readings by engine so one engine's history never mixes with another's.
    g = df.groupby("unit")[sensors]
    # The average of the recent window for each sensor.
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).add_suffix("_mean")
    # How much the recent readings jump around. Fill the early gaps with zero.
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0).add_suffix("_std")
    # How much each sensor changed since the start of the window.
    delta = g.transform(lambda s: s - s.shift(window - 1).fillna(s.iloc[0])).add_suffix("_delta")
    # Stick the engine id, the cycle, the raw sensors, and the new clues into one table.
    return pd.concat([df[["unit", "cycle"]], df[sensors], mean, std, delta], axis=1)


# This lists the columns the model is allowed to look at. The engine id and the cycle number are left out.
def feature_columns(feats):
    return [c for c in feats.columns if c not in ("unit", "cycle")]


# This keeps only the newest row for each engine. That is the moment we want a prediction for.
def last_cycle(feats):
    return feats.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)


# This writes a tiny fake dataset in the same shape as the real files.
# It is only for a quick check that the program runs. These numbers are not real and must not be treated as results.
def make_synthetic(out_dir=RAW, n_train=40, n_test=20, seed=0):
    # A repeatable random-number maker so the fake data looks the same each time.
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    # Create the output folder if it is missing.
    out_dir.mkdir(parents=True, exist_ok=True)
    # These sensor slots stay flat, like the real data where some sensors never move.
    constant = {0, 4, 9, 15, 17, 18}

    # This builds one fake engine: a list of cycles whose sensors slowly wear out.
    def engine(uid, life, cut=None):
        # Use the full life, or stop early if this is a test engine.
        n = life if cut is None else cut
        rows = []
        for i in range(1, n + 1):
            # Wear grows faster near the end of life.
            wear = (i / life) ** 2
            # Build 21 sensor numbers. Some stay flat. Others drift as the engine wears, plus a little noise.
            sens = [500.0 + k if k in constant else 500.0 + k + (8 if k % 2 else -8) * wear + rng.normal(0, 0.5)
                    for k in range(21)]
            # One row: engine id, cycle, three settings, then the sensors.
            rows.append([uid, i, 0.0, 0.0, 100.0, *sens])
        return rows

    # Build many training engines that run until they fail.
    train_rows = [r for u in range(1, n_train + 1) for r in engine(u, int(rng.integers(130, 260)))]
    test_rows, ruls = [], []
    # Build test engines that stop before failure, and remember how much life was left.
    for u in range(1, n_test + 1):
        life = int(rng.integers(130, 260))
        cut = int(rng.integers(50, life - 5))
        test_rows += engine(u, life, cut)
        ruls.append(life - cut)
    # Save the three fake files.
    pd.DataFrame(train_rows).to_csv(out_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(ruls).to_csv(out_dir / "RUL_FD001.txt", header=False, index=False)
