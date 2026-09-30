"""Optional CPU LSTM experiment over rolling 30-cycle sensor windows.

Run: python -m src.deep [--subset FD001] [--epochs 20]
Requires the optional PyTorch dependency. The baseline training path is unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import data as D
from .metrics import nasa_score, rmse

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 30
SEED = 42


def make_windows(values, targets, window=WINDOW):
    """Create left-padded, causal windows; each target belongs to its final cycle."""
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("values must be 2D and have one target per row")
    if not len(values) or window < 1:
        raise ValueError("values must be non-empty and window must be positive")

    padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values], axis=0)
    windows = np.stack([padded[index:index + window] for index in range(len(values))])
    return windows, targets


def last_cycle_indices(frame):
    ordered = frame.sort_values(["unit", "cycle"])
    return ordered.groupby("unit", sort=True).tail(1).index.to_numpy()


def _split_windows(frame, units, sensors, mean, scale):
    selected = frame[frame["unit"].isin(units)].sort_values(["unit", "cycle"])
    all_windows, all_targets = [], []
    for _, engine in selected.groupby("unit", sort=True):
        values = ((engine[sensors].to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)
        windows, targets = make_windows(values, engine["rul"].to_numpy(dtype=np.float32))
        all_windows.append(windows)
        all_targets.append(targets)
    return np.concatenate(all_windows), np.concatenate(all_targets)


def run_experiment(subset="FD001", epochs=20, batch_size=256):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install the optional PyTorch dependency with: python -m pip install torch") from exc

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_raw, test_raw, test_rul = D.load_cmapss(subset)
    train = D.add_rul(train_raw)
    sensors = D.select_sensors(train)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train, groups=train["unit"]))
    train_part = train.iloc[train_idx]
    val_part = train.iloc[val_idx]
    train_units = train_part["unit"].unique()
    val_units = val_part["unit"].unique()

    mean = train_part[sensors].mean().to_numpy(dtype=np.float32)
    scale = train_part[sensors].std().replace(0, 1).fillna(1).to_numpy(dtype=np.float32)
    x_train, y_train = _split_windows(train, train_units, sensors, mean, scale)
    x_val, y_val = _split_windows(train, val_units, sensors, mean, scale)

    test = test_raw.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)
    last_cycles = test.groupby("unit")["cycle"].transform("max")
    test["rul"] = (test_rul.loc[test["unit"]].to_numpy() + last_cycles.to_numpy() - test["cycle"]).clip(upper=D.RUL_CAP)
    test_units = test["unit"].unique()
    test_last_indices = last_cycle_indices(test)
    x_test, y_test = _split_windows(test, test_units, sensors, mean, scale)

    def loader(values, targets, shuffle):
        dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(targets))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val, False)
    test_loader = loader(x_test, y_test, False)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, inputs):
            sequence, _ = self.lstm(inputs)
            return self.head(sequence[:, -1]).squeeze(-1)

    model = LSTMRegressor(len(sensors))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_state, best_val_loss, stale_epochs = None, float("inf"), 0

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total, val_count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_loss_total += loss_fn(model(inputs), targets).item() * len(targets)
                val_count += len(targets)
        val_loss = val_loss_total / val_count
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

    model.load_state_dict(best_state)

    def predict(loader_to_run):
        model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, _ in loader_to_run:
                outputs.append(model(inputs).numpy())
        return np.clip(np.concatenate(outputs), 0, D.RUL_CAP)

    val_predictions = predict(val_loader)
    test_predictions = predict(test_loader)[test_last_indices]
    test_targets = y_test[test_last_indices]
    result = {
        "subset": subset,
        "model": "PyTorch LSTM",
        "window_cycles": WINDOW,
        "epochs_completed": epoch + 1,
        "best_validation_mse": best_val_loss,
        "val_rmse": rmse(y_val, val_predictions),
        "val_nasa_score": nasa_score(y_val, val_predictions),
        "test_rmse": rmse(test_targets, test_predictions),
        "test_nasa_score": nasa_score(test_targets, test_predictions),
        "train_engines": int(len(train_units)),
        "validation_engines": int(len(val_units)),
        "test_engines": int(len(test_units)),
        "synthetic_data": False,
    }
    output_path = ROOT / "reports" / f"deep_results_{subset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved separate experiment results to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    arguments = parser.parse_args()
    run_experiment(arguments.subset, arguments.epochs, arguments.batch_size)