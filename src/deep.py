# This file tries an optional model that reads the last 30 sensor snapshots in a row and guesses how much life is left.
# It is a side experiment. It does not change the main training path.
# The short description below is kept because the run block uses it as the help text.
"""This file tries an optional model that reads the last 30 sensor snapshots in a row. It does not change the main training path."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import data as D
from .metrics import nasa_score, rmse

# The project folder, how many past cycles each example looks at, and a fixed random seed.
ROOT = Path(__file__).resolve().parents[1]
WINDOW = 30
SEED = 42


# This cuts one engine's history into overlapping windows. Each window ends on the cycle whose life-left number we want to guess.
# Early cycles are padded by repeating the first reading, so every window is the same length, and no future reading is used.
def make_windows(values, targets, window=WINDOW):
    # Turn the sensor rows and the life-left answers into number tables.
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    # Stop if the shapes do not match: one answer per row, and rows that are a table.
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("values must be 2D and have one target per row")
    # Stop if there is no data, or the window length is not a positive number.
    if not len(values) or window < 1:
        raise ValueError("values must be non-empty and window must be positive")

    # Repeat the first row so the early windows can still be full length.
    padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values], axis=0)
    # Slice one window per cycle. The last row of each window is the cycle we are guessing for.
    windows = np.stack([padded[index:index + window] for index in range(len(values))])
    return windows, targets


# This finds the row number of the newest cycle for each engine.
def last_cycle_indices(frame):
    ordered = frame.sort_values(["unit", "cycle"])
    return ordered.groupby("unit", sort=True).tail(1).index.to_numpy()


# This builds the windows for a chosen set of engines, after scaling the sensors with numbers learned from the training engines.
def _split_windows(frame, units, sensors, mean, scale):
    # Keep only the requested engines, in time order.
    selected = frame[frame["unit"].isin(units)].sort_values(["unit", "cycle"])
    all_windows, all_targets = [], []
    for _, engine in selected.groupby("unit", sort=True):
        # Shift and scale this engine's sensors so big and small sensors sit on a similar range.
        values = ((engine[sensors].to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)
        windows, targets = make_windows(values, engine["rul"].to_numpy(dtype=np.float32))
        all_windows.append(windows)
        all_targets.append(targets)
    # Stack every engine's windows into two big arrays: the inputs and the answers.
    return np.concatenate(all_windows), np.concatenate(all_targets)


# This runs the whole side experiment: prepare the windows, train the sequence model, score it, and save the result.
def run_experiment(subset="FD001", epochs=20, batch_size=256):
    try:
        # Load the deep-learning tools. If they are not installed, stop with a clear message.
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install the optional PyTorch dependency with: python -m pip install torch") from exc

    # Make the random steps repeatable, and keep this run from using too many processor threads.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    # Load the engines, add life-left numbers, and drop sensors that never move.
    train_raw, test_raw, test_rul = D.load_cmapss(subset)
    train = D.add_rul(train_raw)
    sensors = D.select_sensors(train)

    # Split by engine so the same engine is not used both to learn and to check.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train, groups=train["unit"]))
    train_part = train.iloc[train_idx]
    val_part = train.iloc[val_idx]
    train_units = train_part["unit"].unique()
    val_units = val_part["unit"].unique()

    # Learn the typical value and the spread of each sensor from the learning engines only.
    mean = train_part[sensors].mean().to_numpy(dtype=np.float32)
    scale = train_part[sensors].std().replace(0, 1).fillna(1).to_numpy(dtype=np.float32)
    # Build the windows for learning and for checking.
    x_train, y_train = _split_windows(train, train_units, sensors, mean, scale)
    x_val, y_val = _split_windows(train, val_units, sensors, mean, scale)

    # For the test engines, rebuild life left at every cycle from the final life-left number, then cap it.
    test = test_raw.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)
    last_cycles = test.groupby("unit")["cycle"].transform("max")
    test["rul"] = (test_rul.loc[test["unit"]].to_numpy() + last_cycles.to_numpy() - test["cycle"]).clip(upper=D.RUL_CAP)
    test_units = test["unit"].unique()
    test_last_indices = last_cycle_indices(test)
    x_test, y_test = _split_windows(test, test_units, sensors, mean, scale)

    # This wraps a set of windows into batches the model can train on. Shuffle only the learning batches.
    def loader(values, targets, shuffle):
        dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(targets))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val, False)
    test_loader = loader(x_test, y_test, False)

    # This is the sequence model: it reads the window in order, then turns the last step into one life-left guess.
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            # The memory layer that reads the sensor window from start to end.
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
            # A small finishing layer that turns that memory into a single number.
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, inputs):
            # Read the whole window, then guess from the last moment only.
            sequence, _ = self.lstm(inputs)
            return self.head(sequence[:, -1]).squeeze(-1)

    # Create the model, the update rule, and the "how wrong is this guess" measure.
    model = LSTMRegressor(len(sensors))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    # Remember the best version of the model, its checking error, and how long it has gone without improving.
    best_state, best_val_loss, stale_epochs = None, float("inf"), 0

    for epoch in range(epochs):
        # Teaching pass: update the model on every learning batch.
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # Checking pass: measure the error without changing the model.
        model.eval()
        val_loss_total, val_count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_loss_total += loss_fn(model(inputs), targets).item() * len(targets)
                val_count += len(targets)
        val_loss = val_loss_total / val_count
        # Keep this version if it is the best so far. Otherwise count another round with no improvement.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            # Stop early if four rounds in a row got no better.
            if stale_epochs >= 4:
                break

    # Put the best saved version back into the model.
    model.load_state_dict(best_state)

    # This runs the model on a set of batches and keeps the guesses inside the allowed life-left range.
    def predict(loader_to_run):
        model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, _ in loader_to_run:
                outputs.append(model(inputs).numpy())
        return np.clip(np.concatenate(outputs), 0, D.RUL_CAP)

    # Score the checking engines, and score the test engines only at their last recorded cycle.
    val_predictions = predict(val_loader)
    test_predictions = predict(test_loader)[test_last_indices]
    test_targets = y_test[test_last_indices]
    # Pack the scores and the engine counts into one result.
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
    # Save that result next to the other reports and print it.
    output_path = ROOT / "reports" / f"deep_results_{subset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved separate experiment results to {output_path}")
    return result


# When this file is run directly, read the options and start the experiment.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    arguments = parser.parse_args()
    run_experiment(arguments.subset, arguments.epochs, arguments.batch_size)
