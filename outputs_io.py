"""Saving and loading the two output tables in outputs/.

    outputs/results.csv      long:  algorithm, params, p, k, mse, mae, n
    outputs/predictions.csv  wide:  time, period, obs, AROWR_k1_p5, ..., persistence_k1, ...

Re-running an algorithm replaces only that algorithm's rows (results) or
columns (predictions); everything else in the files is kept.

Usage (notebook):
    from outputs_io import load_results, load_predictions
    res  = load_results()
    pred = load_predictions()
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "outputs"
RESULTS_PATH = OUT_DIR / "results.csv"
PREDICTIONS_PATH = OUT_DIR / "predictions.csv"
BASE_COLS = ["time", "period", "obs"]


def column_name(algorithm, k, p=None):
    """'AROWR_k1_p5' for algorithms with p, 'persistence_k1' for baselines."""
    return f"{algorithm}_k{k}" + ("" if p is None else f"_p{p}")


# ---------------------------------------------------------------- results (long)
def save_results(new):
    """Add rows to results.csv, replacing any existing rows of the same algorithm(s)."""
    OUT_DIR.mkdir(exist_ok=True)
    new = pd.DataFrame(new).astype({"p": float})          # p is NaN for baselines
    if RESULTS_PATH.exists():
        old = load_results().astype({"p": float})
        old = old[~old["algorithm"].isin(new["algorithm"].unique())]
        if len(old):
            new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["algorithm", "p", "k"], na_position="first")
    new.to_csv(RESULTS_PATH, index=False)
    return new


def load_results():
    return pd.read_csv(RESULTS_PATH, keep_default_na=False, na_values={"p": [""]})


# ---------------------------------------------------------------- predictions (wide)
def save_predictions(algorithm, s, n_tune, preds_by_p):
    """Write one algorithm's predictions into predictions.csv.

    s          : the observed series (UTC index) - gives time and obs columns
    n_tune     : number of tuning hours (rows before it are marked 'tune')
    preds_by_p : {p: {k: array}} for algorithms, or {None: {k: array}} for baselines
    Existing columns of this algorithm are replaced; other algorithms' columns are kept.
    """
    OUT_DIR.mkdir(exist_ok=True)
    period = np.where(np.arange(len(s)) < n_tune, "tune", "test")
    base = pd.DataFrame({"time": s.index, "period": period, "obs": s.to_numpy()})

    new_cols = {column_name(algorithm, k, p): preds[k]
                for p, preds in preds_by_p.items() for k in preds}
    new = pd.DataFrame(new_cols)

    if PREDICTIONS_PATH.exists():
        old = load_predictions()
        if len(old) != len(base) or not (old["time"].to_numpy() == base["time"].to_numpy()).all():
            raise ValueError("predictions.csv has different time stamps - delete it and re-run.")
        keep = [c for c in old.columns
                if c not in BASE_COLS and not c.startswith(algorithm + "_")]
        table = pd.concat([base, old[keep], new], axis=1)
    else:
        table = pd.concat([base, new], axis=1)
    table.to_csv(PREDICTIONS_PATH, index=False)
    return table


def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df
