"""Online forecasting of Bristol hourly temperature - main script.

Pipeline, for every model and every p:
    inputs X -> tune parameters on the first 20% (generic grid search)
    -> fresh model from 1 Jan 2020 through run_online -> score the last 80% -> save
and for every baseline:
    predictions for each k -> score the same test hours -> save

Choose what to run in the settings below, then:  python online.py
"""
import itertools

import numpy as np
import pandas as pd

from bristol import load_bristol
from splines import ncs_basis
from outputs_io import save_results, save_predictions
from algorithms.baselines import persistence, yesterday
from algorithms.arowr import AROWR, GRID as AROWR_GRID
from algorithms.arcor import ARCOR, GRID as ARCOR_GRID
from algorithms.rls import RLS, CRRLS, GRID as RLS_GRID, CRRLS_GRID
from algorithms.aar import AAR, GRID as AAR_GRID
from algorithms.laser import LASER, GRID as LASER_GRID

# ---------------------------------------------------------------- available methods
MODELS = {                       # name -> (class with update/predict, parameter grid)
    "AROWR": (AROWR, AROWR_GRID),
    "ARCOR": (ARCOR, ARCOR_GRID),
    "RLS":   (RLS, RLS_GRID),
    "CR-RLS": (CRRLS, CRRLS_GRID),
    "AAR":   (AAR, AAR_GRID),
    "LASER": (LASER, LASER_GRID),
}
BASELINES = {                    # name -> function(y, k)
    "persistence": persistence,
    "yesterday":   yesterday,
}

# ---------------------------------------------------------------- settings
RUN_MODELS = ["AROWR", "ARCOR", "RLS", "CR-RLS", "AAR", "LASER"]
RUN_BASELINES = ["persistence", "yesterday"]
P_GRID = range(3, 7)             # numbers of basis functions / knots: p = 3..6
KS = tuple(range(1, 10))         # horizons k = 1..9
TUNE_FRAC = 0.2                  # first 20% of hours used for tuning


# ---------------------------------------------------------------- inputs
def features(s, p):
    """T x p input matrix X: row t = x_t = basis functions at the UTC hour of time t."""
    B = ncs_basis(np.arange(24), p)          # 24 x p: one row per hour of the day
    return B[s.index.hour]                   # pick each observation's hour row


# ---------------------------------------------------------------- online loop
def run_online(model, X, y, ks):
    """Run one model through the series, online.

    Inputs:
        model : object with model.update(x, y) and model.predict(x)
        X     : T x p array, X[t] = x_t
        y     : length-T array of observed temperatures
        ks    : tuple of horizons, e.g. (1, ..., 9)

    At each time t:
        1. learn from y_t           (model.update with x_t and y_t)
        2. for every k in ks, predict y_{t+k} with x_{t+k} and the updated weights,
           as long as t + k < T

    Returns:
        preds : dict {k: length-T array}, where preds[k][t + k] is the
                prediction of y_{t+k} made at time t, and NaN where no
                prediction exists (the first k entries).
    """
    T = len(y)
    preds = {k: np.full(T,np.nan) for k in ks}  # initialise predictions with NaN

    for t in range(T):
        # 1. learn from y_t (skip if missing)
        if not np.isnan(y[t]):
            model.update(X[t], y[t])

        # 2. predict y_{t+k} for every k in ks, if t + k < T
        for k in ks:
            if t + k < T:
                preds[k][t + k] = model.predict(X[t + k])
                
    return preds


# ---------------------------------------------------------------- errors
def errors(y, y_hat, mask):
    """MSE and MAE of y_hat over the hours in `mask` that have a prediction."""
    ok = mask & ~np.isnan(y) & ~np.isnan(y_hat)
    e = y[ok] - y_hat[ok]
    return {"mse": np.mean(e ** 2), "mae": np.mean(np.abs(e)), "n": int(ok.sum())}


# ---------------------------------------------------------------- tuning
def tune(model_cls, grid, X, y, tune_mask, ks=KS, metric="mse"):
    """Generic grid search on the tuning period, separately for each horizon k.

    Inputs:
        model_cls : the class itself, e.g. AROWR (not an object) - build a FRESH
                    model for every combination with model_cls(p, **params),
                    where p = X.shape[1]
        grid      : dict param name -> list of values, e.g. {"r": [1e-6, ..., 10]}
                    (try every combination: itertools.product(*grid.values()))
        X, y      : full input matrix and observations
        tune_mask : boolean array, True for tuning hours (the first 20%)
        ks        : horizons to tune for
        metric    : "mse" or "mae" (key of errors())

    For each combination: run a fresh model on X[tune_mask], y[tune_mask] with
    run_online (one run gives every k) and compute the metric for every k.
    For each k, pick the combination with the smallest metric.

    Returns:
        best_params : dict {k: params}, e.g. {1: {"r": 1e-05}, 2: {"r": 0.0001}, ...}
        table       : DataFrame, one row per (k, combination) with its metric
    """
    Xt, yt = X[tune_mask], y[tune_mask]
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        try:
            model = model_cls(X.shape[1], **params)
        except ValueError:                       # invalid combination (e.g. LASER needs b < c)
            continue
        preds = run_online(model, Xt, yt, ks)

        for k in ks:
            value = errors(yt, preds[k], np.ones(len(yt), bool))[metric]
            rows.append({"k": k, **params, metric: value})

    table = pd.DataFrame(rows)

    # Find the best parameter combination independently for each k
    best_params = {}
    for k in ks:
        best_row = table[table["k"] == k].sort_values(metric).iloc[0]
        best_params[k] = {param: best_row[param] for param in grid.keys()}

    table = table.sort_values(["k", metric])
    return best_params, table


# ---------------------------------------------------------------- main
def score(name, p, best_params, preds, y, test_mask):
    """One results row per k, labelled with that k's tuned params ({} for baselines)."""
    rows = []
    for k in preds:
        params = best_params.get(k, {})
        label = ", ".join(f"{a}={b:g}" for a, b in params.items())
        rows.append({"algorithm": name, "params": label, "p": p, "k": k,
                     **errors(y, preds[k], test_mask)})
    return rows


def reset_summary(reset_times, n_tune):
    """Reset counts in the tuning / test periods and the mean gap (hours) between test resets.
    All NaN for models without covariance resets (reset_times is None)."""
    if reset_times is None:
        return {"n_resets_tune": np.nan, "n_resets_test": np.nan, "mean_gap_test": np.nan}
    t = np.asarray(reset_times)
    test = t[t >= n_tune]
    return {"n_resets_tune": int((t < n_tune).sum()),
            "n_resets_test": len(test),
            "mean_gap_test": float(np.diff(test).mean()) if len(test) >= 2 else np.nan}


if __name__ == "__main__":
    s = load_bristol()
    y = s.to_numpy()
    n_tune = int(TUNE_FRAC * len(y))
    tune_mask = np.arange(len(y)) < n_tune
    test_mask = ~tune_mask

    print(f"Tuning period: {s.index[0]} to {s.index[n_tune - 1]} ({n_tune} hours)")
    print(f"Test period:   {s.index[n_tune]} to {s.index[-1]} ({len(y) - n_tune} hours)\n")

    for name in RUN_MODELS:
        model_cls, grid = MODELS[name]
        rows, preds_by_p = [], {}
        for p in P_GRID:
            X = features(s, p)
            best_params, table = tune(model_cls, grid, X, y, tune_mask)   # tuned separately for each p and k
            print(f"{name}, p = {p}: tuning ({len(table)} rows, best per k below)")
            print(table.groupby("k").head(1).to_string(index=False), "\n")

            preds, resets = {}, {}
            for k in KS:
                model = model_cls(p, **best_params[k])          # fresh model with k's settings, from 1 Jan 2020
                preds[k] = run_online(model, X, y, (k,))[k]     # keep the k-hours-ahead forecasts
                resets[k] = getattr(model, "reset_times", None) # ARCOR only; None for other models

            preds_by_p[p] = preds
            new_rows = score(name, p, best_params, preds, y, test_mask)
            for row in new_rows:
                row.update(reset_summary(resets[row["k"]], n_tune))
            rows += new_rows
        save_results(rows)
        save_predictions(name, s, n_tune, preds_by_p)

    for name in RUN_BASELINES:
        preds = {k: BASELINES[name](y, k) for k in KS}
        save_results(score(name, None, {}, preds, y, test_mask))
        save_predictions(name, s, n_tune, {None: preds})

    from outputs_io import load_results
    res = load_results()
    print("Test-period MAE (°C) by k")
    res["method"] = res["algorithm"] + res["p"].map(lambda p: "" if pd.isna(p) else f" p={p:g}")
    print(res.pivot_table(index="k", columns="method", values="mae").round(3).to_string())
    print("\nSaved outputs/results.csv and outputs/predictions.csv")
