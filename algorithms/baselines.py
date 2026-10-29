"""Simple benchmark forecasts (Task 5). Plain functions: no learning, no p.

Each returns an array aligned with y: out[t + k] = forecast of y_{t+k} made at time t,
NaN where no forecast exists.
"""
import numpy as np


def persistence(y, k):
    """Predict y_{t+k} by y_t (the latest reading)."""
    out = np.full(len(y), np.nan); out[k:] = y[:-k]
    return out


def yesterday(y, k):
    """Predict y_{t+k} by y_{t+k-24} (same hour the day before).
    Identical for every k; only a fair forecast for k <= 24 (y_{t+k-24} must already be observed)."""
    out = np.full(len(y), np.nan); out[24:] = y[:-24]
    return out
