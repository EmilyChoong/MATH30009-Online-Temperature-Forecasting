"""Loading the Bristol hourly temperature data (2020-2022), indexed by UTC.

Usage:
    from bristol import load_bristol
    s = load_bristol()          # pd.Series of temperature in °C, UTC index
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "bristol_data.csv"
N_HOURS = 26304          # 1 Jan 2020 00:00 to 31 Dec 2022 23:00 (2020 is a leap year)


def load_bristol():
    """Hourly temperature (°C) for Bristol, indexed by UTC time.

    The raw timestamps are UK clock time (GMT in winter, BST in summer):
      - March: 01:00 is skipped (clocks go forward)
      - October: 01:00 appears twice (clocks go back)
    We attach the Europe/London time zone and convert to UTC, so that the
    hour of day follows the sun. ambiguous="infer" uses the row order to tell
    the two October 01:00 readings apart (first = BST, second = GMT).
    """
    df = pd.read_csv(DATA_PATH, usecols=["datetime", "temp"])
    t = pd.DatetimeIndex(pd.to_datetime(df["datetime"]))
    t = t.tz_localize("Europe/London", ambiguous="infer").tz_convert("UTC")
    s = pd.Series(df["temp"].to_numpy(dtype=float), index=t, name="temp")

    # checks: one reading per hour, in order, no gaps or duplicates
    assert len(s) == N_HOURS, f"expected {N_HOURS} rows, got {len(s)}"
    assert s.index.is_unique and s.index.is_monotonic_increasing
    assert (np.diff(s.index.asi8) == 3600 * 10**9).all(), "series is not strictly hourly"
    return s


if __name__ == "__main__":
    s = load_bristol()
    print(s.index[0], "to", s.index[-1], f"({len(s)} hours, {s.isna().sum()} missing)")
