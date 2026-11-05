# MATH30009 – Online Temperature Forecasting

Online forecasting of Bristol hourly temperature (2020–2022). Each model is a regression on natural cubic spline basis functions of the hour of the day. The model updates its weights every hour and forecasts k = 1…9 hours ahead. Algorithms follow Moroshko et al. (2015).

## Repository layout

```
online.py          MAIN SCRIPT – registers algorithms, tunes, runs online, scores, saves
splines.py         natural cubic spline basis functions (ncs_basis)
bristol.py         loads bristol_data.csv, converts UK clock time -> UTC
outputs_io.py      saves / loads the two output tables in outputs/
algorithms/
    arowr.py       AROWR class + its parameter GRID
    arcor.py       ARCOR (AROWR + covariance reset + projection)
    rls.py         RLS (forgetting factor r) and CR-RLS (reset Sigma every T0 hours)
    aar.py         AAR (stationary; LASER with c = infinity)
    laser.py       LASER (AAR + forgetting via c)
    baselines.py   persistence and "same hour yesterday" benchmarks
outputs/
    results.csv      overall performance: one row per algorithm × p × k
    predictions.csv  every prediction at every time step (for plots / analysis)
01_check.ipynb     analysis notebook – tables and plots built from outputs/
```

## How a run works

1. **Data** – `bristol.load_bristol()` returns hourly temperature indexed by UTC, so the hour of day follows the sun.
2. **Basis functions** – `splines.ncs_basis(h, p)` builds the p natural cubic spline basis functions on hours 0–23, with p evenly spaced knots.
3. **Inputs** – during a run, `online.features(s, p)` computes the inputs: row t of X is x_t = (b_1(h_t), …, b_p(h_t)), the basis functions evaluated at the hour of observation t.
4. **Tuning** – `online.tune()` grid-searches each algorithm's parameters on the **first 20%** of hours. It picks the combination with the lowest MSE, separately for each k and each p.
5. **Online run** – `online.run_online()` starts a fresh model with the tuned parameters on 1 Jan 2020. At each hour t it
   - updates on y_t (`model.update(x_t, y_t)`)
   - predicts y_{t+k} for every k (`model.predict(x_{t+k})`)
6. **Scoring** – MSE and MAE are computed on the **last 80%** of hours (the test period).
7. **Saving** – `outputs_io` writes the two CSVs (below).

Settings (which models, p values, k values, tuning fraction) are at the top of `online.py`:

```python
RUN_MODELS    = ["AROWR", "ARCOR", "RLS", "CR-RLS", "AAR", "LASER"]
RUN_BASELINES = ["persistence", "yesterday"]
P_GRID = range(3, 7)        # p = 3..6
KS     = tuple(range(1, 10))  # k = 1..9
TUNE_FRAC = 0.2
```

Run with:

```bash
python online.py
```

## Adding a new algorithm

1. Create `algorithms/<name>.py` containing:
   - a class with `__init__(self, p, **params)`, `predict(self, x)` and `update(self, x, y)` (see `arowr.py` for the template)
   - a `GRID` dict of parameter values to tune, e.g. `GRID = {"r": [1e-4, 1e-3, 1e-2]}`
2. In `online.py`, import it at the top and register it in `MODELS`:
   ```python
   from algorithms.rls import RLS, GRID as RLS_GRID
   MODELS = {
       "AROWR": (AROWR, AROWR_GRID),
       "RLS":   (RLS, RLS_GRID),
   }
   ```
3. Add its name to `RUN_MODELS` and run `python online.py`.

Tuning, the online loop, scoring and saving are shared, so nothing else needs changing. Its results are added to the two CSVs automatically.

## Output files

Re-running an algorithm **replaces only that algorithm's rows/columns**, so results from other algorithms are kept.

**`outputs/results.csv`** – overall performance (one row per algorithm, p, k):

| algorithm | params | p | k | mse | mae | n |
|---|---|---|---|---|---|---|
| AROWR | r=0.1 | 3 | 1 | 30.26 | 4.42 | 21044 |

`p` is empty for baselines. `n` is the number of test hours scored.

**`outputs/predictions.csv`** – prediction at each time step (for plots and in-depth analysis):

> Not stored on GitHub (it is over 100 MB). Run `python online.py` to create it locally.

| time | period | obs | AROWR_k1_p3 | … | persistence_k1 | … |
|---|---|---|---|---|---|---|

- `period` is `tune` or `test`.
- Column names are `<algorithm>_k<k>_p<p>` (algorithms) or `<algorithm>_k<k>` (baselines).
- The value in row t is the forecast of y_t made k hours earlier.

Load either table in a notebook with:

```python
from outputs_io import load_results, load_predictions
res, pred = load_results(), load_predictions()
```

## Requirements

Python 3.9+, `numpy`, `pandas`, `matplotlib` (SciPy is not needed).

```bash
pip install numpy pandas matplotlib
```
