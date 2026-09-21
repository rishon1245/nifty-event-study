"""Synthetic NIFTY-like OHLC data - ONLY for testing the pipeline offline.

GARCH(1,1) volatility with fat-tailed (Student-t) shocks and NO built-in
predictability (``bounce=0``) - i.e. the null hypothesis is true by construction.
``bounce > 0`` injects a known next-day rebound after big falls, which lets the
test-suite verify that the engine can actually detect a real effect.

NEVER report results from this generator as NIFTY findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_nifty(start: str = "2007-09-17", end: str = "2025-12-31", seed: int = 7,
                         bounce: float = 0.0, bounce_trigger: float = -0.02,
                         drift_annual: float = 0.10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    omega, a, b, dfree = 2e-6, 0.09, 0.89, 5
    mu = drift_annual / 252
    r = np.zeros(n)
    var = omega / (1 - a - b)
    scale = np.sqrt((dfree - 2) / dfree)               # unit-variance t shocks
    for t in range(n):
        eps = rng.standard_t(dfree) * scale
        r[t] = mu + np.sqrt(var) * eps
        if bounce and t > 0 and r[t - 1] <= bounce_trigger:
            r[t] += bounce
        var = omega + a * (r[t] - mu) ** 2 + b * var
    close = 4000.0 * np.exp(np.cumsum(np.log1p(np.clip(r, -0.2, 0.2))))
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.003, n))      # overnight gap
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx)
    df.index.name = "Date"
    return df
