"""Statistical tools.  Each function documents WHAT it answers.

Why several methods?  Index returns are fat-tailed, skewed and (for overlapping
windows) autocorrelated, so no single test is trusted on its own:

* t-test         - fast, well understood, but leans on the CLT (weak for n < ~30 and
                   fat tails).
* Wilcoxon       - rank-based; robust to outliers, but tests the median-ish shift.
* Bootstrap CI   - no normality assumption; answers "how big is the effect and how
                   uncertain is it?" rather than only "is it non-zero?".
* Placebo test   - "how often would n RANDOM days look this good?"; the most direct
                   answer to "is this just the market's normal drift?".
* Holm / BH      - correct the p-values when many (threshold x holding) cells are tried.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------- descriptives
def describe_returns(x) -> Dict[str, float]:
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    nan = np.nan
    if n == 0:
        return dict(n=0, mean=nan, median=nan, std=nan, se_mean=nan, win_rate=nan,
                    skew=nan, excess_kurt=nan, p05=nan, p95=nan, min=nan, max=nan, mean_over_std=nan)
    std = x.std(ddof=1) if n > 1 else nan
    return dict(
        n=n,
        mean=x.mean(),
        median=x.median(),
        std=std,
        se_mean=std / np.sqrt(n) if n > 1 else nan,
        win_rate=(x > 0).mean(),
        skew=stats.skew(x, bias=False) if n > 3 else nan,
        excess_kurt=stats.kurtosis(x, bias=False) if n > 3 else nan,
        p05=x.quantile(0.05),
        p95=x.quantile(0.95),
        min=x.min(),
        max=x.max(),
        mean_over_std=x.mean() / std if n > 1 and std > 0 else nan,
    )


# ------------------------------------------------------------------------------ tests
def t_test_greater(x, mu: float = 0.0):
    """One-sided one-sample t-test  H1: mean(x) > mu.  Returns (t, p)."""
    x = np.asarray(pd.Series(x, dtype=float).dropna())
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return np.nan, np.nan
    r = stats.ttest_1samp(x, mu, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def t_test_two_sided(x, mu: float = 0.0):
    x = np.asarray(pd.Series(x, dtype=float).dropna())
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return np.nan, np.nan
    r = stats.ttest_1samp(x, mu)
    return float(r.statistic), float(r.pvalue)


def wilcoxon_greater(x, mu: float = 0.0) -> float:
    d = np.asarray(pd.Series(x, dtype=float).dropna()) - mu
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        return np.nan


def binom_win_rate_greater(x, p0: float) -> float:
    """Is the event win-rate higher than the baseline win-rate ``p0``?"""
    x = pd.Series(x, dtype=float).dropna()
    if len(x) == 0 or not np.isfinite(p0):
        return np.nan
    return float(stats.binomtest(int((x > 0).sum()), len(x), p0, alternative="greater").pvalue)


def cohens_d(x, mu: float = 0.0) -> float:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan
    return float((x.mean() - mu) / x.std(ddof=1))


# --------------------------------------------------------------------------- bootstrap
def bootstrap_mean_ci(x, n_boot: int, alpha: float, rng: np.random.Generator):
    """Percentile bootstrap CI for the mean of i.i.d. observations."""
    x = np.asarray(pd.Series(x, dtype=float).dropna())
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    means = _iid_boot_means(x, n_boot, rng)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def _iid_boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    out = np.empty(n_boot)
    chunk = max(1, int(2_000_000 // max(n, 1)))
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        out[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return out


def block_bootstrap_means(x, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap distribution of the MEAN of an autocorrelated series.

    Overlapping h-day forward returns are autocorrelated up to lag h-1, so an
    i.i.d. bootstrap would understate the uncertainty of the *baseline* mean.
    Resampling whole blocks keeps that dependence.
    """
    x = np.asarray(pd.Series(x, dtype=float).dropna())
    n = len(x)
    if n < 2:
        return np.full(n_boot, np.nan)
    b = int(min(max(block, 1), n))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    block_means = (cs[b:] - cs[:-b]) / b            # mean of every window of length b
    k = int(np.ceil(n / b))
    idx = rng.integers(0, len(block_means), size=(n_boot, k))
    return block_means[idx].mean(axis=1)


def excess_bootstrap(events, baseline, block: int, n_boot: int, alpha: float, rng: np.random.Generator):
    """Bootstrap distribution of  mean(events) - mean(baseline).

    Events: i.i.d. resampling (they are de-clustered).  Baseline: block bootstrap.
    Returns (ci_low, ci_high, one-sided p that the excess <= 0).
    """
    ev = np.asarray(pd.Series(events, dtype=float).dropna())
    if len(ev) < 2:
        return np.nan, np.nan, np.nan
    diff = _iid_boot_means(ev, n_boot, rng) - block_bootstrap_means(baseline, block, n_boot, rng)
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    p = (1 + int((diff <= 0).sum())) / (n_boot + 1)
    return float(lo), float(hi), float(p)


def placebo_p_value(observed_mean: float, pool, n: int, n_sims: int, rng: np.random.Generator) -> float:
    """P(mean of n RANDOM days from ``pool`` >= observed_mean).  (Sampling with replacement;
    with n << len(pool) the duplicate probability is negligible.)"""
    pool = np.asarray(pd.Series(pool, dtype=float).dropna())
    if n < 1 or len(pool) < 2 or not np.isfinite(observed_mean):
        return np.nan
    sims = np.empty(n_sims)
    chunk = max(1, int(2_000_000 // n))
    for s in range(0, n_sims, chunk):
        e = min(n_sims, s + chunk)
        sims[s:e] = pool[rng.integers(0, len(pool), size=(e - s, n))].mean(axis=1)
    return float((1 + int((sims >= observed_mean).sum())) / (n_sims + 1))


# --------------------------------------------------------------------- multiple testing
def _finite_apply(p, fn):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = fn(p[ok])
    return out


def holm_adjust(p) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error)."""
    def fn(v):
        m = len(v)
        order = np.argsort(v)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * v[i])
            adj[i] = min(1.0, running)
        return adj
    return _finite_apply(p, fn)


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls false-discovery rate)."""
    def fn(v):
        m = len(v)
        order = np.argsort(v)
        ranked = v[order] * m / np.arange(1, m + 1)
        adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.clip(adj_sorted, 0, 1)
        return out
    return _finite_apply(p, fn)
