"""Small plotting helpers (matplotlib only)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .events import event_paths
from .backtest import max_drawdown


def plot_return_distributions(exp, ax=None):
    ax = ax or plt.subplots(figsize=(7, 3.6))[1]
    base = exp.baseline["all_days"] * 100
    ev = exp.events["gross"] * 100
    bins = np.linspace(min(base.quantile(0.005), ev.min()), max(base.quantile(0.995), ev.max()), 50)
    ax.hist(base, bins=bins, density=True, alpha=0.5, label=f"all days (n={len(base)})")
    ax.hist(ev, bins=bins, density=True, alpha=0.6, label=f"after event (n={len(ev)})")
    ax.axvline(base.mean(), color="C0", ls="--", lw=1)
    ax.axvline(ev.mean(), color="C1", ls="--", lw=1)
    ax.set_xlabel(f"{exp.h}-day forward return, gross (%)")
    ax.set_ylabel("density")
    ax.set_title("Forward returns: events vs. baseline (dashed = means)")
    ax.legend()
    return ax


def plot_event_paths(feat, exp, pre=5, post=20, ax=None):
    ax = ax or plt.subplots(figsize=(7, 3.6))[1]
    paths = event_paths(feat, exp.events.index, pre, post) * 100
    if paths.empty:
        return ax
    m = paths.mean()
    se = paths.std(ddof=1) / np.sqrt(len(paths)) if len(paths) > 1 else 0 * m
    ax.plot(m.index, m.values, marker="o", ms=3)
    ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, alpha=0.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="r", ls=":", lw=1)
    ax.set_xlabel("trading days relative to event (0 = fall day)")
    ax.set_ylabel("cum. return vs. pre-fall close (%)")
    ax.set_title(f"Average path around events (n={len(paths)}, band = 95% CI of mean)")
    return ax


def plot_grid_heatmap(grid: pd.DataFrame, value="excess", ax=None, alpha=0.05):
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    piv = grid.pivot(index="threshold", columns="h", values=value) * 100
    n = grid.pivot(index="threshold", columns="h", values="n")
    ph = grid.pivot(index="threshold", columns="h", values="p_holm")
    lim = np.nanmax(np.abs(piv.values)) or 1
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), [f"{t:.1%}" for t in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            star = "*" if ph.values[i, j] < alpha else ""
            ax.text(j, i, f"{piv.values[i, j]:.2f}{star}\nn={int(n.values[i, j])}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("holding period (days)")
    ax.set_ylabel("event threshold")
    ax.set_title(f"{value} vs baseline (%)   * = Holm-adjusted p < {alpha}")
    plt.colorbar(im, ax=ax)
    return ax


def plot_equity(result, axes=None):
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    eq, bh = result.equity, result.benchmark
    axes[0].plot(eq.index, eq / eq.iloc[0], label="event strategy")
    axes[0].plot(bh.index, bh / bh.iloc[0], label="buy & hold NIFTY", alpha=0.7)
    axes[0].set_ylabel("growth of 1")
    axes[0].legend()
    axes[0].set_title("Equity curve (net of costs)")
    _, _, dd = max_drawdown(eq)
    _, _, dd_b = max_drawdown(bh)
    axes[1].fill_between(dd.index, dd * 100, 0, alpha=0.6, label="strategy")
    axes[1].plot(dd_b.index, dd_b * 100, color="C1", lw=0.8, label="buy & hold")
    axes[1].set_ylabel("drawdown (%)")
    axes[1].legend()
    return axes
