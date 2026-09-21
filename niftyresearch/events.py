"""Event detection and forward returns.

Timing convention (the single most important thing to get right)
-----------------------------------------------------------------
* Day ``t``     : the fall happens.  We only know the close-to-close return at the
                  15:30 IST close of day ``t``.
* ``next_open`` : we enter at the OPEN of day ``t+1``  (realistic; default).
* ``event_close``: we enter at the CLOSE of day ``t``   (optimistic - needs the close
                  to be known slightly before it happens; kept only as a sensitivity).
* Exit          : the CLOSE of day ``t + h``   (``h`` = holding period, trading days).

Hence with ``next_open`` a holding period of ``h`` means ``h`` sessions
(``t+1 ... t+h``); with ``h = 1`` the trade is the intraday move of day ``t+1``.

Every feature used to *define* an event (return, trailing vol, expanding
percentile) is computed from data available at the close of day ``t`` only -
see ``tests/test_core.py::test_features_are_causal``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


# ---------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Causal features. Row ``t`` uses information up to the close of day ``t`` only."""
    out = df.copy()
    out["ret"] = out["Close"].pct_change()
    # trailing vol is shifted by one day so today's return is NOT in its own denominator
    out["vol"] = out["ret"].rolling(cfg.vol_window, min_periods=cfg.vol_window).std().shift(1)
    out["z"] = out["ret"] / out["vol"]
    # expanding empirical quantile of PAST returns (shifted -> excludes today)
    out["pct_thr"] = out["ret"].expanding(min_periods=250).quantile(cfg.percentile).shift(1)
    return out


def is_event_row(ret: float, z: float, pct_thr: float, cfg: Config) -> bool:
    """Per-bar event test (used by the event-driven backtester)."""
    if cfg.event_mode == "fixed":
        return bool(ret <= cfg.threshold)
    if cfg.event_mode == "zscore":
        return bool(np.isfinite(z) and z <= cfg.z_threshold)
    return bool(np.isfinite(pct_thr) and ret <= pct_thr)


def event_flag(feat: pd.DataFrame, cfg: Config) -> pd.Series:
    """Vectorised version of :func:`is_event_row`."""
    if cfg.event_mode == "fixed":
        flag = feat["ret"] <= cfg.threshold
    elif cfg.event_mode == "zscore":
        flag = feat["z"] <= cfg.z_threshold
    else:
        flag = feat["ret"] <= feat["pct_thr"]
    return flag.fillna(False).astype(bool)


# --------------------------------------------------------------------- forward returns
def forward_table(feat: pd.DataFrame, h: int, cfg: Config) -> pd.DataFrame:
    """For EVERY day t: the trade you would have made had t been an event day.

    Events and the baseline are both taken from this one table, so they are
    guaranteed to use identical entry/exit conventions.
    """
    dates = pd.Series(feat.index, index=feat.index)
    if cfg.entry == "next_open":
        entry_px, entry_date = feat["Open"].shift(-1), dates.shift(-1)
    else:  # event_close
        entry_px, entry_date = feat["Close"], dates
    exit_px, exit_date = feat["Close"].shift(-h), dates.shift(-h)
    gross = exit_px / entry_px - 1.0
    return pd.DataFrame(
        {
            "entry_date": entry_date,
            "entry_px": entry_px,
            "exit_date": exit_date,
            "exit_px": exit_px,
            "gross": gross,
            "net": gross - cfg.round_trip_cost,
            # "recovery" flavour 2: did the index get back to the pre-fall close?
            "retraced": (exit_px >= feat["Close"].shift(1)).where(gross.notna()),
        }
    )


def decluster(events: pd.DataFrame, gap: int) -> pd.DataFrame:
    """Greedy de-clustering: keep an event only if it is >= ``gap`` trading days after
    the previously kept one (so holding windows never overlap).  ``events`` must
    carry an integer ``pos`` column (position in the full price index)."""
    if events.empty or gap <= 0:
        return events
    keep, last = [], -10**9
    for p in events["pos"].to_numpy():
        if p - last >= gap:
            keep.append(True)
            last = p
        else:
            keep.append(False)
    return events[np.array(keep)]


def event_paths(feat: pd.DataFrame, event_dates, pre: int = 5, post: int = 20) -> pd.DataFrame:
    """Cumulative return path around each event, rebased to the close BEFORE the fall.

    Rows = events, columns = offsets ``-pre .. +post`` (0 = event day).  Diagnostic
    only (uses future data by construction) - never used to define trades.
    """
    close = feat["Close"].to_numpy()
    pos = feat.index.get_indexer(pd.DatetimeIndex(event_dates))
    offsets = np.arange(-pre, post + 1)
    rows = []
    for p in pos:
        if p - 1 < 0 or p - pre < 0 or p + post >= len(close):
            continue
        rows.append(close[p + offsets] / close[p - 1] - 1.0)
    return pd.DataFrame(rows, columns=offsets)
