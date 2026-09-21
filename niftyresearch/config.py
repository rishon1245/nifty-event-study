"""Single source of truth for every experiment parameter.

Changing the event threshold, the holding period, the entry rule, the costs or
the train/test split is a config change - the research engine is never edited.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml

EVENT_MODES = ("fixed", "zscore", "percentile")
ENTRY_MODES = ("next_open", "event_close")


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ data
    ticker: str = "^NSEI"                      # Yahoo Finance symbol for NIFTY 50
    data_start: str = "2007-09-17"             # first date Yahoo has for ^NSEI
    data_end: Optional[str] = None             # None = up to latest available
    csv_path: Optional[str] = None             # optional local CSV (e.g. niftyindices.com export)
    cache_path: str = "data/raw/nifty50_daily.csv"

    # ------------------------------------------------------- event definition
    event_mode: str = "fixed"                  # fixed | zscore | percentile
    threshold: float = -0.02                   # fixed: close-to-close return <= -2 %
    z_threshold: float = -2.0                  # zscore: return / trailing daily vol <= -2
    percentile: float = 0.02                   # percentile: worst 2 % of *past* daily returns
    vol_window: int = 60                       # trailing window (days) for vol, shifted by 1 day

    # ------------------------------------------------------------ trade rules
    entry: str = "next_open"                   # next_open (realistic) | event_close (optimistic)
    holding_days: int = 5                      # primary holding period (trading days)
    decluster: bool = True                     # keep only non-overlapping events
    min_gap_days: Optional[int] = None         # None -> gap = holding_days

    # -------------------------------------------------------------- grids (robustness only)
    holding_grid: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    threshold_grid: Tuple[float, ...] = (-0.01, -0.015, -0.02, -0.025, -0.03, -0.04)
    crisis_windows: Tuple[Tuple[str, str], ...] = (
        ("2008-09-01", "2009-06-30"),          # Global Financial Crisis
        ("2020-02-15", "2020-06-30"),          # COVID crash
    )

    # ------------------------------------------------------------------ costs (per side)
    commission_bps: float = 2.0                # brokerage + STT + exchange + stamp (futures/ETF proxy)
    slippage_bps: float = 3.0                  # bid-ask / impact / open-auction noise

    # --------------------------------------------------------- train / test split
    split_date: str = "2018-01-01"             # research: < split_date ; out-of-sample: >= split_date

    # ---------------------------------------------------------------- statistics
    alpha: float = 0.05
    n_boot: int = 10_000
    n_placebo: int = 10_000
    seed: int = 42
    min_events: int = 30                       # below this, statistics are flagged as unreliable

    # ------------------------------------------------------------------ backtest
    initial_capital: float = 1_000_000.0

    def __post_init__(self):
        if self.event_mode not in EVENT_MODES:
            raise ValueError(f"event_mode must be one of {EVENT_MODES}")
        if self.entry not in ENTRY_MODES:
            raise ValueError(f"entry must be one of {ENTRY_MODES}")
        if self.holding_days < 1:
            raise ValueError("holding_days must be >= 1")
        if self.event_mode == "fixed" and self.threshold >= 0:
            raise ValueError("threshold must be negative (a fall)")

    # ---------------------------------------------------------------- helpers
    @property
    def round_trip_cost(self) -> float:
        """Total round-trip cost as a fraction (commission + slippage, both sides)."""
        return 2.0 * (self.commission_bps + self.slippage_bps) / 1e4

    @property
    def gap(self) -> int:
        """Minimum spacing between independent events, in trading days."""
        return self.min_gap_days if self.min_gap_days is not None else self.holding_days

    def replace(self, **kwargs) -> "Config":
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for key in ("holding_grid", "threshold_grid"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if "crisis_windows" in raw:
            raw["crisis_windows"] = tuple(tuple(w) for w in raw["crisis_windows"])
        return cls(**raw)
