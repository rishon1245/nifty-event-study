"""Data sourcing, validation and cleaning for NIFTY 50 daily OHLC.

Source hierarchy
----------------
1. ``cfg.csv_path``  - a local CSV you supply (e.g. an export from niftyindices.com,
                       the official NSE Indices site).
2. ``cfg.cache_path``- a previous download, re-used so results are reproducible.
3. Yahoo Finance ``^NSEI`` via ``yfinance`` (daily, unadjusted; an index has no
   splits/dividends so "adjusted" and "raw" prices coincide).

Every cleaning decision is written to a log (a DataFrame) so nothing is silently
altered.  Suspicious-but-real observations (e.g. the +17.7 % NIFTY day on
2009-05-18, the -13 % day on 2020-03-23) are FLAGGED, never dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import Config

OHLC = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------- loading
def download_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is not installed. Either `pip install yfinance` or set "
            "`csv_path` in config.yaml to a local NIFTY 50 CSV."
        ) from exc
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse dates trying explicit formats first (avoids day/month ambiguity)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    else:
        s = s.astype("string").str.strip()
        best, best_na = None, None
        for fmt in ("%Y-%m-%d", "%d %b %Y", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            na = int(parsed.isna().sum())
            if best is None or na < best_na:
                best, best_na = parsed, na
            if na == 0:
                break
        out = best
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def _standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Uniform column names, parsed dates, numeric OHLC.  Row order is preserved."""
    df = raw.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Date" not in df.columns:
        raise ValueError(f"No 'Date' column found. Columns: {list(df.columns)}")
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Date"] = _parse_dates(df["Date"])
    for c in OHLC:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype("string").str.replace(",", "", regex=False), errors="coerce")
        df[c] = df[c].astype(float)
    return df[["Date"] + OHLC].reset_index(drop=True)


def load_raw(cfg: Config, root: str | Path = ".") -> Tuple[pd.DataFrame, str]:
    """Return (raw dataframe, human-readable source description)."""
    root = Path(root)
    if cfg.csv_path:
        p = root / cfg.csv_path
        return pd.read_csv(p), f"local CSV: {cfg.csv_path}"
    cache = root / cfg.cache_path
    if cache.exists():
        return pd.read_csv(cache), f"cached download: {cfg.cache_path} (originally Yahoo Finance {cfg.ticker})"
    df = download_yfinance(cfg.ticker, cfg.data_start, cfg.data_end)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df, f"Yahoo Finance {cfg.ticker} (downloaded, cached at {cfg.cache_path})"


# ------------------------------------------------------------------------ validation
@dataclass
class ValidationReport:
    n_rows: int
    summary: pd.DataFrame                 # one row per check
    details: Dict[str, pd.DataFrame]      # offending rows per check

    def __repr__(self) -> str:
        return f"ValidationReport(n_rows={self.n_rows})\n{self.summary.to_string(index=False)}"


def validate(raw: pd.DataFrame) -> ValidationReport:
    """Run every integrity check on the RAW data and report (does not modify)."""
    df = _standardize(raw)
    tol = df["Close"].abs() * 1e-6
    checks = {
        "unparseable_date": (df["Date"].isna(), "drop row"),
        "duplicate_date": (df["Date"].notna() & df["Date"].duplicated(keep=False), "keep last occurrence"),
        "out_of_order": (df["Date"].diff().dt.days < 0, "sort ascending"),
        "missing_ohlc": (df[OHLC].isna().any(axis=1), "drop row"),
        "non_positive_ohlc": ((df[OHLC] <= 0).any(axis=1), "drop row"),
        "high_low_inconsistent": (
            (df["High"] < df[["Open", "Low", "Close"]].max(axis=1) - tol)
            | (df["Low"] > df[["Open", "High", "Close"]].min(axis=1) + tol),
            "clip High/Low to the O/C envelope (study only uses Open & Close)",
        ),
        "flat_bar_O=H=L=C": (
            (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]),
            "drop row (stale / non-standard session)",
        ),
    }
    rows, details = [], {}
    for name, (mask, action) in checks.items():
        mask = mask.fillna(False)
        rows.append({"check": name, "rows_affected": int(mask.sum()), "action": action})
        if mask.any():
            details[name] = df[mask]
    return ValidationReport(n_rows=len(df), summary=pd.DataFrame(rows), details=details)


# ---------------------------------------------------------------------------- cleaning
def clean(raw: pd.DataFrame, start: str | None = None, end: str | None = None
          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean OHLC frame indexed by Date, cleaning log)."""
    df = _standardize(raw)
    log: list[dict] = []

    def note(step: str, sub: pd.DataFrame, detail: str):
        for _, r in sub.iterrows():
            log.append({"step": step, "date": r["Date"], "detail": detail})

    m = df["Date"].isna()
    note("drop_unparseable_date", df[m], "date could not be parsed")
    df = df[~m]

    m = df[OHLC].isna().any(axis=1) | (df[OHLC] <= 0).any(axis=1)
    note("drop_invalid_ohlc", df[m], "missing or non-positive OHLC value")
    df = df[~m]

    m = df["Date"].duplicated(keep="last")
    note("drop_duplicate_date", df[m], "earlier duplicate of a later row with same date")
    df = df[~m]

    if not df["Date"].is_monotonic_increasing:
        log.append({"step": "sort", "date": pd.NaT, "detail": "rows were not in ascending date order; sorted"})
        df = df.sort_values("Date", kind="stable")

    m = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    note("drop_flat_bar", df[m], "Open=High=Low=Close (stale or non-standard session)")
    df = df[~m]

    hi = df[["Open", "High", "Low", "Close"]].max(axis=1)
    lo = df[["Open", "High", "Low", "Close"]].min(axis=1)
    m = (df["High"] < hi) | (df["Low"] > lo)
    note("repair_high_low", df[m], "High/Low clipped to include Open and Close")
    df["High"], df["Low"] = hi, lo

    if start:
        m = df["Date"] < pd.Timestamp(start)
        if m.any():
            log.append({"step": "trim_start", "date": pd.NaT, "detail": f"{int(m.sum())} rows before {start}"})
        df = df[~m]
    if end:
        m = df["Date"] > pd.Timestamp(end)
        if m.any():
            log.append({"step": "trim_end", "date": pd.NaT, "detail": f"{int(m.sum())} rows after {end}"})
        df = df[~m]

    out = df.set_index("Date")[OHLC].astype(float)
    out.index.name = "Date"
    log_df = pd.DataFrame(log, columns=["step", "date", "detail"])
    return out, log_df


# ---------------------------------------------------------------- suspicious / coverage
def flag_suspicious(clean_df: pd.DataFrame, ret_limit: float = 0.10, gap_limit: float = 0.05,
                    calendar_gap_days: int = 5, stale_run: int = 3) -> pd.DataFrame:
    """Flag (never drop) observations that deserve a manual look."""
    c = clean_df["Close"]
    ret = c.pct_change()
    gap = clean_df["Open"] / c.shift(1) - 1
    cal_gap = clean_df.index.to_series().diff().dt.days
    same_close = (c.diff() == 0).astype(int)
    stale = same_close.groupby((same_close == 0).cumsum()).cumsum() >= (stale_run - 1)
    flags = pd.DataFrame(
        {
            "extreme_return(>10%)": ret.abs() > ret_limit,
            "large_open_gap(>5%)": gap.abs() > gap_limit,
            "calendar_gap(>5d)": cal_gap > calendar_gap_days,
            "weekend_session": clean_df.index.dayofweek >= 5,
            "stale_close(3+)": stale,
        },
        index=clean_df.index,
    ).fillna(False)
    out = flags[flags.any(axis=1)].copy()
    out.insert(0, "ret", ret.loc[out.index])
    out.insert(1, "open_gap", gap.loc[out.index])
    out["flags"] = [", ".join(flags.columns[row]) for row in flags.loc[out.index].to_numpy()]
    return out[["ret", "open_gap", "flags"]]


def coverage(clean_df: pd.DataFrame, expected_per_year: int = 248) -> pd.DataFrame:
    """Rows per calendar year vs. the ~248 trading days NSE typically has."""
    dates = clean_df.index.to_series()
    by = dates.groupby(dates.dt.year)
    g = by.size().rename("n_days").to_frame()
    g["first"] = by.min().dt.date
    g["last"] = by.max().dt.date
    g["vs_expected"] = g["n_days"] - expected_per_year
    g["low_coverage"] = (g["n_days"] < expected_per_year - 15) & (g.index != g.index.min()) & (g.index != g.index.max())
    return g


def prepare_data(cfg: Config, root: str | Path = ".", save: bool = True) -> dict:
    """Load -> validate -> clean -> flag -> coverage.  Returns everything as a dict."""
    raw, source = load_raw(cfg, root)
    report = validate(raw)
    clean_df, log = clean(raw, cfg.data_start, cfg.data_end)
    flags = flag_suspicious(clean_df)
    cov = coverage(clean_df)
    if save:
        root = Path(root)
        (root / "data/processed").mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(root / "data/processed/nifty50_clean.csv")
        log.to_csv(root / "results/cleaning_log.csv", index=False)
        flags.to_csv(root / "results/suspicious_observations.csv")
    provenance = (
        f"Source: {source}\n"
        f"Fields: {', '.join(OHLC)} (daily, unadjusted index levels)\n"
        f"Coverage: {clean_df.index.min().date()} -> {clean_df.index.max().date()}  "
        f"({len(clean_df)} trading days after cleaning; {report.n_rows} raw rows)\n"
        f"Cleaning actions logged: {len(log)}   Flagged (kept) observations: {len(flags)}"
    )
    return dict(raw=raw, clean=clean_df, report=report, log=log, flags=flags,
                coverage=cov, provenance=provenance, source=source)
