"""The configurable research engine.

    feat = add_features(clean_df, cfg)
    exp  = run_experiment(feat, cfg)                    # one experiment
    grid = robustness_grid(feat, cfg, end=research_end) # threshold x holding grid
    var  = robustness_variants(feat, cfg, end=research_end)

Changing the event threshold / holding period / entry rule / costs means editing
``config.yaml`` (or ``cfg.replace(...)``), never this file.

Return conventions
------------------
``gross``  = exit_close / entry_price - 1
``net``    = gross - round-trip cost (commission + slippage, both sides)
Comparisons with the baseline use GROSS returns (like-for-like: does the fall
carry information?); tradability is judged on NET returns (does it survive costs?).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from . import stats as S
from .config import Config
from .events import decluster, event_flag, forward_table


# ================================================================== experiment container
@dataclass
class Experiment:
    cfg: Config
    h: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    universe: pd.DataFrame            # every day in the sample with a complete forward window
    events_raw: pd.DataFrame          # all qualifying events (may overlap / cluster)
    events: pd.DataFrame              # independent events used for inference
    baseline: Dict[str, pd.Series]    # gross forward returns of comparison groups
    tests: Dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- reporting
    def summary_table(self) -> pd.DataFrame:
        rows = {
            "Events (all, may overlap)": self.events_raw["gross"],
            "Events (independent)": self.events["gross"],
            "Baseline: all days": self.baseline["all_days"],
            "Baseline: non-event days": self.baseline["non_event"],
            "Baseline: high-vol non-event days": self.baseline["high_vol_non_event"],
        }
        tbl = pd.DataFrame({k: S.describe_returns(v) for k, v in rows.items()}).T
        tbl["mean_net"] = tbl["mean"] - self.cfg.round_trip_cost
        return tbl

    def event_table(self) -> pd.DataFrame:
        cols = ["ret", "entry_date", "entry_px", "exit_date", "exit_px", "gross", "net", "retraced"]
        t = self.events[cols].rename(columns={"ret": "event_return"})
        t.index.name = "event_date"
        return t


# ========================================================================== main engine
def run_experiment(feat: pd.DataFrame, cfg: Config, h: Optional[int] = None,
                   start=None, end=None, exclude: Iterable[Tuple[str, str]] = (),
                   tests: bool = True) -> Experiment:
    """Run ONE event study.

    ``start`` / ``end`` restrict the sample: an event counts only if its date >= start
    and its full trade (exit date) <= end.  Features are computed on the full history
    (they are causal) so the out-of-sample period is not starved of warm-up data.
    """
    h = int(h or cfg.holding_days)
    c = cfg if h == cfg.holding_days else cfg.replace(holding_days=h)
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None

    fw = forward_table(feat, h, c)
    u = pd.concat([feat[["ret", "vol", "z"]], fw], axis=1)
    u["is_event"] = event_flag(feat, c)
    u["pos"] = np.arange(len(u))
    u = u[u["gross"].notna()]
    if start is not None:
        u = u[u.index >= start]
    if end is not None:
        u = u[u["exit_date"] <= end]
    for a, b in exclude:
        u = u[~((u.index >= pd.Timestamp(a)) & (u.index <= pd.Timestamp(b)))]

    events_raw = u[u["is_event"]]
    events = decluster(events_raw, c.gap) if c.decluster else events_raw

    non_event = u.loc[~u["is_event"], "gross"]
    ev_vol = events_raw["vol"].dropna()
    if len(ev_vol):
        vol_cut = ev_vol.quantile(0.25)
        hv = u.loc[(~u["is_event"]) & (u["vol"] >= vol_cut), "gross"]
    else:
        hv = pd.Series(dtype=float)
    baseline = {"all_days": u["gross"], "non_event": non_event, "high_vol_non_event": hv}

    exp = Experiment(c, h, start, end, u, events_raw, events, baseline)
    if tests:
        exp.tests = evaluate(exp)
    return exp


def evaluate(exp: Experiment, n_boot: Optional[int] = None, n_placebo: Optional[int] = None) -> Dict[str, float]:
    """All statistical tests for the independent events of one experiment."""
    cfg, ev = exp.cfg, exp.events
    n_boot = n_boot or cfg.n_boot
    n_placebo = n_placebo or cfg.n_placebo
    rng = np.random.default_rng(cfg.seed)
    base = exp.baseline["all_days"]
    mu0 = float(base.mean()) if len(base) else np.nan
    n = len(ev)

    t_zero, p_zero = S.t_test_greater(ev["net"], 0.0)
    t_ex, p_ex = S.t_test_greater(ev["gross"], mu0)
    _, p_ex_two = S.t_test_two_sided(ev["gross"], mu0)
    ci_net = S.bootstrap_mean_ci(ev["net"], n_boot, cfg.alpha, rng)
    ex_lo, ex_hi, p_boot = S.excess_bootstrap(ev["gross"], base, max(2 * exp.h, 10), n_boot, cfg.alpha, rng)
    ev_mean = float(ev["gross"].mean()) if n else np.nan
    hv = exp.baseline["high_vol_non_event"]
    return {
        "n_events_raw": len(exp.events_raw),
        "n_events_independent": n,
        "reliable(n>=min_events)": n >= cfg.min_events,
        "mean_gross": ev_mean,
        "mean_net": float(ev["net"].mean()) if n else np.nan,
        "baseline_mean_gross": mu0,
        "excess_mean_gross": ev_mean - mu0 if n else np.nan,
        "cohens_d_vs_baseline": S.cohens_d(ev["gross"], mu0),
        "t_stat_net_vs_zero": t_zero,
        "p_net_gt_zero(one-sided t)": p_zero,
        "t_stat_excess": t_ex,
        "p_excess(one-sided t)": p_ex,
        "p_excess(two-sided t)": p_ex_two,
        "p_excess(Wilcoxon)": S.wilcoxon_greater(ev["gross"], mu0),
        "boot_ci_mean_net_low": ci_net[0],
        "boot_ci_mean_net_high": ci_net[1],
        "boot_ci_excess_low": ex_lo,
        "boot_ci_excess_high": ex_hi,
        "p_excess(block-bootstrap)": p_boot,
        "p_placebo(all days)": S.placebo_p_value(ev_mean, base, n, n_placebo, rng),
        "p_placebo(high-vol days)": S.placebo_p_value(ev_mean, hv, n, n_placebo, rng),
        "event_win_rate": float((ev["gross"] > 0).mean()) if n else np.nan,
        "baseline_win_rate": float((base > 0).mean()) if len(base) else np.nan,
        "p_win_rate_gt_baseline(binomial)": S.binom_win_rate_greater(
            ev["gross"], float((base > 0).mean()) if len(base) else np.nan),
    }


# ================================================================== robustness machinery
def robustness_grid(feat: pd.DataFrame, cfg: Config, start=None, end=None,
                    exclude: Iterable[Tuple[str, str]] = ()) -> pd.DataFrame:
    """Threshold x holding-period grid with multiple-testing correction.

    The whole grid is ONE family of tests.  Reporting the best cell without the
    Holm / BH columns would be data snooping.
    """
    rows = []
    for thr in cfg.threshold_grid:
        for h in cfg.holding_grid:
            c = cfg.replace(event_mode="fixed", threshold=thr, holding_days=h)
            e = run_experiment(feat, c, h, start, end, exclude, tests=False)
            ev, base = e.events, e.baseline["all_days"]
            mu0 = float(base.mean()) if len(base) else np.nan
            _, p = S.t_test_greater(ev["gross"], mu0)
            rows.append(dict(
                threshold=thr, h=h, n_raw=len(e.events_raw), n=len(ev),
                mean_gross=ev["gross"].mean(), mean_net=ev["net"].mean(), median_net=ev["net"].median(),
                win_rate_net=(ev["net"] > 0).mean() if len(ev) else np.nan,
                base_mean=mu0, excess=ev["gross"].mean() - mu0 if len(ev) else np.nan, p_greater=p,
            ))
    g = pd.DataFrame(rows)
    g["p_holm"] = S.holm_adjust(g["p_greater"])
    g["p_bh"] = S.bh_adjust(g["p_greater"])
    g["reliable"] = g["n"] >= cfg.min_events
    return g


def _variant_row(label: str, e: Experiment) -> dict:
    ev, base = e.events, e.baseline["all_days"]
    mu0 = float(base.mean()) if len(base) else np.nan
    _, p = S.t_test_greater(ev["gross"], mu0)
    return dict(variant=label, n_raw=len(e.events_raw), n=len(ev),
                mean_gross=ev["gross"].mean(), mean_net=ev["net"].mean(), median_net=ev["net"].median(),
                win_rate_net=(ev["net"] > 0).mean() if len(ev) else np.nan,
                base_mean=mu0, excess=ev["gross"].mean() - mu0 if len(ev) else np.nan, p_greater=p)


def robustness_variants(feat: pd.DataFrame, cfg: Config, start=None, end=None) -> pd.DataFrame:
    """Change one *assumption* at a time (not only threshold/holding)."""
    out = []

    def add(label, c, exclude=()):
        out.append(_variant_row(label, run_experiment(feat, c, None, start, end, exclude, tests=False)))

    add("Primary specification", cfg)
    add("No de-clustering (overlapping events)", cfg.replace(decluster=False))
    add("Entry at event-day CLOSE (optimistic)", cfg.replace(entry="event_close"))
    add("Vol-adjusted event (z <= z_threshold)", cfg.replace(event_mode="zscore"))
    add("Percentile event (worst p% of past returns)", cfg.replace(event_mode="percentile"))
    add("Excluding crisis windows (GFC, COVID)", cfg, exclude=cfg.crisis_windows)
    for rt in (0, 20, 40):   # total round-trip bps
        add(f"Round-trip cost = {rt} bps", cfg.replace(commission_bps=0.0, slippage_bps=rt / 2))
    return pd.DataFrame(out).set_index("variant")


def by_year(exp: Experiment) -> pd.DataFrame:
    """Is the effect spread over many years or driven by one or two?"""
    ev, u = exp.events, exp.universe
    if ev.empty:
        return pd.DataFrame()
    g = ev.groupby(ev.index.year).agg(n=("gross", "size"), mean_gross=("gross", "mean"),
                                      mean_net=("net", "mean"), win_rate=("gross", lambda s: (s > 0).mean()))
    g["base_mean"] = u.groupby(u.index.year)["gross"].mean()
    g["excess"] = g["mean_gross"] - g["base_mean"]
    return g


def by_vol_regime(exp: Experiment) -> pd.DataFrame:
    """Same comparison split by trailing-volatility tercile (market regime)."""
    u = exp.universe.dropna(subset=["vol"]).copy()
    if len(u) < 30:
        return pd.DataFrame()
    u["regime"] = pd.qcut(u["vol"], 3, labels=["low vol", "mid vol", "high vol"])
    ev = u[u["is_event"] & u.index.isin(exp.events.index)]
    g = ev.groupby("regime", observed=False).agg(n=("gross", "size"), mean_gross=("gross", "mean"))
    g["base_mean"] = u.groupby("regime", observed=False)["gross"].mean()
    g["excess"] = g["mean_gross"] - g["base_mean"]
    return g


# ============================================================================ OOS split
def research_end(feat: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    """Last trading day of the research period (the day before ``split_date``)."""
    return feat.index[feat.index < pd.Timestamp(cfg.split_date)][-1]


def run_research(feat: pd.DataFrame, cfg: Config, **kw) -> Experiment:
    """DEVELOPMENT sample only.  Everything you tune must use this."""
    return run_experiment(feat, cfg, end=research_end(feat, cfg), **kw)


def run_oos(feat: pd.DataFrame, cfg: Config, **kw) -> Experiment:
    """UNSEEN sample.  Call ONCE, with parameters frozen after the research phase."""
    return run_experiment(feat, cfg, start=cfg.split_date, **kw)


def compare_periods(research: Experiment, oos: Experiment) -> pd.DataFrame:
    """Side-by-side research vs out-of-sample, including market context ('what changed')."""
    def col(e: Experiment) -> dict:
        t, u = e.tests, e.universe
        span_years = max((u.index.max() - u.index.min()).days / 365.25, 1e-9) if len(u) else np.nan
        return {
            "sample": f"{u.index.min().date()} -> {u.index.max().date()}" if len(u) else "n/a",
            "independent events": t["n_events_independent"],
            "events / year (raw)": t["n_events_raw"] / span_years,
            "mean net return": t["mean_net"],
            "mean gross return": t["mean_gross"],
            "baseline mean (gross)": t["baseline_mean_gross"],
            "excess vs baseline": t["excess_mean_gross"],
            "win rate (gross)": t["event_win_rate"],
            "baseline win rate": t["baseline_win_rate"],
            "excess CI low": t["boot_ci_excess_low"],
            "excess CI high": t["boot_ci_excess_high"],
            "p excess (one-sided t)": t["p_excess(one-sided t)"],
            "p placebo (all days)": t["p_placebo(all days)"],
            "market ann. vol": u["ret"].std() * np.sqrt(252),
            "market mean daily ret (bps)": u["ret"].mean() * 1e4,
        }
    return pd.DataFrame({"Research": col(research), "Out-of-sample": col(oos)})


# ==================================================================== falsification
def falsification_scorecard(research: Experiment, grid: pd.DataFrame, variants: pd.DataFrame,
                            oos: Optional[Experiment] = None) -> pd.DataFrame:
    """Pre-registered rejection criteria.  FAIL on any = the hypothesis is not supported.

    The criteria are written down BEFORE looking at the results so that they cannot
    be bent to fit whatever the data says.
    """
    cfg, t = research.cfg, research.tests
    rows = []

    def add(name, observed, passed):
        rows.append({"criterion": name, "observed": observed, "verdict": "PASS" if bool(passed) else "FAIL"})

    fin = lambda x: x is not None and np.isfinite(x)
    add(f"Independent events >= {cfg.min_events}", t["n_events_independent"], t["n_events_independent"] >= cfg.min_events)
    lo, hi = t["boot_ci_excess_low"], t["boot_ci_excess_high"]
    add("Bootstrap CI of excess return over baseline excludes 0", f"[{lo:.4%}, {hi:.4%}]", fin(lo) and lo > 0)
    add("Mean NET return per trade > 0 after costs", f"{t['mean_net']:.4%}", fin(t["mean_net"]) and t["mean_net"] > 0)
    pp = t["p_placebo(all days)"]
    add(f"Placebo p (random days) < {cfg.alpha}", f"{pp:.4f}", fin(pp) and pp < cfg.alpha)
    ph = t["p_placebo(high-vol days)"]
    add(f"Placebo p (random HIGH-VOL days) < {cfg.alpha}", f"{ph:.4f}", fin(ph) and ph < cfg.alpha)

    cell = grid[(np.isclose(grid["threshold"], cfg.threshold)) & (grid["h"] == research.h)]
    if len(cell):
        ph_ = float(cell["p_holm"].iloc[0])
        add(f"Primary cell survives Holm correction over the whole grid (< {cfg.alpha})", f"{ph_:.4f}", fin(ph_) and ph_ < cfg.alpha)
    for label in ("Excluding crisis windows (GFC, COVID)", "No de-clustering (overlapping events)",
                  "Vol-adjusted event (z <= z_threshold)"):
        if label in variants.index:
            ex = variants.loc[label, "excess"]
            add(f"Excess > 0 under: {label}", f"{ex:.4%}", fin(ex) and ex > 0)
    if oos is not None:
        o = oos.tests
        add("OUT-OF-SAMPLE: mean net return > 0", f"{o['mean_net']:.4%}", fin(o["mean_net"]) and o["mean_net"] > 0)
        add("OUT-OF-SAMPLE: excess over baseline > 0", f"{o['excess_mean_gross']:.4%}",
            fin(o["excess_mean_gross"]) and o["excess_mean_gross"] > 0)
    return pd.DataFrame(rows)
