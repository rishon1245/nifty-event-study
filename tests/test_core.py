"""Correctness tests.  Run with either:

    pytest -q
    python tests/test_core.py        # no pytest needed
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from niftyresearch import Config                                            # noqa: E402
from niftyresearch.backtest import EventDrivenBacktester                     # noqa: E402
from niftyresearch.data import clean, flag_suspicious, validate              # noqa: E402
from niftyresearch.events import add_features, decluster, event_flag, forward_table, is_event_row  # noqa: E402
from niftyresearch.research import (run_experiment, research_end, run_research,   # noqa: E402
                                    run_oos, robustness_grid)
from niftyresearch.stats import bh_adjust, holm_adjust                       # noqa: E402
from niftyresearch.synthetic import make_synthetic_nifty                     # noqa: E402

CFG = Config(n_boot=500, n_placebo=500)


# ------------------------------------------------------------------------------- data
def test_validation_and_cleaning():
    raw = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-01", "2020-01-03", "2020-01-03", "2020-01-06",
                 "2020-01-07", "2020-01-08", "bad-date", "2020-01-09"],
        "Open": [100, 99, 101, 101, 102, np.nan, 104, 105, 50],
        "High": [101, 100, 103, 104, 103, 105, 104, 106, 52],
        "Low": [99, 98, 100, 100, 101, 104, 104, 104, 49],
        "Close": [100.5, 99.5, 102, 103, 102.5, 104.5, 104, 105.5, 51],
    })
    rep = validate(raw)
    s = rep.summary.set_index("check")["rows_affected"]
    assert s["duplicate_date"] == 2 and s["out_of_order"] == 1
    assert s["missing_ohlc"] == 1 and s["unparseable_date"] == 1
    assert s["flat_bar_O=H=L=C"] == 1                 # 2020-01-08
    out, log = clean(raw)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert pd.Timestamp("2020-01-07") not in out.index      # missing Open -> dropped
    assert pd.Timestamp("2020-01-08") not in out.index      # flat bar -> dropped
    assert out.loc["2020-01-03", "Close"] == 103            # keep LAST duplicate
    assert len(log) >= 4
    flags = flag_suspicious(out)
    assert "extreme_return(>10%)" in flags["flags"].iloc[-1]   # 102.5 -> 51 is flagged, not dropped


# ------------------------------------------------------------------------- look-ahead
def test_features_are_causal():
    df = make_synthetic_nifty(end="2012-12-31")
    full = add_features(df, CFG)
    cut = 1500
    part = add_features(df.iloc[:cut], CFG)
    for col in ["ret", "vol", "z", "pct_thr"]:
        a, b = full[col].iloc[:cut], part[col]
        assert np.allclose(a.fillna(-999), b.fillna(-999)), f"{col} uses future data"


def test_vector_and_rowwise_event_flags_agree():
    df = make_synthetic_nifty(end="2012-12-31")
    for mode in ("fixed", "zscore", "percentile"):
        c = CFG.replace(event_mode=mode)
        feat = add_features(df, c)
        vec = event_flag(feat, c)
        row = pd.Series([is_event_row(r.ret, r.z, r.pct_thr, c) for r in feat.itertuples()], index=feat.index)
        assert (vec == row).all(), mode


def test_forward_returns_by_hand():
    idx = pd.bdate_range("2021-01-04", periods=6)
    df = pd.DataFrame({"Open": [100, 98, 97, 99, 101, 102.0], "High": 105.0, "Low": 90.0,
                       "Close": [100, 96, 98, 100, 101, 103.0]}, index=idx)
    feat = add_features(df, CFG)
    fw = forward_table(feat, 2, CFG.replace(holding_days=2, entry="next_open"))
    # event on day 1 (close 96): enter day-2 open 97, exit day-3 close 100
    assert np.isclose(fw["gross"].iloc[1], 100 / 97 - 1)
    assert fw["entry_date"].iloc[1] == idx[2] and fw["exit_date"].iloc[1] == idx[3]
    fw2 = forward_table(feat, 2, CFG.replace(holding_days=2, entry="event_close"))
    assert np.isclose(fw2["gross"].iloc[1], 100 / 96 - 1)
    assert np.isnan(fw["gross"].iloc[-1]) and np.isnan(fw["gross"].iloc[-2])   # no future data invented


def test_decluster_gap():
    ev = pd.DataFrame({"pos": [10, 12, 15, 16, 30, 33, 34]})
    kept = decluster(ev, 5)["pos"].tolist()
    assert kept == [10, 15, 30]
    assert all(b - a >= 5 for a, b in zip(kept, kept[1:]))


# -------------------------------------------------------------------------- statistics
def test_multiple_testing_adjustments():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm_adjust(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(bh_adjust(p), [0.02, 0.04, 0.04, 0.02])
    assert np.isnan(holm_adjust([np.nan, 0.01])[0])


def test_engine_detects_injected_effect_and_not_null():
    real = make_synthetic_nifty(seed=3, bounce=0.02)
    null = make_synthetic_nifty(seed=3, bounce=0.0)
    e_real = run_experiment(add_features(real, CFG), CFG.replace(holding_days=1, decluster=False))
    assert e_real.tests["excess_mean_gross"] > 0.004
    assert e_real.tests["p_excess(one-sided t)"] < 0.01
    e_null = run_experiment(add_features(null, CFG), CFG.replace(holding_days=1, decluster=False))
    assert e_null.tests["p_excess(one-sided t)"] > 0.01


def test_config_driven_no_code_change():
    feat = add_features(make_synthetic_nifty(seed=5), CFG)
    a = run_experiment(feat, CFG.replace(threshold=-0.015), tests=False)
    b = run_experiment(feat, CFG.replace(threshold=-0.03), tests=False)
    assert len(a.events_raw) > len(b.events_raw) > 0
    c = run_experiment(feat, CFG.replace(holding_days=10), tests=False)
    assert c.h == 10 and len(c.events) <= len(a.events)


def test_oos_split_has_no_overlap():
    cfg = CFG.replace(split_date="2016-01-01")
    feat = add_features(make_synthetic_nifty(seed=5), cfg)
    end = research_end(feat, cfg)
    r = run_research(feat, cfg, tests=False)
    o = run_oos(feat, cfg, tests=False)
    assert r.events_raw["exit_date"].max() <= end            # research trades never touch OOS prices
    assert o.events_raw.index.min() >= pd.Timestamp(cfg.split_date)
    assert set(r.events_raw.index).isdisjoint(o.events_raw.index)


def test_grid_shape_and_adjustment_monotone():
    cfg = CFG.replace(threshold_grid=(-0.015, -0.02), holding_grid=(1, 5))
    feat = add_features(make_synthetic_nifty(seed=5), cfg)
    g = robustness_grid(feat, cfg)
    assert len(g) == 4
    ok = g["p_greater"].notna()
    assert (g.loc[ok, "p_holm"] >= g.loc[ok, "p_greater"] - 1e-12).all()
    assert (g.loc[ok, "p_bh"] <= g.loc[ok, "p_holm"] + 1e-12).all()


# --------------------------------------------------------------------------- backtest
def _zero_cost(cfg):
    return cfg.replace(commission_bps=0.0, slippage_bps=0.0)


def test_backtest_matches_research_engine_when_costless():
    for entry in ("next_open", "event_close"):
        cfg = _zero_cost(CFG.replace(holding_days=3, entry=entry))
        feat = add_features(make_synthetic_nifty(seed=11, end="2016-12-31"), cfg)
        exp = run_experiment(feat, cfg, tests=False)
        res = EventDrivenBacktester(feat, cfg).run()
        tr = res.trades[~res.trades["truncated"]].reset_index(drop=True)
        ev = exp.events.reset_index()
        assert len(tr) == len(ev), (entry, len(tr), len(ev))
        assert np.allclose(tr["gross_ret"].to_numpy(), ev["gross"].to_numpy())
        assert (pd.to_datetime(tr["signal_date"]).to_numpy() == ev["Date"].to_numpy()).all()


def test_backtest_equity_identity_and_costs():
    cfg = CFG.replace(holding_days=5)
    feat = add_features(make_synthetic_nifty(seed=2, end="2015-12-31"), cfg)
    res = EventDrivenBacktester(feat, _zero_cost(cfg)).run()
    assert np.isclose(res.equity.iloc[-1], cfg.initial_capital * np.prod(1 + res.trades["net_ret"]))
    res_c = EventDrivenBacktester(feat, cfg.replace(commission_bps=5, slippage_bps=10)).run()
    assert res_c.equity.iloc[-1] < res.equity.iloc[-1]        # costs always hurt
    # single position at a time, no overlapping trades
    t = res.trades
    assert (pd.to_datetime(t["entry_date"]).iloc[1:].to_numpy() > pd.to_datetime(t["exit_date"]).iloc[:-1].to_numpy()).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
