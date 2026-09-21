"""Reproduce every table / figure in results/.

    python run_pipeline.py                      # research phase only (safe default)
    python run_pipeline.py --stage oos          # ALSO evaluate the unseen out-of-sample period
    python run_pipeline.py --synthetic          # offline smoke test on fake data (NOT findings)
    python run_pipeline.py --config my.yaml

Out-of-sample discipline: the default stage never touches data on/after
``split_date``.  Run ``--stage oos`` once, after the parameters in the config
have been frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from niftyresearch import Config  # noqa: E402
from niftyresearch.backtest import EventDrivenBacktester  # noqa: E402
from niftyresearch.data import prepare_data  # noqa: E402
from niftyresearch.events import add_features  # noqa: E402
from niftyresearch.plots import (plot_equity, plot_event_paths, plot_grid_heatmap,  # noqa: E402
                                 plot_return_distributions)
from niftyresearch.research import (by_vol_regime, by_year, compare_periods,  # noqa: E402
                                    falsification_scorecard, research_end, robustness_grid,
                                    robustness_variants, run_oos, run_research)
from niftyresearch.synthetic import make_synthetic_nifty  # noqa: E402

pd.options.display.float_format = "{:,.5f}".format
pd.options.display.width = 200
pd.options.display.max_columns = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", choices=["research", "oos"], default="research")
    ap.add_argument("--synthetic", action="store_true", help="use fake data (pipeline test only)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config)
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    if a.synthetic:
        clean_df = make_synthetic_nifty()
        print("!! SYNTHETIC DATA - pipeline test only, not NIFTY results !!")
    else:
        d = prepare_data(cfg)
        clean_df = d["clean"]
        print(d["provenance"], "\n")
        print(d["report"].summary.to_string(index=False), "\n")
        d["coverage"].to_csv(out / "coverage_by_year.csv")
    feat = add_features(clean_df, cfg)
    end_r = research_end(feat, cfg)
    print(f"Research period ends {end_r.date()}; out-of-sample starts {cfg.split_date}\n")

    # ------------------------------------------------- 2-4. research sample only
    exp = run_research(feat, cfg)
    exp.summary_table().to_csv(out / "summary_research.csv")
    exp.event_table().to_csv(out / "events_research.csv")
    pd.Series(exp.tests).to_csv(out / "tests_research.csv", header=["value"])
    print("== Summary (research sample) ==")
    print(exp.summary_table()[["n", "mean", "median", "std", "win_rate", "skew", "excess_kurt", "mean_net"]])
    print("\n== Tests ==")
    print(pd.Series(exp.tests).to_string())

    grid = robustness_grid(feat, cfg, end=end_r)
    grid.to_csv(out / "grid_research.csv", index=False)
    var = robustness_variants(feat, cfg, end=end_r)
    var.to_csv(out / "variants_research.csv")
    by_year(exp).to_csv(out / "by_year_research.csv")
    by_vol_regime(exp).to_csv(out / "by_vol_regime_research.csv")
    print("\n== Robustness variants ==")
    print(var)
    print(f"\nGrid: {len(grid)} cells; Holm p<{cfg.alpha}: {(grid.p_holm < cfg.alpha).sum()}, "
          f"nominal p<{cfg.alpha}: {(grid.p_greater < cfg.alpha).sum()}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    plot_return_distributions(exp, ax[0])
    plot_event_paths(feat, exp, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out / "fig_distribution_and_paths.png", dpi=130)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_grid_heatmap(grid, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "fig_grid_heatmap.png", dpi=130)

    # -------------------------------------------------------- 5-6. OOS + falsification
    oos = None
    if a.stage == "oos":
        oos = run_oos(feat, cfg)
        cmp_ = compare_periods(exp, oos)
        cmp_.to_csv(out / "research_vs_oos.csv")
        print("\n== Research vs Out-of-sample ==")
        print(cmp_.to_string())
    score = falsification_scorecard(exp, grid, var, oos)
    score.to_csv(out / "falsification_scorecard.csv", index=False)
    print("\n== Falsification scorecard ==")
    print(score.to_string(index=False))

    # ---------------------------------------------------------------- 7. backtest
    res = EventDrivenBacktester(feat, cfg, end=end_r).run()
    res.trades.to_csv(out / "backtest_trades_research.csv", index=False)
    pd.DataFrame({"strategy": res.metrics, "buy_and_hold": res.benchmark_metrics}).to_csv(
        out / "backtest_metrics_research.csv")
    print("\n== Backtest (research period) ==")
    print(pd.DataFrame({"strategy": res.metrics, "buy_and_hold": res.benchmark_metrics}).to_string())
    plot_equity(res)
    plt.gcf().savefig(out / "fig_equity_research.png", dpi=130)
    if a.stage == "oos":
        res_o = EventDrivenBacktester(feat, cfg, start=cfg.split_date).run()
        res_o.trades.to_csv(out / "backtest_trades_oos.csv", index=False)
        pd.DataFrame({"strategy": res_o.metrics, "buy_and_hold": res_o.benchmark_metrics}).to_csv(
            out / "backtest_metrics_oos.csv")
        print("\n== Backtest (out-of-sample, frozen parameters) ==")
        print(pd.DataFrame({"strategy": res_o.metrics, "buy_and_hold": res_o.benchmark_metrics}).to_string())
        plot_equity(res_o)
        plt.gcf().savefig(out / "fig_equity_oos.png", dpi=130)
    print(f"\nAll outputs written to {out}/")


if __name__ == "__main__":
    main()
