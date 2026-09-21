"""A deliberately small event-driven backtester (an extension of the research,
NOT a framework).

The loop walks the price history one bar at a time and, at each bar, may only
act on what is known at that moment:

    on bar OPEN  : fill a pending entry order (placed after yesterday's close)
    on bar CLOSE : (1) exit if the holding period is over
                   (2) mark the portfolio to market
                   (3) if flat, look at TODAY's return / features; if it is an event,
                       place an entry order for tomorrow's open (or fill now for
                       the optimistic ``event_close`` mode)

Rules
-----
* long-only, 1 unit of capital fully invested, at most one open position
  (this is the trading equivalent of de-clustering),
* costs are charged on every fill: ``slippage`` moves the fill price against us,
  ``commission`` is a fee on the traded notional,
* cash earns 0 % (conservative), no dividends, no leverage, no margin/roll costs
  (a real implementation would use NIFTY futures or an index ETF such as NIFTYBEES).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .events import is_event_row


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    trades: pd.DataFrame
    in_market: pd.Series
    metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]


class EventDrivenBacktester:
    def __init__(self, feat: pd.DataFrame, cfg: Config, start=None, end=None):
        self.cfg = cfg
        bars = feat
        if start is not None:
            bars = bars[bars.index >= pd.Timestamp(start)]
        if end is not None:
            bars = bars[bars.index <= pd.Timestamp(end)]
        if len(bars) < cfg.holding_days + 2:
            raise ValueError("Not enough bars for a backtest in the requested window.")
        self.bars = bars

    def run(self) -> BacktestResult:
        cfg, bars = self.cfg, self.bars
        h = cfg.holding_days
        comm, slip = cfg.commission_bps / 1e4, cfg.slippage_bps / 1e4

        cash, units = cfg.initial_capital, 0.0
        in_pos, entry_idx, pending = False, -1, False
        eq0 = 0.0
        signal_date = entry_date = None
        raw_entry = fill_entry = np.nan
        equity, flag_in, trades = [], [], []
        n = len(bars)

        def open_position(i, price, date):
            nonlocal cash, units, in_pos, entry_idx, eq0, entry_date, raw_entry, fill_entry
            fill = price * (1 + slip)
            fee = cash * comm
            eq0 = cash
            units, cash = (cash - fee) / fill, 0.0
            in_pos, entry_idx, entry_date = True, i, date
            raw_entry, fill_entry = price, fill

        def close_position(i, price, date, truncated):
            nonlocal cash, units, in_pos
            fill = price * (1 - slip)
            proceeds = units * fill
            cash = proceeds - proceeds * comm
            trades.append(dict(
                signal_date=signal_date, entry_date=entry_date, exit_date=date,
                entry_px=fill_entry, exit_px=fill, bars_held=i - entry_idx + (1 if cfg.entry == "next_open" else 0),
                gross_ret=price / raw_entry - 1.0, net_ret=cash / eq0 - 1.0, truncated=truncated))
            units, in_pos = 0.0, False

        for i, (dt, row) in enumerate(bars.iterrows()):
            # ---- bar open: fill a pending entry order
            if pending and not in_pos:
                open_position(i, row["Open"], dt)
                pending = False

            # ---- bar close: exit if the holding period has elapsed
            if in_pos:
                held = i - entry_idx + (1 if cfg.entry == "next_open" else 0)
                if held >= h or i == n - 1:
                    close_position(i, row["Close"], dt, truncated=held < h)

            # ---- mark to market at the close
            equity.append(cash + units * row["Close"])
            flag_in.append(in_pos)

            # ---- signal on today's close (only information known now)
            if not in_pos and not pending and i < n - 1:
                if is_event_row(row["ret"], row["z"], row["pct_thr"], cfg):
                    signal_date = dt
                    if cfg.entry == "next_open":
                        pending = True
                    else:                                   # optimistic: fill at today's close
                        open_position(i, row["Close"], dt)
                        equity[-1] = cash + units * row["Close"]
                        flag_in[-1] = True

        idx = bars.index
        equity = pd.Series(equity, index=idx, name="equity")
        in_market = pd.Series(flag_in, index=idx, name="in_market")
        bench = cfg.initial_capital * bars["Close"] / bars["Close"].iloc[0]
        bench.name = "buy_and_hold"
        trades_df = pd.DataFrame(trades)
        return BacktestResult(
            equity=equity, benchmark=bench, trades=trades_df, in_market=in_market,
            metrics=performance_metrics(equity, cfg.initial_capital, trades_df, in_market),
            benchmark_metrics=performance_metrics(bench, cfg.initial_capital, None, None),
        )


def max_drawdown(equity: pd.Series):
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()), dd.idxmin(), dd


def performance_metrics(equity: pd.Series, initial: float, trades: Optional[pd.DataFrame],
                        in_market: Optional[pd.Series]) -> Dict[str, float]:
    rets = equity.pct_change()
    rets.iloc[0] = equity.iloc[0] / initial - 1.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / initial - 1.0
    mdd, mdd_date, _ = max_drawdown(equity)
    sd = rets.std()
    cagr = (equity.iloc[-1] / initial) ** (1 / years) - 1.0
    m = dict(
        total_return=total, cagr=cagr,
        ann_vol=sd * np.sqrt(252),
        sharpe=(rets.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        max_drawdown=mdd, max_drawdown_date=mdd_date,
        calmar=(cagr / abs(mdd)) if mdd < 0 else np.nan,
    )
    if trades is not None and in_market is not None:
        m["n_trades"] = int(len(trades))
        m["exposure_fraction_of_days"] = float(in_market.mean())
        if len(trades):
            m["win_rate_net"] = float((trades["net_ret"] > 0).mean())
            m["avg_trade_net"] = float(trades["net_ret"].mean())
            m["median_trade_net"] = float(trades["net_ret"].median())
            gains, losses = trades.loc[trades.net_ret > 0, "net_ret"].sum(), -trades.loc[trades.net_ret < 0, "net_ret"].sum()
            m["profit_factor"] = float(gains / losses) if losses > 0 else np.inf
            m["worst_trade"] = float(trades["net_ret"].min())
            m["trades_truncated_at_end"] = int(trades["truncated"].sum())
    return m
