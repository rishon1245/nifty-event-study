# NIFTY 50: does the market recover after a big one-day fall?

A single research investigation (AlgoChowk Quant Engineer assignment) with a configurable event-study engine,
statistical evidence, robustness / out-of-sample checks, a falsification scorecard and a small event-driven backtest.

> **Hypothesis.** After a NIFTY 50 close-to-close fall of at least 2 %, buying at the next session's open and holding
> 5 trading days earns a positive net return that exceeds the return of the same trade opened on an ordinary day.

> **Status:** The analysis has been run on real NIFTY 50 data from Yahoo Finance.
> Research and out-of-sample results are reported below. Synthetic mode is used only
> for offline pipeline smoke tests and is not used for the findings.

---------------------------------------------------------------------------------------------------

## 1. Quick start

### Install dependencies

```bash
pip install -r requirements.txt
pytest -q
python run_pipeline.py
python run_pipeline.py --stage oos
```

No internet / Yahoo blocked: save a NIFTY 50 CSV with columns `Date,Open,High,Low,Close` (e.g. the historical-data
export on niftyindices.com - it caps each request to ~1 year, so download in chunks and concatenate; duplicates are
handled by the cleaner) and set `csv_path:` in `config.yaml`.

## 2. Repository layout

```
config.yaml                 every experiment parameter (threshold, holding period, entry rule, costs, split, ...)
niftyresearch/
  config.py                 typed, validated config (Config.replace(...) for one-off variations)
  data.py                   load -> validate -> clean -> flag -> coverage, with a cleaning log
  events.py                 causal features, event flags, forward-return table, de-clustering
  stats.py                  descriptive stats, t / Wilcoxon / binomial, (block) bootstrap, placebo, Holm / BH
  research.py               run_experiment, robustness grid / variants, by-year / by-regime, OOS, falsification scorecard
  backtest.py               small event-driven backtester (bar-by-bar loop)
  plots.py, synthetic.py    figures; fake data generator for offline smoke tests
run_pipeline.py             CLI that reproduces all tables and figures in results/
notebooks/nifty_event_study.ipynb   the narrative, following the assignment sections 1-8
tests/test_core.py          look-ahead, hand-computed returns, de-clustering, multiple testing, backtest == engine, ...
docs/                       research-note and AI-usage-note templates
```

## 3. Data 

* **Source:** Yahoo Finance `^NSEI` (NIFTY 50) via `yfinance`, or a local CSV (`csv_path`). Yahoo history starts 2007-09-17.
* **Fields:** Date, Open, High, Low, Close (daily index levels; an index has no splits/dividends, so no adjustment).
* **Cleaning (all logged to `results/cleaning_log.csv`):** drop unparseable dates / missing / non-positive OHLC;
  keep the last of duplicated dates; sort ascending; drop flat `O=H=L=C` bars (stale / non-standard sessions);
  clip High/Low to contain Open & Close (the study only uses Open & Close).
* **Flagged, never dropped** (`results/suspicious_observations.csv`): |daily return| > 10 %, |open gap| > 5 %,
  calendar gaps > 5 days, weekend sessions, 3+ identical closes. Genuine extremes (2009-05-18, 2020-03-23) are
  exactly the events under study.
* **Coverage** per year is checked against the ~248 sessions NSE normally has (`results/coverage_by_year.csv`).
* Results come from a snapshot ending 2026-09-21 and that `python run_pipeline.py` reproduces them from the committed data. 

## 4. Methodology

| Element | Choice | Rationale |
|---|---|---|
| Event | close-to-close return <= -2 % (also: vol-adjusted `z <= -2` on 60-day trailing vol; expanding-percentile) | simple, reproducible; alternatives test regime dependence |
| Timing | signal known at the close of day *t* -> **enter next open** -> exit close of *t+h* | the only entry price realistically known/executable; `event_close` entry is an optimistic sensitivity |
| Recovery | five-trading-day forward return from next open to exit close | tests whether post-fall returns are positive and exceed normal market forward returns |
| Holding | h = 5 primary; 1-20 in the grid | fixed *before* seeing results |
| Independence | keep the first event, skip events < h days after it | overlapping windows share returns |
| Baseline | every day's forward return using the *same* entry/exit; also non-event days and high-vol non-event days | separates drift and volatility regime from the fall effect |
| Costs | 2 bps fees + 3 bps slippage per side = 10 bps round trip; stress 0/20/40 | futures / ETF proxy |
| Split | research < 2018-01-01 <= out-of-sample | chronological; OOS includes COVID and 2022 |

**Statistics** (each answers a different question - see the notebook): one-sided t-test vs baseline mean, Wilcoxon,
percentile bootstrap CI of the mean, **block bootstrap** CI of the *excess* over baseline (baseline windows overlap),
**placebo test** (random days and random high-vol days), binomial test of win-rate vs baseline, Cohen's d, minimum
detectable effect, break-even cost. Multiple testing over the threshold x holding grid: **Holm** and **Benjamini-Hochberg**.

**Falsification.** `falsification_scorecard()` encodes the rejection criteria (CI of excess includes 0; net mean <= 0;
placebo not significant; disappears without crises / without de-clustering / under vol-adjustment; fails Holm; OOS
excess <= 0 or sign flips). They are written down before results are seen.

**Backtest.** Bar-by-bar loop; entry order after the signal close, filled at next open with slippage + fee; exit at the
close of the holding period; one position at a time; cash earns 0. Reports trades, win rate, average net trade, profit
factor, cumulative equity, CAGR, Sharpe, max drawdown, exposure vs buy-and-hold. A test proves that, with zero costs,
the backtester reproduces the research engine's trades exactly.

## 5. Assumptions and limitations

* NIFTY is an index; tradability is approximated by futures / ETF at the open and close. No roll cost, financing,
  dividends, margin, tax; no intraday stop-losses (daily bars only).
* Falls cluster in 2008-09 and 2020: the number of *independent* episodes is small even when the number of days is not.
* Only NIFTY 50, one split, one primary specification - the conclusion is about this sample, not markets in general.
* Yahoo data are not an official source; cross-check a few dates against NSE Indices if they matter.
* The out-of-sample period is only "unseen" if you do not tune after looking at it. The code separates the stages, but
  discipline is yours.

## 6. Results

The analysis was run using the primary specification:

- Event threshold: **−2% close-to-close**
- Entry: **next trading day's Open**
- Holding period: **5 trading days**
- Independence: **5-day de-clustering**
- Round-trip transaction cost: **10 bps**
- Research period: **2007-09-17 to 2017-12-29**
- Out-of-sample period: **2018-01-01 onward**

### 6.1 Research Sample

The research period produced **145 raw events** and **92 independent events**.

| Metric | Event | Baseline |
|---|---:|---:|
| Mean return | 0.070% | 0.131% |
| Median return | 0.037% | 0.249% |
| Standard deviation | 4.726% | 3.273% |
| Win rate | 50.00% | 54.34% |
| Mean net return | −0.030% | — |
| Excess vs baseline | −0.061% | — |

The 95% bootstrap confidence interval for excess return was:

**−1.045% to +0.934%**

Statistical evidence:

| Test | Result |
|---|---:|
| One-sided t-test for excess return | p = 0.549 |
| Two-sided t-test | p = 0.902 |
| Wilcoxon test | p = 0.522 |
| Block bootstrap | p = 0.544 |
| Random-day placebo | p = 0.581 |
| High-volatility placebo | p = 0.502 |

The research-period results do not provide statistically significant evidence of positive excess returns following the defined −2% events.

### 6.2 Robustness

The threshold × holding-period grid contained **36 cells**.

- Nominal p < 0.05: **0 cells**
- Holm-adjusted p < 0.05: **0 cells**
- BH-adjusted significance: **0 cells**

Selected robustness results:

| Variant | Result |
|---|---:|
| No de-clustering | Excess = +0.137% |
| Event-close entry | Excess = −0.115% |
| Volatility-adjusted event | Excess = +0.591% |
| Excluding GFC/COVID | Excess = −0.086% |
| 0 bps round-trip cost | Mean net = +0.070% |
| 20 bps round-trip cost | Mean net = −0.130% |
| 40 bps round-trip cost | Mean net = −0.330% |

None of the robustness variants produced statistically significant evidence at the 5% level.

### 6.3 Out-of-Sample Results

The primary parameters were frozen after the research period and then evaluated on the unseen period beginning in 2018.

| Metric | Research | Out-of-Sample |
|---|---:|---:|
| Independent events | 92 | 39 |
| Mean gross return | +0.070% | −0.107% |
| Mean net return | −0.030% | −0.207% |
| Baseline mean | +0.131% | +0.110% |
| Excess vs baseline | −0.061% | −0.217% |
| Event win rate | 50.00% | 48.72% |
| Baseline win rate | 54.34% | 53.41% |

OOS statistical results:

- One-sided excess-return p-value: **0.613**
- 95% bootstrap CI: **−1.805% to +1.125%**

The proposed recovery effect did not persist out-of-sample.

### 6.4 Falsification Scorecard

| Criterion | Observed | Verdict |
|---|---:|:---:|
| Independent events ≥ 30 | 92 | PASS |
| Excess CI excludes 0 | [−1.045%, +0.934%] | FAIL |
| Mean net return > 0 | −0.030% | FAIL |
| Random-day placebo p < 0.05 | 0.581 | FAIL |
| High-volatility placebo p < 0.05 | 0.502 | FAIL |
| Primary cell survives Holm correction | 1.000 | FAIL |
| Excess positive excluding crises | −0.086% | FAIL |
| Excess positive without de-clustering | +0.137% | PASS |
| Excess positive under volatility adjustment | +0.591% | PASS |
| OOS mean net return > 0 | −0.207% | FAIL |
| OOS excess > 0 | −0.217% | FAIL |

### 6.5 Backtest

The event-driven backtest was run using the same frozen primary specification.

**Research period:**

- Total return: **−12.32%**
- CAGR: **−1.27%**
- Sharpe: **−0.014**
- Maximum drawdown: **−52.64%**
- Number of trades: **92**
- Profit factor: **0.983**

**Out-of-sample:**

- Total return: **−11.86%**
- CAGR: **−1.44%**
- Sharpe: **−0.096**
- Maximum drawdown: **−35.45%**
- Number of trades: **39**
- Profit factor: **0.867**

The backtest is treated as an extension of the event-study evidence rather than as a strategy-optimization exercise.

### 6.6 Conclusion

Under the predefined −2% event threshold, next-open entry and five-day holding period, the analysis does **not support the hypothesis** that large one-day NIFTY 50 declines are followed by reliable positive five-day excess returns.

The research-period excess return was **−0.061%**, while the out-of-sample excess return was **−0.217%**. Statistical tests, placebo tests, robustness checks and multiple-testing corrections did not provide significant evidence of a recovery effect.

The result should be interpreted within the limitations of the sample, event definition, market regimes, execution assumptions and relatively small number of independent out-of-sample events.
