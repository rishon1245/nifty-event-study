# Research Note — Do NIFTY 50 falls get recovered?

## 1. Hypothesis
**H1.** After a NIFTY 50 close-to-close fall of at least 2%, a long position bought at the next session's open and sold at the close 5 trading days after the fall earns a positive return that exceeds the normal NIFTY 50 5-day return (same entry/exit rule applied to every day of the period). **H0.** The mean 5-day return after such falls is no greater than the normal return. The primary specification was fixed in `config.yaml` before the out-of-sample period was evaluated and never changed; results reproduce with `python run_pipeline.py`.

## 2. Data
**Source:** Yahoo Finance `^NSEI` (cached at `data/raw/nifty50_daily.csv`). **Range:** 2007-09-17 to 2026-09-21. **Fields:** Open, High, Low, Close. **Rows:** 4,664 raw, 4,664 clean. **Checks:** unparseable/duplicate dates, ordering, missing or non-positive OHLC, flat bars, High/Low consistency, coverage per year. No rows were removed and no High/Low repairs were needed (cleaning log empty). 9 unusual observations were flagged and kept (`results/suspicious_observations.csv`); all are genuine market days (e.g. 2008-10-24, 2009-05-18, 2020-03-23) and are exactly the events under study. **Periods:** research 2007-09-17 to 2017-12-29; out-of-sample (OOS) from 2018-01-01 (each trade needs a full 5-day window, so usable events end slightly earlier).

## 3. Definitions and design
| Element | Definition | Reason |
|---|---|---|
| Event | close-to-close return ≤ −2% | substantial yet frequent; fixed, not tuned |
| Recovery | 5-day return positive net of costs **and** above the normal 5-day return | a positive return alone is just market drift |
| Entry | next session's **open** | the fall is only known at the close; avoids look-ahead |
| Exit / holding | close of day t+5 (5 sessions) | about one trading week |
| Independence | drop events within 5 trading days of the last kept event | overlapping windows are not independent |
| Test period | research < 2018-01-01 ≤ OOS, chronological | no leakage; OOS untouched until frozen |
| Costs | 2 bps commission + 3 bps slippage per side = 10 bps round trip (stress: 20, 40) | futures/ETF proxy |
| Assumptions | OHLC accurate; open executable; constant costs; long-only, one position, cash earns 0, daily bars | simplest honest set |

## 4. Statistical evidence (research sample)
145 raw events; **92 independent** after de-clustering.

| | After falls | Baseline (all days) |
|---|---:|---:|
| Mean 5-day return (gross) | 0.070% | 0.131% |
| Median | 0.037% | 0.249% |
| Standard deviation | 4.726% | 3.273% |
| Win rate | 50.00% | 54.34% |
| Mean net of costs | −0.030% | — |

Excess over baseline **−0.061%**; 95% bootstrap CI **[−1.045%, +0.934%]**, which includes zero. One-sided p for "excess > 0": t-test 0.549, Wilcoxon 0.522, block bootstrap 0.544; two-sided t-test 0.902; placebo (random day-sets doing at least as well) 0.581, or 0.502 against random *high-volatility* days. *Why these methods:* the t-test leans on the CLT, so Wilcoxon and the bootstrap check it under fat tails; the block bootstrap respects overlapping baseline windows; the placebo asks whether random days look this good. Post-fall returns are no better than normal, more volatile, and negative after costs: no statistical and no economic significance.

## 5. Baseline and robustness
A 36-cell threshold × holding-period grid (one family of tests) had **0 cells with p < 0.05, 0 after Holm, 0 after Benjamini–Hochberg**. Changing one assumption at a time gave excess returns of: no de-clustering +0.137%; entry at event-day close −0.115%; volatility-adjusted event +0.591%; excluding GFC/COVID −0.086%; mean net at 20 / 40 bps round trip −0.130% / −0.330%. None was significant at 5% and the sign flips across variants, so there is no stable effect. The volatility-adjusted variant is exploratory (different event definition, one of several tried) and is **not** promoted; choosing it now would be post-hoc selection. Snooping controls: fixed primary specification, Holm/BH over the whole grid, no best-cell selection, OOS opened once.

## 6. Out-of-sample validation (from 2018-01-01)
| | Research | Out-of-sample |
|---|---:|---:|
| Independent events | 92 | 39 |
| Mean gross / net return | +0.070% / −0.030% | −0.107% / −0.207% |
| Baseline mean | +0.131% | +0.110% |
| Excess vs baseline | −0.061% | −0.217% |
| Event / baseline win rate | 50.00% / 54.34% | 48.72% / 53.41% |

OOS one-sided p = 0.613; 95% CI [−1.805%, +1.125%]. The research excess was already about zero, so there was no effect to persist; OOS again shows **no positive effect** (slightly negative, statistically indistinguishable from zero). What changed: the market was calmer (annualised volatility 16.9% vs 23.2%), so −2% falls were rarer (6.3 vs 14.1 per year) and there are fewer events (39 vs 92, hence a wider interval); baseline drift was similar.

## 7. Challenging the result
*Rejection criteria (set before evaluating OOS):* CI of the excess includes zero; mean net return ≤ 0; placebo p ≥ 0.05; primary cell fails Holm correction; effect disappears without crisis windows, without de-clustering or under a volatility-adjusted event; OOS net return or excess ≤ 0. **The main criteria were met:** CI [−1.045%, +0.934%] includes zero, mean net return −0.030%, both placebo p-values above 0.5, Holm-adjusted p = 1.0, and OOS net return (−0.207%) and excess (−0.217%) negative, so **H1 is not supported**. Two variants showed a positive excess (no de-clustering, volatility-adjusted) but neither was significant (p ≥ 0.17).
*Risks:* look-ahead (features use data up to the close only, unit-tested by truncating future data; entry at next open); costs (0/20/40 bps stress); overlap (de-clustering, block bootstrap); regimes (crisis exclusion, volatility-adjusted event); data quality (validation checks, empty cleaning log, 9 flagged days that are real market events). *Sample size:* with σ ≈ 4.7% and n = 92, the smallest excess detectable with 80% power is about 1.2% per trade, so failing to find an effect is not proof that none exists.

## 8. Backtest, conclusion and limitations
**Backtest** (event-driven, one position, costs and slippage on every fill). Research: 92 trades, average net trade −0.030%, total return −12.3%, max drawdown −52.6% (buy-and-hold +134%, −59.9%). OOS: 39 trades, average net trade −0.207%, total return −11.9%, max drawdown −35.5% (buy-and-hold +124%, −38.4%). It illustrates the rule and is not evidence of a deployable strategy.
**Conclusion.** The hypothesis is **not supported** (not the same as disproved). With a −2% threshold, next-open entry and 5-day holding, the research excess was −0.061% with a CI spanning zero; tests, placebos and the Holm-corrected grid show nothing; net returns are negative after costs. OOS gave excess −0.217% and net −0.207%: again no positive effect.
**Limitations.** Few independent events (falls cluster in 2008–09 and 2020); one index, one split, one primary specification; futures/ETF fills at open/close assumed; daily bars; Yahoo Finance is not an official source.
