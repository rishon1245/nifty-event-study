# AI Usage Note

## Tools Used

I used **ChatGPT** and **Claude** during this project mainly as a coding and research assistant. I used it to understand the assignment, clarify statistical concepts, review parts of the Python code, debug issues, and improve the documentation.

## How I Used AI

I asked ChatGPT questions about how to structure the event study, how to avoid look-ahead bias, how to calculate and interpret returns, and how to use tests such as bootstrap, placebo tests and multiple-testing corrections.

I also used it to review the results produced by my code and help explain what the numbers meant. I did not simply use an AI-generated result as the final answer. I ran the code myself and used the actual outputs from the project for the research note and README.

## My Own Decisions

I made the main research choices for the experiment:

- **Event:** NIFTY 50 falls by at least 2% in one day.
- **Entry:** Next trading day's open.
- **Holding period:** 5 trading days.
- **De-clustering:** 5 days between independent events.
- **Transaction costs:** 10 bps round trip.
- **Research/OOS split:** 2018-01-01.
- **Recovery:** Compare the five-day forward return after an event with the normal-market baseline.

The main reason for these choices was to keep the experiment simple, realistic and reproducible. I kept the primary settings fixed instead of choosing the combination that gave the best backtest result.

## Suggestions I Changed or Disagreed With

One important choice was using the **next day's open** instead of entering at the event-day close. The fall is only known after the day's close, so using the event-day close as the main entry would not represent the same information available to a trader.

I also did not pick the best result from the threshold and holding-period grid. The −2% event and 5-day holding period remained the primary specification even when other combinations showed different results.

I kept the out-of-sample period separate and did not use it to tune the strategy.

## Incorrect or Risky Suggestions I Identified

While working on the project, I paid particular attention to look-ahead bias. Features used for identifying events needed to be based only on information available at that point in time.

I also learned that event windows can overlap, meaning the observations cannot always be treated as completely independent. This is why the project uses de-clustering and block-bootstrap analysis.

Another important point was that a low p-value by itself does not mean a strategy is profitable or tradable. Transaction costs, the baseline, robustness tests and out-of-sample results also need to be considered.

## Things I Verified

I ran the project's automated tests and checked the actual outputs from the research pipeline.

I also checked the forward-return calculations, de-clustering behaviour, look-ahead protection, multiple-testing corrections and the consistency between the research engine and the event-driven backtest.

The final results in the research note and README come from the actual runs of the project rather than from AI-generated numbers.

## What I Learned

The biggest thing I learned from this project is that finding a pattern in the market is not the same as finding a reliable trading strategy.

The initial idea that the market might recover after a large fall sounded reasonable, but the actual results did not provide strong evidence for it. The research-period excess return was negative and the out-of-sample result was also negative.

I also learned how important it is to define the experiment before looking at the results, account for costs and dependence, test alternative explanations, and use unseen data to check whether an observed effect actually persists.