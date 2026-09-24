# Methodology — trade-sentiment-vs-price

## The small-indicators methodology

This module implements the suite's standing methodology: **build many small
tools/outputs/indicators, then combine their outputs later to drive trading
ideas and decisions.** Each function here is one measurement with one honest
number (estimate + p-value + n). Nothing here places trades. Composition
happens downstream — in `trade-agents` (reasoning), `trade-backtest`
(simulation), or `trade-factors` (a sentiment factor series).

## What "sentiment vs price" means

Five sub-questions, one tool each:

1. **Correlation** — contemporaneous Pearson/Spearman plus a lead/lag
   profile. Lag `L` = corr(sentiment[t+1], returns[t+L]); `L>0` ⇒ sentiment
   leads. A peak at positive lag is necessary (not sufficient) for
   predictiveness.
2. **Event study** — do prices drift *after* sentiment bursts? Bursts =
   causal trailing z-score on mentions (or |Δ sentiment|) > threshold.
   Abnormal return = realized − estimation-window mean; CAR averaged over
   events with a cross-event t-stat. The classic Fama-style test.
3. **Information coefficient** — corr(score_t, forward return), the
   quant-standard gauge of a signal's predictive power, with rank IC,
   forward OLS (optional lagged-return control), and sign accuracy.
4. **Indicators** — per-day building blocks: sentiment momentum, attention
   z-score, sentiment z-score, price-vs-sentiment divergence, and a
   composite bundle. Rolling IC tracks whether predictiveness is stable
   or a one-off regime.
5. **Signal evaluation** — two toy rules (sentiment z-score, attention
   burst + sentiment direction) scored on forward returns: hit rate, mean
   trade, t-stat. A sniff test before spending backtest budget.

## Statistics notes

- **Exact p-values.** Correlation and OLS t-stats use the Student-t CDF via
  the regularized incomplete beta function — no normal approximations.
- **Causality.** Burst detection and rule signals use only data available
  before the signal date. The demo DGP bakes sentiment→returns with a
  1-day lead, which the lead/lag tool recovers.
- **Multiple comparisons.** Scanning lags −5…+5 is 11 tests; treat the
  "best lag" p-value as exploratory unless Bonferroni/Holm-adjusted
  downstream.
- **In-sample.** IC, regressions, and rule stats are in-sample. Honest
  out-of-sample validation belongs in `trade-backtest`.

## What this is not

- Not a sentiment engine (that's `trade-sentiment` — this consumes it).
- Not a backtester (that's `trade-backtest` — this feeds it signals).
- Not investment advice. Research tooling only.
