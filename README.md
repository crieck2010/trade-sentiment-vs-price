> [!IMPORTANT]
> **Consolidated into [TradeSuite](https://github.com/crieck2010/trade-suite)**  
> This engine has been consolidated into the unified [TradeSuite monorepo](https://github.com/crieck2010/trade-suite). Active development, bug fixes, releases, and unified testing now live in `trade-suite`. This repository is preserved as an archive.

# trade-sentiment-vs-price

Does sentiment lead, lag, or just mirror price? Small, honest measurement
tools for the sentiment↔price relationship — built for the "many small
indicators" methodology: each module is one clean measurement, and their
outputs are combined downstream (agents, backtests, factor models) to drive
trading ideas.

Part of the [trade-suite](https://github.com/crieck2010/trade-suite)
algorithmic/agentic trading system. **Research only** — not investment advice,
never live trading.

## What it does

| Tool | Question it answers |
|---|---|
| `correlate` | Contemporaneous correlation + lead/lag profile (−5…+5 days): does sentiment move first? |
| `events` | Sentiment-burst detection → event-study CAR: does the market move *after* the crowd gets loud? |
| `ic` | Information Coefficient, forward regression, sign accuracy: does sentiment forecast returns? |
| `signals` | Evaluate simple sentiment rules (z-score, attention-burst) → hit rate, mean trade, t-stat |
| `indicators` | Small per-day indicator bundle: momentum, attention z, sentiment z, price/sentiment divergence |
| `report` | Full agent-ready JSON: lead/lag verdict, IC, event study, signal eval in one bundle |

## Quick start

```bash
pip install -e .
trade-sentiment-vs-price correlate --demo --symbol DEMO
trade-sentiment-vs-price events --demo --symbol DEMO
trade-sentiment-vs-price ic --demo --symbol DEMO --horizon 1
trade-sentiment-vs-price signals --demo --symbol DEMO
trade-sentiment-vs-price report --demo --symbol DEMO
```

With your own data (CSVs):

```bash
trade-sentiment-vs-price correlate --symbol AAPL \
  --sentiment sentiment.csv --prices prices.csv
```

- `sentiment.csv`: `date,symbol,bullishness,conviction,mentions` (bullishness/conviction on the 0–10 `trade-sentiment` scale; also accepts `bullishness_10`/`conviction_10` headers)
- `prices.csv`: `date,close` (also accepts `Close`/`c`, `Date`/`timestamp`)

Python API:

```python
from trade_sentiment_vs_price import (
    align, lead_lag, information_coefficient, detect_bursts, event_study,
    zscore_rule, summarize_trades, to_agent_report, demo_data,
)

sent, prices = demo_data(n_days=250, seed=11, lead=1, symbol="DEMO")
a = align("DEMO", sent, prices)

ll = lead_lag(a, max_lag=5)
print(ll["best_lag"], ll["verdict"])          # 1  sentiment leads

ic = information_coefficient(a, horizon=1)
print(ic["ic"], ic["pvalue"])                 # IC with exact t p-value

ev = detect_bursts(a, z_thresh=2.0, on="mentions")
es = event_study(a, ev, pre=5, post=10)
print(es["mean_car"], es["pvalue"])           # cumulative abnormal return

rep = to_agent_report("DEMO", sent, prices)   # one JSON bundle for agents
```

## Design

- **Stdlib only.** No numpy/pandas/scipy — every stat (Pearson/Spearman,
  exact t p-values via the incomplete beta function, OLS, event-study
  t-stats) is implemented in pure Python so the engine is dependency-light
  and auditable.
- **Causal by construction.** Burst z-scores, trailing windows, and rule
  signals only ever use history available *before* the signal date —
  no lookahead.
- **Plain-data interop.** Inputs and outputs are lists of dicts/dataclasses,
  so sibling modules plug in without importing each other:
  - ← `trade-sentiment`: `from_sentiment_scan()` parses its JSON rows
  - ← `trade-data-equities` / `trade-data-crypto`: `prices_from_dict_bars()` takes duck-typed bars
  - → `trade-agents`: `to_agent_report()` — one JSON verdict bundle
  - → `trade-backtest`: `to_backtest_signals()` — rule trades as signal rows
  - → `trade-factors`: `sentiment_mimicking_returns()` — long-short sentiment-factor return series
- **Scales** from one symbol (CLI) to a panel: every function takes an
  `AlignedSeries`; loop over symbols and aggregate — no global state.

## Lead/lag convention

Lag `L` = corr(sentiment[t+1], returns[t+L]). `L = 0` is contemporaneous;
`L > 0` means sentiment **leads** (predictive); `L < 0` means it lags.
The demo data bakes in a 1-day lead, which `correlate` recovers.

## Limitations (honest)

- Correlation is not causation; a lead can reflect a third driver (news).
- Event studies assume bursts are exogenous — clustered events overlap.
- IC and rule evaluation are in-sample; use `trade-backtest` for honest OOS.
- Demo data is synthetic AR(1) sentiment — real sentiment is messier.
- Daily granularity; intraday lead/lag needs finer bars.

## Docs

- `docs/ARCHITECTURE.md` — module map and data flow
- `docs/METHODOLOGY.md` — the small-indicators methodology and stats notes
- `CHANGELOG.md` — version history

## The maths

**What you learn.** Whether sentiment moves before price, after it, or
not at all — and whether it's strong enough to matter. The engine measures
five sub-questions: contemporaneous correlation, the lead/lag profile,
event-study drift after sentiment bursts, the Information Coefficient, and
a sniff-test of two toy sentiment rules.

**Why it matters.** "Sentiment drives markets" is a story; "sentiment leads
by one day with IC 0.05" is a measurement you can size a position on — or
reject. Each tool here produces one honest number (estimate + exact p-value
+ n), and composition happens downstream in agents, backtests, or factor
models.

**The maths.**

- *Lead/lag*: L = corr(sentiment[t+1], returns[t+L]) for L in −5…+5; L = 0
  is contemporaneous, L > 0 means sentiment **leads** (predictive). The demo
  DGP bakes in a 1-day lead, which the tool recovers.
- *Event study*: sentiment bursts = causal trailing z-score on mentions (or
  |Δ sentiment|) above threshold; abnormal return = realized − estimation-
  window mean; CAR averaged over events with a cross-event t-stat (Fama-style).
- *Information coefficient*: corr(score_t, forward return) — Pearson and
  rank (Spearman) IC, forward OLS with optional lagged-return control, and
  sign accuracy.
- *Inference*: correlation and OLS t-stats use the exact Student-t CDF via
  the regularized incomplete beta function — no normal approximations.
- *Causality*: burst z-scores, trailing windows, and rule signals only ever
  use history available *before* the signal date — no lookahead.

**Honest limitations.**

- Correlation is not causation; a lead can reflect a third driver (news).
- Scanning lags −5…+5 is 11 tests — treat the "best lag" p-value as
  exploratory unless Bonferroni/Holm-adjusted downstream.
- IC and rule evaluation are in-sample; honest out-of-sample validation
  belongs in `trade-backtest`.
- Event studies assume bursts are exogenous — clustered events overlap.

## License

MIT. `trade-sentiment-vs-price license` / `update-check` hooks included per
suite convention.
