# Architecture — trade-sentiment-vs-price

```
src/trade_sentiment_vs_price/
├── __init__.py        public API + __version__
├── __main__.py        python -m entry
├── cli.py             correlate|events|ic|signals|indicators|report|license|update-check
├── series.py          SentimentPoint, PriceBar, AlignedSeries, align(), demo_data()
├── stats.py           pearson/spearman, exact t p-values (incomplete beta),
│                      causal trailing z-scores
├── ols_lite.py        minimal OLS (params, SEs, t, p, R²) for forward regressions
├── correlation.py     contemporaneous(), lead_lag(), rolling_correlation()
├── events.py          detect_bursts(), event_study()  (CAR + t-stat)
├── predict.py         information_coefficient(), forward_regression(),
│                      sign_accuracy()
├── indicators.py      sentiment_momentum(), attention_zscore(), sentiment_zscore(),
│                      divergence(), composite_attention(), rolling_ic_stability()
├── signals.py         zscore_rule(), attention_rule(), summarize_trades()
├── adapters.py        from_sentiment_scan, prices_from_dict_bars,
│                      to_agent_report, to_backtest_signals,
│                      sentiment_mimicking_returns   (all lazy — no sibling imports)
└── licensing.py       license-key + update-check hooks (suite convention)
tests/test_svp.py      26 tests, stdlib unittest, deterministic seeds
examples/svp_example.py  end-to-end demo on synthetic data
```

## Data flow

```
trade-sentiment JSON ──from_sentiment_scan──┐
trade-data-* bars ──prices_from_dict_bars───┤
                                            ▼
                                   align() → AlignedSeries
                                   (inner join on date, close→returns)
                                            │
            ┌───────────────┬───────────────┼───────────────┬───────────────┐
            ▼               ▼               ▼               ▼               ▼
      correlation.py   events.py      predict.py     indicators.py    signals.py
      lead/lag,        bursts →       IC, fwd        momentum,        z/attention
      rolling corr     CAR            regression     attention,       rules →
                                     sign acc       divergence       trade stats
            └───────────────┴───────────────┴───────────────┴───────────────┘
                                            ▼
                              to_agent_report() → trade-agents
                              to_backtest_signals() → trade-backtest
                              sentiment_mimicking_returns() → trade-factors
```

## Key decisions

- **Stdlib only.** Exact p-values need a t CDF; instead of scipy we
  implement the regularized incomplete beta function (~40 lines). Keeps the
  package dependency-free and the stats auditable.
- **Causal windows.** Every trailing computation (`trailing_zscore`,
  burst detection, rule signals) uses history strictly before the signal
  date. `AlignedSeries.paired_returns()` documents the date alignment so
  off-by-one lead/lag errors are visible.
- **Adapters, not imports.** Sibling modules are never imported at module
  load; adapters translate plain dicts/dataclasses. The package installs,
  imports, and tests standalone.
- **Deterministic.** `demo_data(seed=…)` uses its own `random.Random`; tests
  never touch global RNG state.
- **One honest number per tool.** Each public function returns a small dict
  with the estimate, its p-value, and n — the "small indicator" contract.

## Scaling

- Per-symbol `AlignedSeries` are independent → embarrassingly parallel over
  a symbol panel (multiprocessing or chunked loops; no shared state).
- All hot loops are O(n·window); the beta continued fraction converges in
  < 200 iterations. A 10-year daily panel analyzes in milliseconds.
- For intraday bars the same code works — just pass finer-grained dates.
