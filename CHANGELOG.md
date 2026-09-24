# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### Added
- Initial release.
- `series`: `SentimentPoint`/`PriceBar`/`AlignedSeries`, date inner-join
  `align()`, seeded `demo_data()` with a known 1-day sentiment lead.
- `stats`: Pearson/Spearman, exact t p-values via incomplete beta function,
  causal trailing z-scores. Stdlib-only.
- `correlation`: contemporaneous, lead/lag profile (−5…+5, documented
  convention), rolling correlation.
- `events`: causal burst detection (mentions / sentiment-change) and
  event-study CAR with t-stat.
- `predict`: information coefficient (+ rank IC), forward OLS regression
  with lagged-return control, sign accuracy.
- `indicators`: sentiment momentum, attention z-score, sentiment z-score,
  price/sentiment divergence, composite bundle, rolling IC stability.
- `signals`: z-score rule and attention-burst rule with per-trade stats
  (hit rate, mean return, t-stat).
- `adapters`: `from_sentiment_scan` (trade-sentiment), `prices_from_dict_bars`
  (trade-data-*), `to_agent_report` (trade-agents), `to_backtest_signals`
  (trade-backtest), `sentiment_mimicking_returns` (trade-factors).
- CLI: `correlate`, `events`, `ic`, `signals`, `indicators`, `report`,
  `license`, `update-check`; `--demo` mode and CSV inputs.
- Docs: README, `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md`.
- `examples/svp_example.py` end-to-end demo.
- 26 deterministic tests, all passing.
