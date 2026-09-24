"""Tests for trade-sentiment-vs-price."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trade_sentiment_vs_price import (  # noqa: E402
    align,
    attention_zscore,
    composite_attention,
    contemporaneous,
    demo_data,
    detect_bursts,
    divergence,
    event_study,
    forward_regression,
    from_sentiment_scan,
    information_coefficient,
    lead_lag,
    prices_from_dict_bars,
    rolling_correlation,
    sentiment_mimicking_returns,
    sentiment_momentum,
    sentiment_zscore,
    sign_accuracy,
    summarize_trades,
    to_agent_report,
    to_backtest_signals,
    zscore_rule,
    attention_rule,
)
from trade_sentiment_vs_price import ols_lite, stats  # noqa: E402
from trade_sentiment_vs_price.cli import main as cli_main  # noqa: E402
from trade_sentiment_vs_price.series import PriceBar, SentimentPoint  # noqa: E402


def make_pair(n=200, seed=11, lead=1, symbol="TST"):
    sent, prices = demo_data(n_days=n, seed=seed, lead=lead, symbol=symbol)
    return align(symbol, sent, prices)


class TestStats(unittest.TestCase):
    def test_pearson(self):
        self.assertAlmostEqual(stats.pearson([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(stats.pearson([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertEqual(stats.pearson([1, 1, 1], [1, 2, 3]), 0.0)

    def test_spearman_monotonic(self):
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)

    def test_corr_pvalue(self):
        import random
        rng = random.Random(5)
        xs = [rng.gauss(0, 1) for _ in range(200)]
        ys = [x + rng.gauss(0, 0.2) for x in xs]
        r = stats.pearson(xs, ys)
        self.assertLess(stats.corr_pvalue(r, 200), 0.001)
        self.assertGreater(stats.corr_pvalue(0.01, 200), 0.5)

    def test_trailing_zscore_causal(self):
        zs = stats.trailing_zscore([1.0, 2.0, 1.0, 2.0, 10.0], window=10)
        self.assertEqual(zs[0], 0.0)
        self.assertEqual(zs[1], 0.0)  # single-point history → 0
        self.assertGreater(zs[4], 2.0)  # spike vs trailing history only


class TestSeries(unittest.TestCase):
    def test_align_inner_join(self):
        sent = [SentimentPoint("2026-01-01", "A", 6, 5, 10),
                SentimentPoint("2026-01-02", "A", 7, 5, 12),
                SentimentPoint("2026-01-03", "B", 4, 5, 3)]
        prices = [PriceBar("2026-01-01", 100.0), PriceBar("2026-01-02", 110.0),
                  PriceBar("2026-01-04", 120.0)]
        a = align("A", sent, prices)
        self.assertEqual(a.dates, ["2026-01-01", "2026-01-02"])
        self.assertAlmostEqual(a.returns[0], 0.1)

    def test_align_needs_overlap(self):
        with self.assertRaises(ValueError):
            align("A", [SentimentPoint("2026-01-01", "A", 5, 5, 1)],
                  [PriceBar("2026-02-01", 100.0)])


class TestCorrelation(unittest.TestCase):
    def test_lead_lag_recovers_lead(self):
        a = make_pair()
        ll = lead_lag(a, max_lag=5)
        self.assertEqual(ll["best_lag"], 1)
        self.assertEqual(ll["verdict"], "sentiment leads")
        self.assertLess(ll["best_pvalue"], 0.05)

    def test_contemporaneous_weaker_than_lead(self):
        a = make_pair()
        c = contemporaneous(a)
        ll = lead_lag(a, max_lag=5)
        self.assertLess(abs(c["r"]), abs(ll["best_r"]))

    def test_rolling_length(self):
        a = make_pair(n=100)
        rc = rolling_correlation(a, window=30)
        self.assertEqual(len(rc), 100 - 30)
        self.assertIn("date", rc[0])


class TestEvents(unittest.TestCase):
    def _burst_data(self):
        import random
        rng = random.Random(77)
        n = 300
        sent, prices = demo_data(n_days=n, seed=77, symbol="E")
        # inject mention bursts every 30 days + positive 5-day drift after
        closes = [b.close for b in prices]
        rets = [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, n)]
        for t in range(40, n - 12, 30):
            sent[t].mentions = 500
            for h in range(1, 6):
                rets[t + h] += 0.01
        px = 100.0
        new_prices = []
        for i, (b, r) in enumerate(zip(prices, rets)):
            px = 100.0 * (1 + r) if i == 0 else new_prices[-1].close * (1 + r)
            new_prices.append(PriceBar(b.date, px))
        return align("E", sent, new_prices)

    def test_detect_bursts(self):
        a = self._burst_data()
        ev = detect_bursts(a, z_thresh=2.0, on="mentions")
        self.assertGreaterEqual(len(ev), 5)

    def test_event_study_positive_car(self):
        a = self._burst_data()
        ev = detect_bursts(a, z_thresh=2.0, on="mentions")
        es = event_study(a, ev, pre=2, post=5, est=30)
        self.assertGreater(es["mean_car"], 0.02)
        self.assertLess(es["pvalue"], 0.05)


class TestPredict(unittest.TestCase):
    def test_ic_positive(self):
        a = make_pair()
        ic = information_coefficient(a, horizon=1)
        self.assertGreater(ic["ic"], 0.0)
        self.assertLess(ic["pvalue"], 0.05)

    def test_sign_accuracy_beats_coin(self):
        a = make_pair()
        sa = sign_accuracy(a, horizon=1)
        self.assertGreater(sa["accuracy"], 0.5)

    def test_forward_regression(self):
        a = make_pair()
        fr = forward_regression(a, horizon=1)
        self.assertGreater(fr["params"]["sentiment"], 0.0)
        self.assertLess(fr["pvalues"]["sentiment"], 0.05)


class TestIndicators(unittest.TestCase):
    def test_shapes(self):
        a = make_pair(n=100)
        self.assertEqual(len(sentiment_momentum(a)), 100)
        self.assertEqual(len(attention_zscore(a)), 100)
        self.assertEqual(len(sentiment_zscore(a)), 100)
        self.assertEqual(len(divergence(a)), 99)  # return-aligned
        comp = composite_attention(a)
        self.assertEqual(len(comp), 100)
        self.assertIn("divergence", comp[-1])

    def test_attention_flags_spike(self):
        a = make_pair(n=100)
        att = attention_zscore(a, window=20)
        self.assertTrue(all(isinstance(z, float) for z in att))


class TestSignals(unittest.TestCase):
    def test_zscore_rule_profitable_on_demo(self):
        a = make_pair(n=300)
        trades = zscore_rule(a, thresh=1.5, horizon=5)
        s = summarize_trades(trades)
        self.assertGreater(s["n_trades"], 5)
        self.assertGreater(s["mean_return"], 0.0)

    def test_summarize_empty(self):
        self.assertEqual(summarize_trades([])["n_trades"], 0)

    def test_attention_rule_runs(self):
        a = make_pair(n=200)
        s = summarize_trades(attention_rule(a, horizon=5))
        self.assertIn("hit_rate", s)


class TestOlsLite(unittest.TestCase):
    def test_perfect_fit(self):
        y = [2 + 3 * x for x in range(10)]
        r = ols_lite.ols(y, [[float(x)] for x in range(10)])
        self.assertAlmostEqual(r.params[0], 2.0, places=9)
        self.assertAlmostEqual(r.params[1], 3.0, places=9)
        self.assertAlmostEqual(r.rsquared, 1.0, places=9)


class TestAdapters(unittest.TestCase):
    def test_from_sentiment_scan_variants(self):
        rows = [{"date": "2026-01-01", "symbol": "A",
                 "bullishness_10": 7.0, "conviction_10": 6.0, "mentions": 5},
                {"date": "2026-01-02", "symbol": "A",
                 "bullishness": 4.0, "conviction": 3.0}]
        pts = from_sentiment_scan(rows)
        self.assertEqual(pts[0].bullishness, 7.0)
        self.assertEqual(pts[1].bullishness, 4.0)
        self.assertEqual(pts[1].mentions, 0)

    def test_prices_from_dict_bars(self):
        bars = prices_from_dict_bars([{"date": "2026-01-01", "Close": 100.0},
                                      {"Date": "2026-01-02", "c": 101.0}])
        self.assertEqual(len(bars), 2)
        self.assertAlmostEqual(bars[1].close, 101.0)

    def test_agent_report(self):
        sent, prices = demo_data(n_days=250, seed=11, symbol="R")
        rep = to_agent_report("R", sent, prices)
        self.assertGreater(rep["lead_lag"]["best_lag"], 0)
        self.assertEqual(rep["lead_lag"]["verdict"], "sentiment leads")
        self.assertIn("ic", rep)
        self.assertIn("event_study", rep)

    def test_backtest_signals(self):
        rows = to_backtest_signals("A", [{"date": "2026-01-01", "side": 1,
                                          "return": 0.02}])
        self.assertEqual(rows[0]["side"], "long")

    def test_sentiment_mimicking(self):
        import random
        rng = random.Random(3)
        dates = [f"d{i}" for i in range(20)]
        scores = {d: {f"s{i}": rng.gauss(0, 1) for i in range(9)}
                  for d in dates}
        # returns after date d load on scores AT date d (predictive)
        rets = {}
        for i, d in enumerate(dates):
            prev = dates[i - 1] if i > 0 else d
            rets[d] = {s: 0.01 * scores[prev][s] + rng.gauss(0, 0.005)
                       for s in scores[d]}
        fac = sentiment_mimicking_returns(scores, rets)
        self.assertEqual(len(fac), 19)
        self.assertGreater(sum(fac) / len(fac), 0.0)


class TestCLI(unittest.TestCase):
    def test_commands(self):
        for cmd in (["correlate", "--demo", "--symbol", "DEMO"],
                    ["events", "--demo", "--symbol", "DEMO"],
                    ["ic", "--demo", "--symbol", "DEMO"],
                    ["signals", "--demo", "--symbol", "DEMO"],
                    ["indicators", "--demo", "--symbol", "DEMO"],
                    ["report", "--demo", "--symbol", "DEMO"],
                    ["license"], ["update-check"]):
            self.assertEqual(cli_main(cmd), 0, cmd)


if __name__ == "__main__":
    unittest.main()
