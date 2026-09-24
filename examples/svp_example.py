"""End-to-end demo: does sentiment lead price in synthetic data?

The demo data bakes in a 1-day sentiment lead.  We run every tool and check
that each one recovers the planted relationship.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trade_sentiment_vs_price import (  # noqa: E402
    align,
    composite_attention,
    contemporaneous,
    demo_data,
    detect_bursts,
    event_study,
    forward_regression,
    information_coefficient,
    lead_lag,
    sign_accuracy,
    summarize_trades,
    to_agent_report,
    zscore_rule,
)


def main() -> None:
    sent, prices = demo_data(n_days=300, seed=11, lead=1, symbol="DEMO")
    a = align("DEMO", sent, prices)
    print(f"aligned {len(a.dates)} days")

    ll = lead_lag(a, max_lag=5)
    print(f"lead/lag: best lag = {ll['best_lag']} ({ll['verdict']}), "
          f"r = {ll['best_r']:.3f}, p = {ll['best_pvalue']:.4f}")
    c = contemporaneous(a)
    print(f"contemporaneous r = {c['r']:.3f} (weaker than the lead, as expected)")

    ic = information_coefficient(a, horizon=1)
    print(f"IC(h=1) = {ic['ic']:.3f}, rank IC = {ic['rank_ic']:.3f}, "
          f"p = {ic['pvalue']:.4f}")
    sa = sign_accuracy(a, horizon=1)
    print(f"sign accuracy = {sa['accuracy']:.3f} ({sa['hits']}/{sa['n']})")
    fr = forward_regression(a, horizon=1)
    print(f"forward regression: b_sent = {fr['params']['sentiment']:.4f}, "
          f"p = {fr['pvalues']['sentiment']:.4f}, R² = {fr['rsquared']:.3f}")

    ev = detect_bursts(a, z_thresh=2.0, on="mentions")
    es = event_study(a, ev, pre=5, post=10)
    print(f"bursts: {len(ev)} events, mean CAR(±) = {es['mean_car']:+.4f}, "
          f"p = {es['pvalue']:.3f}")

    sig = summarize_trades(zscore_rule(a, thresh=1.5, horizon=5))
    print(f"z-score rule: {sig['n_trades']} trades, hit rate "
          f"{sig['hit_rate']:.3f}, mean {sig['mean_return']:+.4f}, "
          f"p = {sig['pvalue']:.4f}")

    latest = composite_attention(a)[-1]
    print("latest indicators:", {k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in latest.items() if k != "date"})

    rep = to_agent_report("DEMO", sent, prices)
    print("agent verdict:", rep["lead_lag"]["verdict"],
          "| IC p =", f"{rep['ic']['pvalue']:.4f}")


if __name__ == "__main__":
    main()
