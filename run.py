"""End-to-end pipeline: data -> features -> RF forecast -> Black-Litterman ->
covariance -> optimization -> transaction costs -> out-of-sample backtest.

Usage:
    python run.py                 # uses data/prices.csv if present
    python run.py --refresh       # force a fresh yfinance download
    python run.py --cost-bps 20   # change the proportional cost rate
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.backtest import STRATEGIES, generate_weights, out_of_sample_dates, simulate
from src.data import load_prices, month_end_dates, synthetic_prices
from src.evaluation import make_plots, ml_evaluation, performance_metrics
from src.features import FEATURES, build_panel, compute_features

CONFIG = {
    # Mix of broad equity, international, bonds, gold and large single stocks.
    "tickers": ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "AAPL", "MSFT", "JPM", "XOM"],
    "start": "2012-01-01",
    "end": "2026-07-01",
    "cov_window": 252,          # trailing trading days for covariance / historical mean
    "min_train_months": 36,     # labelled months required before the first forecast
    "risk_aversion": 2.5,       # delta in pi = delta * Sigma * w_mkt
    "tau": 0.05,                # scales prior uncertainty tau * Sigma
    "view_confidence": 0.5,     # c in Omega = diag(P tau Sigma P') / c
    "max_weight": 0.40,         # per-asset cap on top of w >= 0, sum(w) = 1
    "rf_rate": 0.0,
    "cost_bps": 10,             # proportional cost per unit of turnover
    "robustness_cost_bps": [0, 10, 25, 50],
    "rf": {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 50,
           "retrain_every": 12, "random_state": 42},
}


def check_no_lookahead(prices, date):
    """Features at `date` must be identical whether or not later prices exist."""
    full = compute_features(prices)
    trunc = compute_features(prices.loc[:date])
    for f in FEATURES:
        pd.testing.assert_series_equal(full[f].loc[date], trunc[f].loc[date])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    parser.add_argument("--cost-bps", type=float, default=CONFIG["cost_bps"])
    parser.add_argument("--synthetic", action="store_true",
                        help="offline smoke test on simulated prices (not market data)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    cost_rate = args.cost_bps / 1e4
    RESULTS = args.results_dir
    os.makedirs(RESULTS, exist_ok=True)

    if args.synthetic:
        print("WARNING: using synthetic prices - results are NOT market results")
        prices = synthetic_prices(CONFIG["tickers"], CONFIG["start"], CONFIG["end"])
    else:
        prices = load_prices(CONFIG["tickers"], CONFIG["start"], CONFIG["end"],
                             refresh=args.refresh)
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} assets, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    rebal = month_end_dates(prices.index)
    panel = build_panel(prices, rebal)
    oos = out_of_sample_dates(panel, rebal, CONFIG["min_train_months"])
    print(f"Out-of-sample rebalances: {len(oos)} ({oos[0].date()} to {oos[-1].date()})")

    check_no_lookahead(prices, oos[len(oos) // 2])

    weights, forecasts, bl, model = generate_weights(prices, panel, oos, CONFIG)
    for t, last_train, n_rows in model.fit_log:
        assert last_train < t, "training data must predate the forecast date"
    print(f"Random Forest refits: {len(model.fit_log)} "
          f"(first on {model.fit_log[0][2]} rows, last on {model.fit_log[-1][2]} rows)")

    # ---- main backtest (net of costs) ----
    rows, nets, logs = {}, {}, {}
    for s in STRATEGIES:
        gross, net, log = simulate(weights[s], prices, cost_rate)
        rows[s] = performance_metrics(gross, net, log)
        nets[s], logs[s] = net, log
    assert len({(n.index[0], n.index[-1], len(n)) for n in nets.values()}) == 1  # same period
    perf = pd.DataFrame(rows).T
    perf.to_csv(f"{RESULTS}/performance.csv", float_format="%.4f")

    # ---- ML evaluation (independent of portfolio results) ----
    ml_table = ml_evaluation(forecasts)
    ml_table.to_csv(f"{RESULTS}/ml_metrics.csv", float_format="%.5f")
    forecasts.to_csv(f"{RESULTS}/oos_forecasts.csv", index=False, float_format="%.6f")

    # ---- Black-Litterman snapshot at the last rebalance ----
    last_bl = bl[bl["date"] == bl["date"].max()].set_index("asset").drop(columns="date")
    last_bl.to_csv(f"{RESULTS}/black_litterman_last.csv", float_format="%.4f")

    # ---- robustness: identical weights, different cost rates ----
    robust = {}
    for bps in CONFIG["robustness_cost_bps"]:
        for s in STRATEGIES:
            g, n, lg = simulate(weights[s], prices, bps / 1e4)
            m = performance_metrics(g, n, lg)
            robust[(s, f"{bps:g} bps")] = m["Sharpe Ratio (net)"]
            if bps == 0:
                assert np.allclose(g, n)
    robust = pd.Series(robust).unstack()[[f"{b:g} bps" for b in CONFIG["robustness_cost_bps"]]]
    robust = robust.loc[STRATEGIES]
    robust.to_csv(f"{RESULTS}/robustness_costs_sharpe.csv", float_format="%.3f")

    make_plots(nets, weights, forecasts, model.mean_importance(), RESULTS)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"\n=== Out-of-sample performance, net of {args.cost_bps:g} bps costs ===")
    print(perf.round(3).T.to_string())
    print("\n=== Forecast accuracy: next-month return (out-of-sample) ===")
    print(ml_table.round(4).to_string())
    print("\n=== Black-Litterman at last rebalance (annualized) ===")
    print(last_bl.round(4).to_string())
    print("\n=== Robustness: net Sharpe ratio by cost rate ===")
    print(robust.round(3).to_string())
    print(f"\nResults and plots saved in {RESULTS}/")


if __name__ == "__main__":
    main()
