"""Walk-forward signal generation and a cost-aware monthly-rebalanced simulator."""
import numpy as np
import pandas as pd

from .black_litterman import black_litterman, implied_equilibrium_returns
from .models import WalkForwardRF, historical_mean_forecast
from .portfolio import equal_weight, max_sharpe, min_volatility

STRATEGIES = ["Equal Weight", "Min Volatility", "Historical Mean MaxSharpe",
              "ML Forecast MaxSharpe", "Black-Litterman + ML"]


def out_of_sample_dates(panel, rebalance_dates, min_train_months):
    """Rebalance dates with >= `min_train_months` of labelled history before them.

    The final month-end is excluded because it has no following holding period.
    """
    panel_dates = panel.index.get_level_values("date").unique()
    ok = [t for t in rebalance_dates[:-1] if (panel_dates < t).sum() >= min_train_months]
    return pd.DatetimeIndex(ok)


def generate_weights(prices, panel, oos_dates, cfg):
    """Compute every strategy's target weights at each rebalance date.

    Everything at date t is built from data up to t's close: the covariance and
    historical mean from the trailing `cov_window` daily returns ending at t, and
    the ML forecast from a model fit on labels realized by t. Weights are then
    held from the next trading day onward (see `simulate`).
    """
    daily = prices.pct_change().dropna()
    assets = list(prices.columns)
    n = len(assets)
    w_eq = equal_weight(n)
    model = WalkForwardRF(**cfg["rf"])
    weights = {s: [] for s in STRATEGIES}
    forecast_rows, bl_rows = [], []

    for t in oos_dates:
        hist = daily.loc[:t].iloc[-cfg["cov_window"]:]
        assert hist.index[-1] <= t  # nothing after the rebalance date
        cov = hist.cov().values * 252                          # annualized covariance
        mu_hist = historical_mean_forecast(hist, horizon=21)    # monthly, baseline
        ml = model.predict(panel, t).reindex(assets)            # monthly, ML

        # Black-Litterman: equal-weight "market" prior + one absolute ML view per asset
        pi = implied_equilibrium_returns(cov, w_eq, cfg["risk_aversion"])
        q = ml.values * 12
        mu_bl = black_litterman(cov, pi, np.eye(n), q, cfg["tau"], cfg["view_confidence"])

        mw, rf = cfg["max_weight"], cfg["rf_rate"]
        targets = {
            "Equal Weight": w_eq,
            "Min Volatility": min_volatility(cov, mw),
            "Historical Mean MaxSharpe": max_sharpe(mu_hist.values * 12, cov, rf, mw),
            "ML Forecast MaxSharpe": max_sharpe(q, cov, rf, mw),
            "Black-Litterman + ML": max_sharpe(mu_bl, cov, rf, mw),
        }
        for s, w in targets.items():
            assert np.all(w >= -1e-9) and abs(w.sum() - 1) < 1e-6 and w.max() <= mw + 1e-6
            weights[s].append(pd.Series(w, index=assets, name=t))

        realized = panel.xs(t, level="date")["target"].reindex(assets)
        for a in assets:
            forecast_rows.append((t, a, ml[a], mu_hist[a], realized[a]))
        bl_rows.append(pd.DataFrame({"date": t, "asset": assets, "prior_pi": pi,
                                     "ml_view_q": q, "posterior_mu": mu_bl}))

    weights = {s: pd.DataFrame(v) for s, v in weights.items()}
    forecasts = pd.DataFrame(forecast_rows, columns=["date", "asset", "ml", "baseline", "realized"])
    return weights, forecasts, pd.concat(bl_rows, ignore_index=True), model


def simulate(target_weights, prices, cost_rate):
    """Daily out-of-sample returns for one strategy with proportional costs.

    At each rebalance date t_k (trading at t_k's close):
      turnover_k = sum_i |w_target_i - w_drifted_i|   (w_drifted = last weights after drift)
      cost_k     = turnover_k * cost_rate              (fraction of portfolio value)
    The target is then held over (t_k, t_{k+1}], weights drifting with prices.
    The first rebalance builds the portfolio from cash (turnover = 1).
    """
    daily = prices.pct_change()
    dates = target_weights.index
    ends = list(dates[1:]) + [prices.index[-1]]
    w_prev = np.zeros(target_weights.shape[1])
    gross, net, log = [], [], []

    for t, t_next in zip(dates, ends):
        w = target_weights.loc[t].values
        turnover = np.abs(w - w_prev).sum()
        cost = turnover * cost_rate
        period = daily.loc[(daily.index > t) & (daily.index <= t_next)]
        growth = (1 + period).cumprod().values          # per-asset growth since t
        value = growth @ w                               # portfolio value path (start = 1)
        r = np.diff(np.concatenate([[1.0], value])) / np.concatenate([[1.0], value[:-1]])
        r_net = r.copy()
        r_net[0] = (1 - cost) * (1 + r[0]) - 1           # cost paid at the rebalance
        gross.append(pd.Series(r, index=period.index))
        net.append(pd.Series(r_net, index=period.index))
        log.append((t, turnover, cost))
        w_prev = w * growth[-1] / value[-1]              # drifted weights at t_next

    log = pd.DataFrame(log, columns=["date", "turnover", "cost"]).set_index("date")
    return pd.concat(gross), pd.concat(net), log
