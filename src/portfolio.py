"""Mean-variance building blocks and long-only optimizers (scipy SLSQP)."""
import numpy as np
from scipy.optimize import minimize


def portfolio_return(w, mu):
    return float(w @ mu)


def portfolio_variance(w, cov):
    return float(w @ cov @ w)


def portfolio_volatility(w, cov):
    return float(np.sqrt(portfolio_variance(w, cov)))


def sharpe_ratio(w, mu, cov, rf=0.0):
    return (portfolio_return(w, mu) - rf) / portfolio_volatility(w, cov)


def equal_weight(n):
    return np.full(n, 1.0 / n)


def _solve(objective, n, max_weight):
    """Minimize `objective` subject to w >= 0, w <= max_weight, sum(w) = 1."""
    res = minimize(objective, equal_weight(n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        return equal_weight(n)
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def min_volatility(cov, max_weight=1.0):
    cov = np.asarray(cov)
    return _solve(lambda w: portfolio_variance(w, cov), len(cov), max_weight)


def max_sharpe(mu, cov, rf=0.0, max_weight=1.0):
    """Tangency portfolio. If no asset is expected to beat rf, fall back to min-vol."""
    mu, cov = np.asarray(mu), np.asarray(cov)
    if mu.max() <= rf:
        return min_volatility(cov, max_weight)
    return _solve(lambda w: -sharpe_ratio(w, mu, cov, rf), len(mu), max_weight)
