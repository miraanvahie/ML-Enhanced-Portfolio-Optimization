"""Compact Black-Litterman: equilibrium prior + ML views -> posterior expected returns."""
import numpy as np


def implied_equilibrium_returns(cov, market_weights, risk_aversion=2.5):
    """Reverse optimization: pi = delta * Sigma * w_mkt (annualized)."""
    return risk_aversion * cov @ market_weights


def black_litterman(cov, pi, P, Q, tau=0.05, view_confidence=0.5):
    """Posterior mean of the Black-Litterman model.

    mu_BL = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q]

    View uncertainty follows He-Litterman, Omega = diag(P tau Sigma P') / c, where
    c = `view_confidence`. c = 1 trusts each view as much as the prior; c < 1 trusts
    the (noisy) ML views less and shrinks the posterior toward equilibrium.
    """
    cov, pi, P, Q = map(np.asarray, (cov, pi, P, Q))
    tau_cov = tau * cov
    omega = np.diag(np.diag(P @ tau_cov @ P.T)) / view_confidence
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(omega)
    A = tau_cov_inv + P.T @ omega_inv @ P
    b = tau_cov_inv @ pi + P.T @ omega_inv @ Q
    return np.linalg.solve(A, b)
