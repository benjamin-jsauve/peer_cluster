"""Statistical tests for signal and portfolio evaluation."""

from __future__ import annotations
import numpy as np
from scipy import stats


def newey_west_tstat(series: np.ndarray, max_lag: int = 6) -> float:
    """
    HAC t-statistic for the mean of a time series.

    Corrects for serial correlation up to max_lag using the
    Bartlett kernel (Newey & West 1987).
    """
    n      = len(series)
    mu     = series.mean()
    dm     = series - mu
    gamma0 = np.var(series, ddof=1)
    gamma  = sum(
        (1 - l / (max_lag + 1)) * np.mean(dm[l:] * dm[:-l])
        for l in range(1, max_lag + 1)
    )
    se = np.sqrt((gamma0 + 2 * gamma) / n)
    return float(mu / se) if se > 0 else np.nan


def icir(ic_series: np.ndarray) -> float:
    """Information Coefficient Information Ratio: mean(IC) / std(IC)."""
    std = ic_series.std(ddof=1)
    return float(ic_series.mean() / std) if std > 0 else np.nan


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: int = 12,
    n_boot: int = 5000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the annualised Sharpe ratio.

    Stationary block bootstrap with block length = sqrt(T).
    Returns (lower, upper) bounds of the ci-level interval.
    """
    T          = len(returns)
    block      = max(1, int(np.sqrt(T)))
    sharpes    = []
    rng        = np.random.default_rng(42)

    for _ in range(n_boot):
        starts = rng.integers(0, T, size=T // block + 1)
        sample = np.concatenate([
            returns[s : s + block] for s in starts
        ])[:T]
        mu  = sample.mean()
        sig = sample.std(ddof=1)
        if sig > 0:
            sharpes.append(mu / sig * np.sqrt(periods_per_year))

    sharpes = np.array(sharpes)
    lo = float(np.percentile(sharpes, (1 - ci) / 2 * 100))
    hi = float(np.percentile(sharpes, (1 + ci) / 2 * 100))
    return lo, hi


def psr(
    sr_hat: float,
    sr_star: float,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (Lopez de Prado 2012).

    Probability that the true Sharpe exceeds sr_star given an observed
    sr_hat over T periods.

    Parameters
    ----------
    sr_hat  : observed annualised Sharpe (already annualised)
    sr_star : benchmark Sharpe to test against (typically 0)
    T       : number of observations
    skew    : skewness of return distribution
    kurt    : excess kurtosis of return distribution
    """
    se = np.sqrt(
        (1 + (skew * sr_hat) + ((kurt - 1) / 4) * sr_hat**2) / (T - 1)
    )
    z  = (sr_hat - sr_star) / se
    return float(stats.norm.cdf(z))


def deflated_psr(
    sr_hat: float,
    sr_trials: np.ndarray,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Lopez de Prado & Lewis 2019).

    Adjusts PSR for the maximum expected Sharpe across N_trials
    independent strategy evaluations (multiple testing correction).

    Parameters
    ----------
    sr_hat    : observed Sharpe
    sr_trials : array of Sharpe ratios from all trials (including sr_hat)
    T         : number of observations
    """
    N      = len(sr_trials)
    sr_std = sr_trials.std(ddof=1)

    e      = np.euler_gamma
    sr_star = sr_std * (
        (1 - e) * stats.norm.ppf(1 - 1 / N)
        + e * stats.norm.ppf(1 - 1 / (N * np.e))
    )
    return psr(sr_hat, sr_star, T, skew, kurt)


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[L(e1)] = E[L(e2)] where L is squared error loss.
    Uses HAC variance estimation for the loss differential.

    Parameters
    ----------
    e1, e2 : forecast error series from two models
    h      : forecast horizon (for multi-step HAC correction)

    Returns
    -------
    (dm_stat, p_value)
    """
    d    = e1**2 - e2**2
    n    = len(d)
    dbar = d.mean()

    # HAC variance with h-1 lags
    gamma0 = np.var(d, ddof=1)
    gamma  = sum(
        (1 - k / h) * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        for k in range(1, h)
    ) if h > 1 else 0.0

    var_dbar = (gamma0 + 2 * gamma) / n
    dm_stat  = dbar / np.sqrt(var_dbar) if var_dbar > 0 else np.nan
    p_value  = float(2 * stats.norm.sf(abs(dm_stat)))

    return float(dm_stat), p_value


def factor_regression(
    port_returns: np.ndarray,
    factor_returns: np.ndarray,
    factor_names: list[str] | None = None,
) -> dict:
    """
    OLS regression of portfolio returns on factor returns.

    Returns alpha, betas, t-stats, and R-squared.
    Standard errors are Newey-West corrected with lag=6.

    Parameters
    ----------
    port_returns   : (T,) portfolio return series
    factor_returns : (T, K) factor return matrix
    factor_names   : list of K factor names (optional)
    """
    T, K   = factor_returns.shape
    X      = np.column_stack([np.ones(T), factor_returns])
    theta, _, _, _ = np.linalg.lstsq(X, port_returns, rcond=None)
    resid  = port_returns - X @ theta

    # Newey-West covariance of OLS estimator (6 lags)
    max_lag = 6
    S      = np.zeros((K + 1, K + 1))
    XTX_inv = np.linalg.inv(X.T @ X)
    for l in range(max_lag + 1):
        w    = 1.0 if l == 0 else 1 - l / (max_lag + 1)
        xu   = X[l:].T * resid[l:]
        xu0  = X[:T-l].T * resid[:T-l]
        gam  = xu @ xu0.T / T
        S   += w * (gam + gam.T) if l > 0 else w * gam

    cov    = T * XTX_inv @ S @ XTX_inv
    se     = np.sqrt(np.diag(cov))
    tstats = theta / se

    names = ["alpha"] + (factor_names or [f"f{i}" for i in range(K)])
    return {
        "params":   dict(zip(names, theta)),
        "se":       dict(zip(names, se)),
        "tstat":    dict(zip(names, tstats)),
        "r_squared": float(1 - resid.var() / port_returns.var()),
    }