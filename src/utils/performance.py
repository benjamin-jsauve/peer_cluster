"""Portfolio and signal performance metrics."""

from __future__ import annotations
import numpy as np
import polars as pl


def sharpe(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio."""
    mu  = returns.mean()
    sig = returns.std(ddof=1)
    return (mu / sig) * np.sqrt(periods_per_year) if sig > 0 else np.nan


def sortino(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    mu       = returns.mean()
    downside = returns[returns < 0]
    dd       = downside.std(ddof=1) if len(downside) > 1 else np.nan
    return (mu / dd) * np.sqrt(periods_per_year) if dd and dd > 0 else np.nan


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative log returns."""
    cum = np.cumsum(returns)
    dd  = cum - np.maximum.accumulate(cum)
    return float(np.exp(dd.min()) - 1)


def drawdown_duration(returns: np.ndarray) -> int:
    """Longest drawdown duration in number of periods."""
    cum       = np.cumsum(returns)
    peak      = np.maximum.accumulate(cum)
    in_dd     = cum < peak
    max_dur   = 0
    current   = 0
    for x in in_dd:
        current = current + 1 if x else 0
        max_dur = max(max_dur, current)
    return max_dur


def calmar(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Calmar ratio: annualised return / |max drawdown|."""
    ann = returns.mean() * periods_per_year
    mdd = max_drawdown(returns)
    return ann / abs(mdd) if mdd != 0 else np.nan


def turnover(weights: pl.DataFrame) -> float:
    """
    Mean monthly one-way turnover.

    Parameters
    ----------
    weights : pl.DataFrame with columns rebal_date | symbol | weight,
              sorted by rebal_date.
    """
    dates   = sorted(weights["rebal_date"].unique().to_list())
    to_list = []

    for i in range(1, len(dates)):
        prev = weights.filter(pl.col("rebal_date") == dates[i-1])
        curr = weights.filter(pl.col("rebal_date") == dates[i])
        joined = prev.join(curr, on="symbol", how="outer", suffix="_curr").fill_null(0.0)
        to_list.append(
            (joined["weight"] - joined["weight_curr"]).abs().sum() / 2
        )

    return float(np.mean(to_list)) if to_list else np.nan


def hit_rate(returns: np.ndarray) -> float:
    """Fraction of periods with positive return."""
    return float((returns > 0).mean())


def summary_table(returns: np.ndarray, periods_per_year: int = 12) -> dict:
    """Full performance summary as a dict."""
    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return {
        "ann_return":        ann_ret,
        "ann_vol":           ann_vol,
        "sharpe":            sharpe(returns, periods_per_year),
        "sortino":           sortino(returns, periods_per_year),
        "max_drawdown":      max_drawdown(returns),
        "drawdown_duration": drawdown_duration(returns),
        "calmar":            calmar(returns, periods_per_year),
        "hit_rate":          hit_rate(returns),
        "total_return":      float(np.exp(returns.sum()) - 1),
        "n_periods":         len(returns),
    }