"""Reusable plot functions for portfolio and signal analysis."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _fmt_year(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def cumulative_return(
    dates: list,
    returns_dict: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
    title: str = "Cumulative return",
) -> plt.Axes:
    """
    Cumulative log-return plot for one or more series.

    Parameters
    ----------
    dates         : list of datetime objects (length T)
    returns_dict  : {label: return_array} for each series to plot
    """
    ax = ax or plt.gca()
    for label, ret in returns_dict.items():
        ax.plot(dates, (np.exp(np.cumsum(ret)) - 1) * 100,
                linewidth=1.2, label=label)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    _fmt_year(ax)
    return ax


def drawdown(
    dates: list,
    returns: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Drawdown",
) -> plt.Axes:
    """Drawdown area plot from a log-return series."""
    ax  = ax or plt.gca()
    cum = np.cumsum(returns)
    dd  = np.exp(cum - np.maximum.accumulate(cum)) - 1
    ax.fill_between(dates, dd * 100, 0, color="tomato", alpha=0.7)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    _fmt_year(ax)
    return ax


def rolling_sharpe(
    dates: list,
    returns: np.ndarray,
    window: int = 24,
    periods_per_year: int = 12,
    ax: plt.Axes | None = None,
    title: str = "Rolling Sharpe",
) -> plt.Axes:
    """Rolling annualised Sharpe ratio."""
    ax = ax or plt.gca()
    sr = [
        returns[max(0, i - window + 1):i + 1].mean()
        / returns[max(0, i - window + 1):i + 1].std(ddof=1)
        * np.sqrt(periods_per_year)
        if i >= window - 1 else np.nan
        for i in range(len(returns))
    ]
    ax.plot(dates, sr, linewidth=1.0)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Sharpe")
    ax.set_title(title)
    _fmt_year(ax)
    return ax


def ic_decay(
    horizons: list[int],
    ic_means: list[float],
    ic_stds: list[float],
    ax: plt.Axes | None = None,
    title: str = "IC decay by forecast horizon",
) -> plt.Axes:
    """IC mean +/- 1 std across forecast horizons."""
    ax     = ax or plt.gca()
    mu     = np.array(ic_means)
    sigma  = np.array(ic_stds)
    ax.bar(horizons, mu, alpha=0.7, width=0.6)
    ax.errorbar(horizons, mu, yerr=sigma, fmt="none", color="k",
                capsize=4, linewidth=1)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Mean IC")
    ax.set_title(title)
    return ax


def quintile_returns(
    quintile_means: np.ndarray,
    periods_per_year: int = 12,
    ax: plt.Axes | None = None,
    title: str = "Mean return by signal quintile",
) -> plt.Axes:
    """Bar chart of annualised returns by signal quintile (Q1=worst, Q5=best)."""
    ax  = ax or plt.gca()
    ann = quintile_means * periods_per_year * 100
    ax.bar(range(1, len(ann) + 1), ann,
           color=["tomato" if v < 0 else "steelblue" for v in ann],
           alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Signal quintile")
    ax.set_ylabel("Annualised return (%)")
    ax.set_title(title)
    ax.set_xticks(range(1, len(ann) + 1))
    ax.set_xticklabels([f"Q{i}" for i in range(1, len(ann) + 1)])
    return ax


def performance_table(
    summaries: dict[str, dict],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Render a performance comparison table as a matplotlib figure.

    Parameters
    ----------
    summaries : {strategy_name: summary_dict} where summary_dict
                is the output of src.utils.performance.summary_table
    """
    ax  = ax or plt.gca()
    ax.axis("off")

    rows  = list(summaries.keys())
    cols  = ["Ann Ret", "Ann Vol", "Sharpe", "Max DD", "Hit Rate"]
    fmts  = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]
    pcts  = {0, 1, 3, 4}

    cell_data = []
    for name in rows:
        s    = summaries[name]
        row  = []
        for i, f in enumerate(fmts):
            v = s.get(f, np.nan)
            row.append(f"{v:.1%}" if i in pcts else f"{v:.3f}")
        cell_data.append(row)

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    return ax