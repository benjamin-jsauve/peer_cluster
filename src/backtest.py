"""Signal-agnostic backtester."""

from __future__ import annotations
import numpy as np
import polars as pl
from loguru import logger

from src.config import cfg
from src.portfolio.portfolio import compute_portfolio
from src.data.prices import load_returns


def compute_returns(
    weights: pl.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """Compute portfolio returns from weights and price returns."""
    returns = load_returns()
    dates = sorted(weights["rebal_date"].unique().to_list())
    if start is not None:
        dates = [d for d in dates if str(d) >= start]
    if end is not None:
        dates = [d for d in dates if str(d) <= end]

    records = []
    for i in range(len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]

        w = weights.filter(pl.col("rebal_date") == d).select(["symbol", "weight"])
        period_ret = (
            returns
            .filter(pl.col("date") > d)
            .filter(pl.col("date") <= d_next)
            .filter(pl.col("symbol").is_in(w["symbol"].to_list()))
            .group_by("symbol")
            .agg(pl.col("log_return").sum().alias("ret"))
            .drop_nulls()
        )

        merged = w.join(period_ret, on="symbol", how="inner")
        if len(merged) < 10:
            continue

        port_ret = float((merged["weight"] * merged["ret"]).sum())
        records.append({"date": d_next, "return": port_ret})

    return pl.DataFrame(records).sort("date")


def backtest(
    signals_df: pl.DataFrame,
    alpha_col: str = "alpha",
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """Run portfolio optimization and compute returns for a signal DataFrame."""
    weights = compute_portfolio(
        start=start,
        end=end,
        signals_df=signals_df,
        alpha_col=alpha_col,
    )
    ret_df = compute_returns(weights, start=start, end=end)
    logger.info(f"Backtest returns: {ret_df.shape}")
    return ret_df


def default_window() -> tuple[str, str]:
    return cfg.backtest.start, cfg.backtest.end
