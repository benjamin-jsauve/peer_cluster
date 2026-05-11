"""Within-cluster residual momentum signals."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger

from src.signals.residuals import load_residuals
from src.models.clustering import load_clusters
from src.data.universe import get_rebalancing_dates
from src.config import cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_PATH = PROJECT_ROOT / cfg.signal.output
LOOKBACK = cfg.signal.lookback
SKIP = cfg.signal.skip
START = cfg.signal.start
END = cfg.signal.end


def compute_signals(
    start: str = START,
    end: str = END,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
) -> pl.DataFrame:
    """Compute signals."""
    residuals = load_residuals()
    rebalancing_dates = get_rebalancing_dates(start, end)

    from src.models.clustering import CLUSTERS_DIR
    available_dates = {p.stem for p in CLUSTERS_DIR.glob("*.parquet")}
    rebalancing_dates = [d for d in rebalancing_dates if str(d) in available_dates]

    logger.info(f"Computing signals for {len(rebalancing_dates)} rebalancing dates.")

    IC = cfg.signal.ic
    records = []

    for rebal_date in rebalancing_dates:
        try:
            clusters_df = load_clusters(str(rebal_date))
        except FileNotFoundError:
            continue

        symbols = clusters_df["symbol"].to_list()

        window = (
            residuals
            .filter(pl.col("symbol").is_in(symbols))
            .filter(pl.col("date") <= rebal_date)
            .sort(["symbol", "date"])
            .with_columns(
                pl.col("date")
                  .rank("ordinal", descending=True)
                  .over("symbol")
                  .alias("rank_from_end")
            )
            .filter(pl.col("rank_from_end") > skip)
            .group_by("symbol")
            .tail(lookback)
        )

        latest_vol = (
            residuals
            .filter(pl.col("symbol").is_in(symbols))
            .filter(pl.col("date") <= rebal_date)
            .sort("date")
            .group_by("symbol")
            .agg(pl.col("idio_vol").last())
        )

        summary = (
            window
            .group_by("symbol")
            .agg([
                pl.col("residual").sum().alias("cum_residual"),
                pl.col("residual").count().alias("n_obs"),
            ])
            .filter(pl.col("n_obs") >= int(0.8 * lookback))
            .join(latest_vol, on="symbol", how="inner")
            .join(clusters_df, on="symbol", how="inner")
        )

        if len(summary) < 10:
            continue

        summary = (
            summary
            .with_columns([
                pl.col("cum_residual").mean().over("cluster").alias("cluster_mean"),
                pl.col("cum_residual").std().over("cluster").alias("cluster_std"),
            ])
            .with_columns(
                pl.when(pl.col("cluster_std") > 0)
                  .then((pl.col("cum_residual") - pl.col("cluster_mean")) / pl.col("cluster_std"))
                  .otherwise(0.0)
                  .alias("z_score")
            )
            .with_columns(
                (IC * pl.col("idio_vol") * pl.col("z_score")).alias("alpha")
            )
            .with_columns(pl.lit(rebal_date).alias("rebal_date"))
            .select(["rebal_date", "symbol", "cluster", "z_score", "idio_vol", "alpha"])
        )

        records.append(summary)

    if not records:
        raise RuntimeError("No signal records produced.")

    signals = pl.concat(records).sort(["rebal_date", "symbol"])
    float_cols = ["z_score", "idio_vol", "alpha"]
    signals = signals.with_columns([pl.col(c).round(6) for c in float_cols])

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals.write_parquet(SIGNALS_PATH)
    logger.info(f"Signals saved: {SIGNALS_PATH}")
    return signals


def load_signals() -> pl.DataFrame:
    """Load signals from Parquet cache."""
    if not SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Signals not found at {SIGNALS_PATH}. Run compute_signals() first."
        )
    return pl.read_parquet(SIGNALS_PATH)


if __name__ == "__main__":
    compute_signals()