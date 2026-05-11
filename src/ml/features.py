"""Feature matrix construction for the meta-labeling model (AFML Ch. 5)."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger

from src.config import cfg
from src.signals.signal import load_signals
from src.signals.residuals import load_residuals
from src.ml.fracdiff import fracdiff_ffd

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data/processed/ml/features.parquet"

FEATURE_COLS = [
    "fracdiff_resid",
    "cluster_coherence",
    "cluster_dispersion",
    "idio_vol",
    "dvol_1m",
    "dvol_3m",
    "z_score",
    "z_score_abs",
    "z_rank_within",
    "beta_mkt",
    "beta_smb",
    "beta_hml",
]


def _fracdiff_panel(residuals: pl.DataFrame, d: float, thresh: float) -> pl.DataFrame:
    residuals = residuals.sort(["symbol", "date"]).with_columns(
        pl.col("residual").cum_sum().over("symbol").alias("cum_resid")
    )

    def _apply(group: pl.DataFrame) -> pl.DataFrame:
        arr = group["cum_resid"].to_numpy()
        fd = fracdiff_ffd(arr, d=d, thresh=thresh)
        return group.with_columns(pl.Series(name="fracdiff_resid", values=fd))

    fd = (
        residuals
        .group_by("symbol")
        .map_groups(_apply)
        .select(["date", "symbol", "fracdiff_resid"])
    )
    return fd


def compute_features() -> pl.DataFrame:
    """Build the feature matrix aligned to (rebal_date, symbol)."""
    S = load_signals()
    R = load_residuals().sort(["symbol", "date"])

    logger.info("Computing fractional differentiation feature...")
    d = cfg.ml.fracdiff.d_fixed
    thresh = cfg.ml.fracdiff.thresh
    fd = _fracdiff_panel(R, d=d, thresh=thresh)

    logger.info("Computing volatility and beta features...")
    R_vol = R.with_columns([
        (pl.col("idio_vol") - pl.col("idio_vol").shift(22)).over("symbol").alias("dvol_1m"),
        (pl.col("idio_vol") - pl.col("idio_vol").shift(63)).over("symbol").alias("dvol_3m"),
    ]).select([
        "date", "symbol",
        "idio_vol", "dvol_1m", "dvol_3m",
        "beta_mkt", "beta_smb", "beta_hml",
    ])

    logger.info("Joining all features...")
    features = (
        S.join(fd, left_on=["rebal_date", "symbol"], right_on=["date", "symbol"], how="left")
         .join(R_vol, left_on=["rebal_date", "symbol"], right_on=["date", "symbol"], how="left")
         .with_columns(pl.col("z_score").abs().alias("z_score_abs"))
         .with_columns(
             (
                 pl.col("z_score").rank("ordinal").over(["rebal_date", "cluster"])
                 / pl.col("z_score").count().over(["rebal_date", "cluster"])
             ).alias("z_rank_within")
         )
         .with_columns(pl.col("z_score").sign().alias("z_sign"))
         .with_columns(
             pl.col("z_sign").mean().over(["rebal_date", "cluster"]).alias("cluster_mean_sign")
         )
         .with_columns(
             (pl.col("z_sign") * pl.col("cluster_mean_sign")).alias("cluster_coherence")
         )
         .with_columns(
             pl.col("z_score").std().over(["rebal_date", "cluster"]).alias("cluster_dispersion")
         )
         .select([
             "rebal_date", "symbol", "cluster",
             *FEATURE_COLS,
         ])
         .sort(["rebal_date", "symbol"])
    )

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(FEATURES_PATH)
    logger.info(f"Features: {features.shape} -> {FEATURES_PATH}")

    return features


def load_features() -> pl.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features not found at {FEATURES_PATH}.")
    return pl.read_parquet(FEATURES_PATH)


if __name__ == "__main__":
    compute_features()