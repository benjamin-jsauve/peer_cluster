"""Fama-French factor data loader."""

from __future__ import annotations
from pathlib import Path
import polars as pl
import pandas_datareader.data as web
from loguru import logger

from src.config import cfg

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas_datareader")

FF5_DATASET = "F-F_Research_Data_5_Factors_2x3_Daily"
MOM_DATASET = "F-F_Momentum_Factor_Daily"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTORS_PATH = PROJECT_ROOT / cfg.universe.ff5m_file
FACTOR_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
RF_COL = "rf"

def download_ff5m(
    start: str = cfg.universe.start,
    end: str = cfg.universe.end,
) -> pl.DataFrame:
    """Download daily FF5 plus momentum and return a clean panel."""
    logger.info("Downloading FF5 factors")

    ff5_raw = web.DataReader(
        FF5_DATASET,
        "famafrench",
        start=start,
        end=end,
    )[0]

    logger.info(f"FF5 rows: {len(ff5_raw)}")

    mom_raw = web.DataReader(
        MOM_DATASET,
        "famafrench",
        start=start,
        end=end,
    )[0]

    logger.info(f"Momentum rows: {len(mom_raw)}")

    ff5_raw = ff5_raw / 100.0
    mom_raw = mom_raw / 100.0

    mom_raw.columns = [c.strip() for c in mom_raw.columns]
    ff5_raw.columns = [c.strip() for c in ff5_raw.columns]
    mom_col = [c for c in mom_raw.columns if c != "RF"][0]

    merged = ff5_raw.join(mom_raw[[mom_col]], how="inner")

    factors = (
        pl.from_pandas(merged.reset_index())

        .rename({
            "Date": "date",
            "Mkt-RF": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "RF": "rf",
            mom_col: "mom",
        })
        .with_columns(
            pl.col("date").cast(pl.Date)
        )
        .filter(
            pl.col("date").is_between(
                pl.lit(start).str.to_date(),
                pl.lit(end).str.to_date(),
            )
        )
        .select(["date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"])
        .sort("date")
    )

    _validate_factors(factors)

    FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    factors.write_parquet(FACTORS_PATH)

    logger.info(f"Factors saved: {FACTORS_PATH}")

    return factors


def _validate_factors(factors: pl.DataFrame) -> None:
    """
    Run basic sanity checks on the factor panel before saving.

    These checks catch common issues:
    - Still in percentage form (forgot to divide by 100)
    - Missing columns
    - Extreme values suggesting data errors

    Raises ValueError with a descriptive message if any check fails.
    """
    expected = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
    missing = [c for c in expected if c not in factors.columns]
    if missing:
        raise ValueError(
            f"Factor panel missing columns: {missing}. "
            f"Available: {factors.columns}"
        )

    for col in FACTOR_COLS:
        max_abs = factors[col].abs().max()
        if max_abs > 0.5:
            raise ValueError(
                f"Factor {col} has max absolute value {max_abs:.4f}. "
                f"Expected < 0.5. Factors may still be in percentage form."
            )

    null_counts = {
        col: factors[col].null_count()
        for col in FACTOR_COLS + [RF_COL]
    }
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        raise ValueError(
            f"Factor panel contains nulls: {null_counts}"
        )

    n_days = len(factors)
    logger.info(f"Factor validation passed: {n_days} days")


def load_factors() -> pl.DataFrame:
    """
    Load FF5+M factors from Parquet cache.

    Raises FileNotFoundError if factors have not been downloaded yet.
    """
    if not FACTORS_PATH.exists():
        raise FileNotFoundError(
            f"Factors not found at {FACTORS_PATH}. "
            f"Run download_ff5m() first."
        )
    return pl.read_parquet(FACTORS_PATH)

if __name__ == "__main__":
    download_ff5m()