"""VIX daily close prices, downloaded from yfinance."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import yfinance as yf
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIX_PATH     = PROJECT_ROOT / "data/external/vix.parquet"


def download_vix(start: str = "2005-01-01", end: str = "2024-12-31") -> pl.DataFrame:
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    pdf = raw[["Close"]].reset_index()
    if "Date" not in pdf.columns:
        # yfinance may name the index column differently; normalize to Date.
        pdf = pdf.rename(columns={pdf.columns[0]: "Date"})
    df = pl.from_pandas(pdf)
    cols = df.columns
    if "date" in cols:
        pass
    elif "Date" in cols:
        df = df.rename({"Date": "date"})
    else:
        df = df.rename({cols[0]: "date"})

    cols = df.columns
    if "vix" in cols:
        pass
    elif "Close" in cols:
        df = df.rename({"Close": "vix"})
    elif "close" in cols:
        df = df.rename({"close": "vix"})
    else:
        df = df.rename({cols[1]: "vix"})

    df = (
        df
        .with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
    )
    VIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(VIX_PATH)
    logger.info(f"VIX saved: {len(df)} rows -> {VIX_PATH}")
    return df


def load_vix() -> pl.DataFrame:
    if not VIX_PATH.exists():
        return download_vix()
    return pl.read_parquet(VIX_PATH)


def vix_features(vix: pl.DataFrame) -> pl.DataFrame:
    """
    Derive monthly VIX features aligned to rebalancing dates.

    Returns a DataFrame with columns:
        date | vix_level | vix_chg_1m | high_vol
    """
    return (
        vix
        .sort("date")
        .with_columns([
            pl.col("vix").shift(22).alias("vix_lag_1m"),
        ])
        .with_columns(
            (pl.col("vix") - pl.col("vix_lag_1m")).alias("vix_chg_1m")
        )
        .with_columns(
            (pl.col("vix") > 25).cast(pl.Float64).alias("high_vol")
        )
        .select(["date", "vix", "vix_chg_1m", "high_vol"])
        .rename({"vix": "vix_level"})
    )


if __name__ == "__main__":
    download_vix()