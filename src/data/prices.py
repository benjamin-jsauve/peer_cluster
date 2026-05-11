"""Price, return, and market-cap panels."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import polars as pl
import yfinance as yf
from loguru import logger
from src.data.universe import get_seed_tickers
from src.config import cfg

BATCH_SIZE = cfg.prices.batch_size
BATCH_DELAY = cfg.prices.batch_delay
WINSOR_THRESHOLD = cfg.prices.winsor_threshold
MIN_STOCKS_PER_DATE = cfg.prices.min_stocks_per_date
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = PROJECT_ROOT / cfg.prices.prices_output
RETURNS_PATH = PROJECT_ROOT / cfg.prices.returns_output
MARKET_CAP_PATH = PROJECT_ROOT / cfg.prices.market_cap_output

def download_prices(
    tickers: list[str] | None = None,
    start: str = cfg.universe.start,
    end: str = cfg.universe.end,
) -> pl.DataFrame:
    """Download adjusted daily close prices and volume."""
    if tickers is None:
        tickers = get_seed_tickers()

    logger.info(f"Downloading prices for {len(tickers)} tickers")

    batches = [
        tickers[i : i + BATCH_SIZE]
        for i in range(0, len(tickers), BATCH_SIZE)
    ]

    logger.info(f"Batches: {len(batches)}")

    all_frames: list[pl.DataFrame] = []
    failed_tickers: list[str] = []

    for batch_idx, batch in enumerate(batches):
        logger.info(f"Batch {batch_idx + 1}/{len(batches)}")

        batch_df = _download_batch(batch, start, end)

        if batch_df is not None:
            all_frames.append(batch_df)
        else:
            failed_tickers.extend(batch)

        if batch_idx < len(batches) - 1:
            time.sleep(BATCH_DELAY)

    if not all_frames:
        raise RuntimeError(
            "No price data downloaded. "
            "Check internet connection and yfinance version."
        )

    if failed_tickers:
        logger.warning(
            f"{len(failed_tickers)} tickers failed to download: "
            f"{failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}"
        )
    prices_raw = pl.concat(all_frames)
    logger.info(f"Raw rows: {len(prices_raw):,}")
    prices_clean = _clean_prices(prices_raw)

    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices_clean.write_parquet(PRICES_PATH)

    logger.info(f"Prices saved: {PRICES_PATH}")

    return prices_clean


def _download_batch(
    tickers: list[str],
    start: str,
    end: str,
) -> pl.DataFrame | None:
    """Download one batch and return long-format data."""
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if raw.empty:
            logger.warning(f"Empty response for batch starting {tickers[0]}")
            return None

        frames = []

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_df = raw
                else:
                    ticker_df = raw[ticker]

                if ticker_df["Close"].isna().all():
                    continue

                frame = (
                    pl.from_pandas(ticker_df.reset_index())
                    .rename({c: c.lower() for c in pl.from_pandas(
                        ticker_df.reset_index()
                    ).columns})
                    .select([
                        pl.col("date").cast(pl.Date),
                        pl.lit(ticker).alias("symbol"),
                        pl.col("close").cast(pl.Float64),
                        pl.col("volume").cast(pl.Float64),
                    ])
                    .filter(pl.col("close").is_not_null())
                    .filter(pl.col("close") > 0)
                )

                frames.append(frame)

            except KeyError:
                continue

        if not frames:
            return None

        return pl.concat(frames)

    except Exception as e:
        logger.warning(f"Batch download failed: {e}")
        return None


def _clean_prices(prices: pl.DataFrame) -> pl.DataFrame:
    """Clean the raw price panel."""
    date_coverage = (
        prices
        .group_by("date")
        .agg(pl.col("close").is_not_null().sum().alias("n_valid"))
    )

    valid_dates = (
        date_coverage
        .filter(pl.col("n_valid") >= MIN_STOCKS_PER_DATE)
        .select("date")
    )

    prices = prices.join(valid_dates, on="date", how="inner")

    logger.info(f"Dates after filter: {prices['date'].n_unique()}")

    prices = (
        prices
        .sort(["symbol", "date"])
        .with_columns([
            pl.col("close")
            .fill_null(strategy="forward", limit=5)
            .over("symbol"),
            pl.col("volume")
            .fill_null(strategy="forward", limit=5)
            .over("symbol"),
        ])
        .filter(pl.col("close").is_not_null())
    )

    prices = prices.sort(["symbol", "date"])

    return prices


def compute_log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute daily log returns from adjusted close prices."""
    logger.info("Computing log returns...")

    returns = (
        prices
        .sort(["symbol", "date"])
        .with_columns([
            (pl.col("close").log(base=np.e) -
             pl.col("close").shift(1).log(base=np.e))
            .over("symbol")
            .alias("log_return")
        ])
        .filter(pl.col("log_return").is_not_null())
        .with_columns([
            pl.col("log_return")
            .clip(
                lower_bound=-WINSOR_THRESHOLD,
                upper_bound=WINSOR_THRESHOLD,
            )
            .alias("log_return")
        ])
        .select(["date", "symbol", "log_return"])
    )

    RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    returns.write_parquet(RETURNS_PATH)

    logger.info(f"Returns saved: {RETURNS_PATH}")

    return returns

def compute_market_cap(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute market cap and cumulative observation count per ticker."""
    logger.info("Fetching shares outstanding from yfinance...")

    tickers = prices["symbol"].unique().to_list()
    shares_records = []

    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            logger.info(
                f"Fetching shares outstanding: "
                f"{i}/{len(tickers)}..."
            )

        try:
            info = yf.Ticker(ticker).info
            shares = info.get("sharesOutstanding", None)

            if shares and shares > 0:
                shares_records.append(
                    {"symbol": ticker, "shares_outstanding": float(shares)}
                )
            else:
                # Fall back to impliedSharesOutstanding if available
                shares_impl = info.get("impliedSharesOutstanding", None)
                if shares_impl and shares_impl > 0:
                    shares_records.append(
                        {"symbol": ticker, "shares_outstanding": float(shares_impl)}
                    )
                else:
                    logger.warning(
                        f"No shares outstanding for {ticker} - "
                        f"will be excluded from market cap panel."
                    )

        except Exception as e:
            logger.warning(
                f"Failed to fetch shares for {ticker}: {e}"
            )

    shares_df = pl.DataFrame(shares_records)

    logger.info(f"Shares retrieved: {len(shares_df)}/{len(tickers)}")

    market_cap = (
        prices
        .join(shares_df, on="symbol", how="inner")
        .with_columns([
            (pl.col("close") * pl.col("shares_outstanding"))
            .alias("market_cap")
        ])
        .with_columns([
            pl.col("close")
            .is_not_null()
            .cum_sum()
            .over("symbol")
            .cast(pl.Int64)
            .alias("n_obs")
        ])
        .select(["date", "symbol", "market_cap", "n_obs"])
        .filter(pl.col("market_cap") > 0)
    )

    MARKET_CAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    market_cap.write_parquet(MARKET_CAP_PATH)

    logger.info(f"Market cap saved: {MARKET_CAP_PATH}")

    return market_cap


def load_prices() -> pl.DataFrame:
    """Load cleaned prices from Parquet cache."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Prices not found at {PRICES_PATH}. "
            f"Run download_prices() first."
        )
    return pl.read_parquet(PRICES_PATH)


def load_returns() -> pl.DataFrame:
    """Load log returns from Parquet cache."""
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"Returns not found at {RETURNS_PATH}. "
            f"Run compute_log_returns() first."
        )
    return pl.read_parquet(RETURNS_PATH)


def load_market_cap() -> pl.DataFrame:
    """Load market cap panel from Parquet cache."""
    if not MARKET_CAP_PATH.exists():
        raise FileNotFoundError(
            f"Market cap not found at {MARKET_CAP_PATH}. "
            f"Run compute_market_cap() first."
        )
    return pl.read_parquet(MARKET_CAP_PATH)


if __name__ == "__main__":
    prices = download_prices()
    compute_log_returns(prices)
    compute_market_cap(prices)

    print(f"\nMarket cap shape: {market_cap.shape}")
    print(market_cap.head(10))

    last_date = market_cap["date"].max()
    print(f"\nTop 10 by market cap on {last_date}:")
    print(
        market_cap
        .filter(pl.col("date") == last_date)
        .sort("market_cap", descending=True)
        .head(10)
        .select(["symbol", "market_cap"])
    )