"""Russell 1000 universe construction."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import requests
from loguru import logger

from src.config import cfg

IWB_URL = cfg.universe.iwb_url
MIN_MARKET_CAP_USD = cfg.universe.min_market_cap_usd
MIN_HISTORY_DAYS = cfg.universe.min_history_days
UNIVERSE_SIZE = cfg.universe.universe_size
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = PROJECT_ROOT / cfg.universe.seed_file

def download_iwb_seed() -> pl.DataFrame:
    """Download current IWB holdings and extract tickers."""

    logger.info("Downloading IWB holdings")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; research pipeline;"
            "contact: research@example.com)"
        )
    }

    response = requests.get(IWB_URL, headers=headers, timeout=30)
    response.raise_for_status()

    lines = response.text.splitlines()
    holdings_header_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped.lower().startswith("ticker"):
            holdings_header_idx = i

            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip():
                    holdings_header_idx = i
                    break
    
    if holdings_header_idx is None:
        preview = "\n".join(lines[:20])
        raise ValueError(
            f"Could not find holdings. See first 20 lines: \n{preview}"
        )

    logger.info(f"Holdings header at line {holdings_header_idx}")

    end_idx = len(lines)
    for i in range(holdings_header_idx + 1, len(lines)):
        if lines[i].strip() == "":
            end_idx = i
            break

    holdings_csv = "\n".join(lines[holdings_header_idx:end_idx])
    raw = pl.read_csv(
        io.StringIO(holdings_csv),
        infer_schema_length=500,
        ignore_errors=True,
    )

    logger.info(
        f"Parsed {len(raw)} rows with columns: {raw.columns}"
    )

    ticker_col = _find_column(raw.columns, keywords=["ticker"])
    name_col = _find_column(raw.columns, keywords=["name"], required=False)
    weight_col = _find_column(raw.columns, keywords=["weight"], required=False)
    asset_class_col = _find_column(
        raw.columns, keywords=["asset", "class"], required=False
    )

    if asset_class_col:
        raw = raw.filter(
            pl.col(asset_class_col).str.to_lowercase() == "equity"
        )
        logger.info(f"Equity rows: {len(raw)}")

    select_exprs = [
        pl.col(ticker_col)
        .str.strip_chars()
        .str.replace(r"\.", "-", literal=False)
        .alias("symbol"),
    ]

    if weight_col:
        select_exprs.append(
            pl.col(weight_col).cast(pl.Float64, strict=False).alias("weight_pct")
        )
    
    df = (
        raw
        .select(select_exprs)

        .filter(pl.col("symbol").is_not_null())
        .filter(pl.col("symbol").str.len_chars() > 1)
        .filter(pl.col("symbol") != "-")
        .filter(~pl.col("symbol").str.contains(r"^\d"))
        .unique(subset=["symbol"], keep="first")
        .sort("symbol")
    )

    logger.info(f"Seed tickers: {len(df)}")

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(SEED_PATH)

    logger.info(f"Seed saved: {SEED_PATH}")

    return df


def _find_column(
    columns: list[str],
    keywords: list[str],
    required: bool = True,
) -> str | None:
    """Find a column name that contains all keywords (case-insensitive)."""

    for col in columns:
        col_lower = col.lower()
        if all(kw.lower() in col_lower for kw in keywords):
            return col

    if required:
        raise ValueError(
            f"Could not find column matching keywords {keywords} "
            f"in columns: {columns}. "
            f"BlackRock may have changed the file format."
        )

    return None

def load_seed() -> pl.DataFrame:
    """Load cached seed tickers, downloading if needed."""
    if SEED_PATH.exists():
        logger.info(f"Loading seed from cache: {SEED_PATH}")
        return pl.read_parquet(SEED_PATH)

    logger.info("No cached seed found. Downloading from BlackRock...")
    return download_iwb_seed()


def get_seed_tickers() -> list[str]:
    """Return seed tickers as a list of strings."""
    return load_seed()["symbol"].to_list()


def build_dynamic_universe(
    market_cap: pl.DataFrame,
    rebalancing_dates: list,
    universe_size: int = UNIVERSE_SIZE,
    min_history_days: int = MIN_HISTORY_DAYS,
    min_market_cap: float = MIN_MARKET_CAP_USD,
) -> pl.DataFrame:
    """Select top market-cap names at each rebalancing date."""
    snapshots = []

    for rebal_date in rebalancing_dates:

        snapshot = (
            market_cap
            .filter(pl.col("date") == rebal_date)
            .filter(pl.col("market_cap") >= min_market_cap)
            .filter(pl.col("n_obs") >= min_history_days)
            .sort("market_cap", descending=True)
            .head(universe_size)
            .with_columns(
                pl.lit(rebal_date).alias("rebal_date"),
            )
            .with_columns(
                pl.int_range(1, pl.len() + 1).cast(pl.Int32).alias("rank")
            )
            .select(["rebal_date", "symbol", "market_cap", "rank"])
        )

        snapshots.append(snapshot)

    if not snapshots:
        raise ValueError(
            "No valid universe snapshots produced. "
            "Check that rebalancing_dates fall within the market_cap panel."
        )
    universe = pl.concat(snapshots)
    n_dates = universe["rebal_date"].n_unique()
    avg_size = len(universe) / n_dates

    logger.info(
        f"Universe built: {n_dates} dates, "
        f"avg {avg_size:.0f} stocks, "
        f"{universe['symbol'].n_unique()} tickers"
    )

    return universe

def get_rebalancing_dates(
    start: str,
    end: str,
) -> list:
    """Generate month-end rebalancing dates."""

    dates = (
        pl.date_range(
            pl.lit(start).str.to_date(),
            pl.lit(end).str.to_date(),
            interval="1mo",
            eager=True,
        )

        .dt.month_end()
        .to_list()
    )

    logger.info(f"Rebalancing dates: {len(dates)}")

    return dates

if __name__ == "__main__":
    tickers = get_seed_tickers()
    print(f"\nSeed tickers: {len(tickers)}")
    print(f"First 5:  {tickers[:5]}")
    print(f"Last 5:   {tickers[-5:]}")
    dates = get_rebalancing_dates(cfg.universe.start, cfg.universe.end)
    print(f"\nRebalancing dates: {len(dates)}")
    print(f"First: {dates[0]}")
    print(f"Last:  {dates[-1]}")