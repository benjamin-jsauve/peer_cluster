"""Rolling factor-model residuals and idiosyncratic volatility."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import statsmodels.api as sm
from tqdm import tqdm
from loguru import logger
 
from src.data.prices import load_returns
from src.data.factors import load_factors, FACTOR_COLS
from src.config import cfg
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW = cfg.residuals.window
MIN_OBS = cfg.residuals.min_obs
RESIDUALS_PATH = PROJECT_ROOT / cfg.residuals.output
 
def compute_residuals(
    returns: pl.DataFrame | None = None,
    factors: pl.DataFrame | None = None,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
) -> pl.DataFrame:
    """Compute rolling FF5+M residuals and idiosyncratic volatility."""
    if returns is None:
        returns = load_returns()
    if factors is None:
        factors = load_factors()
 
    logger.info("Computing residuals")
    factor_dates = factors["date"].to_numpy().astype("datetime64[D]")
    factor_arrays = {
        col: factors[col].to_numpy()
        for col in FACTOR_COLS + ["rf"]
    }
 
    tickers = sorted(returns["symbol"].unique().to_list())
 
    all_records: list[dict] = []
 
    for ticker in tqdm(tickers, desc="Rolling OLS"):
        ticker_data = (
            returns
            .filter(pl.col("symbol") == ticker)
            .sort("date")
        )
 
        ticker_dates = ticker_data["date"].to_numpy().astype("datetime64[D]")
        ticker_returns = ticker_data["log_return"].to_numpy()
 
        date_idx = np.searchsorted(factor_dates, ticker_dates)
 
        valid_mask = (
            (date_idx < len(factor_dates)) &
            (factor_dates[np.minimum(date_idx, len(factor_dates) - 1)] == ticker_dates)
        )
 
        aligned_dates = ticker_dates[valid_mask]
        aligned_returns = ticker_returns[valid_mask]
        aligned_idx = date_idx[valid_mask]
 
        F = np.column_stack([
            factor_arrays[col][aligned_idx]
            for col in FACTOR_COLS
        ])
 
        rf = factor_arrays["rf"][aligned_idx]
 
        y = aligned_returns - rf
        T = len(y)
 
        for t in range(window, T + 1):
            y_window = y[t - window : t]
            F_window = F[t - window : t]
            valid = ~(np.isnan(y_window) | np.any(np.isnan(F_window), axis=1))
            n_valid = valid.sum()
 
            if n_valid < min_obs:
                continue
 
            X_window = sm.add_constant(F_window[valid], has_constant="add")
            y_clean = y_window[valid]
 
            try:
                model = sm.OLS(y_clean, X_window).fit()

                betas = model.params[1:]
                r_squared = model.rsquared
                if np.isnan(y_window[-1]) or np.any(np.isnan(F_window[-1])):
                    continue

                y_hat = model.params[0] + F_window[-1] @ betas
                residual = y_window[-1] - y_hat
                record_date = aligned_dates[t - 1].astype("datetime64[D]").item()
 
                all_records.append({
                    "date": record_date,
                    "symbol": ticker,
                    "residual": float(residual),
                    "beta_mkt": float(betas[0]),
                    "beta_smb": float(betas[1]),
                    "beta_hml": float(betas[2]),
                    "beta_rmw": float(betas[3]),
                    "beta_cma": float(betas[4]),
                    "beta_mom": float(betas[5]),
                    "r_squared": float(r_squared),
                })
 
            except Exception as e:
                logger.debug(f"OLS failed for {ticker} t={t}: {e}")
                continue
 
    logger.info(f"Residual records: {len(all_records):,}")
 
    residuals = pl.DataFrame(all_records)
 
    residuals = (
        residuals
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("residual")
            .rolling_std(window_size=window)
            .over("symbol")
            .alias("idio_vol")
        )
        .with_columns(
            (pl.col("idio_vol") * np.sqrt(252))
            .alias("idio_vol")
        )
        .filter(pl.col("idio_vol").is_not_null())
    )
 
    float_cols = [
        "residual",
        "beta_mkt",
        "beta_smb",
        "beta_hml",
        "beta_rmw",
        "beta_cma",
        "beta_mom",
        "r_squared",
        "idio_vol",
    ]
    residuals = residuals.with_columns([
        pl.col(c).round(6) for c in float_cols
    ])
 
    RESIDUALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    residuals.write_parquet(RESIDUALS_PATH)
 
    logger.info(f"Residuals saved: {RESIDUALS_PATH}")
 
    return residuals
 
def load_residuals() -> pl.DataFrame:
    """Load residuals from Parquet cache."""
    if not RESIDUALS_PATH.exists():
        raise FileNotFoundError(
            f"Residuals not found at {RESIDUALS_PATH}. "
            "Run compute_residuals() first."
        )
    return pl.read_parquet(RESIDUALS_PATH)
