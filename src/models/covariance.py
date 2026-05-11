"""Marchenko-Pastur covariance cleaning."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from loguru import logger

from src.signals.residuals import load_residuals
from src.data.universe import build_dynamic_universe, get_rebalancing_dates
from src.data.prices import load_market_cap
from src.config import cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW = cfg.covariance.window
MIN_STOCKS = cfg.covariance.min_stocks
MAX_STOCKS = cfg.covariance.max_stocks
COV_DIR = PROJECT_ROOT / cfg.covariance.output_dir
START = cfg.covariance.start
END = cfg.covariance.end


def mp_upper_edge(q, sigma2=1.0):
    """Marchenko-Pastur upper spectral edge. q = N/T."""
    return sigma2 * (1 + q ** 0.5) ** 2


def mp_pdf(var, q, pts):
    """Analytical MP density. q = N/T."""
    eMin = var * (1 - q ** 0.5) ** 2
    eMax = var * (1 + q ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = ((1 / q) / (2 * np.pi * var * eVal)) * np.sqrt(
        np.maximum((eMax - eVal) * (eVal - eMin), 0)
    )
    return eVal, pdf


def fit_sigma2(eVal, q, bWidth=0.01):
    """
    Fit sigma2 by minimising SSE between empirical KDE and MP density.
    Applied to correlation matrix eigenvalues so sigma2 <= 1.
    Lopez de Prado (2018) Ch.2 fitKDE + errPDFs pattern, q = N/T convention.
    """
    def err(var):
        eVal_, theory = mp_pdf(var[0], q, len(eVal))
        kde = KernelDensity(kernel='gaussian', bandwidth=bWidth).fit(eVal.reshape(-1, 1))
        emp = np.exp(kde.score_samples(eVal_.reshape(-1, 1)))
        return np.sum((emp - theory) ** 2)

    res = minimize(err, x0=0.5, bounds=((1e-5, 1 - 1e-5),), method='L-BFGS-B')
    return float(res.x[0]) if res.success else float(np.mean(eVal))


def cov2corr(cov):
    """Derive correlation matrix from covariance matrix."""
    std = np.sqrt(np.diag(cov))
    std = np.where(std == 0, 1e-8, std)
    corr = (cov / std[:, None]) / std[None, :]
    np.fill_diagonal(corr, 1.0)
    return corr, std


def corr2cov(corr, std):
    """Recover covariance matrix from correlation matrix and std vector."""
    return (corr * std[:, None]) * std[None, :]


def clean_covariance(E):
    """
    Apply Marchenko-Pastur cleaning to the sample covariance matrix.

    Normalises to correlation, applies MP eigenvalue thresholding on the
    correlation matrix (where sigma2 <= 1 holds), replaces noise eigenvalues
    with their mean (Bai-Yao trace-preserving correction), then converts back.

    Parameters
    ----------
    E : np.ndarray (T, N)
        Residual matrix. Rows are dates, columns are stocks. Requires T > N.

    Returns
    -------
    np.ndarray (N, N) cleaned covariance matrix.
    """
    T, N = E.shape
    q = N / T

    Sigma = np.cov(E, rowvar=False)
    C, std = cov2corr(Sigma)
    eVal, eVec = np.linalg.eigh(C)

    sigma2 = fit_sigma2(eVal, q)
    threshold = mp_upper_edge(q, sigma2)

    noiseMask = eVal <= threshold
    noiseMean = float(eVal[noiseMask].mean()) if noiseMask.any() else 0.0
    eValCleaned = eVal.copy()
    eValCleaned[noiseMask] = noiseMean

    C_tilde = eVec @ np.diag(eValCleaned) @ eVec.T
    Sigma_tilde = corr2cov(C_tilde, std)

    logger.debug(
        f"N={N} T={T} q={q:.3f} sigma2={sigma2:.4f} "
        f"lambda+={threshold:.4f} signal={int((~noiseMask).sum())} "
        f"noise={int(noiseMask.sum())}"
    )
    return Sigma_tilde


def compute_covariance_snapshot(residuals, symbols, rebal_date, window=WINDOW):
    """
    Build cleaned covariance matrix for a given rebalancing date.

    Selects MAX_STOCKS stocks with the most complete residual history,
    fills missing values with 0, and applies MP cleaning.
    """
    window_resid = (
        residuals
        .filter(pl.col('symbol').is_in(symbols))
        .filter(pl.col('date') <= rebal_date)
        .sort('date')
        .group_by('symbol')
        .tail(window)
    )
    wide = (
        window_resid
        .pivot(index='date', on='symbol', values='residual')
        .sort('date')
    )

    allCols = [c for c in wide.columns if c != 'date']
    coverage = {c: wide[c].is_not_null().sum() for c in allCols}
    selected = sorted(allCols, key=lambda c: coverage[c], reverse=True)[:MAX_STOCKS]

    if len(selected) < MIN_STOCKS:
        logger.warning(f"{rebal_date}: only {len(selected)} stocks, skipping.")
        return None, []

    wide = wide.select(['date'] + selected).fill_null(0.0)
    E = wide.select(selected).to_numpy()
    T, N = E.shape

    if T < N:
        logger.warning(f"{rebal_date}: T={T} < N={N}, skipping.")
        return None, []

    return clean_covariance(E), selected


def save_covariance(Sigma, symbols, rebal_date):
    """
    Save cleaned covariance matrix as long-format Parquet.
    Schema: i | j | value
    """
    N = len(symbols)
    rows = {
        'i': [symbols[r] for r in range(N) for c in range(N)],
        'j': [symbols[c] for r in range(N) for c in range(N)],
        'value': [float(Sigma[r, c]) for r in range(N) for c in range(N)],
    }
    pl.DataFrame(rows).write_parquet(COV_DIR / f"{rebal_date}.parquet")


def load_covariance(rebal_date):
    """Load cleaned covariance matrix for a given rebalancing date."""
    path = COV_DIR / f"{rebal_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No covariance matrix for {rebal_date}.")

    df = pl.read_parquet(path)
    symbols = df['i'].unique().sort().to_list()
    N = len(symbols)
    idx = {s: i for i, s in enumerate(symbols)}
    Sigma = np.zeros((N, N))

    for row in df.iter_rows(named=True):
        Sigma[idx[row['i']], idx[row['j']]] = row['value']

    return Sigma, symbols


def compute_all_covariances(start=START, end=END, window=WINDOW):
    """
    Compute and save cleaned covariance matrices for all rebalancing dates.

    For each month-end date: determine universe, pull trailing residuals,
    apply MP cleaning, save to data/processed/covariance/{date}.parquet.
    Skips dates already computed. Safe to re-run.
    """
    COV_DIR.mkdir(parents=True, exist_ok=True)

    residuals = load_residuals()
    market_cap = load_market_cap()
    rebalancing_dates = get_rebalancing_dates(start, end)
    trading_days = sorted(market_cap['date'].unique().to_list())
    universe = build_dynamic_universe(market_cap, rebalancing_dates)

    logger.info(f"Computing covariance matrices for {len(rebalancing_dates)} dates.")

    for rebal_date in tqdm(rebalancing_dates, desc="Covariance cleaning"):
        path = COV_DIR / f"{rebal_date}.parquet"
        if path.exists():
            continue

        valid = [d for d in trading_days if d <= rebal_date]
        if not valid:
            continue
        lookup_date = valid[-1]

        symbols = universe.filter(pl.col('rebal_date') == rebal_date)['symbol'].to_list()

        if not symbols:
            symbols = (
                market_cap
                .filter(pl.col('date') == lookup_date)
                .filter(pl.col('n_obs') >= 252)
                .sort('market_cap', descending=True)
                .head(1000)['symbol'].to_list()
            )

        if not symbols:
            continue

        Sigma, valid_symbols = compute_covariance_snapshot(residuals, symbols, rebal_date, window)
        if Sigma is None:
            continue

        save_covariance(Sigma, valid_symbols, rebal_date)

    logger.info("Done.")

def detone(C, n_factors=1):
    """
    Remove the market component from a denoised correlation matrix.

    The detoned matrix is singular (one eigenvector removed) and cannot
    be used directly for mean-variance optimization. Use the denoised
    matrix (clean_covariance output) for the portfolio optimizer, and
    this detoned matrix for spectral clustering only.

    Parameters
    ----------
    C        : np.ndarray (N, N) denoised correlation matrix
    n_factors: int, number of market components to remove (default 1)

    Returns
    -------
    np.ndarray (N, N) detoned correlation matrix with unit diagonal.
    """
    eVal, eVec = np.linalg.eigh(C)

    eVec_mkt = eVec[:, -n_factors:]
    eVal_mkt = np.diag(eVal[-n_factors:])

    C_mkt = eVec_mkt @ eVal_mkt @ eVec_mkt.T
    C_detone = C - C_mkt

    d = np.sqrt(np.diag(C_detone))
    d = np.where(d == 0, 1e-8, d)
    C_detone = (C_detone / d[:, None]) / d[None, :]
    np.fill_diagonal(C_detone, 1.0)

    return C_detone

if __name__ == "__main__":
    compute_all_covariances()