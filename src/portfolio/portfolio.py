"""Transaction-cost aware mean-variance portfolio."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import cvxpy as cp
from tqdm import tqdm
from loguru import logger

from src.config import cfg
from src.signals.signal import load_signals
from src.models.covariance import load_covariance

WEIGHTS_PATH = Path(cfg.portfolio.output)
GAMMA = cfg.portfolio.gamma
TC_LAMBDA = cfg.portfolio.tc_lambda
MAX_W = cfg.portfolio.max_weight
MIN_W = cfg.portfolio.min_weight


def solve_qp(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = GAMMA,
    lam: float = TC_LAMBDA,
    max_w: float = MAX_W,
    min_w: float = MIN_W,
) -> np.ndarray | None:
    """Solve the TC-aware mean-variance QP for one date."""
    N = len(alpha)
    w = cp.Variable(N)

    u = cp.Variable(N, nonneg=True)

    objective = cp.Maximize(
        alpha @ w
        - (gamma / 2) * cp.quad_form(w, Sigma)
        - lam * cp.sum(u)
    )

    constraints = [
        cp.sum(w) == 0,
        w >= min_w,
        w <= max_w,
        u >= w - w_prev,
        u >= w_prev - w,
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            return None

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None

    return w.value


def compute_portfolio(
    gamma: float = GAMMA,
    lam: float = TC_LAMBDA,
    max_w: float = MAX_W,
    min_w: float = MIN_W,
    start: str | None = None,
    end: str | None = None,
    signals_df: pl.DataFrame | None = None,
    alpha_col: str = "alpha",
) -> pl.DataFrame:
    """Compute and save weights for all rebalancing dates."""
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    signals = signals_df if signals_df is not None else load_signals()
    if alpha_col not in signals.columns:
        raise ValueError(f"Missing alpha column: {alpha_col}")
    if start is not None:
        signals = signals.filter(pl.col("rebal_date") >= pl.lit(start).str.to_date())
    if end is not None:
        signals = signals.filter(pl.col("rebal_date") <= pl.lit(end).str.to_date())
    dates = signals["rebal_date"].unique().sort().to_list()

    logger.info(f"Solving portfolio QP for {len(dates)} dates")

    records = []
    w_prev_map: dict[str, float] = {}

    for rebal_date in tqdm(dates, desc="Portfolio QP"):

        date_signals = (
            signals
            .filter(pl.col("rebal_date") == rebal_date)
            .select(["symbol", alpha_col])
            .drop_nulls()
        )

        try:
            Sigma, cov_symbols = load_covariance(str(rebal_date))
        except FileNotFoundError:
            logger.warning(f"{rebal_date}: no covariance matrix, skipping.")
            continue

        signal_symbols = set(date_signals["symbol"].to_list())
        cov_set = set(cov_symbols)
        common = sorted(signal_symbols & cov_set)

        if len(common) < 10:
            logger.warning(f"{rebal_date}: only {len(common)} stocks in common, skipping.")
            continue

        alpha_map = dict(zip(
            date_signals["symbol"].to_list(),
            date_signals[alpha_col].to_list(),
        ))
        alpha_vec = np.array([alpha_map[s] for s in common])

        cov_idx = {s: i for i, s in enumerate(cov_symbols)}
        idx = [cov_idx[s] for s in common]
        Sigma_sub = Sigma[np.ix_(idx, idx)]
        Sigma_sub = Sigma_sub * 252.0

        w_prev = np.array([w_prev_map.get(s, 0.0) for s in common])

        w_opt = solve_qp(alpha_vec, Sigma_sub, w_prev, gamma, lam, max_w, min_w)

        if w_opt is None:
            logger.warning(f"{rebal_date}: solver failed, carrying previous weights.")
            w_opt = w_prev

        for sym, wt in zip(common, w_opt):
            records.append({
                "rebal_date": rebal_date,
                "symbol": sym,
                "weight": float(wt),
            })

        w_prev_map = {sym: float(wt) for sym, wt in zip(common, w_opt)}

    weights = pl.DataFrame(records).sort(["rebal_date", "symbol"])

    float_cols = ["weight"]
    weights = weights.with_columns([pl.col(c).round(8) for c in float_cols])

    weights.write_parquet(WEIGHTS_PATH)
    logger.info(f"Weights saved: {WEIGHTS_PATH}")

    return weights


def load_weights() -> pl.DataFrame:
    """Load portfolio weights from Parquet cache."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS_PATH}. "
            "Run compute_portfolio() first."
        )
    return pl.read_parquet(WEIGHTS_PATH)


if __name__ == "__main__":
    compute_portfolio()