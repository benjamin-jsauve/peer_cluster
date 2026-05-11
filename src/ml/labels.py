"""Triple-barrier labels on residual paths."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger
from src.signals.residuals import load_residuals
from src.config import cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = PROJECT_ROOT / "data/processed/ml/labels.parquet"

T_MAX = cfg.ml.labels.t_max
H = cfg.ml.labels.h


def _daily_vol(idio_vol_annual: float) -> float:
    return idio_vol_annual / np.sqrt(252)


def label_observation(path: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> int:
    """Return the triple-barrier label for a residual path."""
    for i, r in enumerate(path):
        if r >= upper[i]:
            return 1
        if r <= lower[i]:
            return -1
    return 0


def compute_labels(h: float = H, t_max: int = T_MAX) -> pl.DataFrame:
    """Compute labels for all (symbol, rebal_date) pairs."""
    from src.signals.signal import load_signals

    residuals = load_residuals().sort(["symbol", "date"])
    signals = load_signals()

    all_dates = residuals["date"].unique().sort().to_list()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    records = []

    for row in signals.iter_rows(named=True):
        sym = row["symbol"]
        rebal_date = row["rebal_date"]
        idio_vol = row["idio_vol"]

        start_idx = date_to_idx.get(rebal_date)
        if start_idx is None or start_idx + t_max >= len(all_dates):
            continue

        path_dates = all_dates[start_idx + 1 : start_idx + t_max + 1]

        path = (
            residuals
            .filter(pl.col("symbol") == sym)
            .filter(pl.col("date").is_in(path_dates))
            .sort("date")["residual"]
            .to_numpy()
        )

        if len(path) < int(0.8 * t_max):
            continue

        cum_path = np.cumsum(path)
        vol_d = _daily_vol(idio_vol)
        upper = h * vol_d * np.sqrt(np.arange(1, len(cum_path) + 1))
        lower = -h * vol_d * np.sqrt(np.arange(1, len(cum_path) + 1))

        label = label_observation(cum_path, upper, lower)
        ret_fwd = float(cum_path[-1])

        records.append({
            "rebal_date": rebal_date,
            "symbol": sym,
            "label": label,
            "ret_fwd": ret_fwd,
        })

    labels = pl.DataFrame(records).sort(["rebal_date", "symbol"])

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(LABELS_PATH)

    logger.info(f"Labels saved: {LABELS_PATH}")

    return labels


def load_labels() -> pl.DataFrame:
    """Load labels from cache."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels not found at {LABELS_PATH}.")
    return pl.read_parquet(LABELS_PATH)


if __name__ == "__main__":
    compute_labels()