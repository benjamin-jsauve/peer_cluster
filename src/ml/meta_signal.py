"""Combine AFML components into a final meta-labeled alpha signal."""

from __future__ import annotations
from pathlib import Path
import polars as pl
from loguru import logger

from src.signals.signal import load_signals
from src.ml.model import load_predictions
from src.ml.bet_sizing import bet_size_from_prob

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SIGNALS_PATH = PROJECT_ROOT / "data/processed/ml/ml_signals.parquet"


def compute_meta_signal() -> pl.DataFrame:
    """Build the meta-labeled signal using predictions and bet sizing."""
    signals = load_signals()
    preds = load_predictions()

    meta = (
        signals.join(
            preds.select(["rebal_date", "symbol", "prob_1", "prob_0", "prob_-1"]),
            on=["rebal_date", "symbol"],
            how="inner",
        )
        .with_columns(
            pl.when(pl.col("z_score") > 0)
              .then(pl.col("prob_1"))
              .when(pl.col("z_score") < 0)
              .then(pl.col("prob_-1"))
              .otherwise(0.5)
              .alias("prob_correct")
        )
    )

    prob = meta["prob_correct"].to_numpy()
    pred = meta["z_score"].sign().to_numpy()
    bet_scalar = bet_size_from_prob(meta["prob_correct"].to_numpy(), n_classes=3)

    meta = (
        meta
        .rename({"alpha": "alpha_gk"})
        .with_columns([
            pl.Series("bet_size", bet_scalar),
            (pl.col("alpha_gk") * pl.Series(bet_scalar)).alias("alpha_ml"),
        ])
        .select([
            "rebal_date", "symbol", "cluster",
            "z_score", "idio_vol",
            "alpha_gk", "prob_correct", "bet_size", "alpha_ml",
        ])
    )

    ML_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.write_parquet(ML_SIGNALS_PATH)
    logger.info(f"ML signals saved: {ML_SIGNALS_PATH}")

    return meta


def load_meta_signal() -> pl.DataFrame:
    if not ML_SIGNALS_PATH.exists():
        raise FileNotFoundError(f"ML signals not found at {ML_SIGNALS_PATH}.")
    return pl.read_parquet(ML_SIGNALS_PATH)


if __name__ == "__main__":
    compute_meta_signal()
