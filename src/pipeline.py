"""Pipeline runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import importlib

import numpy as np
import polars as pl
from loguru import logger

from src.config import cfg


def _filter_dates(dates: list, start: str | None, end: str | None) -> list:
    filtered = dates
    if start is not None:
        filtered = [d for d in filtered if str(d) >= start]
    if end is not None:
        filtered = [d for d in filtered if str(d) <= end]
    return filtered


def _set_experiment_paths(run_id: str, use_ml: bool) -> None:
    base = Path("data") / "experiments" / run_id
    cfg.residuals.output = str(base / "residuals" / "residuals.parquet")
    cfg.covariance.output_dir = str(base / "covariance")
    cfg.clustering.output_dir = str(base / "clusters")
    cfg.signal.output = str(base / "signals" / "signals.parquet")
    weights_name = "weights_ml.parquet" if use_ml else "weights.parquet"
    cfg.portfolio.output = str(base / "portfolio" / weights_name)


def _reload_pipeline_modules() -> None:
    modules = [
        "src.signals.residuals",
        "src.models.covariance",
        "src.models.clustering",
        "src.signals.signal",
        "src.portfolio.portfolio",
    ]
    for name in modules:
        importlib.reload(importlib.import_module(name))


def _has_parquet(dir_path: Path) -> bool:
    return dir_path.exists() and any(dir_path.glob("*.parquet"))


def run_pipeline(
    start: str | None,
    end: str | None,
    run_id: str,
    use_ml: bool,
    force: bool,
) -> None:
    logger.info("Running pipeline")
    _set_experiment_paths(run_id, use_ml)
    _reload_pipeline_modules()

    from src.signals.residuals import compute_residuals
    from src.models.covariance import compute_all_covariances
    from src.models.clustering import compute_all_clusters
    from src.signals.signal import compute_signals
    from src.portfolio.portfolio import compute_portfolio

    residuals_path = Path(cfg.residuals.output)
    cov_dir = Path(cfg.covariance.output_dir)
    clusters_dir = Path(cfg.clustering.output_dir)
    signals_path = Path(cfg.signal.output)

    if force or not residuals_path.exists():
        compute_residuals()

    cov_start = start or cfg.covariance.start
    cov_end = end or cfg.covariance.end
    if force or not _has_parquet(cov_dir):
        compute_all_covariances(start=cov_start, end=cov_end)
    if force or not _has_parquet(clusters_dir):
        compute_all_clusters()

    sig_start = start or cfg.signal.start
    sig_end = end or cfg.signal.end
    if force or not signals_path.exists():
        compute_signals(start=sig_start, end=sig_end)

    if use_ml:
        from src.ml.labels import compute_labels, LABELS_PATH
        from src.ml.features import compute_features, FEATURES_PATH
        from src.ml.model import train, model_exists, PREDS_PATH
        from src.ml.meta_signal import compute_meta_signal, ML_SIGNALS_PATH

        if force or not LABELS_PATH.exists():
            compute_labels()
        if force or not FEATURES_PATH.exists():
            compute_features()
        if force or not (model_exists() and PREDS_PATH.exists()):
            train()
        if force or not ML_SIGNALS_PATH.exists():
            compute_meta_signal()
        ml_signals = pl.read_parquet(ML_SIGNALS_PATH)
        compute_portfolio(
            start=sig_start,
            end=sig_end,
            signals_df=ml_signals,
            alpha_col="alpha_ml",
        )
    else:
        compute_portfolio(start=sig_start, end=sig_end)

    logger.info("Pipeline complete")


def evaluate_portfolio(
    start: str | None,
    end: str | None,
    run_id: str,
    use_ml: bool,
) -> None:
    _set_experiment_paths(run_id, use_ml)
    _reload_pipeline_modules()

    from src.portfolio.portfolio import load_weights
    from src.backtest import compute_returns

    weights = load_weights()
    ret_df = compute_returns(weights, start=start, end=end)
    ret_arr = ret_df["return"].to_numpy()

    ann_ret = ret_arr.mean() * 12
    ann_vol = ret_arr.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum_log = np.cumsum(ret_arr)
    running_max = np.maximum.accumulate(cum_log)
    drawdown = cum_log - running_max
    max_dd = np.exp(drawdown.min()) - 1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    hit_rate = (ret_arr > 0).mean() if len(ret_arr) else 0.0

    start_s = str(ret_df["date"].min())
    end_s = str(ret_df["date"].max())

    logger.info("Performance")
    logger.info(f"Window: {start_s} to {end_s}")
    logger.info(f"Ann ret: {ann_ret:.2%}")
    logger.info(f"Ann vol: {ann_vol:.2%}")
    logger.info(f"Sharpe: {sharpe:.3f}")
    logger.info(f"Max DD: {max_dd:.2%}")
    logger.info(f"Calmar: {calmar:.3f}")
    logger.info(f"Hit rate: {hit_rate:.1%}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy strategy pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run full pipeline")
    run_p.add_argument("--start", type=str, default=None)
    run_p.add_argument("--end", type=str, default=None)
    run_p.add_argument("--run-id", type=str, default="default")
    run_p.add_argument("--ml", action="store_true", help="Use ML alpha")
    run_p.add_argument("--force", action="store_true", help="Recompute all steps")

    eval_p = sub.add_parser("eval", help="Evaluate portfolio returns")
    eval_p.add_argument("--start", type=str, default=None)
    eval_p.add_argument("--end", type=str, default=None)
    eval_p.add_argument("--run-id", type=str, default="default")
    eval_p.add_argument("--ml", action="store_true", help="Use ML weights")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        run_pipeline(args.start, args.end, args.run_id, args.ml, args.force)
    elif args.command == "eval":
        evaluate_portfolio(args.start, args.end, args.run_id, args.ml)


if __name__ == "__main__":
    main()
