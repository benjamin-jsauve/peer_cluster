"""Meta-labeling model with purged cross-validation."""

from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
from loguru import logger
from src.config import cfg
from src.ml.labels import load_labels
from src.ml.features import load_features, FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
LEGACY_MODEL_PATH = PROJECT_ROOT / "data/processed/ml/model.joblib"
PREDS_PATH = PROJECT_ROOT / "data/processed/ml/predictions.parquet"

N_SPLITS = cfg.ml.model.n_splits
EMBARGO = cfg.ml.model.embargo
FEAT_COLS = FEATURE_COLS


def purged_kfold_indices(dates: np.ndarray, n_splits: int, embargo: int):
    """Yield train/test indices with purging and embargo."""
    kf = KFold(n_splits=n_splits, shuffle=False)
    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)

    for _, test_date_idx in kf.split(unique_dates):
        test_dates = unique_dates[test_date_idx]
        t_start = test_dates.min()
        t_end = test_dates.max()

        # Embargo boundary: t_end + embargo days in the date array
        embargo_end_pos = np.searchsorted(unique_dates, t_end) + embargo
        embargo_end = unique_dates[min(embargo_end_pos, n_dates - 1)]

        test_mask = np.isin(dates, test_dates)
        purge_mask = (dates >= t_start) & (dates <= embargo_end)
        train_mask = ~purge_mask

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


def build_dataset():
    """Align features and labels."""
    features = load_features()
    labels = load_labels()

    data = (
        features.join(
            labels.select(["rebal_date", "symbol", "label"]),
            on=["rebal_date", "symbol"],
            how="inner",
        ).drop_nulls(subset=FEAT_COLS + ["label"])
    )

    dates = data["rebal_date"].cast(pl.Utf8).to_numpy()
    X = data.select(FEAT_COLS).to_numpy().astype(np.float32)
    y = data["label"].to_numpy()
    symbols = data.select(["rebal_date", "symbol"]).to_numpy()

    return X, y, dates, symbols, data


def _score_fold(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> float:
    return roc_auc_score(
        y_true, probs, multi_class="ovr", labels=list(range(len(classes)))
    )


def _grid_search(
    X: np.ndarray,
    y_enc: np.ndarray,
    dates: np.ndarray,
    classes: np.ndarray,
    n_splits: int,
    embargo: int,
) -> dict:
    grid = cfg.ml.tuning
    candidates = [
        {
            "n_estimators": n,
            "max_features": mf,
            "min_samples_leaf": msl,
        }
        for n in grid.n_estimators
        for mf in grid.max_features
        for msl in grid.min_samples_leaf
    ]

    best_params = None
    best_score = -np.inf

    for params in candidates:
        scores = []
        for train_idx, test_idx in purged_kfold_indices(dates, n_splits, embargo):
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_features=params["max_features"],
                min_samples_leaf=params["min_samples_leaf"],
                n_jobs=-1,
                random_state=cfg.ml.model.random_state,
            )
            clf.fit(X[train_idx], y_enc[train_idx])
            probs = clf.predict_proba(X[test_idx])
            scores.append(_score_fold(y_enc[test_idx], probs, classes))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        logger.info(f"Params {params} AUC={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        best_params = {
            "n_estimators": cfg.ml.model.n_estimators,
            "max_features": cfg.ml.model.max_features,
            "min_samples_leaf": cfg.ml.model.min_samples_leaf,
        }

    logger.info(f"Best params: {best_params}  AUC={best_score:.4f}")
    return best_params


def train(n_splits: int = N_SPLITS, embargo: int = EMBARGO) -> pl.DataFrame:
    """Train with purged cross-validation and save OOF predictions."""
    X, y, dates, symbols, data = build_dataset()

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    classes = le.classes_

    oof_probs = np.zeros((len(X), len(classes)))
    oof_mask = np.zeros(len(X), dtype=bool)

    auc_scores = []

    best_params = _grid_search(X, y_enc, dates, classes, n_splits, embargo)

    for fold, (train_idx, test_idx) in enumerate(
        purged_kfold_indices(dates, n_splits, embargo)
    ):
        clf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            min_samples_leaf=best_params["min_samples_leaf"],
            n_jobs=-1,
            random_state=cfg.ml.model.random_state,
        )
        clf.fit(X[train_idx], y_enc[train_idx])

        probs = clf.predict_proba(X[test_idx])
        oof_probs[test_idx] = probs
        oof_mask[test_idx] = True

        auc = _score_fold(y_enc[test_idx], probs, classes)
        auc_scores.append(auc)
        logger.info(f"Fold {fold + 1}/{n_splits} AUC={auc:.4f}")

    logger.info(f"Mean OOF AUC: {np.mean(auc_scores):.4f}")

    # Retrain on full dataset
    clf_full = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_features=best_params["max_features"],
        min_samples_leaf=best_params["min_samples_leaf"],
        n_jobs=-1,
        random_state=cfg.ml.model.random_state,
    )
    clf_full.fit(X, y_enc)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf_full, "label_encoder": le, "params": best_params}, MODEL_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")

    # Build predictions dataframe
    label_cols = {f"prob_{c}": oof_probs[:, i] for i, c in enumerate(classes)}
    preds = data.select(["rebal_date", "symbol", "label"]).with_columns([
        pl.Series(name=col, values=vals)
        for col, vals in label_cols.items()
    ])

    preds.write_parquet(PREDS_PATH)
    logger.info(f"Predictions saved: {PREDS_PATH}")

    return preds


def load_predictions() -> pl.DataFrame:
    """Load OOF predictions from cache."""
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Predictions not found at {PREDS_PATH}.")
    return pl.read_parquet(PREDS_PATH)


def _migrate_legacy_model() -> None:
    if MODEL_PATH.exists() or not LEGACY_MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_MODEL_PATH, MODEL_PATH)
    logger.info(f"Copied legacy model to {MODEL_PATH}")


def model_exists() -> bool:
    """Return True if a model exists, migrating from legacy path if needed."""
    _migrate_legacy_model()
    return MODEL_PATH.exists()


if __name__ == "__main__":
    train()