"""Feature importance utilities (AFML Ch. 8)."""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def mdi_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Mean Decrease in Impurity (MDI) feature importance."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no feature_importances_.")
    return dict(zip(feature_names, model.feature_importances_.tolist()))


def mda_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
) -> dict[str, float]:
    """Mean Decrease in Accuracy (MDA) feature importance."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rng = np.random.default_rng(42)

    base_scores = []
    perm_scores = {name: [] for name in feature_names}

    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])
        base_scores.append(roc_auc_score(y[test_idx], probs, multi_class="ovr"))

        for j, name in enumerate(feature_names):
            X_perm = X[test_idx].copy()
            rng.shuffle(X_perm[:, j])
            probs_perm = model.predict_proba(X_perm)
            score = roc_auc_score(y[test_idx], probs_perm, multi_class="ovr")
            perm_scores[name].append(score)

    base = float(np.mean(base_scores))
    return {k: base - float(np.mean(v)) for k, v in perm_scores.items()}
