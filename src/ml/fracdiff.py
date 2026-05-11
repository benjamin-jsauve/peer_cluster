"""Fractional differentiation utilities (AFML Ch. 5)."""

from __future__ import annotations
import numpy as np


def fracdiff_weights(d: float, size: int, thresh: float = 1e-5) -> np.ndarray:
    """Return fractional differencing weights up to a size or threshold."""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w[::-1])


def fracdiff_ffd(series: np.ndarray, d: float, thresh: float = 1e-5) -> np.ndarray:
    """Fixed-width fractional differentiation (AFML Snippet 5.3)."""
    if len(series) == 0:
        return series

    w = fracdiff_weights(d, len(series), thresh)
    width = len(w)
    out = np.full(len(series), np.nan, dtype=float)

    for i in range(width - 1, len(series)):
        window = series[i - width + 1 : i + 1]
        if np.any(~np.isfinite(window)):
            continue
        out[i] = float(np.dot(w, window))

    return out
