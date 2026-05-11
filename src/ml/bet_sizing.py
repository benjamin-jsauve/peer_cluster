"""Bet sizing from predicted probabilities (AFML Ch. 10)."""

from __future__ import annotations
import numpy as np


def bet_size_from_prob(prob: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """
    Scale bet size by ML confidence. Maps [1/n_classes, 1] -> [0, 1].
    Direction comes from the primary signal, not from here.
    """
    base_rate = 1.0 / n_classes
    return np.clip((prob - base_rate) / (1.0 - base_rate), 0.0, 1.0)
