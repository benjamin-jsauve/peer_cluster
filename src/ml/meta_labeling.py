"""Legacy meta-labeling wrapper (use meta_signal instead)."""

from __future__ import annotations
import polars as pl
from src.ml.meta_signal import compute_meta_signal, load_meta_signal


def compute_meta_signals() -> pl.DataFrame:
    return compute_meta_signal()


def load_meta_signals() -> pl.DataFrame:
    return load_meta_signal()


if __name__ == "__main__":
    compute_meta_signals()