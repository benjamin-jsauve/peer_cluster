"""Load config and expose typed dataclasses."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@dataclass
class UniverseConfig:
    start: str
    end: str
    iwb_url: str
    seed_file: str
    ff5m_file: str
    min_market_cap_usd: int
    min_history_days: int
    universe_size: int


@dataclass
class PricesConfig:
    batch_size: int
    batch_delay: float
    winsor_threshold: float
    min_stocks_per_date: int
    prices_output: str
    returns_output: str
    market_cap_output: str


@dataclass
class ResidualsConfig:
    window: int
    min_obs: int
    output: str


@dataclass
class CovarianceConfig:
    window: int
    min_stocks: int
    max_stocks: int
    start: str
    end: str
    output_dir: str


@dataclass
class ClusteringConfig:
    k_target: int
    linkage_method: str
    output_dir: str


@dataclass
class SignalConfig:
    lookback: int
    skip: int
    ic: float
    start: str
    end: str
    output: str


@dataclass
class PortfolioConfig:
    gamma: float
    tc_lambda: float
    max_weight: float
    min_weight: float
    output: str


@dataclass
class MLLabelsConfig:
    t_max: int
    h: float


@dataclass
class MLFracdiffConfig:
    d_fixed: float
    thresh: float


@dataclass
class MLModelConfig:
    n_splits: int
    embargo: int
    n_estimators: int
    max_features: str
    min_samples_leaf: int
    random_state: int


@dataclass
class MLTuningConfig:
    n_estimators: list[int]
    max_features: list[str]
    min_samples_leaf: list[int]


@dataclass
class MLConfig:
    labels: MLLabelsConfig
    fracdiff: MLFracdiffConfig
    model: MLModelConfig
    tuning: MLTuningConfig


@dataclass
class BacktestConfig:
    start: str
    end: str


@dataclass
class Config:
    universe: UniverseConfig
    prices: PricesConfig
    residuals: ResidualsConfig
    covariance: CovarianceConfig
    clustering: ClusteringConfig
    signal: SignalConfig
    portfolio: PortfolioConfig
    ml: MLConfig
    backtest: BacktestConfig


def _load() -> Config:
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    return Config(
        universe=UniverseConfig(**raw["universe"]),
        prices=PricesConfig(**raw["prices"]),
        residuals=ResidualsConfig(**raw["residuals"]),
        covariance = CovarianceConfig(**raw["covariance"]),
        clustering=ClusteringConfig(**raw["clustering"]),
        signal=SignalConfig(**raw["signal"]),
        portfolio=PortfolioConfig(**raw["portfolio"]),
        ml=MLConfig(
            labels=MLLabelsConfig(**raw["ml"]["labels"]),
            fracdiff=MLFracdiffConfig(**raw["ml"]["fracdiff"]),
            model=MLModelConfig(**raw["ml"]["model"]),
            tuning=MLTuningConfig(**raw["ml"]["tuning"]),
        ),
        backtest=BacktestConfig(**raw["backtest"]),
    )


cfg = _load()