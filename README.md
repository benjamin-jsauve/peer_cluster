# Peer Cluster Residual Momentum

End-to-end toy factor strategy pipeline with optional ML meta-labeling. Cached outputs live in `data/processed/` so you do not have to recompute every run.

This repository is a simple end-to-end pipeline that turns raw price data into a monthly long/short portfolio. It removes broad market effects, groups similar stocks, builds a signal inside each group, and then picks weights while accounting for risk and trading costs. An optional ML step learns when to size the signal up or down. Outputs are cached in `data/processed/` so reruns are fast. At its core we use,

$$
r_t^i - r_t^f = \beta_t^{i,\top} f_t + \varepsilon_t^i
$$

$$
\alpha_t^i = \mathrm{IC} \cdot \hat{\sigma}_t^i \cdot z_t^i
$$

$$
w_t^* = \arg\max_w\; w^\top \alpha_t - \tfrac{\gamma}{2} w^\top \tilde{\Sigma}_t w - \lambda \lVert w - w_{t-1} \rVert_1
$$

## Quick start

1) Create environment and install deps

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Initialize folders

```bash
make init
```

3) Run the pipeline (uses cached outputs when available)

```bash
make run
make run-ml
```

4) Evaluate

```bash
make eval
make eval-ml
```

If you do not have `make` installed, use:

```bash
python -m src.pipeline run --run-id default
python -m src.pipeline run --run-id default --ml
python -m src.pipeline eval --run-id default
python -m src.pipeline eval --run-id default --ml
```

## Data

- `data/external/` contains static inputs like `iwb_seed_tickers.parquet` and `ff5m_daily.parquet`.
- `data/raw/` is for raw downloads before processing.
- `data/processed/` stores cached outputs. The pipeline will only recompute missing steps unless you pass `--force`.

## Models

Trained ML models are stored in `models/`. If an older model exists in `data/processed/ml/model.joblib`, it is automatically copied into `models/` on first use.

## References

Core references used in the report:

1) Bai, J. and Yao, J. (2012). On sample eigenvalues in a generalized spiked population model. *Annals of Statistics*, 40(3), 1466--1492.
2) Blitz, D., Hanauer, M. X., and Vidojevic, M. (2017). The idiosyncratic momentum anomaly. *Working paper, Erasmus University*.
3) Daniel, K. and Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221--247.
4) Devarakonda, P. et al. (2023). Meta-labeling: theory and framework. *Hudson & Thames Research*.
5) Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1--22.
6) Garleanu, N. and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance*, 68(6), 2309--2340.
7) Grinold, R. C. (1994). Alpha is volatility times IC times score. *Journal of Portfolio Management*, 20(4), 9--16.
8) Grinold, R. C. and Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill.
9) Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881--898.
10) Laloux, L., Cizeau, P., Bouchaud, J.-P., and Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters*, 83(7), 1467--1470.
11) Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
12) Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
13) Mantegna, R. N. (1999). Hierarchical structure in financial markets. *European Physical Journal B*, 11(1), 193--197.
14) Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703--708.

