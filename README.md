# Toy Strategy

End-to-end toy factor strategy pipeline with optional ML meta-labeling. Cached outputs live in `data/processed/` so you do not have to recompute every run.

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

Place the PDF papers in `references/`. They are not included in this repository.

## Reports

Save figures for slides or writeups in `reports/figures/`.
