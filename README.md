# Multi-Horizon Probabilistic Energy Forecasting

Probabilistic time series forecasting on the **ETT (Electricity Transformer Temperature)** benchmarks using three model families — **PatchTST**, **Temporal Fusion Transformer (TFT)**, and **N-BEATS** — with **quantile regression** for prediction intervals and **CRPS** evaluation.

## Architecture

Three forecasting models, each producing quantile predictions (10th, 25th, 50th, 75th, 90th percentiles):

```
┌──────────────────────────────────────────────────────────────┐
│  PatchTST                                                    │
│  Input → Patch(16, stride=8) → Linear Embed → Positional Enc│
│  → 6-layer Transformer Encoder → Flatten → Quantile Heads   │
├──────────────────────────────────────────────────────────────┤
│  Temporal Fusion Transformer (SimpleTFT)                     │
│  Input → Variable Selection (softmax gating)                 │
│  → 2-layer LSTM → Multi-Head Self-Attention → Quantile Heads│
├──────────────────────────────────────────────────────────────┤
│  N-BEATS                                                     │
│  Input → 6 Generic Blocks (FC → backcast/forecast residual)  │
│  → Aggregated Forecast → Quantile Heads                      │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Patching (PatchTST) | 16-step patches, stride 8 | Captures local temporal patterns while reducing sequence length |
| Variable selection (TFT) | Softmax-gated feature weighting | Learns which input features matter most |
| Residual stacking (N-BEATS) | 6 generic blocks | Iterative signal decomposition via backcast/forecast |
| Output | 5 quantiles (0.1, 0.25, 0.5, 0.75, 0.9) | Full predictive distribution with 50% and 80% prediction intervals |
| Loss | Pinball / Quantile Loss | Proper scoring rule for probabilistic forecasts |
| Baselines | Seasonal Naive + Exponential Smoothing | Statistical baselines for sanity checking |

## Dataset

**ETT (Electricity Transformer Temperature)** — 4 real-world datasets from power transformers in China.

| Dataset | Granularity | Records | Features |
|---------|------------|---------|----------|
| ETTh1 | Hourly | 17,420 | 7 (6 power load features + oil temperature target) |
| ETTh2 | Hourly | 17,420 | 7 |
| ETTm1 | 15-minute | 69,680 | 7 |
| ETTm2 | 15-minute | 69,680 | 7 |

Splits: 60% train / 20% validation / 20% test. Lookback window: 336 steps.

## Results

Trained on a single NVIDIA RTX 4090. Prediction horizons: 24, 96, 336 steps.

### ETTh1 — MSE / MAE / CRPS (lower is better)

| Model | H=24 | H=96 | H=336 |
|-------|------|------|-------|
| **PatchTST** | **0.047** / **0.165** / **0.121** | 0.059 / 0.186 / 0.136 | 0.054 / **0.176** / **0.127** |
| TFT | 0.071 / 0.203 / — | 0.117 / 0.253 / — | 0.097 / 0.242 / 0.173 |
| N-BEATS | 0.176 / 0.333 / 0.234 | 0.105 / 0.260 / 0.180 | 0.147 / 0.295 / — |

### Coverage (80% prediction interval, target = 80%)

| Model | ETTh1 H=24 | ETTm1 H=24 |
|-------|------------|------------|
| PatchTST | 89.8% | 89.1% |
| TFT | — | — |
| N-BEATS | — | — |

### ETTm1 — MSE / MAE (best per horizon)

| Model | H=24 | H=96 | H=336 |
|-------|------|------|-------|
| **PatchTST** | **0.020** / **0.100** | **0.019** / **0.095** | **0.020** / **0.100** |
| TFT | 0.038 / 0.145 | 0.039 / 0.131 | 0.045 / 0.157 |
| N-BEATS | 0.091 / 0.233 | 0.049 / 0.160 | 0.061 / 0.186 |

## Installation

```bash
git clone https://github.com/A-SHOJAEI/multi-horizon-probabilistic-energy-forecasting.git
cd multi-horizon-probabilistic-energy-forecasting

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Download data

```bash
python scripts/download_data.py
```

Downloads ETTh1, ETTh2, ETTm1, ETTm2 CSV files (~25 MB total).

### Train

```bash
python scripts/train.py
```

Trains all 3 models across all 4 datasets and 3 prediction horizons (36 experiments total). Configuration in `configs/default.yaml`.

Key training settings:
- PatchTST: 100 epochs, lr=1e-4, AdamW
- TFT: 50 epochs, lr=1e-3
- N-BEATS: 50 epochs, lr=1e-3
- Early stopping (patience=10), mixed precision, gradient clipping

### Evaluate

```bash
python scripts/evaluate.py
```

Generates model comparison bar charts and training curve plots from saved results.

### Predict / forecast

```bash
python scripts/predict.py
```

Loads trained models and produces example forecasts with prediction intervals.

## Project Structure

```
├── configs/
│   └── default.yaml             # All hyperparameters
├── scripts/
│   ├── download_data.py         # ETT dataset download
│   ├── train.py                 # Multi-model, multi-dataset training pipeline
│   ├── evaluate.py              # Visualization of results
│   └── predict.py               # Forecast generation
├── src/energy_forecast/
│   ├── data/
│   │   ├── loader.py            # TimeSeriesDataset + DataLoader
│   │   └── preprocessing.py     # CSV loading, splitting, scaling
│   ├── models/
│   │   ├── patchtst.py          # PatchTST with quantile heads
│   │   ├── components.py        # N-BEATS + SimpleTFT
│   │   └── baselines.py         # Seasonal Naive + Exponential Smoothing
│   ├── training/
│   │   └── trainer.py           # Training loop (early stopping, AMP)
│   ├── evaluation/
│   │   ├── metrics.py           # MSE, MAE, RMSE, CRPS, coverage, interval width
│   │   └── analysis.py          # Plotting utilities
│   └── utils/
│       └── config.py            # YAML config loader
├── tests/
│   └── test_models.py           # Unit tests for all 3 models
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## References

- Nie, Y., et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR.
- Lim, B., et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* IJF.
- Oreshkin, B. N., et al. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.* ICLR.
- Zhou, H., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI.

## License

MIT
