"""Statistical baseline models for time series forecasting."""

import numpy as np
from typing import Dict, List


class SeasonalNaive:
    """Seasonal Naive forecaster.

    Predicts by repeating the last seasonal period.
    """

    def __init__(self, season_len: int = 24):
        self.season_len = season_len
        self.history = None

    def fit(self, y: np.ndarray):
        """Store the last few seasonal periods."""
        self.history = y[-self.season_len * 4:]
        return self

    def predict(self, pred_len: int) -> np.ndarray:
        """Generate naive seasonal forecast."""
        preds = []
        for i in range(pred_len):
            idx = -(self.season_len - (i % self.season_len))
            preds.append(self.history[idx])
        return np.array(preds)


class ExponentialSmoothing:
    """Simple Exponential Smoothing with trend (Holt's method)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta

    def fit(self, y: np.ndarray):
        """Fit the model."""
        self.level = y[0]
        self.trend = y[1] - y[0]

        for t in range(1, len(y)):
            prev_level = self.level
            self.level = self.alpha * y[t] + (1 - self.alpha) * (prev_level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

        return self

    def predict(self, pred_len: int) -> np.ndarray:
        """Generate forecast."""
        preds = []
        for h in range(1, pred_len + 1):
            preds.append(self.level + h * self.trend)
        return np.array(preds)


def run_baseline_evaluation(
    train_y: np.ndarray,
    test_data: np.ndarray,
    seq_len: int,
    pred_len: int,
    target_idx: int = -1,
) -> Dict[str, Dict[str, float]]:
    """Run all baseline models on test data."""
    results = {}

    # Determine season length from dataset
    season_len = 24  # hourly data → daily seasonality

    all_y = test_data[:, target_idx]

    for name, model_class, kwargs in [
        ("seasonal_naive", SeasonalNaive, {"season_len": season_len}),
        ("exp_smoothing", ExponentialSmoothing, {}),
    ]:
        all_preds = []
        all_targets = []

        for i in range(len(test_data) - seq_len - pred_len + 1):
            context = test_data[i : i + seq_len, target_idx]
            target = test_data[i + seq_len : i + seq_len + pred_len, target_idx]

            model = model_class(**kwargs)
            model.fit(context)
            pred = model.predict(pred_len)

            all_preds.append(pred)
            all_targets.append(target)

        preds = np.stack(all_preds)
        targets = np.stack(all_targets)

        mse = float(np.mean((preds - targets) ** 2))
        mae = float(np.mean(np.abs(preds - targets)))

        results[name] = {"mse": mse, "mae": mae}

    return results
