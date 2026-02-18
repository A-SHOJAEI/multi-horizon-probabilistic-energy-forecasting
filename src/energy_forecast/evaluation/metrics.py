"""Evaluation metrics for probabilistic time series forecasting."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - targets)))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(predictions, targets)))


def crps_quantile(quantile_preds: np.ndarray, targets: np.ndarray, quantiles: List[float]) -> float:
    """Continuous Ranked Probability Score estimated from quantile forecasts.

    Args:
        quantile_preds: [N, pred_len, n_quantiles]
        targets: [N, pred_len]
        quantiles: list of quantile levels
    """
    n_q = len(quantiles)
    total_crps = 0.0

    for i in range(n_q):
        q = quantiles[i]
        errors = targets - quantile_preds[:, :, i]
        total_crps += np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))

    return float(total_crps / n_q * 2)


def coverage(quantile_preds: np.ndarray, targets: np.ndarray, lower_idx: int, upper_idx: int) -> float:
    """Compute prediction interval coverage.

    Args:
        quantile_preds: [N, pred_len, n_quantiles]
        targets: [N, pred_len]
        lower_idx: index of lower quantile
        upper_idx: index of upper quantile
    """
    lower = quantile_preds[:, :, lower_idx]
    upper = quantile_preds[:, :, upper_idx]
    covered = (targets >= lower) & (targets <= upper)
    return float(np.mean(covered))


def interval_width(quantile_preds: np.ndarray, lower_idx: int, upper_idx: int) -> float:
    """Mean prediction interval width."""
    lower = quantile_preds[:, :, lower_idx]
    upper = quantile_preds[:, :, upper_idx]
    return float(np.mean(upper - lower))


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float] = None,
) -> Dict:
    """Compute comprehensive forecasting metrics."""
    results = {}

    # Point metrics (use median if quantiles available, else direct)
    if predictions.ndim == 3 and quantiles:
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        point_preds = predictions[:, :, median_idx]
    else:
        point_preds = predictions

    results["mse"] = mse(point_preds, targets)
    results["mae"] = mae(point_preds, targets)
    results["rmse"] = rmse(point_preds, targets)

    # Probabilistic metrics
    if predictions.ndim == 3 and quantiles:
        results["crps"] = crps_quantile(predictions, targets, quantiles)

        # 80% interval (0.1 to 0.9)
        if 0.1 in quantiles and 0.9 in quantiles:
            lower_idx = quantiles.index(0.1)
            upper_idx = quantiles.index(0.9)
            results["coverage_80"] = coverage(predictions, targets, lower_idx, upper_idx)
            results["width_80"] = interval_width(predictions, lower_idx, upper_idx)

        # 50% interval (0.25 to 0.75)
        if 0.25 in quantiles and 0.75 in quantiles:
            lower_idx = quantiles.index(0.25)
            upper_idx = quantiles.index(0.75)
            results["coverage_50"] = coverage(predictions, targets, lower_idx, upper_idx)
            results["width_50"] = interval_width(predictions, lower_idx, upper_idx)

    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
